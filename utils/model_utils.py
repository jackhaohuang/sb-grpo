"""
Model loading and batched inference utilities for LLaMA 3-8B.

Supports three backends:
  - "hf"   : HuggingFace Transformers (default, works everywhere)
  - "vllm" : vLLM (fast, requires separate GPU allocation)
  - "4bit" : HF + bitsandbytes 4-bit quantisation (low VRAM path)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

# Default model id – swap for the instruct variant as needed
LLAMA3_8B_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Chat template roles
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def build_chat_prompt(
    user_message: str,
    system_prompt: Optional[str] = None,
    tokenizer=None,
) -> str:
    """Format a single user turn using LLaMA 3's chat template.

    Falls back to a minimal hand-written template when no tokenizer is given.
    """
    messages = []
    if system_prompt:
        messages.append({"role": ROLE_SYSTEM, "content": system_prompt})
    messages.append({"role": ROLE_USER, "content": user_message})

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Minimal fallback for LLaMA 3
    parts = ["<|begin_of_text|>"]
    for msg in messages:
        parts.append(
            f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
            f"{msg['content']}<|eot_id|>"
        )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------

class HFModel:
    """Thin wrapper around a HuggingFace causal LM for batched generation."""

    def __init__(
        self,
        model_id: str = LLAMA3_8B_MODEL_ID,
        device_map: str = "auto",
        load_in_4bit: bool = False,
        torch_dtype=None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info("Loading tokenizer: %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        logger.info("Loading model weights: %s", model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2" if _flash_attn_available() else "eager",
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompts: List[str], batch_size: int = 8) -> List[str]:
        """Generate responses for a list of already-formatted prompt strings."""
        all_responses: List[str] = []

        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.model.device)

            do_sample = self.temperature > 0
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Decode only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            new_ids = output_ids[:, input_len:]
            decoded = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            all_responses.extend(decoded)

        return all_responses


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------

class VLLMModel:
    """Wrapper around vLLM's LLM class for high-throughput offline inference."""

    def __init__(
        self,
        model_id: str = LLAMA3_8B_MODEL_ID,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.SamplingParams = SamplingParams

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        logger.info("Initialising vLLM engine: %s", model_id)
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
        )

    def generate(self, prompts: List[str], batch_size: int = 0) -> List[str]:
        """batch_size is ignored; vLLM handles scheduling internally."""
        sampling_params = self.SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return [out.outputs[0].text for out in outputs]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_model(
    backend: str = "hf",
    model_id: str = LLAMA3_8B_MODEL_ID,
    load_in_4bit: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
):
    """Instantiate a model with the requested backend.

    Args:
        backend: "hf" | "4bit" | "vllm"
        model_id: HuggingFace model repository ID.
        load_in_4bit: Force 4-bit quantisation (only relevant for "hf" backend).
        max_new_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature (0 = greedy).

    Returns:
        An HFModel or VLLMModel instance.
    """
    backend = backend.lower()

    if backend == "vllm":
        return VLLMModel(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    if backend in ("hf", "4bit"):
        return HFModel(
            model_id=model_id,
            load_in_4bit=(backend == "4bit" or load_in_4bit),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    raise ValueError(f"Unknown backend '{backend}'. Choose one of: hf, 4bit, vllm")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False
