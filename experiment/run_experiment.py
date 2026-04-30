from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment.directions import (
    build_directions,
    choose_best_layer_actor_style,
    choose_best_layer_by_silhouette,
    collect_prompt_activations,
    compute_refusal_score_first_token,
)
from experiment.grpo import GRPOConfig, train_grpo
from experiment.io_utils import (
    PromptExample,
    load_alpaca_safe,
    load_harmful_split,
    load_orbench_split,
    load_safe_split,
    to_chat_prompts,
)
from experiment.reward import DirectionBundle, RewardConfig
from utils import build_chat_prompt


DEFAULT_REFUSAL_TOKENS = [
    "Sorry",
    "Unfortunately",
    "cannot",
    "unable",
]
DEFAULT_REFUSAL_PHRASES = [
    "i'm sorry",
    "i am sorry",
    "i apologize",
    "as an ai",
    "as a language model",
    "as an assistant",
    "i cannot",
    "i can't",
    "i am unable to",
    "i'm unable to",
    "i am not able to",
    "i'm not able to",
    "i'm just an ai",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run sb-GRPO experiment.")
    p.add_argument("--model_id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument(
        "--ref_device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Device placement for frozen reference model.",
    )
    p.add_argument("--harmful_train_path", default="data/splits/harmful_train.jsonl")
    p.add_argument("--orbench_train_path", default="data/splits/orbench_hard_train.jsonl")
    p.add_argument("--safe_train_path", default="data/splits/safe_train.jsonl")
    p.add_argument("--output_dir", default="experiment/artifacts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--safe_samples", type=int, default=600)
    p.add_argument("--direction_batch_size", type=int, default=8)
    p.add_argument(
        "--simple_layer",
        action="store_true",
        help="If set, use simple middle index over collected layers (overrides --layer-selector).",
    )
    p.add_argument(
        "--layer_selector",
        choices=["actor", "silhouette"],
        default="actor",
        help="Layer choice: pseudo-aware ACTOR-style score (default), or silhouette harm vs safe (ablation).",
    )
    p.add_argument("--orth_lambda", type=float, default=1.0)
    p.add_argument(
        "--use_raw_false_refusal_dir",
        action="store_true",
        help="Use raw w_hat instead of orthogonalized w_hat' in geometry reward.",
    )
    p.add_argument("--pseudo_refusal_threshold", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--clip_m", type=float, default=2.0)
    p.add_argument("--harmful_compliance_penalty", type=float, default=2.0)
    p.add_argument("--benign_refusal_penalty", type=float, default=0.2)
    p.add_argument("--grpo_group_size", type=int, default=4)
    p.add_argument("--grpo_steps", type=int, default=100)
    p.add_argument("--grpo_batch_prompts", type=int, default=4)
    p.add_argument("--grpo_max_new_tokens", type=int, default=128)
    p.add_argument("--grpo_max_prompt_length", type=int, default=1024)
    p.add_argument("--grpo_temperature", type=float, default=0.8)
    p.add_argument("--grpo_top_p", type=float, default=0.95)
    p.add_argument("--grpo_kl_coef", type=float, default=0.02)
    p.add_argument("--grpo_optimizer", choices=["adafactor", "adamw"], default="adafactor")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names for LoRA targets.",
    )
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--normalize_geom_reward", action="store_true")
    p.add_argument("--geom_reward_scale", type=float, default=1.0)
    p.add_argument(
        "--geom_token_pool",
        choices=["first_k", "last_k"],
        default="first_k",
    )
    p.add_argument("--geom_k_tokens", type=int, default=4)
    p.add_argument(
        "--lora_layer_window",
        type=int,
        default=-1,
        help="If >=0, apply LoRA only to layers within +/- this window around selected layer.",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        padding_side="left",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.generation_config.pad_token_id = tokenizer.pad_token_id
    policy.generation_config.eos_token_id = tokenizer.eos_token_id
    policy.gradient_checkpointing_enable()
    if hasattr(policy, "enable_input_require_grads"):
        policy.enable_input_require_grads()
    policy.config.use_cache = False

    policy.train()

    if args.ref_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--ref_device=cuda requested, but CUDA is unavailable.")
    ref_device_map = "auto" if args.ref_device == "auto" else {"": args.ref_device}
    ref_dtype = (
        torch.float16
        if args.ref_device in {"cuda", "auto"} and torch.cuda.is_available()
        else torch.float32
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=ref_dtype,
        device_map=ref_device_map,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    harmful_prompts = load_harmful_split(args.harmful_train_path)
    pseudo_prompts = load_orbench_split(args.orbench_train_path)
    if os.path.exists(args.safe_train_path):
        safe_prompts = load_safe_split(args.safe_train_path)
        if args.safe_samples > 0 and len(safe_prompts) > args.safe_samples:
            rng = random.Random(args.seed)
            rng.shuffle(safe_prompts)
            safe_prompts = safe_prompts[: args.safe_samples]
    else:
        safe_prompts = load_alpaca_safe(max_samples=args.safe_samples, seed=args.seed)

    harm_chat = to_chat_prompts(harmful_prompts, tokenizer)
    safe_chat = to_chat_prompts(safe_prompts, tokenizer)
    pseudo_chat = to_chat_prompts(pseudo_prompts, tokenizer)

    pseudo_rs = compute_refusal_score_first_token(
        model=ref_model,
        tokenizer=tokenizer,
        chat_prompts=pseudo_chat,
        refusal_tokens=DEFAULT_REFUSAL_TOKENS,
        batch_size=args.direction_batch_size,
    )
    keep_idx = (pseudo_rs > args.pseudo_refusal_threshold).nonzero().squeeze(-1).tolist()
    if not keep_idx:
        raise RuntimeError("No pseudo-harmful prompts passed refusal-score filter; lower threshold.")
    pseudo_chat_filtered = [pseudo_chat[i] for i in keep_idx]
    pseudo_prompts_filtered = [pseudo_prompts[i] for i in keep_idx]

    print(f"Pseudo filter: kept {len(pseudo_chat_filtered)} / {len(pseudo_chat)}")
    print(f"Pseudo RS stats: min={pseudo_rs.min().item():.2f}, "
        f"mean={pseudo_rs.mean().item():.2f}, "
        f"max={pseudo_rs.max().item():.2f}")

    harm_acts, layer_ids = collect_prompt_activations(
        model=ref_model,
        tokenizer=tokenizer,
        prompts=harm_chat,
        batch_size=args.direction_batch_size,
    )
    safe_acts, _ = collect_prompt_activations(
        model=ref_model,
        tokenizer=tokenizer,
        prompts=safe_chat,
        batch_size=args.direction_batch_size,
    )
    pseudo_acts, _ = collect_prompt_activations(
        model=ref_model,
        tokenizer=tokenizer,
        prompts=pseudo_chat_filtered,
        batch_size=args.direction_batch_size,
    )

    if args.simple_layer:
        local_idx = choose_best_layer_by_silhouette(
            harm_acts,
            safe_acts,
            simple_layer=True,
        )
        true_layer_idx = layer_ids[local_idx]
    elif args.layer_selector == "silhouette":
        local_idx = choose_best_layer_by_silhouette(
            harm_acts,
            safe_acts,
            simple_layer=False,
        )
        true_layer_idx = layer_ids[local_idx]
    else:
        true_layer_idx = choose_best_layer_actor_style(
            harm_acts,
            safe_acts,
            pseudo_acts,
            layer_ids=layer_ids,
        )

    layer_idx = true_layer_idx

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        if not target_modules:
            raise ValueError("--lora_target_modules must contain at least one module name.")

        lora_kwargs = {}
        if args.lora_layer_window >= 0:
            n_layers = policy.config.num_hidden_layers
            lo = max(0, layer_idx - args.lora_layer_window)
            hi = min(n_layers - 1, layer_idx + args.lora_layer_window)
            lora_kwargs["layers_to_transform"] = list(range(lo, hi + 1))
            lora_kwargs["layers_pattern"] = "layers"
            print(f"Applying LoRA only to layers {lo}..{hi} around selected layer {layer_idx}")

        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            **lora_kwargs,
        )
        policy = get_peft_model(policy, lora_cfg)
        policy.print_trainable_parameters()
    else:
        # Full fine-tune only a narrow window to avoid OOM.
        layers = policy.model.layers
        n_layers = len(layers)
        window = args.lora_layer_window if args.lora_layer_window >= 0 else 0
        lo = max(0, layer_idx - window)
        hi = min(n_layers - 1, layer_idx + window)

        for p in policy.parameters():
            p.requires_grad = False

        for i in range(lo, hi + 1):
            for p in layers[i].parameters():
                p.requires_grad = True

        print(f"Full fine-tuning only layers {lo}..{hi}; all other params frozen.")

    artifacts = build_directions(
        harm_acts=harm_acts,
        safe_acts=safe_acts,
        pseudo_acts=pseudo_acts,
        model_layer_idx=true_layer_idx,
        layer_ids=layer_ids,
        orth_lambda=args.orth_lambda,
    )
    false_refusal_dir_for_reward = (
        artifacts.false_refusal_dir
        if args.use_raw_false_refusal_dir
        else artifacts.false_refusal_ortho_dir
    )
    directions = DirectionBundle(
        layer_idx=artifacts.layer_idx,
        refusal_dir=artifacts.refusal_dir,
        false_refusal_dir=false_refusal_dir_for_reward,
    )

    benign_pool = safe_prompts + pseudo_prompts_filtered
    train_examples: list[PromptExample] = (
        [PromptExample(prompt=p, label="harmful", source="harmful_train") for p in harmful_prompts]
        + [PromptExample(prompt=p, label="benign", source="safe_or_pseudo") for p in benign_pool]
    )

    train_chat_prompts = [build_chat_prompt(ex.prompt, tokenizer=tokenizer) for ex in train_examples]
    train_labels = [ex.label for ex in train_examples]

    reward_cfg = RewardConfig(
        alpha=args.alpha,
        beta=args.beta,
        clip_m=args.clip_m,
        harmful_compliance_penalty=args.harmful_compliance_penalty,
        benign_refusal_penalty=args.benign_refusal_penalty,
        normalize_geom_reward=args.normalize_geom_reward,
        geom_reward_scale=args.geom_reward_scale,
    )
    grpo_cfg = GRPOConfig(
        group_size=args.grpo_group_size,
        max_new_tokens=args.grpo_max_new_tokens,
        max_prompt_length=args.grpo_max_prompt_length,
        temperature=args.grpo_temperature,
        top_p=args.grpo_top_p,
        kl_coef=args.grpo_kl_coef,
        optimizer=args.grpo_optimizer,
        lr=args.lr,
        steps=args.grpo_steps,
        batch_prompts=args.grpo_batch_prompts,
        geom_token_pool=args.geom_token_pool,
        geom_k_tokens=args.geom_k_tokens,
    )

    history = train_grpo(
        policy=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
        prompts=train_chat_prompts,
        labels=train_labels,
        cfg=grpo_cfg,
        reward_cfg=reward_cfg,
        directions=directions,
        refusal_phrases=DEFAULT_REFUSAL_PHRASES,
    )

    policy.eval()
    policy.config.use_cache = True
    policy.generation_config.use_cache = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    torch.save(
        {
            "layer_idx": artifacts.layer_idx,
            "refusal_dir": artifacts.refusal_dir.cpu(),
            "false_refusal_dir": artifacts.false_refusal_dir.cpu(),
            "false_refusal_ortho_dir": artifacts.false_refusal_ortho_dir.cpu(),
            "v_harm": artifacts.v_harm.cpu(),
            "v_safe": artifacts.v_safe.cpu(),
            "v_pseudo": artifacts.v_pseudo.cpu(),
        },
        os.path.join(out_dir, "directions.pt"),
    )
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "reward_cfg": asdict(reward_cfg),
                "grpo_cfg": asdict(grpo_cfg),
                "n_harmful": len(harmful_prompts),
                "n_safe": len(safe_prompts),
                "n_pseudo_all": len(pseudo_prompts),
                "n_pseudo_filtered": len(pseudo_prompts_filtered),
                "selected_layer": true_layer_idx,
                "layer_selector": (
                    "simple_middle" if args.simple_layer else args.layer_selector
                ),
                "false_refusal_dir_used_for_reward": (
                    "raw_w_hat" if args.use_raw_false_refusal_dir else "orthogonalized_w_hat_prime"
                ),
                "use_lora": args.use_lora,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_target_modules": args.lora_target_modules,
            },
            f,
            indent=2,
        )

    if args.save_model:
        model_dir = "policy_adapter" if args.use_lora else "policy"
        policy.save_pretrained(os.path.join(out_dir, model_dir))
        tokenizer.save_pretrained(os.path.join(out_dir, "policy"))

    print(f"Saved run artifacts to: {out_dir}")


if __name__ == "__main__":
    main()

