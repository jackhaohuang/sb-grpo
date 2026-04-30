from __future__ import annotations

import modal


APP_NAME = "sb-grpo"
REPO_ROOT = "/root/sb_grpo"

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime",
        add_python="3.10",
    )
    .add_local_dir(
        "/accounts/projects/binyu/hao_huang/sb_grpo/experiment",
        remote_path=f"{REPO_ROOT}/experiment",
        copy=True,
        ignore=["__pycache__/"],
    )
    .add_local_dir(
        "/accounts/projects/binyu/hao_huang/sb_grpo/utils",
        remote_path=f"{REPO_ROOT}/utils",
        copy=True,
        ignore=["__pycache__/"],
    )
    .add_local_dir(
        "/accounts/projects/binyu/hao_huang/sb_grpo/data/splits",
        remote_path=f"{REPO_ROOT}/data/splits",
        copy=True,
    )
    .pip_install(
        "transformers>=4.40.0,<5",
        "accelerate>=0.29.0",
        "peft>=0.11.0",
        "datasets>=2.18.0",
        "tqdm>=4.66.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "sentencepiece>=0.1.99",
        "protobuf>=4.25.0",
        "huggingface_hub>=0.22.0",
        "wandb>=0.17.0",
    )
)

app = modal.App(APP_NAME)

cache_vol = modal.Volume.from_name("sb-grpo", create_if_missing=True)
results_vol = modal.Volume.from_name("sb-grpo-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 12,
    volumes={
        "/cache": cache_vol,
        "/accounts/projects/binyu/hao_huang/sb_grpo/results": results_vol,
    },
    secrets=[
        # Create these once with:
        # modal secret create huggingface HF_TOKEN=...
        # modal secret create wandb WANDB_API_KEY=...
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def run_experiment(
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    output_dir: str = "/accounts/projects/binyu/hao_huang/sb_grpo/results/trained_grpo",
    safe_samples: int = 600,
    grpo_steps: int = 100,
    grpo_batch_prompts: int = 4,
    grpo_group_size: int = 4,
    grpo_max_new_tokens: int = 128,
    grpo_max_prompt_length: int = 1024,
    grpo_optimizer: str = "adafactor",
    ref_device: str = "cpu",
    simple_layer: bool = False,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj",
    alpha: float = 0.1,
    beta: float = 0.1,
    clip_m: float = 2.0,
    benign_refusal_penalty: float = 0.2,
    lr: float = 1e-5,
    kl_coef: float = 0.02,
    save_model: bool = False,
    normalize_geom_reward: bool = False,
    geom_reward_scale: float = 1.0,
    geom_token_pool: str = "first_k",
    geom_k_tokens: int = 4,
    lora_layer_window: int = -1,
    pseudo_refusal_threshold: float = -6.0,
) -> None:
    import os
    import subprocess

    def ensure_output_dir(path: str) -> str:
        normalized = os.path.abspath(path)
        os.makedirs(normalized, exist_ok=True)
        return normalized

    def ensure_safe_layer_name(path: str) -> str:
        cleaned = path.rstrip("/")
        if not cleaned:
            return "safe-layer"
        base = os.path.basename(cleaned)
        if "safe-layer" in base:
            return cleaned
        return f"{cleaned}-safe-layer"

    env = os.environ.copy()
    env["HF_HOME"] = "/cache/huggingface"
    env["HF_HUB_CACHE"] = "/cache/huggingface/hub"
    env["HF_DATASETS_CACHE"] = "/cache/huggingface/datasets"
    env.setdefault("WANDB_PROJECT", "sb-grpo")
    env.setdefault("WANDB_MODE", "online")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    os.makedirs(env["HF_HUB_CACHE"], exist_ok=True)
    os.makedirs(env["HF_DATASETS_CACHE"], exist_ok=True)
    if simple_layer:
        output_dir = ensure_safe_layer_name(output_dir)
    output_dir = ensure_output_dir(output_dir)

    cmd = [
        "python",
        "-m",
        "experiment.run_experiment",
        "--model_id",
        model_id,
        "--harmful_train_path",
        "data/splits/harmful_train.jsonl",
        "--orbench_train_path",
        "data/splits/orbench_hard_train.jsonl",
        "--safe_train_path",
        "data/splits/safe_train.jsonl",
        "--output_dir",
        output_dir,
        "--safe_samples",
        str(safe_samples),
        "--grpo_steps",
        str(grpo_steps),
        "--grpo_batch_prompts",
        str(grpo_batch_prompts),
        "--grpo_group_size",
        str(grpo_group_size),
        "--grpo_max_new_tokens",
        str(grpo_max_new_tokens),
        "--grpo_max_prompt_length",
        str(grpo_max_prompt_length),
        "--grpo_optimizer",
        grpo_optimizer,
        "--ref_device",
        ref_device,
        "--alpha",
        str(alpha),
        "--beta",
        str(beta),
        "--clip_m", str(clip_m),
        "--benign_refusal_penalty", str(benign_refusal_penalty),
        "--lr", str(lr),
        "--grpo_kl_coef", str(kl_coef),
        "--geom_reward_scale", str(geom_reward_scale),
        "--geom_token_pool", geom_token_pool,
        "--geom_k_tokens", str(geom_k_tokens),
        "--lora_layer_window", str(lora_layer_window),
        "--pseudo_refusal_threshold", str(pseudo_refusal_threshold),
    ]
    if normalize_geom_reward:
        cmd.append("--normalize_geom_reward")
    if simple_layer:
        cmd.append("--simple_layer")
    if use_lora:
        cmd.extend(
            [
                "--use_lora",
                "--lora_r",
                str(lora_r),
                "--lora_alpha",
                str(lora_alpha),
                "--lora_dropout",
                str(lora_dropout),
                "--lora_target_modules",
                lora_target_modules,
            ]
        )
    if save_model:
        cmd.append("--save_model")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
    results_vol.commit()


@app.local_entrypoint()
def main(
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    output_dir: str = "/accounts/projects/binyu/hao_huang/sb_grpo/results/trained_grpo",
    safe_samples: int = 600,
    grpo_steps: int = 100,
    grpo_batch_prompts: int = 4,
    grpo_group_size: int = 4,
    grpo_max_new_tokens: int = 128,
    grpo_max_prompt_length: int = 1024,
    grpo_optimizer: str = "adafactor",
    ref_device: str = "cpu",
    simple_layer: bool = False,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj",
    alpha: float = 0.1,
    beta: float = 0.1,
    clip_m: float = 2.0,
    benign_refusal_penalty: float = 0.2,
    lr: float = 1e-5,
    kl_coef: float = 0.02,
    save_model: bool = False,
    normalize_geom_reward: bool = False,
    geom_reward_scale: float = 1.0,
    geom_token_pool: str = "first_k",
    geom_k_tokens: int = 4,
    lora_layer_window: int = -1,
    pseudo_refusal_threshold: float = -6.0,
) -> None:
    run_experiment.spawn(
        model_id=model_id,
        output_dir=output_dir,
        safe_samples=safe_samples,
        grpo_steps=grpo_steps,
        grpo_batch_prompts=grpo_batch_prompts,
        grpo_group_size=grpo_group_size,
        grpo_max_new_tokens=grpo_max_new_tokens,
        grpo_max_prompt_length=grpo_max_prompt_length,
        grpo_optimizer=grpo_optimizer,
        ref_device=ref_device,
        simple_layer=simple_layer,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        alpha=alpha,
        beta=beta,
        clip_m=clip_m,
        benign_refusal_penalty=benign_refusal_penalty,
        lr=lr,
        kl_coef=kl_coef,
        save_model=save_model,
        normalize_geom_reward=normalize_geom_reward,
        geom_reward_scale=geom_reward_scale,
        geom_token_pool=geom_token_pool,
        geom_k_tokens=geom_k_tokens,
        lora_layer_window=lora_layer_window,
        pseudo_refusal_threshold=pseudo_refusal_threshold,
    )
