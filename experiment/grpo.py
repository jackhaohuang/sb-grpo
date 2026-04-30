from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import Adafactor

from experiment.reward import DirectionBundle, RewardConfig, base_reward, geometry_reward
from experiment.reward import _contains_refusal_phrase



@dataclass
class GRPOConfig:
    group_size: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    kl_coef: float = 0.02
    lr: float = 1e-5
    grad_clip: float = 1.0
    steps: int = 200
    batch_prompts: int = 4
    max_prompt_length: int = 2048
    optimizer: str = "adafactor"
    clip_eps: float = 0.2
    geom_token_pool: str = "first_k"  # "first_k" or "last_k"
    geom_k_tokens: int = 4



@dataclass
class SampleRollout:
    prompt: str
    label: str
    completion: str
    reward: float
    advantage: float


def _last_non_pad_index(attn_mask: torch.Tensor) -> torch.Tensor:
    return attn_mask.sum(dim=1) - 1


def _pool_completion_hidden(
    layer_hidden: torch.Tensor,
    completion_mask: torch.Tensor,
    mode: str,
    k: int,
) -> torch.Tensor:
    """
    layer_hidden: [B, S, D]
    completion_mask: [B, S-1], mask for predicted tokens.
    Returns [B, D].
    """
    device = layer_hidden.device
    bsz = layer_hidden.shape[0]
    h_list = []

    # completion_mask[j] corresponds to token position j+1 in gen_ids/layer_hidden
    token_positions = completion_mask.nonzero(as_tuple=False)

    for b in range(bsz):
        pos = token_positions[token_positions[:, 0] == b][:, 1] + 1
        if pos.numel() == 0:
            pos = torch.tensor([layer_hidden.shape[1] - 1], device=device)

        if mode == "first_k":
            chosen = pos[:k]
        elif mode == "last_k":
            chosen = pos[-k:]
        else:
            raise ValueError(f"Unknown geom_token_pool: {mode}")

        h_list.append(layer_hidden[b, chosen].mean(dim=0))

    return torch.stack(h_list, dim=0)


def _get_causal_decoder_layers(model):
    """
    Return the ModuleList of transformer blocks for a CausalLM, including
    PEFT/LoRA-wrapped models (where base_model.model holds the inner LM).
    """
    try:
        from peft import PeftModel
    except ImportError:  # pragma: no cover
        PeftModel = ()  # type: ignore[assignment, misc]
    if isinstance(model, PeftModel):
        inner = model.get_base_model()
        for candidate in (inner, getattr(inner, "model", None)):
            if candidate is not None and hasattr(candidate, "layers"):
                return candidate.layers
    prefix = getattr(model, "base_model_prefix", "model")
    top = getattr(model, prefix, None)
    if top is not None and hasattr(top, "layers"):
        return top.layers
    if top is not None and hasattr(top, "model") and hasattr(top.model, "layers"):
        return top.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Unable to access transformer layers for policy model.")


def _decode_completions(tokenizer, gen_ids: torch.Tensor, input_len: int) -> list[str]:
    out = []
    for i in range(gen_ids.shape[0]):
        completion_ids = gen_ids[i, input_len:]
        out.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
    return out


def rollout_group(
    policy,
    ref_model,
    tokenizer,
    prompts: Sequence[str],
    labels: Sequence[str],
    cfg: GRPOConfig,
    reward_cfg: RewardConfig,
    directions: DirectionBundle,
    refusal_phrases: Sequence[str],
) -> tuple[list[SampleRollout], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    device = next(policy.parameters()).device
    ref_device = next(ref_model.parameters()).device
    toks = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_prompt_length,
    ).to(device)
    prompt_lens = toks["attention_mask"].sum(dim=1)

    do_sample = cfg.temperature > 0

    policy.eval()
    policy.config.use_cache = True
    with torch.no_grad():
        gen_ids = policy.generate(
            **toks,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=do_sample,
            temperature=cfg.temperature if do_sample else None,
            top_p=cfg.top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False,
        )
    policy.train()
    policy.config.use_cache = False

    input_len = toks["input_ids"].shape[1]
    completions = _decode_completions(tokenizer, gen_ids, input_len)

    #debug prints
    # if torch.rand((), device=device) < 0.05:
    #     for i in range(min(5, len(completions))):
    #         print("=" * 80)
    #         print("PROMPT:", prompts[i])
    #         print("LABEL:", labels[i])
    #         print("COMPLETION:", completions[i])
    #         print("REFUSED?", _contains_refusal_phrase(completions[i], refusal_phrases))
    #         print("BASE REWARD:", base_reward(completions[i], labels[i], refusal_phrases, reward_cfg))

    attn = (gen_ids != tokenizer.pad_token_id).long()

    # Keep gradients for policy forward, but capture only the target layer hidden state
    # instead of materializing hidden states for every transformer layer.
    hidden_capture: dict[str, torch.Tensor] = {}
    layers = _get_causal_decoder_layers(policy)

    def _capture_layer_output(_, __, output):
        hidden_capture["layer_hidden"] = output[0] if isinstance(output, tuple) else output

    hook = layers[directions.layer_idx].register_forward_hook(_capture_layer_output)
    try:
        pol_out = policy(
            input_ids=gen_ids,
            attention_mask=attn,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    finally:
        hook.remove()

    # ref model is frozen, so no gradients needed
    with torch.no_grad():
        ref_out = ref_model(
            input_ids=gen_ids.to(ref_device),
            attention_mask=attn.to(ref_device),
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )

    shift_labels = gen_ids[:, 1:]
    shift_mask = attn[:, 1:]
    prompt_token_mask = (
        torch.arange(shift_labels.shape[1], device=device).unsqueeze(0)
        < (input_len - 1)
    )
    completion_mask = shift_mask.bool() & (~prompt_token_mask)

    with torch.no_grad():
        old_out = policy(
            input_ids=gen_ids,
            attention_mask=attn,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    old_log_probs = old_out.logits[:, :-1, :].log_softmax(dim=-1).gather(
        -1, shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    shift_logits = pol_out.logits[:, :-1, :]
    pol_log_probs = shift_logits.log_softmax(dim=-1).gather(
        -1, shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    ref_log_probs = (
        ref_out.logits[:, :-1, :]
        .log_softmax(dim=-1)
        .gather(-1, shift_labels.to(ref_device).unsqueeze(-1))
        .squeeze(-1)
        .to(device)
    )

    layer_hidden = hidden_capture["layer_hidden"]
    h_geom = _pool_completion_hidden(
        layer_hidden=layer_hidden,
        completion_mask=completion_mask,
        mode=cfg.geom_token_pool,
        k=cfg.geom_k_tokens,
    )

    base_rewards = []
    geom_rewards = []
    refused_flags = []
    dot_r_vals = []
    dot_w_vals = []
    harmful_refused_flags = []
    benign_refused_flags = []
    for i in range(len(completions)):
        base = base_reward(completions[i], labels[i], refusal_phrases, reward_cfg)
        geom_t = geometry_reward(
            h_geom[i],
            labels[i],
            completions[i],
            directions,
            refusal_phrases,
            reward_cfg,
        )
        geom = float(geom_t.item())

        refused = _contains_refusal_phrase(completions[i], refusal_phrases)

        h_norm = F.normalize(h_geom[i], dim=-1)
        r = directions.refusal_dir.to(device=h_norm.device, dtype=h_norm.dtype)
        w = directions.false_refusal_dir.to(device=h_norm.device, dtype=h_norm.dtype)

        dot_r = float(torch.dot(h_norm, r).item())
        dot_w = float(torch.dot(h_norm, w).item())

        base_rewards.append(base)
        geom_rewards.append(geom)
        refused_flags.append(refused)
        dot_r_vals.append(dot_r)
        dot_w_vals.append(dot_w)
        label_i = labels[i].strip().lower()
        if label_i == "harmful":
            harmful_refused_flags.append(refused)
        elif label_i == "benign":
            benign_refused_flags.append(refused)

    base_rewards_t = torch.tensor(base_rewards, dtype=torch.float32, device=device)
    geom_rewards_t = torch.tensor(geom_rewards, dtype=torch.float32, device=device)

    if reward_cfg.normalize_geom_reward:
        geom_mean = geom_rewards_t.mean()
        geom_std = geom_rewards_t.std().clamp_min(reward_cfg.geom_reward_eps)
        geom_rewards_t = (geom_rewards_t - geom_mean) / geom_std
        geom_rewards_t = reward_cfg.geom_reward_scale * geom_rewards_t
        geom_rewards_t = geom_rewards_t.clamp(-reward_cfg.clip_m, reward_cfg.clip_m)

    rewards_t = base_rewards_t + geom_rewards_t
    rewards = rewards_t.detach().cpu().tolist()

    diagnostics: dict[str, float] = {
        "base_reward_mean": float(torch.tensor(base_rewards, dtype=torch.float32).mean().item()),
        "geom_reward_mean": float(geom_rewards_t.mean().item()),
        "geom_reward_abs_mean": float(geom_rewards_t.abs().mean().item()),
        "dot_r_mean": float(torch.tensor(dot_r_vals, dtype=torch.float32).mean().item()),
        "dot_w_mean": float(torch.tensor(dot_w_vals, dtype=torch.float32).mean().item()),
        "refusal_rate": float(sum(refused_flags) / len(refused_flags)),
        "harmful_refusal_rate": float(sum(harmful_refused_flags) / len(harmful_refused_flags)) if harmful_refused_flags else 0.0,
        "benign_refusal_rate": float(sum(benign_refused_flags) / len(benign_refused_flags)) if benign_refused_flags else 0.0,
    }
    return (
        [
            SampleRollout(
                prompt=prompts[i],
                label=labels[i],
                completion=completions[i],
                reward=float(rewards[i]),
                advantage=0.0,
            )
            for i in range(len(completions))
        ],
        pol_log_probs,
        old_log_probs,
        ref_log_probs,
        completion_mask,
        rewards_t,
        diagnostics,
    )


def _group_normalized_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    if rewards.shape[0] % group_size != 0:
        raise ValueError("Batch size must be divisible by group_size.")
    grouped = rewards.view(-1, group_size)
    mu = grouped.mean(dim=1, keepdim=True)
    sigma = grouped.std(dim=1, keepdim=True).clamp_min(1e-6)
    adv = (grouped - mu) / sigma
    adv = adv.clamp(-2.0, 2.0)
    return adv.view(-1)


def grpo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    completion_mask: torch.Tensor,
    token_advantages: torch.Tensor,
    kl_coef: float,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    mask = completion_mask.float()

    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    unclipped = ratio * token_advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * token_advantages

    pg_loss = -torch.minimum(unclipped, clipped)
    pg_loss = (pg_loss * mask).sum() / mask.sum().clamp_min(1)

    ref_log_ratio = ref_log_probs - new_log_probs
    kl_per_token = torch.exp(ref_log_ratio) - ref_log_ratio - 1
    kl = (kl_per_token * mask).sum() / mask.sum().clamp_min(1)

    return pg_loss + kl_coef * kl


def train_grpo(
    policy,
    ref_model,
    tokenizer,
    prompts: Sequence[str],
    labels: Sequence[str],
    cfg: GRPOConfig,
    reward_cfg: RewardConfig,
    directions: DirectionBundle,
    refusal_phrases: Sequence[str],
) -> list[dict]:
    device = next(policy.parameters()).device
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    elif cfg.optimizer == "adafactor":
        optimizer = Adafactor(
            policy.parameters(),
            lr=cfg.lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")
    history: list[dict] = []

    if len(prompts) != len(labels):
        raise ValueError("prompts and labels must have same length")
    if cfg.batch_prompts <= 0:
        raise ValueError("batch_prompts must be > 0")

    for step in range(cfg.steps):
        print(f"[GRPO] starting step {step}", flush=True)
        harm_idx = [i for i, y in enumerate(labels) if y == "harmful"]
        benign_idx = [i for i, y in enumerate(labels) if y == "benign"]

        n_harm = cfg.batch_prompts // 2
        n_benign = cfg.batch_prompts - n_harm

        hidx = torch.randint(0, len(harm_idx), size=(n_harm,), device=device).tolist()
        bidx = torch.randint(0, len(benign_idx), size=(n_benign,), device=device).tolist()

        idx = [harm_idx[i] for i in hidx] + [benign_idx[i] for i in bidx]
        batch_prompts = [prompts[i] for i in idx]
        batch_labels = [labels[i] for i in idx]

        expanded_prompts = []
        expanded_labels = []
        for p, y in zip(batch_prompts, batch_labels):
            for _ in range(cfg.group_size):
                expanded_prompts.append(p)
                expanded_labels.append(y)
        print(f"[GRPO] calling rollout", flush=True)
        rollouts, pol_lp, old_lp, ref_lp, completion_mask, rewards, diagnostics = rollout_group(
            policy=policy,
            ref_model=ref_model,
            tokenizer=tokenizer,
            prompts=expanded_prompts,
            labels=expanded_labels,
            cfg=cfg,
            reward_cfg=reward_cfg,
            directions=directions,
            refusal_phrases=refusal_phrases,
        )
        print(f"[GRPO] finished rollout step {step}", flush=True)
        advantages = _group_normalized_advantages(rewards, cfg.group_size)
        adv_abs_mean = float(advantages.abs().mean().item())
        reward_std = float(rewards.view(-1, cfg.group_size).std(dim=1).mean().item())
        for i in range(len(rollouts)):
            rollouts[i].advantage = float(advantages[i].item())

        # if step % 10 == 0:
        #     print(f"\n=== STEP {step} SAMPLE OUTPUT ===")
        #     print("PROMPT:", rollouts[0].prompt[:200])
        #     print("COMPLETION:", rollouts[0].completion[:300])
        #     print("REWARD:", rollouts[0].reward)
        #     print("ADV:", rollouts[0].advantage)
        #     print("===============================\n")

        token_advantages = advantages.unsqueeze(1).expand_as(pol_lp)
        loss = grpo_loss(
            new_log_probs=pol_lp,
            old_log_probs=old_lp,
            ref_log_probs=ref_lp,
            completion_mask=completion_mask.float(),
            token_advantages=token_advantages,
            kl_coef=cfg.kl_coef,
            clip_eps=cfg.clip_eps,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()

        mean_reward = float(rewards.mean().item())
        mean_adv = float(advantages.mean().item())
        history.append({
            "step": step,
            "loss": float(loss.item()),
            "mean_reward": mean_reward,
            "mean_adv": mean_adv,
            "adv_abs_mean": adv_abs_mean,
            "reward_std": reward_std,
            **diagnostics,
        })

        print(
            f"[GRPO] step {step} done | loss={loss.item():.4f} "
            f"reward={mean_reward:.4f} refusal={diagnostics['refusal_rate']:.3f} "
            f"harm_refusal={diagnostics['harmful_refusal_rate']:.3f} "
            f"benign_refusal={diagnostics['benign_refusal_rate']:.3f}",
            flush=True,
        )

    return history

