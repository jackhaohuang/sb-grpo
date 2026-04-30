from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from tqdm import tqdm


@dataclass
class DirectionArtifacts:
    layer_idx: int
    refusal_dir: torch.Tensor
    false_refusal_dir: torch.Tensor
    false_refusal_ortho_dir: torch.Tensor
    v_harm: torch.Tensor
    v_safe: torch.Tensor
    v_pseudo: torch.Tensor


def _last_non_pad_index(attn_mask: torch.Tensor) -> torch.Tensor:
    return attn_mask.sum(dim=1) - 1


@torch.inference_mode()
def collect_prompt_activations(
    model,
    tokenizer,
    prompts: Sequence[str],
    batch_size: int,
    max_length: int = 2048,
) -> tuple[torch.Tensor, list[int]]:
    """
    Returns (activations, layer_ids).

    activations: [n_prompts, n_local_layers, hidden_size] — residual stream at the
    last prompt token; column i corresponds to model layer layer_ids[i].

    layer_ids: model layer indices included (e.g. range(4, n_layers) skipping early layers).
    """
    device = next(model.parameters()).device
    all_batches: list[torch.Tensor] = []
    n_layers = model.config.num_hidden_layers
    layer_ids = list(range(4, n_layers))

    for start in tqdm(range(0, len(prompts), batch_size), desc="Collect activations"):
        batch_prompts = prompts[start : start + batch_size]
        toks = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(
            **toks,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # hidden_states[0] is embeddings; hidden_states[l+1] is post-layer-l
        last_idx = _last_non_pad_index(toks["attention_mask"])
        batch_out = []
        for l in layer_ids:
            h = out.hidden_states[l + 1]  # [b, s, d]
            gathered = h[torch.arange(h.shape[0], device=device), last_idx]
            batch_out.append(gathered)
        stacked = torch.stack(batch_out, dim=1)  # [b, L, d]
        all_batches.append(stacked.detach().cpu())
    return torch.cat(all_batches, dim=0), layer_ids


def choose_best_layer_by_silhouette(
    harm_acts: torch.Tensor,  # [n_h, L_local, d]
    safe_acts: torch.Tensor,  # [n_s, L_local, d]
    simple_layer: bool = False,
) -> int:
    """Returns local layer index into harm_acts[:, local, :] (0 .. L_local-1). Map to a model
    layer with true_layer_idx = layer_ids[local_idx] when activations were collected with
    collect_prompt_activations (which skips early layers).
    """
    n_local_layers = harm_acts.shape[1]
    if simple_layer:
        return n_local_layers // 2

    y = torch.cat(
        [
            torch.ones(harm_acts.shape[0], dtype=torch.long),
            torch.zeros(safe_acts.shape[0], dtype=torch.long),
        ]
    ).numpy()
    best_local = 0
    best_score = float("-inf")
    for l in range(n_local_layers):
        x = torch.cat([harm_acts[:, l, :], safe_acts[:, l, :]], dim=0).numpy()
        score = silhouette_score(x, y, metric="cosine")
        if score > best_score:
            best_score = score
            best_local = l
    return best_local


def choose_best_layer_actor_style(
    harm_acts: torch.Tensor,
    safe_acts: torch.Tensor,
    pseudo_acts: torch.Tensor,
    layer_ids: Optional[Sequence[int]] = None,
    lambda_pseudo: float = 0.5,
    search_start_frac: float = 0.33,
    search_end_frac: float = 0.80,
    direction_strength_coef: float = 0.5,
) -> int:
    """
    Pick a layer that separates harmful from safe while keeping pseudo-harmful (over-refusal)
    prompts close to safe. Uses cosine margins on mean centroids (direction over magnitude), plus
    mean projection spread along the harm–safe direction (Arditi-style “clean” direction). Returns
    a model layer index if layer_ids is set, otherwise the local index into harm_acts[:, l, :].
    """
    n_local_layers = harm_acts.shape[1]

    lo = int(n_local_layers * search_start_frac)
    hi = max(lo + 1, int(n_local_layers * search_end_frac))

    best_local = lo
    best_score = float("-inf")

    for l in range(lo, hi):
        h = harm_acts[:, l, :]
        s = safe_acts[:, l, :]
        p = pseudo_acts[:, l, :]

        v_h = h.mean(dim=0)
        v_s = s.mean(dim=0)
        v_p = p.mean(dim=0)

        harm_safe_cos = F.cosine_similarity(
            v_h.unsqueeze(0), v_s.unsqueeze(0), dim=1
        ).squeeze(0)
        pseudo_safe_cos = F.cosine_similarity(
            v_p.unsqueeze(0), v_s.unsqueeze(0), dim=1
        ).squeeze(0)

        harm_safe_margin = 1.0 - harm_safe_cos
        pseudo_safe_margin = 1.0 - pseudo_safe_cos

        r = F.normalize(v_h - v_s, dim=0)
        proj_h = (h @ r).mean()
        proj_s = (s @ r).mean()
        direction_strength = proj_h - proj_s

        score = (
            harm_safe_margin
            - lambda_pseudo * pseudo_safe_margin
            + direction_strength_coef * direction_strength
        )

        if score.item() > best_score:
            best_score = score.item()
            best_local = l

    if layer_ids is None:
        return best_local

    return layer_ids[best_local]


def _mean_dir(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    va = a.mean(dim=0)
    vb = b.mean(dim=0)
    d = F.normalize(va - vb, dim=0)
    return va, vb, d


@torch.inference_mode()
def compute_refusal_score_first_token(
    model,
    tokenizer,
    chat_prompts: Sequence[str],
    refusal_tokens: Sequence[str],
    batch_size: int,
    max_length: int = 2048,
) -> torch.Tensor:
    """
    RS = log sum_{t in R} p(t) - log sum_{t notin R} p(t), computed at
    the first generated token distribution from prompt-only forward pass.
    """
    device = next(model.parameters()).device
    refusal_ids = []
    for t in refusal_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) == 1:
            refusal_ids.append(ids[0])
    if not refusal_ids:
        raise ValueError("No single-token refusal token ids found. Provide model-specific tokens.")
    refusal_ids_t = torch.tensor(sorted(set(refusal_ids)), device=device, dtype=torch.long)

    scores: list[torch.Tensor] = []
    for start in tqdm(range(0, len(chat_prompts), batch_size), desc="Compute refusal score"):
        batch_prompts = chat_prompts[start : start + batch_size]
        toks = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**toks, use_cache=False, return_dict=True)
        last_idx = _last_non_pad_index(toks["attention_mask"])
        logits = out.logits[torch.arange(out.logits.shape[0], device=device), last_idx]
        log_probs = logits.log_softmax(dim=-1)
        refusal_log_mass = torch.logsumexp(log_probs.index_select(-1, refusal_ids_t), dim=-1)
        all_log_mass = torch.logsumexp(log_probs, dim=-1)  # == 0.0 after log_softmax
        non_refusal_log_mass = torch.logsumexp(
            log_probs + _mask_non_refusal(log_probs.shape[-1], refusal_ids_t, device),
            dim=-1,
        )
        scores.append((refusal_log_mass - non_refusal_log_mass).detach().cpu())
    return torch.cat(scores, dim=0)


def _mask_non_refusal(vocab_size: int, refusal_ids_t: torch.Tensor, device) -> torch.Tensor:
    mask = torch.zeros(vocab_size, device=device)
    mask[refusal_ids_t] = float("-inf")
    return mask


def build_directions(
    harm_acts: torch.Tensor,
    safe_acts: torch.Tensor,
    pseudo_acts: torch.Tensor,
    model_layer_idx: int,
    layer_ids: Sequence[int],
    orth_lambda: float,
) -> DirectionArtifacts:
    try:
        local_idx = list(layer_ids).index(model_layer_idx)
    except ValueError as e:
        raise ValueError(
            f"model_layer_idx={model_layer_idx} not in layer_ids={list(layer_ids)}"
        ) from e
    v_harm, v_safe, refusal = _mean_dir(harm_acts[:, local_idx, :], safe_acts[:, local_idx, :])
    v_pseudo = pseudo_acts[:, local_idx, :].mean(dim=0)
    false_refusal = F.normalize(v_pseudo - v_safe, dim=0)
    false_refusal_ortho = false_refusal - orth_lambda * torch.dot(refusal, false_refusal) * refusal
    false_refusal_ortho = F.normalize(false_refusal_ortho, dim=0)

    return DirectionArtifacts(
        layer_idx=model_layer_idx,
        refusal_dir=refusal,
        false_refusal_dir=false_refusal,
        false_refusal_ortho_dir=false_refusal_ortho,
        v_harm=v_harm,
        v_safe=v_safe,
        v_pseudo=v_pseudo,
    )

