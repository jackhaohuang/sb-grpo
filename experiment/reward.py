from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass
class RewardConfig:
    alpha: float = 0.1
    beta: float = 0.1
    clip_m: float = 0.2
    harmful_refusal_reward: float = 1.0
    harmful_compliance_penalty: float = 2.0
    benign_compliance_reward: float = 0.2
    benign_refusal_penalty: float = 0.2

    normalize_geom_reward: bool = True
    geom_reward_scale: float = 1.0
    geom_reward_eps: float = 1e-6


@dataclass
class DirectionBundle:
    layer_idx: int
    refusal_dir: torch.Tensor
    false_refusal_dir: torch.Tensor


def _contains_refusal_phrase(text: str, phrases: Sequence[str]) -> bool:
    lower = text.lower()
    return any(p.lower() in lower for p in phrases)


def base_reward(
    completion: str,
    label: str,
    refusal_phrases: Sequence[str],
    cfg: RewardConfig,
) -> float:
    label = label.strip().lower()
    refused = _contains_refusal_phrase(completion, refusal_phrases)

    if label == "harmful":
        return cfg.harmful_refusal_reward if refused else -cfg.harmful_compliance_penalty

    return -cfg.benign_refusal_penalty if refused else cfg.benign_compliance_reward


def geometry_reward(
    h_last_completion_token: torch.Tensor,
    label: str,
    completion: str,
    directions: DirectionBundle,
    refusal_phrases: Sequence[str],
    cfg: RewardConfig,
) -> torch.Tensor:
    label = label.strip().lower()
    h = F.normalize(h_last_completion_token, dim=-1)

    if label == "harmful":
        r = directions.refusal_dir.to(device=h.device, dtype=h.dtype)
        geom = cfg.alpha * torch.dot(h, r)
    else:
        refused = _contains_refusal_phrase(completion, refusal_phrases)
        if refused:
            w = directions.false_refusal_dir.to(device=h.device, dtype=h.dtype)
            geom = -cfg.beta * torch.dot(h, w)
        else:
            geom = torch.zeros((), device=h.device, dtype=h.dtype)

    return torch.clamp(geom, min=-cfg.clip_m, max=cfg.clip_m)