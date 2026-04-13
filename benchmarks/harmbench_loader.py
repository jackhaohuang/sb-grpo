"""
HarmBench dataset loader.

HarmBench (https://arxiv.org/abs/2402.04249) is a standardized evaluation framework
for automated red-teaming of LLMs. We load the "standard" behaviors split here,
which contains direct harmful requests.

HuggingFace: https://huggingface.co/datasets/walledai/HarmBench
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from datasets import load_dataset


# All functional categories in HarmBench
HARMBENCH_CATEGORIES = [
    "standard",
    "contextual",
    "copyright"
]

# Semantic categories used in the standard split
HARMBENCH_SEMANTIC_CATEGORIES = [
    "Cybercrime & Unauthorized Intrusion",
    "Chemical & Biological Weapons",
    "Copyright Violations",
    "Misinformation & Disinformation",
    "Harassment & Bullying",
    "Illegal Activities",
    "General Harm",
]


@dataclass
class HarmBenchSample:
    behavior_id: str
    behavior: str
    semantic_category: str
    tags: List[str] = field(default_factory=list)
    source: str = "harmbench"
    response: Optional[str] = None
    is_harmful: Optional[bool] = None


def _behavior_text(row: dict) -> str:
    """HarmBench HF revisions use ``prompt``; older schemas use ``Behavior`` / ``behavior``."""
    text = (
        row.get("Behavior")
        or row.get("behavior")
        or row.get("prompt")
        or ""
    )
    return text.strip() if isinstance(text, str) else ""


def _semantic_category(row: dict) -> str:
    return str(row.get("SemanticCategory") or row.get("category") or "")


def load_harmbench(
    config: str | list[str] = "standard",
    semantic_category: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[HarmBenchSample]:

    if isinstance(config, str):
        configs = [config]
    else:
        configs = config

    samples: List[HarmBenchSample] = []

    for cfg in configs:
        ds = load_dataset("walledai/HarmBench", cfg, split="train")

        for idx, row in enumerate(ds):
            row = dict(row)
            func_cat = (row.get("FunctionalCategory") or "").lower()
            sem_cat = _semantic_category(row)

            if func_cat and func_cat != cfg.lower():
                continue

            if semantic_category and sem_cat != semantic_category:
                continue

            behavior = _behavior_text(row)
            bid = row.get("BehaviorID")
            behavior_id = str(bid).strip() if bid else f"{cfg}_{idx}"

            samples.append(
                HarmBenchSample(
                    behavior_id=behavior_id,
                    behavior=behavior,
                    semantic_category=sem_cat,
                    tags=[],
                )
            )

            if max_samples is not None and len(samples) >= max_samples:
                return samples

    return samples