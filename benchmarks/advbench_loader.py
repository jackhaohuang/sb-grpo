from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset


@dataclass
class AdvBenchSample:
    prompt: str
    category: Optional[str] = None
    sample_id: Optional[str] = None
    source: str = "advbench"
    response: Optional[str] = None
    is_harmful: Optional[bool] = None


def load_advbench(max_samples: Optional[int] = None) -> List[AdvBenchSample]:
    ds = load_dataset("walledai/AdvBench", split="train")

    samples: List[AdvBenchSample] = []
    for idx, row in enumerate(ds):
        prompt = (
            row.get("prompt")
            or row.get("goal")
            or row.get("instruction")
            or row.get("Behavior")
            or row.get("behavior")
            or ""
        )
        prompt = prompt.strip() if isinstance(prompt, str) else ""
        if not prompt:
            continue

        samples.append(
            AdvBenchSample(
                prompt=prompt,
                category=row.get("category"),
                sample_id=str(row.get("id") or row.get("BehaviorID") or idx),
            )
        )

        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples