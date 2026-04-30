from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset


@dataclass
class AlpacaSample:
    prompt: str
    source: str = "alpaca"
    response: Optional[str] = None
    is_refusal: Optional[bool] = None


def load_alpaca_safe(max_samples: Optional[int] = None) -> List[AlpacaSample]:
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    samples: List[AlpacaSample] = []
    for row in ds:
        instruction = str(row.get("instruction") or "").strip()
        inp = str(row.get("input") or "").strip()
        if not instruction:
            continue

        prompt = f"{instruction}\n\nInput: {inp}" if inp else instruction
        samples.append(AlpacaSample(prompt=prompt))

        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples
