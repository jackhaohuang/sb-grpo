from dataclasses import dataclass, field
from typing import List, Optional

from datasets import load_dataset


ORBENCH_CONFIGS = [
    "or-bench-80k",
    "or-bench-hard-1k",
    "or-bench-toxic",
]


@dataclass
class ORBenchSample:
    prompt: str
    category: Optional[str] = None
    source: str = "orbench"
    response: Optional[str] = None
    is_refusal: Optional[bool] = None


def load_orbench(
    config: str = "or-bench-hard-1k",
    max_samples: Optional[int] = None,
) -> List[ORBenchSample]:
    if config not in ORBENCH_CONFIGS:
        raise ValueError(f"config must be one of {ORBENCH_CONFIGS}, got {config!r}")

    ds = load_dataset("bench-llm/or-bench", config, split="train")

    samples: List[ORBenchSample] = []
    for row in ds:
        samples.append(
            ORBenchSample(
                prompt=row.get("prompt", row.get("Prompt", "")),
                category=row.get("category", row.get("Category")),
            )
        )

        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples