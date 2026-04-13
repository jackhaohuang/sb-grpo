from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from typing import Any, Sequence


def _shuffle_copy(items: Sequence[Any], seed: int) -> list[Any]:
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def split_three_way(
    items: Sequence[Any],
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[list[Any], list[Any], list[Any]]:
    items = _shuffle_copy(items, seed)

    if len(items) < n_train + n_test:
        raise ValueError(
            f"Not enough samples: have {len(items)}, need at least {n_train + n_test}"
        )

    train = items[:n_train]
    test = items[n_train:n_train + n_test]
    eval_ = items[n_train + n_test:]
    return train, test, eval_


def save_jsonl(path: str, rows: Sequence[Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                payload = asdict(row)
            else:
                payload = row
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows