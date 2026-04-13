from __future__ import annotations

import os
import random
import json
from datetime import datetime

from benchmarks.advbench_loader import load_advbench
from benchmarks.harmbench_loader import load_harmbench
from benchmarks.orbench_loader import load_orbench
from benchmarks.splits import save_jsonl, split_three_way


SPLIT_DIR = "data/splits"
SEED = 42


def main() -> None:
    os.makedirs(SPLIT_DIR, exist_ok=True)
    advbench = load_advbench()

    hb_standard = load_harmbench(config="standard")
    hb_contextual = load_harmbench(config="contextual")
    hb_copyright = load_harmbench(config="copyright")

    rng = random.Random(SEED)

    advbench = list(advbench)
    hb_standard = list(hb_standard)
    hb_contextual = list(hb_contextual)
    hb_copyright = list(hb_copyright)

    rng.shuffle(advbench)
    rng.shuffle(hb_standard)
    rng.shuffle(hb_contextual)
    rng.shuffle(hb_copyright)

    if len(advbench) < 500:
        raise ValueError(f"Need at least 500 AdvBench samples, got {len(advbench)}")

    if len(hb_standard) < 200:
        raise ValueError(f"Need at least 200 standard HarmBench samples, got {len(hb_standard)}")
    if len(hb_contextual) < 100:
        raise ValueError(f"Need at least 100 contextual HarmBench samples, got {len(hb_contextual)}")
    if len(hb_copyright) < 100:
        raise ValueError(f"Need at least 100 copyright HarmBench samples, got {len(hb_copyright)}")

    # AdvBench stays the same
    adv_train = advbench[:300]
    adv_test = advbench[300:400]
    adv_eval = advbench[400:]

    # HarmBench: 2:1:1 within each config
    hb_standard_train = hb_standard[:100]
    hb_standard_test = hb_standard[100:150]
    hb_standard_eval = hb_standard[150:]

    hb_contextual_train = hb_contextual[:50]
    hb_contextual_test = hb_contextual[50:75]
    hb_contextual_eval = hb_contextual[75:]

    hb_copyright_train = hb_copyright[:50]
    hb_copyright_test = hb_copyright[50:75]
    hb_copyright_eval = hb_copyright[75:]

    hb_train = hb_standard_train + hb_contextual_train + hb_copyright_train
    hb_test = hb_standard_test + hb_contextual_test + hb_copyright_test
    hb_eval = hb_standard_eval + hb_contextual_eval + hb_copyright_eval

    harmful_train = adv_train + hb_train
    harmful_test = adv_test + hb_test
    harmful_eval = adv_eval + hb_eval

    rng.shuffle(harmful_train)
    rng.shuffle(harmful_test)
    rng.shuffle(harmful_eval)

    save_jsonl(f"{SPLIT_DIR}/harmful_train.jsonl", harmful_train)
    save_jsonl(f"{SPLIT_DIR}/harmful_test.jsonl", harmful_test)
    save_jsonl(f"{SPLIT_DIR}/harmful_eval.jsonl", harmful_eval)

    orbench = load_orbench(config="or-bench-hard-1k")
    or_train, or_test, or_eval = split_three_way(
        orbench,
        n_train=600,
        n_test=200,
        seed=SEED,
    )

    save_jsonl(f"{SPLIT_DIR}/orbench_hard_train.jsonl", or_train)
    save_jsonl(f"{SPLIT_DIR}/orbench_hard_test.jsonl", or_test)
    save_jsonl(f"{SPLIT_DIR}/orbench_hard_eval.jsonl", or_eval)

    print("Saved splits:")
    print("  harmful_train.jsonl")
    print("  harmful_test.jsonl")
    print("  harmful_eval.jsonl")
    print("  orbench_hard_train.jsonl")
    print("  orbench_hard_test.jsonl")
    print("  orbench_hard_eval.jsonl")

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": SEED,
        "harmful": {
            "advbench": {"train": 300, "test": 100, "eval": 100},
            "harmbench": {
                "standard": {"train": 100, "test": 50, "eval": 50},
                "contextual": {"train": 50, "test": 25, "eval": 25},
                "copyright": {"train": 50, "test": 25, "eval": 25},
                "total": {"train": 200, "test": 100, "eval": 100},
            },
            "mixed_total": {
                "train": len(harmful_train),
                "test": len(harmful_test),
                "eval": len(harmful_eval),
            },
        },
        "orbench": {
            "config": "or-bench-hard-1k",
            "train": len(or_train),
            "test": len(or_test),
            "eval": len(or_eval),
        },
    }

    with open(f"{SPLIT_DIR}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()