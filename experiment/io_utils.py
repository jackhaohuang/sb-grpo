from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from utils import build_chat_prompt


@dataclass
class PromptExample:
    prompt: str
    label: str  # "harmful" | "benign"
    source: str


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _harmful_prompt(row: dict) -> str:
    text = row.get("behavior") or row.get("prompt") or row.get("goal") or ""
    return text.strip() if isinstance(text, str) else ""


def _benign_prompt(row: dict) -> str:
    text = row.get("prompt") or row.get("instruction") or ""
    return text.strip() if isinstance(text, str) else ""


def load_harmful_split(path: str) -> list[str]:
    rows = load_jsonl(path)
    prompts = [_harmful_prompt(r) for r in rows]
    return [p for p in prompts if p]


def load_orbench_split(path: str) -> list[str]:
    rows = load_jsonl(path)
    prompts = [_benign_prompt(r) for r in rows]
    return [p for p in prompts if p]


def load_safe_split(path: str) -> list[str]:
    rows = load_jsonl(path)
    prompts = [_benign_prompt(r) for r in rows]
    return [p for p in prompts if p]


def load_alpaca_safe(max_samples: int, seed: int) -> list[str]:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = []
    for row in ds:
        instruction = str(row.get("instruction") or "").strip()
        inp = str(row.get("input") or "").strip()
        if not instruction:
            continue
        if inp:
            prompts.append(f"{instruction}\n\nInput: {inp}")
        else:
            prompts.append(instruction)
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:max_samples]


def to_chat_prompts(prompts: Iterable[str], tokenizer) -> list[str]:
    return [build_chat_prompt(p, tokenizer=tokenizer) for p in prompts]

