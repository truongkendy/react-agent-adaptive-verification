from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class HotpotExample:
    id: str
    question: str
    answer: str
    type: str
    level: str
    supporting_facts: list[tuple[str, int]] = field(default_factory=list)
    context: list[tuple[str, list[str]]] = field(default_factory=list)

    @property
    def gold_titles(self) -> set[str]:
        return {title for title, _ in self.supporting_facts}

    @classmethod
    def from_raw(cls, r: dict) -> "HotpotExample":
        return cls(
            id=r["_id"],
            question=r["question"],
            answer=r.get("answer", ""),
            type=r.get("type", ""),
            level=r.get("level", ""),
            supporting_facts=[tuple(sf) for sf in r.get("supporting_facts", [])],
            context=[tuple(c) for c in r.get("context", [])],
        )


class HotpotQADataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run `python experiments/download_data.py` first."
            )
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.examples: list[HotpotExample] = [HotpotExample.from_raw(r) for r in raw]

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[HotpotExample]:
        return iter(self.examples)

    def __getitem__(self, i: int) -> HotpotExample:
        return self.examples[i]

    def sample_subset(self, n: int, seed: int = 42) -> list[HotpotExample]:
        rng = random.Random(seed)
        n = min(n, len(self.examples))
        return rng.sample(self.examples, n)

    def filter_by(self, level: str | None = None,
                  qtype: str | None = None) -> list[HotpotExample]:
        out = self.examples
        if level:
            out = [e for e in out if e.level == level]
        if qtype:
            out = [e for e in out if e.type == qtype]
        return out


def load_examples(data_path: str, n: int, seed: int) -> list[dict]:
    raw = json.loads(Path(data_path).read_text(encoding="utf-8"))
    rng = random.Random(seed)
    if 0 < n < len(raw):
        raw = rng.sample(raw, n)
    return raw
