"""Offline distillation dataset: schema, loading, and a synthetic corpus for tests.

The central design decision is that **teacher generation and student training are
separate operations**. The teacher runs once — possibly on rented hardware — and writes
a JSONL file; the student trains from that file later, on whatever GPU is available. A
16 GB card therefore never has to hold a 27B teacher and a student simultaneously,
which is what makes the project's hardware situation workable at all.

The schema records not just what the teacher said but how much it cost to say it, so
reasoning-efficiency training can supervise on length without the teacher present.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Bump when the schema changes incompatibly; every record carries it so a mixed
#: directory of old and new files is detectable rather than silently misread.
SCHEMA_VERSION = "1.0"


@dataclass
class DistillationExample:
    """One teacher-generated training example."""

    example_id: str
    prompt: str
    teacher_answer: str

    teacher_reasoning: str | None = None
    task_category: str = "unknown"
    difficulty: str = "unknown"
    teacher_reasoning_setting: str | None = None
    teacher_thinking_tokens: int | None = None
    teacher_total_tokens: int | None = None
    teacher_metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    dataset_version: str = SCHEMA_VERSION

    # --- optional, for later phases -------------------------------------
    #: Path to stored teacher logits. Full distributions over a ~248k vocabulary are
    #: expensive, so this stays optional and top-k is the intended first step.
    teacher_logits_path: str | None = None
    #: Top-k teacher logits inline: [[token_id, logprob], ...]. Cheap enough to store.
    teacher_top_logits: list[list[float]] | None = None
    candidate_answers: list[str] | None = None
    preference_rank: int | None = None
    verifier_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DistillationExample:
        known = set(cls.__dataclass_fields__)
        extra = {k: v for k, v in data.items() if k not in known}
        kwargs = {k: v for k, v in data.items() if k in known}
        example = cls(**kwargs)
        if extra:
            example.teacher_metadata = {**example.teacher_metadata, "_unknown_fields": extra}
        return example

    def validate(self) -> list[str]:
        """Return a list of problems; empty means usable."""
        problems: list[str] = []
        if not self.prompt.strip():
            problems.append("empty prompt")
        if not self.teacher_answer.strip():
            problems.append("empty teacher_answer")
        if self.teacher_thinking_tokens is not None and self.teacher_thinking_tokens < 0:
            problems.append("negative teacher_thinking_tokens")
        return problems

    @property
    def has_kd_targets(self) -> bool:
        """Whether this example can support a logit-KD objective."""
        return self.teacher_top_logits is not None or self.teacher_logits_path is not None


def write_jsonl(examples: list[DistillationExample], path: str | Path) -> int:
    """Write examples as JSONL. Returns the number written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    return len(examples)


def read_jsonl(path: str | Path, *, skip_invalid: bool = True) -> Iterator[DistillationExample]:
    """Stream examples from JSONL.

    Streams rather than loading, so a dataset larger than RAM is usable on a small
    machine. Malformed lines are skipped with a count rather than aborting a long run.
    """
    source = Path(path)
    skipped = 0
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                example = DistillationExample.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError) as exc:
                if not skip_invalid:
                    raise ValueError(f"{source}:{number}: {exc}") from exc
                skipped += 1
                continue
            problems = example.validate()
            if problems:
                if not skip_invalid:
                    raise ValueError(f"{source}:{number}: {', '.join(problems)}")
                skipped += 1
                continue
            yield example
    if skipped:
        print(f"  note: skipped {skipped} invalid record(s) in {source}")


def synthetic_corpus(n: int = 256, seed: int = 0) -> list[DistillationExample]:
    """A deterministic toy corpus for pipeline tests.

    Arithmetic, so correctness is mechanically checkable and the model has something
    learnable. This validates plumbing; it teaches nothing worth keeping.
    """
    import random

    rng = random.Random(seed)
    examples: list[DistillationExample] = []
    for index in range(n):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        examples.append(
            DistillationExample(
                example_id=f"synthetic-{index:05d}",
                prompt=f"What is {a} + {b}?",
                teacher_answer=str(a + b),
                task_category="arithmetic",
                difficulty="trivial",
                source="synthetic",
                teacher_metadata={"generator": "synthetic_corpus", "seed": seed},
            )
        )
    return examples
