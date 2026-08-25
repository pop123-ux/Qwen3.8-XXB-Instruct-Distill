"""Reading a teacher dataset back, without the teacher.

The architectural point of the whole pipeline: the expensive stage produces a durable
artifact, and the cheap stage consumes it. A 27B teacher on a rented A100 writes JSONL;
a 94M student on a free T4 trains from that JSONL weeks later. The two never need to
coexist, which is what makes the project affordable.

So this loader depends on nothing but the files. No tokenizer download, no teacher, no
network.

Integrity is checked rather than assumed, because a dataset that arrives over Drive from
another machine can be short, duplicated or edited, and each of those corrupts training
in a way that looks like a modelling problem later.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..training.data import DistillationExample
from .generation import iter_records
from .manifest import DatasetManifest


@dataclass
class DatasetFilter:
    """Which records to keep. Every exclusion is counted, never silent."""

    max_prompt_chars: int | None = None
    max_answer_chars: int | None = None
    min_answer_chars: int = 1
    categories: tuple[str, ...] | None = None
    reasoning_modes: tuple[str, ...] | None = None
    require_reasoning: bool = False
    #: Drop records whose recorded digests no longer match their content. On by default:
    #: a record that was edited after generation is not teacher output any more.
    require_intact_hashes: bool = True
    deduplicate_by: str = "example_id"     # "example_id" | "prompt" | "none"

    def reason_to_drop(self, example: DistillationExample) -> str | None:
        problems = example.validate()
        if problems:
            return f"invalid: {problems[0]}"
        if self.require_intact_hashes and not example.hashes_match():
            return "content hash mismatch (record edited or corrupted after generation)"
        if self.max_prompt_chars and len(example.prompt) > self.max_prompt_chars:
            return "prompt too long"
        if self.max_answer_chars and len(example.teacher_answer) > self.max_answer_chars:
            return "answer too long"
        if len(example.teacher_answer.strip()) < self.min_answer_chars:
            return "answer too short"
        if self.categories and example.task_category not in self.categories:
            return "category excluded"
        if self.reasoning_modes and example.teacher_reasoning_setting not in self.reasoning_modes:
            return "reasoning mode excluded"
        if self.require_reasoning and not (example.teacher_reasoning or "").strip():
            return "no reasoning trace"
        return None


@dataclass
class DatasetStats:
    """What was loaded, and what was left out and why."""

    n_seen: int = 0
    n_kept: int = 0
    n_dropped: int = 0
    n_duplicates: int = 0
    dropped_reasons: Counter = field(default_factory=Counter)
    categories: Counter = field(default_factory=Counter)
    reasoning_modes: Counter = field(default_factory=Counter)
    teacher_models: Counter = field(default_factory=Counter)
    n_with_kd_targets: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_seen": self.n_seen, "n_kept": self.n_kept, "n_dropped": self.n_dropped,
            "n_duplicates": self.n_duplicates,
            "dropped_reasons": dict(self.dropped_reasons),
            "categories": dict(self.categories),
            "reasoning_modes": dict(self.reasoning_modes),
            "teacher_models": dict(self.teacher_models),
            "n_with_kd_targets": self.n_with_kd_targets,
        }

    def render(self) -> str:
        lines = [f"  kept {self.n_kept:,} of {self.n_seen:,} record(s)"]
        if self.n_duplicates:
            lines.append(f"  {self.n_duplicates:,} duplicate(s) removed")
        for reason, count in self.dropped_reasons.most_common():
            lines.append(f"  dropped {count:,}: {reason}")
        if len(self.teacher_models) > 1:
            lines.append(f"  ! {len(self.teacher_models)} different teacher models present: "
                         f"{dict(self.teacher_models)}")
        if len(self.reasoning_modes) > 1:
            lines.append(f"  ! mixed reasoning modes: {dict(self.reasoning_modes)}")
        return "\n".join(lines)


@dataclass
class TeacherDataset:
    """A loaded teacher dataset plus how it was assembled."""

    examples: list[DistillationExample]
    stats: DatasetStats
    manifest: dict[str, Any] | None = None
    source: str = ""

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[DistillationExample]:
        return iter(self.examples)

    def split(self, validation_fraction: float = 0.05, seed: int = 0
              ) -> tuple[list[DistillationExample], list[DistillationExample]]:
        """Deterministic train/validation split.

        Shuffled with a fixed seed and split by *example id*, so the same dataset always
        splits the same way regardless of shard order or how many sessions produced it.
        """
        ordered = sorted(self.examples, key=lambda e: e.example_id)
        rng = random.Random(seed)
        rng.shuffle(ordered)
        n_validation = max(1, int(len(ordered) * validation_fraction)) if ordered else 0
        return ordered[n_validation:], ordered[:n_validation]

    def kd_ready(self) -> bool:
        """Whether every record carries teacher logits, which logit KD would need."""
        return bool(self.examples) and all(e.has_kd_targets for e in self.examples)


def load_teacher_dataset(
    path: str | Path,
    *,
    filter: DatasetFilter | None = None,
    limit: int | None = None,
    verified_only: bool = True,
    on_record: Callable[[DistillationExample], None] | None = None,
) -> TeacherDataset:
    """Load a sharded teacher dataset directory, or a single JSONL file.

    ``verified_only`` reads only shards the manifest records as complete. An interrupted
    generation leaves an open shard, and training on a truncated file silently changes
    the dataset out from under a comparison.
    """
    source = Path(path)
    rules = filter or DatasetFilter()
    stats = DatasetStats()
    kept: list[DistillationExample] = []
    seen_keys: set[str] = set()

    manifest_data = None
    if source.is_dir():
        manifest = DatasetManifest.read(source)
        manifest_data = manifest.to_dict() if manifest else None
        records = iter_records(source, verified_only=verified_only)
    elif source.is_file():
        from ..training.data import read_jsonl

        records = read_jsonl(source)
    else:
        raise FileNotFoundError(f"teacher dataset not found: {source}")

    for example in records:
        stats.n_seen += 1
        if on_record is not None:
            on_record(example)

        reason = rules.reason_to_drop(example)
        if reason is not None:
            stats.n_dropped += 1
            stats.dropped_reasons[reason] += 1
            continue

        if rules.deduplicate_by != "none":
            key = (example.example_id if rules.deduplicate_by == "example_id"
                   else example.prompt)
            if key in seen_keys:
                stats.n_duplicates += 1
                continue
            seen_keys.add(key)

        kept.append(example)
        stats.categories[example.task_category] += 1
        stats.reasoning_modes[example.teacher_reasoning_setting or "unknown"] += 1
        stats.teacher_models[example.teacher_model or "unknown"] += 1
        if example.has_kd_targets:
            stats.n_with_kd_targets += 1
        if limit is not None and len(kept) >= limit:
            break

    stats.n_kept = len(kept)
    return TeacherDataset(
        examples=kept, stats=stats, manifest=manifest_data, source=str(source)
    )


def format_sft_example(
    example: DistillationExample, *, include_reasoning: bool = False
) -> tuple[str, str]:
    """Split a record into ``(context, target)`` for supervised fine-tuning.

    ``include_reasoning`` decides whether the student is trained to reproduce the
    teacher's reasoning trace or only its answer — a real experimental choice, not a
    detail, since it changes what the student learns to spend tokens on. The default is
    answer-only, which is the cheaper hypothesis and the one this project's efficiency
    goal favours; it must be measured rather than assumed.
    """
    context = example.prompt
    if include_reasoning and (example.teacher_reasoning or "").strip():
        target = f"<think>{example.teacher_reasoning}</think>{example.teacher_answer}"
    else:
        target = example.teacher_answer
    return context, target
