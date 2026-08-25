"""Teacher generation that survives a rented GPU disappearing mid-run.

Generating teacher responses is the most expensive step in this project: it needs a GPU
big enough for a 27B model, and it produces thousands of long generations. Losing that
work to a terminated instance is the failure this module is built around — the same
lesson the Level-2 training run learned at step 1925.

The design:

* **Every completed record is durable immediately.** Records append to an open shard and
  are flushed; a kill loses at most the record in flight.
* **Resume is by prompt id, read back from what is actually on disk.** Not from a
  separate counter that can disagree with the data — the file is the truth.
* **A shard is complete only when it is closed and checksummed.** An open shard is
  visible as incomplete, so a truncated file is never read as a short one.
* **A partial final line is dropped on resume.** A process killed mid-write leaves one,
  and it must cost that record rather than corrupting the shard.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..training.data import DistillationExample
from .backends import TeacherBackend, TeacherResponse
from .manifest import DatasetManifest, ShardRecord, shard_name
from .provenance import RunManifest, sha256_file, sha256_json, sha256_text, utc_now
from .reasoning_modes import ReasoningMode


@dataclass
class Prompt:
    """One input to generate against."""

    id: str
    prompt: str
    category: str = "unknown"
    difficulty: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return sha256_text(self.prompt)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "prompt": self.prompt, "category": self.category,
                "difficulty": self.difficulty, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Prompt:
        return cls(
            id=str(data.get("id") or data.get("example_id") or ""),
            prompt=str(data.get("prompt", "")),
            category=str(data.get("category", data.get("task_category", "unknown"))),
            difficulty=str(data.get("difficulty", "unknown")),
            metadata=data.get("metadata", {}) or {},
        )


def read_prompts(path: str | Path) -> list[Prompt]:
    """Load a prompt set, rejecting duplicate ids.

    Duplicate ids would make resume ambiguous — "have I done this one?" stops having a
    single answer — so this fails loudly rather than picking one.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"prompt file not found: {source}")
    prompts: list[Prompt] = []
    seen: dict[str, int] = {}
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
        prompt = Prompt.from_dict(payload)
        if not prompt.id:
            raise ValueError(f"{source}:{line_number}: record has no id")
        if not prompt.prompt.strip():
            raise ValueError(f"{source}:{line_number}: record {prompt.id!r} has empty prompt")
        if prompt.id in seen:
            raise ValueError(
                f"{source}:{line_number}: duplicate prompt id {prompt.id!r} "
                f"(first seen on line {seen[prompt.id]}). Ids must be unique or resume "
                "cannot tell which records are done."
            )
        seen[prompt.id] = line_number
        prompts.append(prompt)
    return prompts


def write_prompts(prompts: list[Prompt], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt.to_dict(), ensure_ascii=False) + "\n")
    return len(prompts)


def prompt_set_digest(prompts: list[Prompt]) -> str:
    """Digest over ids and content, so a changed prompt set is detectable."""
    return sha256_json([[p.id, p.sha256] for p in prompts])


def scan_completed_ids(directory: str | Path) -> tuple[set[str], dict[str, int]]:
    """Which prompt ids already have records on disk, read from the shards themselves.

    Returns the ids and a per-shard count. A trailing partial line — what a killed
    process leaves — is skipped rather than parsed, costing one record instead of the
    shard.
    """
    base = Path(directory)
    done: set[str] = set()
    counts: dict[str, int] = {}
    if not base.is_dir():
        return done, counts
    for shard in sorted(base.glob("shard-*.jsonl")):
        count = 0
        for line in shard.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a kill mid-append; this record is lost, the rest are fine
            identifier = record.get("example_id") or record.get("id")
            if identifier:
                done.add(str(identifier))
                count += 1
        counts[shard.name] = count
    return done, counts


def build_example(
    prompt: Prompt,
    response: TeacherResponse,
    *,
    mode: ReasoningMode,
    teacher_model: str,
    teacher_revision: str | None,
    chat_template_sha256: str | None,
    generation_config_sha256: str | None,
    source: str,
    dataset_version: str,
) -> DistillationExample:
    """Turn a generation into a durable record, with everything needed to trust it."""
    example = DistillationExample(
        example_id=prompt.id,
        prompt=prompt.prompt,
        teacher_answer=response.answer,
        teacher_reasoning=response.thinking or None,
        task_category=prompt.category,
        difficulty=prompt.difficulty,
        teacher_reasoning_setting=mode.name,
        teacher_thinking_tokens=response.thinking_tokens,
        teacher_answer_tokens=response.answer_tokens,
        teacher_total_tokens=response.total_tokens,
        teacher_prompt_tokens=response.prompt_tokens,
        finish_reason=response.finish_reason,
        reasoning_enabled=mode.reasoning_enabled,
        teacher_model=teacher_model,
        teacher_revision=teacher_revision,
        chat_template_sha256=chat_template_sha256,
        generation_config_sha256=generation_config_sha256,
        source=source,
        dataset_version=dataset_version,
        created_at=utc_now(),
        teacher_metadata={
            "backend": response.backend,
            "latency_s": round(response.latency_s, 6),
            "token_counting_method": response.token_counting_method,
            **({"prompt_metadata": prompt.metadata} if prompt.metadata else {}),
        },
    )
    example.prompt_sha256, example.response_sha256 = example.content_hashes()
    return example


@dataclass
class GenerationStats:
    """What a generation run did, reported honestly."""

    requested: int = 0
    skipped_existing: int = 0
    generated: int = 0
    failed: int = 0
    failed_ids: list[str] = field(default_factory=list)
    shards_written: list[str] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        return self.generated + self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "skipped_existing": self.skipped_existing,
            "attempted": self.attempted,
            "generated": self.generated,
            "failed": self.failed,
            "failed_ids": self.failed_ids[:100],
            "shards_written": self.shards_written,
        }


class ShardWriter:
    """Append records to size-bounded shards, closing each with a checksum.

    A shard is registered in the manifest as incomplete while open and rewritten as
    complete once closed and hashed, so an interrupted run leaves a manifest that says
    so rather than one that overstates what is there.
    """

    def __init__(self, directory: str | Path, manifest: DatasetManifest,
                 *, shard_size: int = 1000, start_index: int = 0) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.shard_size = max(1, shard_size)
        self.index = start_index
        self._handle = None
        self._count = 0
        self._failed = 0
        self._name: str | None = None

    @property
    def current_shard(self) -> str | None:
        return self._name

    def _open(self) -> None:
        while (self.directory / shard_name(self.index)).is_file():
            existing = self.manifest.shard(shard_name(self.index))
            if existing is None or not existing.complete:
                break  # resume into a shard that was never closed
            self.index += 1
        self._name = shard_name(self.index)
        path = self.directory / self._name
        self._count = sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                          if line.strip()) if path.is_file() else 0
        self._handle = path.open("a", encoding="utf-8")
        self.manifest.add_shard(ShardRecord(
            name=self._name, n_records=self._count, complete=False
        ))

    def write(self, example: DistillationExample) -> None:
        if self._handle is None:
            self._open()
        assert self._handle is not None and self._name is not None
        self._handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
        # Flush and fsync per record: the whole point is that a terminated instance
        # loses at most the generation in flight, not the last N.
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._count += 1
        if self._count >= self.shard_size:
            self.close()
            self.index += 1

    def note_failure(self) -> None:
        self._failed += 1

    def close(self) -> None:
        """Close the open shard and record it as complete, with its checksum."""
        if self._handle is None or self._name is None:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        path = self.directory / self._name
        self.manifest.add_shard(ShardRecord(
            name=self._name,
            n_records=self._count,
            n_failed=self._failed,
            sha256=sha256_file(path),
            bytes=path.stat().st_size if path.is_file() else 0,
            complete=True,
        ))
        self._handle = None
        self._name = None
        self._count = 0
        self._failed = 0


def generate_dataset(
    prompts: list[Prompt],
    backend: TeacherBackend,
    mode: ReasoningMode,
    output_dir: str | Path,
    *,
    teacher_model: str = "unknown",
    teacher_revision: str | None = None,
    chat_template_sha256: str | None = None,
    generation_config: dict[str, Any] | None = None,
    shard_size: int = 1000,
    resume: bool = True,
    limit: int | None = None,
    source: str = "teacher_generation",
    on_progress: Callable[[int, int, str], None] | None = None,
    manifest_every: int = 50,
) -> tuple[DatasetManifest, GenerationStats]:
    """Generate teacher responses into sharded JSONL, resumably.

    Returns the manifest and honest statistics. Failures are counted and their ids
    recorded rather than being dropped, because a dataset that is 3% short for unknown
    reasons is a different object from one that is complete.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(generation_config or {})

    manifest = DatasetManifest.read(directory) if resume else None
    if manifest is None:
        manifest = DatasetManifest()
    manifest.run_manifest = RunManifest(
        kind="teacher_generation",
        teacher={
            "model": teacher_model, "revision": teacher_revision,
            "chat_template_sha256": chat_template_sha256,
            "is_pinned": teacher_revision is not None,
            **backend.describe(),
        },
        reasoning_mode=mode.name,
        generation_config=config,
        dataset={"path": str(directory), "shard_size": shard_size},
    ).to_dict()

    already: set[str] = set()
    if resume:
        already, _ = scan_completed_ids(directory)

    pending = [p for p in prompts if p.id not in already]
    if limit is not None:
        pending = pending[:limit]

    stats = GenerationStats(
        requested=len(prompts), skipped_existing=len(prompts) - len(pending)
    )
    manifest.n_prompts_requested = len(prompts)
    manifest.prompt_set_sha256 = prompt_set_digest(prompts)
    if backend.describe().get("is_synthetic"):
        note = ("SYNTHETIC: produced by the mock teacher. Not real teacher output; "
                "must never be used as training data or reported as teacher behaviour.")
        if note not in manifest.notes:
            manifest.notes.append(note)

    writer = ShardWriter(directory, manifest, shard_size=shard_size)
    config_digest = sha256_json(config) if config else None

    try:
        for position, prompt in enumerate(pending, 1):
            response = backend.generate(prompt.prompt, mode=mode)
            if not response.ok:
                stats.failed += 1
                stats.failed_ids.append(prompt.id)
                writer.note_failure()
            else:
                writer.write(build_example(
                    prompt, response, mode=mode,
                    teacher_model=teacher_model, teacher_revision=teacher_revision,
                    chat_template_sha256=chat_template_sha256,
                    generation_config_sha256=config_digest,
                    source=source, dataset_version=manifest.dataset_version,
                ))
                stats.generated += 1
            if on_progress is not None:
                on_progress(position, len(pending), prompt.id)
            # Periodic manifest writes so a killed run leaves a usable record of how far
            # it got, not just the shards.
            if manifest_every and position % manifest_every == 0:
                manifest.n_generated = len(already) + stats.generated
                manifest.n_failed = stats.failed
                manifest.write(directory)
    finally:
        writer.close()
        stats.shards_written = [s.name for s in manifest.shards if s.complete]
        manifest.n_generated = len(already) + stats.generated
        manifest.n_failed = stats.failed
        manifest.failed_ids = stats.failed_ids
        manifest.complete = (
            manifest.n_generated + stats.failed >= manifest.n_prompts_requested
        )
        manifest.write(directory)

    return manifest, stats


def iter_records(directory: str | Path, *, verified_only: bool = True) -> Iterator[DistillationExample]:
    """Stream records from a dataset directory, newest-manifest-aware.

    ``verified_only`` restricts reading to shards the manifest records as complete, so a
    partially written shard from an interrupted run is never silently trained on.
    """
    base = Path(directory)
    manifest = DatasetManifest.read(base)
    if verified_only and manifest is not None:
        names = [s.name for s in manifest.shards if s.complete]
    else:
        names = sorted(p.name for p in base.glob("shard-*.jsonl"))
    for name in names:
        path = base / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield DistillationExample.from_dict(payload)
