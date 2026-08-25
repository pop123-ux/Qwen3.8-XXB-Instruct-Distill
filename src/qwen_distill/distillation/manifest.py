"""The manifest that makes a sharded teacher dataset trustworthy.

Teacher generation will happen across rented GPU sessions, possibly weeks apart and on
different machines. What arrives afterwards is a directory of JSONL shards, and the only
questions that matter about it are: is this complete, is it intact, and what produced
it? A directory listing cannot answer any of them.

So a shard is only counted once it is closed and checksummed, and the manifest records
the provenance alongside. A shard still being written is visible as incomplete rather
than being silently treated as a short one — the difference between "1,000 examples" and
"1,000 examples so far" is the difference between a usable dataset and a truncated one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .provenance import sha256_file, utc_now

MANIFEST_NAME = "manifest.json"
SHARD_PREFIX = "shard-"
SHARD_SUFFIX = ".jsonl"

#: Bumped when the manifest's meaning changes incompatibly.
MANIFEST_VERSION = "1.0"


def shard_name(index: int) -> str:
    """Zero-padded so lexical order matches numeric order in any listing."""
    return f"{SHARD_PREFIX}{index:05d}{SHARD_SUFFIX}"


@dataclass
class ShardRecord:
    """One closed shard: what is in it, and proof it has not changed since."""

    name: str
    n_records: int
    n_failed: int = 0
    sha256: str | None = None
    bytes: int = 0
    complete: bool = False
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShardRecord:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def verify(self, directory: str | Path) -> tuple[bool, str | None]:
        """Check the shard on disk still matches what was recorded."""
        path = Path(directory) / self.name
        if not path.is_file():
            return False, "missing"
        if self.sha256 is None:
            return True, "no checksum recorded"
        actual = sha256_file(path)
        if actual != self.sha256:
            found = actual[:12] if actual else "unreadable"
            return False, (
                f"checksum mismatch (recorded {self.sha256[:12]}, found {found})"
            )
        return True, None


@dataclass
class DatasetManifest:
    """What a teacher-output directory contains, and what produced it."""

    dataset_version: str = MANIFEST_VERSION
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    #: Everything about the teacher, reasoning mode and software — see provenance.py.
    run_manifest: dict[str, Any] = field(default_factory=dict)
    shards: list[ShardRecord] = field(default_factory=list)
    #: Prompts requested, which is not the same as records produced: failures and
    #: interruptions make the difference, and hiding it would overstate the dataset.
    n_prompts_requested: int = 0
    n_generated: int = 0
    n_failed: int = 0
    failed_ids: list[str] = field(default_factory=list)
    #: Digest over the ordered prompt ids, so two datasets can be compared without
    #: reading either in full.
    prompt_set_sha256: str | None = None
    complete: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def n_records(self) -> int:
        return sum(shard.n_records for shard in self.shards if shard.complete)

    @property
    def n_incomplete_shards(self) -> int:
        return sum(1 for shard in self.shards if not shard.complete)

    def shard(self, name: str) -> ShardRecord | None:
        return next((s for s in self.shards if s.name == name), None)

    def add_shard(self, record: ShardRecord) -> None:
        existing = self.shard(record.name)
        if existing is not None:
            self.shards.remove(existing)
        self.shards.append(record)
        self.shards.sort(key=lambda s: s.name)
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "run_manifest": self.run_manifest,
            "shards": [s.to_dict() for s in self.shards],
            "n_prompts_requested": self.n_prompts_requested,
            "n_generated": self.n_generated,
            "n_failed": self.n_failed,
            "failed_ids": self.failed_ids[:100],
            "n_records": self.n_records,
            "n_incomplete_shards": self.n_incomplete_shards,
            "prompt_set_sha256": self.prompt_set_sha256,
            "complete": self.complete,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        manifest = cls(
            dataset_version=data.get("dataset_version", MANIFEST_VERSION),
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
            run_manifest=data.get("run_manifest", {}),
            n_prompts_requested=data.get("n_prompts_requested", 0),
            n_generated=data.get("n_generated", 0),
            n_failed=data.get("n_failed", 0),
            failed_ids=data.get("failed_ids", []),
            prompt_set_sha256=data.get("prompt_set_sha256"),
            complete=data.get("complete", False),
            notes=data.get("notes", []),
        )
        manifest.shards = [ShardRecord.from_dict(s) for s in data.get("shards", [])]
        return manifest

    def write(self, directory: str | Path) -> Path:
        """Write atomically: a reader must never see a half-written manifest."""
        from ..training.checkpoints import atomic_write_json

        target = Path(directory) / MANIFEST_NAME
        self.updated_at = utc_now()
        atomic_write_json(target, self.to_dict())
        return target

    @classmethod
    def read(cls, directory: str | Path) -> DatasetManifest | None:
        path = Path(directory) / MANIFEST_NAME
        if not path.is_file():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return None

    def verify(self, directory: str | Path) -> dict[str, Any]:
        """Check every recorded shard is present and unchanged."""
        problems: list[str] = []
        verified: list[str] = []
        for record in self.shards:
            if not record.complete:
                problems.append(f"{record.name}: recorded as incomplete")
                continue
            ok, reason = record.verify(directory)
            (verified if ok else problems).append(
                record.name if ok else f"{record.name}: {reason}"
            )
        # A shard on disk that the manifest does not know about is usually an
        # interrupted write; it must not be read as data.
        known = {s.name for s in self.shards}
        stray = sorted(
            p.name for p in Path(directory).glob(f"{SHARD_PREFIX}*{SHARD_SUFFIX}")
            if p.name not in known
        )
        return {
            "ok": not problems and not stray,
            "verified_shards": verified,
            "problems": problems,
            "unlisted_shards": stray,
            "n_records": self.n_records,
        }

    def render(self) -> str:
        lines = [
            f"dataset manifest v{self.dataset_version}  ({self.updated_at})",
            f"  records    : {self.n_records:,} across {len(self.shards)} shard(s)",
            f"  requested  : {self.n_prompts_requested:,} prompt(s)",
            f"  generated  : {self.n_generated:,}   failed: {self.n_failed:,}",
            f"  complete   : {self.complete}",
        ]
        if self.n_incomplete_shards:
            lines.append(f"  ! {self.n_incomplete_shards} shard(s) recorded incomplete")
        teacher = (self.run_manifest or {}).get("teacher") or {}
        if teacher:
            lines.append(f"  teacher    : {teacher.get('model')} "
                         f"@ {teacher.get('revision') or 'UNPINNED'}")
        if (self.run_manifest or {}).get("reasoning_mode"):
            lines.append(f"  reasoning  : {self.run_manifest['reasoning_mode']}")
        for note in self.notes:
            lines.append(f"  ! {note}")
        return "\n".join(lines)
