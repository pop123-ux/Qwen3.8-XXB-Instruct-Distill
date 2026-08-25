"""What a run has to record to be reproducible, in one place.

Teacher generation and benchmarking will happen on rented GPUs, months apart, possibly
on different accounts. The result is only worth anything if it can be tied back to the
exact inputs that produced it — and the way that fails is not dramatic: a field gets
forgotten in one script and not another, and two runs stop being comparable without
anyone noticing.

So this is a single reusable record rather than a convention each script re-implements.
Everything is best-effort: an unavailable field is recorded as ``None``, never invented
and never silently dropped, because "we did not capture this" and "this was absent" are
different claims.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Bumped when a manifest's meaning changes incompatibly.
MANIFEST_VERSION = "1.0"

#: Packages whose version can change a result. Absence is recorded, not an error.
TRACKED_PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "numpy")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str | None:
    """Digest a file without reading it all into memory. ``None`` if unreadable."""
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    """Digest a structure by its canonical form, so key order cannot change it."""
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def git_dirty() -> bool | None:
    """Whether the working tree has uncommitted changes.

    A result produced from a dirty tree is not reproducible from its commit alone, and
    that is worth recording rather than discovering later.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10,
            check=False,
        )
        return bool(result.stdout.strip()) if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except ImportError:
            versions[name] = None
    return versions


def hardware_summary() -> dict[str, Any]:
    """Where this ran. GPU details only when a GPU is actually present."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or None,
        "gpu_name": None,
        "gpu_count": 0,
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except (ImportError, AttributeError, RuntimeError):
        pass
    return info


@dataclass
class TeacherIdentity:
    """Which teacher produced something, precisely enough to reproduce it.

    ``revision`` is separate from ``model`` on purpose. A repo id alone does not pin a
    result: the same id serves different weights over time. An unpinned revision is
    recorded as ``None`` and flagged, rather than being treated as fine.
    """

    model: str
    revision: str | None = None
    tokenizer_sha256: str | None = None
    chat_template_sha256: str | None = None
    config_sha256: str | None = None

    @property
    def is_pinned(self) -> bool:
        return self.revision is not None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_pinned"] = self.is_pinned
        if not self.is_pinned:
            data["pinning_note"] = (
                "revision is unset, so this result is not reproducible from the model id "
                "alone: the same repo id can serve different weights over time"
            )
        return data

    @classmethod
    def from_metadata_dir(cls, model: str, directory: str | Path,
                          revision: str | None = None) -> TeacherIdentity:
        """Hash the files that decide how a prompt is rendered and tokenised."""
        base = Path(directory)
        return cls(
            model=model,
            revision=revision,
            tokenizer_sha256=sha256_file(base / "tokenizer_config.json"),
            chat_template_sha256=sha256_file(base / "chat_template.jinja"),
            config_sha256=sha256_file(base / "config.json"),
        )


@dataclass
class RunManifest:
    """Everything needed to reproduce one generation or evaluation run."""

    kind: str                        # "teacher_generation" | "evaluation" | "benchmark"
    manifest_version: str = MANIFEST_VERSION
    created_at: str = field(default_factory=utc_now)
    git_commit: str | None = field(default_factory=git_commit)
    git_dirty: bool | None = field(default_factory=git_dirty)
    teacher: dict[str, Any] | None = None
    reasoning_mode: str | None = None
    generation_config: dict[str, Any] = field(default_factory=dict)
    generation_config_sha256: str | None = None
    dataset: dict[str, Any] = field(default_factory=dict)
    software: dict[str, str | None] = field(default_factory=package_versions)
    hardware: dict[str, Any] = field(default_factory=hardware_summary)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.generation_config and self.generation_config_sha256 is None:
            self.generation_config_sha256 = sha256_json(self.generation_config)

    def gaps(self) -> list[str]:
        """What would stop someone reproducing this. Empty means nothing known is missing."""
        problems: list[str] = []
        if self.git_commit is None:
            problems.append("git commit unknown")
        if self.git_dirty:
            problems.append("working tree had uncommitted changes")
        if self.teacher and not self.teacher.get("is_pinned"):
            problems.append("teacher revision not pinned")
        if self.teacher and not self.teacher.get("chat_template_sha256"):
            problems.append("chat template not hashed — prompt rendering is unverifiable")
        return problems

    @property
    def fully_reproducible(self) -> bool:
        return not self.gaps()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gaps"] = self.gaps()
        data["fully_reproducible"] = self.fully_reproducible
        return data

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def read(cls, path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def render(self) -> str:
        lines = [f"manifest: {self.kind} ({self.created_at})"]
        lines.append(f"  git          : {self.git_commit or 'unknown'}"
                     + ("  (DIRTY TREE)" if self.git_dirty else ""))
        if self.teacher:
            lines.append(f"  teacher      : {self.teacher.get('model')} "
                         f"@ {self.teacher.get('revision') or 'UNPINNED'}")
        if self.reasoning_mode:
            lines.append(f"  reasoning    : {self.reasoning_mode}")
        if self.dataset:
            lines.append(f"  dataset      : {self.dataset.get('path', '-')}")
        gaps = self.gaps()
        if gaps:
            lines.append("  NOT fully reproducible:")
            lines += [f"    - {gap}" for gap in gaps]
        else:
            lines.append("  fully reproducible")
        return "\n".join(lines)
