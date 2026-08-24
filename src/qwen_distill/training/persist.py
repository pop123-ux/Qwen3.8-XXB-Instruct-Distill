"""Copy completed checkpoints somewhere that outlives the Colab runtime.

A checkpoint on `/content` is not saved; it is staged. When the runtime recycles it is
gone — which is how one Level-2 run lost ~500 steps, and how a later one lost **1925 of
2000 steps** because it ran with `persistent copy : off (local only)`.

Three rules, each earned:

**Never publish a partial checkpoint.** The source must be complete before the copy
starts, and the copy must verify as complete at the destination before anything points
at it. A half-written checkpoint on Drive is worse than none, because it is the newest
thing there and recovery reaches for the newest.

**Never claim a copy that did not happen.** A failed copy leaves training running — the
local checkpoint is intact and losing the copy is recoverable — but it is reported
loudly and recorded, so no one reads "persisted" and believes it.

**Never mistake an unmounted Drive for a mounted one.** `mkdir(parents=True)` on
`/content/drive/MyDrive/...` succeeds when Drive is *not* mounted: it silently creates
an ordinary local directory that looks exactly like the real thing, and every checkpoint
"persists" into the ephemeral filesystem. That reproduces the failure this module exists
to prevent, while reporting success. So the mount point is verified to pre-exist before
anything is created under it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .checkpoints import (
    COMPLETE_MARKER,
    atomic_write_json,
    is_complete,
    list_checkpoints,
    read_latest_pointer,
)

#: Run-level files worth mirroring. Kilobytes each, and they are what makes a recovered
#: checkpoint interpretable rather than just loadable.
METADATA_FILES = ("metrics.jsonl", "summary.json", "hardware.json", "git_commit.txt")

#: Path prefixes that only exist because a cloud drive is mounted there. If the mount
#: root is missing, a write under it lands on the ephemeral disk instead.
MOUNT_ROOTS = ("/content/drive/MyDrive", "/content/drive/Shareddrives", "/gdrive/MyDrive")


@dataclass
class PersistenceTarget:
    """A run directory on persistent storage, in the canonical layout.

    The configured path is the **run** directory, not its `checkpoints/` subdirectory,
    so the persistent copy mirrors the local one exactly::

        <destination>/checkpoints/step_000200/ ... latest.json
        <destination>/progress/latest.json
        <destination>/metrics.jsonl, summary.json
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        # Tolerate a path aimed at the checkpoints subdirectory: that was the older
        # meaning of this setting, and silently nesting checkpoints/checkpoints/ would
        # be a confusing way to punish it.
        if self.root.name == "checkpoints":
            self.root = self.root.parent

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def progress(self) -> Path:
        return self.root / "progress"

    @property
    def pointer(self) -> Path:
        return self.checkpoints / "latest.json"


@dataclass
class PreflightResult:
    """Whether persistence can actually work, checked before training starts."""

    destination: Path
    usable: bool = False
    mount_root: str | None = None
    reason: str | None = None
    existing_checkpoints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "destination": str(self.destination),
            "usable": self.usable,
            "mount_root": self.mount_root,
            "reason": self.reason,
            "existing_checkpoints": self.existing_checkpoints,
        }

    def render(self) -> str:
        if self.usable:
            lines = [f"    persistent copy : {self.destination}"]
            if self.existing_checkpoints:
                lines.append(
                    f"                      {len(self.existing_checkpoints)} checkpoint(s) "
                    f"already there, newest {self.existing_checkpoints[-1]}"
                )
            return "\n".join(lines)
        return (
            f"    persistent copy : UNUSABLE — {self.reason}\n"
            f"                      destination: {self.destination}"
        )


def preflight(destination: str | Path) -> PreflightResult:
    """Check that persistence will really persist, before a single step is trained.

    Catching this at startup rather than at step 200 is the whole point: the failure
    mode is silent, and by step 200 there is real work to lose.
    """
    target = PersistenceTarget(Path(destination))
    result = PreflightResult(destination=target.root)

    resolved = str(target.root.resolve()) if target.root.is_absolute() else str(target.root)
    for mount in MOUNT_ROOTS:
        if resolved == mount or resolved.startswith(mount + "/"):
            result.mount_root = mount
            if not Path(mount).is_dir():
                result.reason = (
                    f"{mount} does not exist, so Drive is not mounted. Writing here "
                    "would create an ordinary local directory that disappears with the "
                    "runtime — the exact loss this is meant to prevent. Mount Drive "
                    "first:\n"
                    "                        from google.colab import drive\n"
                    "                        drive.mount('/content/drive')"
                )
                return result
            break

    parent = target.root if target.root.is_dir() else target.root.parent
    while not parent.is_dir() and parent != parent.parent:
        parent = parent.parent
    try:
        target.checkpoints.mkdir(parents=True, exist_ok=True)
        probe = target.checkpoints / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        result.reason = f"cannot write to the destination: {type(exc).__name__}: {exc}"
        return result

    result.usable = True
    result.existing_checkpoints = [p.name for p in list_checkpoints(target.checkpoints)]
    return result


def persist_checkpoint(
    checkpoint: str | Path,
    destination: str | Path,
    *,
    checkpoint_root: str | Path | None = None,
) -> Path:
    """Copy one complete checkpoint to persistent storage and publish it.

    The order is the guarantee: copy to a staging name, rename into place, **verify the
    destination copy is itself complete**, and only then update the destination pointer.
    A reader on the far side gets the same promise as a reader here — `latest.json`
    never names a directory that is not a whole checkpoint.
    """
    source = Path(checkpoint)
    if not is_complete(source):
        raise ValueError(
            f"refusing to persist {source}: it is not a complete checkpoint. Copying a "
            "partial write would put a corrupt checkpoint where recovery looks first."
        )

    target = PersistenceTarget(Path(destination))
    target.checkpoints.mkdir(parents=True, exist_ok=True)
    final = target.checkpoints / source.name
    staging = target.checkpoints / f".{source.name}.incomplete"

    shutil.rmtree(staging, ignore_errors=True)
    try:
        # symlinks=False: never follow a link out of the checkpoint directory.
        shutil.copytree(source, staging, symlinks=False)
        if not (staging / COMPLETE_MARKER).is_file():
            raise OSError(f"copy of {source} did not carry its completion marker")
        shutil.rmtree(final, ignore_errors=True)
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Verify what actually landed, not what we believe we sent. A truncated file on a
    # full or flaky Drive would otherwise be advertised as the newest good checkpoint.
    if not is_complete(final):
        raise OSError(
            f"the copy at {final} does not verify as a complete checkpoint. The "
            "destination pointer was NOT advanced; the previous persisted checkpoint "
            "remains the newest one there."
        )

    step = _checkpoint_step(final, fallback_root=checkpoint_root, name=source.name)
    atomic_write_json(target.pointer, {
        "step": step,
        "path": final.name,
        "created_at": _created_at(final),
        "complete": True,
    })
    return final


def _checkpoint_step(checkpoint: Path, *, fallback_root: str | Path | None, name: str) -> int:
    """The step a persisted checkpoint represents, read from the copy itself."""
    try:
        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        if isinstance(metadata.get("step"), int):
            return metadata["step"]
    except (OSError, json.JSONDecodeError):
        pass
    if fallback_root is not None:
        pointer = read_latest_pointer(fallback_root)
        if pointer and pointer.get("path") == name and isinstance(pointer.get("step"), int):
            return pointer["step"]
    return int(name.rsplit("_", 1)[-1]) if name.rsplit("_", 1)[-1].isdigit() else 0


def _created_at(checkpoint: Path) -> str:
    try:
        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        return str(metadata.get("created_at", ""))
    except (OSError, json.JSONDecodeError):
        return ""


def persist_run_metadata(run_directory: str | Path, destination: str | Path) -> list[str]:
    """Mirror the small files that make a persisted checkpoint interpretable.

    Kilobytes, so this runs alongside every checkpoint. Without them a recovered
    checkpoint loads but tells you nothing about the run it came from.
    """
    source = Path(run_directory)
    target = PersistenceTarget(Path(destination))
    target.root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in METADATA_FILES:
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, target.root / name)
            copied.append(name)
    latest = source / "progress" / "latest.json"
    if latest.is_file():
        target.progress.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest, target.progress / "latest.json")
        copied.append("progress/latest.json")
    return copied


def persistent_status(destination: str | Path) -> dict[str, Any]:
    """What is actually on persistent storage — the question a fresh session asks."""
    target = PersistenceTarget(Path(destination))
    checkpoints = list_checkpoints(target.checkpoints)
    pointer = read_latest_pointer(target.checkpoints)

    latest_progress = None
    progress_file = target.progress / "latest.json"
    if progress_file.is_file():
        try:
            latest_progress = json.loads(progress_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            latest_progress = None

    resumable = None
    if pointer and pointer.get("complete"):
        candidate = target.checkpoints / str(pointer.get("path", ""))
        if is_complete(candidate):
            resumable = candidate
    if resumable is None and checkpoints:
        resumable = checkpoints[-1]

    return {
        "destination": str(target.root),
        "exists": target.root.is_dir(),
        "checkpoints": [p.name for p in checkpoints],
        "pointer": pointer,
        "latest_progress": latest_progress,
        "resumable_checkpoint": str(resumable) if resumable else None,
        "resumable_step": pointer.get("step") if pointer else None,
    }


def restore_run(destination: str | Path, local_directory: str | Path) -> dict[str, Any]:
    """Bring a persisted run back onto local disk, for a fresh Colab session.

    Only complete checkpoints are restored — a partial directory on Drive (from a copy
    interrupted by a dying runtime) must not become the thing a resume reaches for.
    """
    target = PersistenceTarget(Path(destination))
    local = Path(local_directory)
    if not target.root.is_dir():
        raise FileNotFoundError(
            f"nothing to restore: {target.root} does not exist. If this is a Drive path, "
            "check that Drive is mounted and the run name is right."
        )

    local_checkpoints = local / "checkpoints"
    local_checkpoints.mkdir(parents=True, exist_ok=True)

    restored: list[str] = []
    skipped: list[str] = []
    for source in sorted(target.checkpoints.iterdir()) if target.checkpoints.is_dir() else []:
        if not source.is_dir() or not source.name.startswith("step_"):
            continue
        if not is_complete(source):
            skipped.append(source.name)
            continue
        local_copy = local_checkpoints / source.name
        if is_complete(local_copy):
            continue  # already here and verified; copying again buys nothing
        staging = local_checkpoints / f".{source.name}.incomplete"
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(source, staging, symlinks=False)
        shutil.rmtree(local_copy, ignore_errors=True)
        staging.rename(local_copy)
        restored.append(source.name)

    for name in METADATA_FILES:
        candidate = target.root / name
        if candidate.is_file():
            shutil.copy2(candidate, local / name)
    progress_file = target.progress / "latest.json"
    if progress_file.is_file():
        (local / "progress").mkdir(parents=True, exist_ok=True)
        shutil.copy2(progress_file, local / "progress" / "latest.json")

    # Rebuild the local pointer from what verifiably arrived, rather than copying the
    # remote pointer: it may name a checkpoint that failed to restore.
    available = list_checkpoints(local_checkpoints)
    pointer = None
    if available:
        newest = available[-1]
        step = _checkpoint_step(newest, fallback_root=None, name=newest.name)
        pointer = {"step": step, "path": newest.name,
                   "created_at": _created_at(newest), "complete": True}
        atomic_write_json(local_checkpoints / "latest.json", pointer)

    return {
        "destination": str(target.root),
        "local": str(local),
        "restored": restored,
        "skipped_incomplete": skipped,
        "available_locally": [p.name for p in available],
        "pointer": pointer,
    }
