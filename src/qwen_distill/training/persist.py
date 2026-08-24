"""Copy completed checkpoints somewhere that outlives the Colab runtime.

A checkpoint on `/content` is not saved; it is staged. When the runtime recycles it is
gone, which is exactly how the Level-2 run lost ~500 steps of real training.

This is deliberately a thin layer. It reuses the safety rules already in
`scripts/backup_colab_to_drive.py` rather than adding a second backup mechanism, and it
adds one rule of its own: **an incomplete checkpoint is never copied**. Publishing a
half-written checkpoint to Drive is worse than not copying it, because it would then be
the newest thing there.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .checkpoints import (
    COMPLETE_MARKER,
    atomic_write_json,
    is_complete,
    read_latest_pointer,
)


def persist_checkpoint(
    checkpoint: str | Path,
    destination: str | Path,
    *,
    checkpoint_root: str | Path | None = None,
) -> Path:
    """Copy one complete checkpoint to ``destination``, then publish the latest pointer.

    Order matters: the checkpoint is copied to a staging name and renamed into place
    before the destination's `latest.json` is updated, so a reader over there sees the
    same guarantee as a reader here — the pointer never names a partial directory.
    """
    source = Path(checkpoint)
    if not is_complete(source):
        raise ValueError(
            f"refusing to persist {source}: it is not a complete checkpoint. "
            "Copying a partial write would put a corrupt checkpoint where the recovery "
            "flow looks first."
        )

    target_root = Path(destination)
    target_root.mkdir(parents=True, exist_ok=True)
    final = target_root / source.name
    staging = target_root / f".{source.name}.incomplete"

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

    # Mirror the pointer only after the copy landed, and only when it agrees.
    if checkpoint_root is not None:
        pointer = read_latest_pointer(checkpoint_root)
        if pointer and pointer.get("path") == source.name:
            atomic_write_json(target_root / "latest.json", pointer)
    return final


def persist_run_metadata(run_directory: str | Path, destination: str | Path) -> list[str]:
    """Copy the small, high-value files that describe a run: metrics, summary, progress.

    These are kilobytes and are what makes a recovered checkpoint interpretable. Copying
    them on every checkpoint is cheap; copying weights that often would not be.
    """
    source = Path(run_directory)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ("metrics.jsonl", "summary.json", "hardware.json", "git_commit.txt"):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, target / name)
            copied.append(name)
    latest = source / "progress" / "latest.json"
    if latest.is_file():
        (target / "progress").mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest, target / "progress" / "latest.json")
        copied.append("progress/latest.json")
    return copied
