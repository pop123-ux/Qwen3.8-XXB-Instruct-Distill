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
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .checkpoint_validation import (
    COMPLETE_MARKER,
    LOAD,
    MANIFEST,
    MANIFEST_FILENAME,
    STRUCTURE,
    CheckpointValidation,
    build_manifest,
    checkpoint_directories,
    read_manifest,
    resolve_latest,
    sha256_file,
    validate_checkpoint_dir,
    validate_checkpoint_root,
)
from .checkpoints import (
    atomic_write_json,
    read_latest_pointer,
)

#: How thoroughly a copy is checked *at the destination* before it is called persisted.
#: ``manifest`` — every file's size and SHA-256 compared against the source — is the
#: default because byte-identity to a source that loads is a stronger statement than "it
#: loaded once", and it costs one read of the copy rather than a full deserialization.
#: ``load`` additionally deserializes the weights and optimizer state from the
#: destination; use it when the destination is a filesystem you distrust.
DEFAULT_VERIFY_LEVEL = MANIFEST

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
    #: Checkpoints that verify. Named ``existing_`` for compatibility, but the contents
    #: are verified, not merely present.
    existing_checkpoints: list[str] = field(default_factory=list)
    #: Checkpoints that are there and are not usable, with the reason. Reported at
    #: startup because an operator who does not learn about them here learns about them
    #: when a resume fails hours later.
    invalid_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    newest_valid: str | None = None
    newest_valid_step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "destination": str(self.destination),
            "usable": self.usable,
            "mount_root": self.mount_root,
            "reason": self.reason,
            "existing_checkpoints": self.existing_checkpoints,
            "invalid_checkpoints": self.invalid_checkpoints,
            "newest_valid": self.newest_valid,
            "newest_valid_step": self.newest_valid_step,
        }

    def render(self) -> str:
        if not self.usable:
            return (
                f"    persistent copy : UNUSABLE — {self.reason}\n"
                f"                      destination: {self.destination}"
            )
        lines = [f"    persistent copy : {self.destination}"]
        total = len(self.existing_checkpoints) + len(self.invalid_checkpoints)
        if total:
            lines.append(
                f"                      {total} checkpoint(s) there, "
                f"{len(self.existing_checkpoints)} verified resumable"
            )
        for entry in self.invalid_checkpoints:
            # Loud on purpose: a hollow checkpoint on Drive is the failure this whole
            # module exists to make impossible to overlook.
            lines.append(
                f"                      ! {entry['name']} INVALID — {entry['reason']}"
            )
        if self.newest_valid:
            lines.append(
                f"                      newest valid: {self.newest_valid} "
                f"(step {self.newest_valid_step})"
            )
        elif total:
            lines.append("                      ! NO VALID CHECKPOINT on persistent storage")
        return "\n".join(lines)


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

    # Storage health, before a single step is trained. Every checkpoint already there is
    # validated now, so a run does not start believing it has recovery points it lost.
    validations = validate_checkpoint_root(target.checkpoints, level=STRUCTURE)
    result.existing_checkpoints = [Path(v.path).name for v in validations if v.valid]
    result.invalid_checkpoints = [
        {"name": Path(v.path).name, "reason": v.invalid_reason}
        for v in validations if not v.valid
    ]
    resolution = resolve_latest(target.checkpoints, validations=validations)
    if resolution.resolved:
        result.newest_valid = Path(resolution.resolved).name
        result.newest_valid_step = resolution.resolved_step
    return result


@dataclass
class FileTransfer:
    """One file's journey, checked at the far end."""

    name: str
    size_bytes: int | None = None
    expected_size: int | None = None
    sha256: str | None = None
    expected_sha256: str | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "size_bytes": self.size_bytes,
            "expected_size": self.expected_size, "sha256": self.sha256,
            "expected_sha256": self.expected_sha256,
            "ok": self.ok, "problems": self.problems,
        }


@dataclass
class PersistResult:
    """Whether a checkpoint is *actually* on persistent storage, and how that is known.

    Returned instead of a bare path so the caller can print the per-file verdicts the
    operator needs, rather than a word that has been wrong before.
    """

    checkpoint: str
    step: int | None = None
    destination: str | None = None
    verify_level: str = DEFAULT_VERIFY_LEVEL
    verified: bool = False
    pointer_updated: bool = False
    files: list[FileTransfer] = field(default_factory=list)
    source_validation: CheckpointValidation | None = None
    destination_validation: CheckpointValidation | None = None
    failure: str | None = None
    staging_left_behind: str | None = None

    @property
    def failed_files(self) -> list[FileTransfer]:
        return [f for f in self.files if not f.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint, "step": self.step,
            "destination": self.destination, "verify_level": self.verify_level,
            "verified": self.verified, "pointer_updated": self.pointer_updated,
            "failure": self.failure,
            "staging_left_behind": self.staging_left_behind,
            "files": [f.to_dict() for f in self.files],
            "destination_validation": (
                self.destination_validation.to_dict() if self.destination_validation else None
            ),
        }

    def render(self) -> str:
        """The operator-facing report. ``persisted ->`` appears only on success."""
        lines = [f"    checkpoint: {self.checkpoint}"]
        if self.verified:
            lines.append(f"    persistent verification: PASS ({self.verify_level})")
            for transfer in self.files:
                if transfer.name in ("model.safetensors", "optimizer.pt"):
                    lines.append(f"      {transfer.name}: PASS ({transfer.size_bytes:,} bytes)")
            lines.append("      checksum verification: PASS")
            if self.verify_level == LOAD:
                lines.append("      checkpoint reload: PASS")
            lines.append(f"    persisted -> {self.destination}")
            return "\n".join(lines)

        lines.append("    persistent verification: FAILED")
        if self.failure:
            lines.append(f"      {self.failure}")
        missing = [f.name for f in self.failed_files if "MISSING" in " ".join(f.problems)]
        if missing:
            lines.append("")
            lines.append("    missing:")
            lines.extend(f"      {name}" for name in missing)
        other = [f for f in self.failed_files if f.name not in missing]
        if other:
            lines.append("")
            lines.append("    failed verification:")
            for transfer in other:
                lines.append(f"      {transfer.name}: {'; '.join(transfer.problems)}")
        lines.append("")
        lines.append("    latest pointer NOT updated.")
        if self.staging_left_behind:
            lines.append(f"    incomplete copy left at {self.staging_left_behind}")
        return "\n".join(lines)


def _copy_and_fsync(source: Path, destination: Path) -> None:
    """Copy one file and force it to durable storage.

    ``shutil.copytree`` does neither of the two things that matter on a Drive FUSE mount:
    it never fsyncs, and ``close()`` there can return long before the bytes are actually
    uploaded. A checkpoint verified immediately after such a copy can read back correctly
    from cache and be absent or truncated on disk minutes later. fsync is the strongest
    request available; it is not a guarantee on every FUSE implementation, which is why
    the destination is re-read and digested afterwards rather than trusted.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as reader, open(destination, "wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    shutil.copystat(source, destination)


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # not every filesystem supports it; the rename is still atomic
    finally:
        os.close(fd)


def _source_expectations(source: Path, step: int) -> dict[str, dict[str, Any]]:
    """What every file in the source actually is: size and SHA-256.

    Prefers the checkpoint's own ``checkpoint_manifest.json`` when it has one — that was
    computed when the bytes were fresh — and otherwise digests the source now. Either way
    the destination is compared against **the source's real bytes**, so a passing copy is
    byte-identical to a checkpoint that has itself been validated.
    """
    recorded = read_manifest(source)
    if recorded and recorded.get("files"):
        expectations = dict(recorded["files"])
        # The manifest cannot describe itself, so it is digested live.
        if (source / MANIFEST_FILENAME).is_file():
            expectations[MANIFEST_FILENAME] = {
                "size_bytes": (source / MANIFEST_FILENAME).stat().st_size,
                "sha256": sha256_file(source / MANIFEST_FILENAME),
            }
        return expectations
    return dict(build_manifest(source, step=step)["files"])


def persist_checkpoint(
    checkpoint: str | Path,
    destination: str | Path,
    *,
    checkpoint_root: str | Path | None = None,
    verify_level: str = DEFAULT_VERIFY_LEVEL,
) -> PersistResult:
    """Copy one checkpoint to persistent storage and prove it arrived intact.

    The source being valid is not evidence about the destination. A Level-2R run left
    three directories on Drive carrying ``COMPLETE``, ``metadata.json`` and every small
    file — and no ``model.safetensors``, no ``optimizer.pt``. The old code copied with
    ``shutil.copytree`` (no fsync), then asked whether the required *filenames* existed at
    the far end, then printed ``persisted ->``.

    The order here is the guarantee, and every step is a gate:

    1. validate the **source**; refuse to copy a checkpoint that is already damaged;
    2. take the source's real sizes and digests;
    3. copy every file except ``COMPLETE`` into ``.step_NNNNNN.incomplete/``, fsyncing each;
    4. **re-read the destination** and compare size and SHA-256 against the source;
    5. write ``COMPLETE`` at the destination — only now, so a staging directory that
       failed verification can never look finished;
    6. run the full validator against the staging copy;
    7. promote atomically and re-validate the promoted directory;
    8. only then update the destination's ``latest.json``.

    A failure at any step leaves the previously persisted checkpoint untouched and the
    pointer where it was, and returns a result whose ``verified`` is ``False``. Nothing
    prints ``persisted`` on that path.
    """
    source = Path(checkpoint)
    result = PersistResult(checkpoint=source.name, verify_level=verify_level)

    source_validation = validate_checkpoint_dir(source, level=STRUCTURE)
    result.source_validation = source_validation
    result.step = source_validation.step
    if not source_validation.valid:
        result.failure = (
            f"refusing to persist {source}: {source_validation.invalid_reason}. Copying a "
            "damaged checkpoint would put it where recovery looks first."
        )
        return result

    target = PersistenceTarget(Path(destination))
    target.checkpoints.mkdir(parents=True, exist_ok=True)
    final = target.checkpoints / source.name
    staging = target.checkpoints / f".{source.name}.incomplete"
    result.destination = str(final)

    expectations = _source_expectations(source, result.step or 0)

    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for child in sorted(source.iterdir()):
            if not child.is_file() or child.name == COMPLETE_MARKER:
                continue
            _copy_and_fsync(child, staging / child.name)
        _fsync_directory(staging)
    except OSError as exc:
        result.failure = f"copy failed: {type(exc).__name__}: {exc}"
        result.staging_left_behind = str(staging)
        return result

    # --- verify what actually landed, by re-reading it --------------------------
    for name in sorted(expectations):
        expected = expectations[name]
        transfer = FileTransfer(
            name=name,
            expected_size=expected.get("size_bytes"),
            expected_sha256=expected.get("sha256"),
        )
        arrived = staging / name
        if not arrived.is_file():
            transfer.problems.append("MISSING at destination")
            result.files.append(transfer)
            continue
        transfer.size_bytes = arrived.stat().st_size
        if transfer.size_bytes == 0:
            transfer.problems.append("ZERO LENGTH at destination")
        elif transfer.expected_size is not None and transfer.size_bytes != transfer.expected_size:
            transfer.problems.append(
                f"size {transfer.size_bytes:,} != source {transfer.expected_size:,}"
            )
        if not transfer.problems and transfer.expected_sha256:
            try:
                transfer.sha256 = sha256_file(arrived)
            except OSError as exc:
                transfer.problems.append(f"unreadable at destination: {type(exc).__name__}")
            else:
                if transfer.sha256 != transfer.expected_sha256:
                    transfer.problems.append(
                        f"sha256 {transfer.sha256[:16]} != source "
                        f"{transfer.expected_sha256[:16]}"
                    )
        result.files.append(transfer)

    if result.failed_files:
        result.failure = (
            f"{len(result.failed_files)} file(s) did not arrive intact at {final}"
        )
        result.staging_left_behind = str(staging)
        return result

    # --- the marker is written at the destination, after verification -----------
    try:
        marker = staging / COMPLETE_MARKER
        with open(marker, "w", encoding="utf-8") as stream:
            stream.write(
                f"step {result.step} verified at destination {_utc_now()}\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(staging)
    except OSError as exc:
        result.failure = f"could not write the completion marker: {exc}"
        result.staging_left_behind = str(staging)
        return result

    staging_validation = validate_checkpoint_dir(staging, level=verify_level)
    if not staging_validation.valid:
        result.destination_validation = staging_validation
        result.failure = (
            f"the copy at {staging} does not validate: {staging_validation.invalid_reason}"
        )
        result.staging_left_behind = str(staging)
        return result

    # --- promote, then verify the promoted directory ----------------------------
    try:
        shutil.rmtree(final, ignore_errors=True)
        os.replace(staging, final)
        _fsync_directory(target.checkpoints)
    except OSError as exc:
        result.failure = f"could not promote {staging} to {final}: {exc}"
        result.staging_left_behind = str(staging)
        return result

    promoted = validate_checkpoint_dir(final, level=STRUCTURE)
    result.destination_validation = promoted
    if not promoted.valid:
        result.failure = (
            f"the promoted copy at {final} does not validate: {promoted.invalid_reason}. "
            "The destination pointer was NOT advanced; the previous persisted checkpoint "
            "remains the newest one there."
        )
        return result

    result.verified = True
    step = promoted.step if promoted.step is not None else _checkpoint_step(
        final, fallback_root=checkpoint_root, name=source.name
    )
    result.step = step
    atomic_write_json(target.pointer, {
        "step": step,
        "path": final.name,
        "created_at": _created_at(final),
        "verified_at": _utc_now(),
        "verify_level": verify_level,
        "complete": True,
    })
    result.pointer_updated = True
    return result


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


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


def persistent_status(
    destination: str | Path, *, level: str = STRUCTURE
) -> dict[str, Any]:
    """What is *verifiably* on persistent storage — the question a fresh session asks.

    Every checkpoint is validated **now**, not read out of a pointer written weeks ago.
    That is what turns an accidental deletion into a visible INVALID line instead of a
    checkpoint count that quietly stays at four while three of them are hollow.
    """
    target = PersistenceTarget(Path(destination))
    validations = validate_checkpoint_root(target.checkpoints, level=level)
    resolution = resolve_latest(target.checkpoints, validations=validations)
    pointer = read_latest_pointer(target.checkpoints)

    latest_progress = None
    progress_file = target.progress / "latest.json"
    if progress_file.is_file():
        try:
            latest_progress = json.loads(progress_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            latest_progress = None

    invalid = [v for v in validations if not v.valid]
    return {
        "destination": str(target.root),
        "exists": target.root.is_dir(),
        # Only verified checkpoints are counted. Counting directories is what made a run
        # with three hollow checkpoints look like it had four recovery points.
        "checkpoints": [Path(v.path).name for v in validations if v.valid],
        "invalid_checkpoints": [
            {"name": Path(v.path).name, "reason": v.invalid_reason} for v in invalid
        ],
        "all_checkpoints": [Path(v.path).name for v in validations],
        "validations": [v.to_dict() for v in validations],
        "pointer": pointer,
        "pointer_valid": resolution.pointer_valid,
        "pointer_invalid_reason": resolution.pointer_invalid_reason,
        "fell_back": resolution.fell_back,
        "latest_progress": latest_progress,
        "resumable_checkpoint": resolution.resolved,
        # From the checkpoint that actually verified, never from the pointer's own claim.
        "resumable_step": resolution.resolved_step,
        "verify_level": level,
        "inventory": resolution.render_inventory(label="persistent checkpoints found"),
    }


def audit_persistent_checkpoints(
    destination: str | Path, *, level: str = STRUCTURE
) -> list[CheckpointValidation]:
    """Validate every checkpoint on persistent storage, valid or not.

    Used by the startup health check and by ``scripts/validate_checkpoint.py``. Returns
    the invalid ones too — a checkpoint that has lost its weights is the single most
    important thing to show an operator, and filtering it out is how it stays unnoticed.
    """
    target = PersistenceTarget(Path(destination))
    return validate_checkpoint_root(target.checkpoints, level=level)


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
    skipped: list[dict[str, Any]] = []
    for source in checkpoint_directories(target.checkpoints):
        remote = validate_checkpoint_dir(source, level=STRUCTURE)
        if not remote.valid:
            # A checkpoint that lost its weights on Drive — by deletion, by an
            # interrupted copy, by sync — must never be restored and never become the
            # thing a resume reaches for. It is named, not silently dropped.
            skipped.append({"name": source.name, "reason": remote.invalid_reason})
            continue
        local_copy = local_checkpoints / source.name
        if validate_checkpoint_dir(local_copy, level=STRUCTURE).valid:
            continue  # already here and verified; copying again buys nothing
        staging = local_checkpoints / f".{source.name}.incomplete"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        for child in sorted(source.iterdir()):
            if child.is_file():
                _copy_and_fsync(child, staging / child.name)
        _fsync_directory(staging)
        # The restored copy is verified before it is promoted, for the same reason the
        # persisted copy is: arriving is not the same as arriving intact.
        arrived = validate_checkpoint_dir(staging, level=STRUCTURE)
        if not arrived.valid:
            skipped.append({
                "name": source.name,
                "reason": f"restored copy did not verify: {arrived.invalid_reason}",
            })
            shutil.rmtree(staging, ignore_errors=True)
            continue
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
    # remote pointer: it may name a checkpoint that failed to restore, or one that was
    # valid when the pointer was written and is not any more.
    resolution = resolve_latest(local_checkpoints)
    pointer = None
    if resolution.resolved:
        newest = Path(resolution.resolved)
        step = resolution.resolved_step
        if step is None:
            step = _checkpoint_step(newest, fallback_root=None, name=newest.name)
        pointer = {
            "step": step, "path": newest.name, "created_at": _created_at(newest),
            "verified_at": _utc_now(), "complete": True,
        }
        atomic_write_json(local_checkpoints / "latest.json", pointer)

    return {
        "destination": str(target.root),
        "local": str(local),
        "restored": restored,
        "skipped_incomplete": [entry["name"] for entry in skipped],
        "skipped": skipped,
        "available_locally": [Path(v.path).name for v in resolution.checked if v.valid],
        "invalid_locally": [
            {"name": Path(v.path).name, "reason": v.invalid_reason}
            for v in resolution.checked if not v.valid
        ],
        "pointer": pointer,
        "inventory": resolution.render_inventory(label="restored checkpoints"),
    }
