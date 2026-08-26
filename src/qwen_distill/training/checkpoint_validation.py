"""The single definition of "is this checkpoint actually usable?".

A Level-2R run reached ~step 800. Persistent storage held `step_000200`, `step_000400`
and `step_000600`, each with `COMPLETE`, `config.json`, `metadata.json`, `rng.pt`,
`scaler.pt`, `scheduler.pt` and `training_state.json` — and **no `model.safetensors`, no
`optimizer.pt`**. `latest.json` said `"complete": true`. Whether those files were deleted
by hand, dropped by Drive, or lost to an interrupted copy is not knowable after the fact,
and it does not matter: **the system must never depend on knowing.**

The vulnerability was not any one of those causes. It was that "complete" meant
*"these filenames were present when someone last looked"*:

    if not all((directory / name).is_file() for name in REQUIRED_FILES):
        return False

``Path.is_file()`` is ``True`` for a zero-byte file. It is ``True`` for a file truncated
to 50 bytes. It is ``True`` for 360 MB of zeros. Nothing recorded how large the files
should have been or what they should hash to, so nothing downstream *could* tell a real
checkpoint from a hollow one — and every caller (the trainer, the backup, the restore,
the status report) shared that one weak definition.

This module replaces it. A checkpoint is valid only if it survives, at the requested
level:

``structure``
    every artifact the checkpoint's own metadata says it contains is present, non-empty,
    and of a size that is physically possible for the recorded parameter count. Cheap —
    no large file is read. This is the floor, and it alone would have caught the failure
    above.
``manifest``
    ...and every artifact's SHA-256 matches ``checkpoint_manifest.json``, which is
    written at save time. This is what makes a *truncated* or *silently rewritten* file
    detectable, which no amount of existence checking can do.
``load``
    ...and the model tensors and optimizer state actually deserialize.

Three properties are deliberate:

**Nothing is inferred from the ``COMPLETE`` marker.** The marker is necessary and never
sufficient. It records that a write finished; it says nothing about what happened to the
directory afterwards.

**Metadata is used against itself.** ``CheckpointMetadata.contents`` already lists every
file the checkpoint was written with. Any file named there and now absent is a deletion,
and that is detectable on checkpoints written long before this module existed — including
the three broken ones on Drive.

**A failure is described, not just reported.** ``valid: False`` with no reason is an
obstacle. Callers get the missing names, the zero-length names, the size and checksum
mismatches, and the load error.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Validation depth. Ordered: each level includes the ones before it.
STRUCTURE = "structure"
MANIFEST = "manifest"
LOAD = "load"
LEVELS: tuple[str, ...] = (STRUCTURE, MANIFEST, LOAD)

#: Written last inside a checkpoint directory, after everything it vouches for is
#: durable. Necessary, never sufficient.
COMPLETE_MARKER = "COMPLETE"

#: Sizes and digests of every artifact, written at save time. Its absence is not fatal —
#: checkpoints predating it still validate at ``structure`` level — but without it a
#: truncated file cannot be distinguished from a whole one.
MANIFEST_FILENAME = "checkpoint_manifest.json"
MANIFEST_VERSION = "1.0"

#: How each artifact is treated. ``core`` must be present in any checkpoint at all;
#: ``resume`` must additionally be present for the checkpoint to continue training;
#: ``optional`` is required only if the checkpoint's own metadata says it was written.
CORE = "core"
RESUME = "resume"
OPTIONAL = "optional"


#: Framing that does not scale with the model: safetensors' JSON header, torch's
#: pickle protocol, tensor names. Added to the upper bound so a two-layer test model,
#: whose optimizer state is mostly overhead, is not reported as "larger than its
#: parameter count can explain". Negligible against a real checkpoint.
SERIALIZATION_OVERHEAD_BYTES = 1_048_576


@dataclass(frozen=True)
class Artifact:
    """One file a checkpoint may contain, and what makes its size impossible.

    Two independent floors, because they catch different things:

    ``min_bytes``
        a **format** floor, needing no metadata. A safetensors file carries a header
        before it carries a byte of tensor data; a torch pickle carries protocol
        framing. Below this the file is not small, it is destroyed.
    ``bytes_per_param``
        a **model** band, applied whenever the parameter count is recorded — which it
        now always is. This is what makes "50 bytes" decisively wrong for a
        94.48M-parameter model: the floor becomes 47 MB, not 64 bytes. Deliberately
        loose, because dtype, tied embeddings and optimizer choice all move the true
        figure; this is a plausibility check, not a prediction, and a false rejection
        would be worse than the gap it closes.
    """

    name: str
    role: str
    min_bytes: int
    bytes_per_param: tuple[float, float] | None = None
    description: str = ""


#: The explicit artifact set. A resumable full-training checkpoint carries all of
#: ``core`` and ``resume``; anything in ``optional`` is required exactly when the
#: checkpoint's metadata records that it was written.
#:
#: Serialization formats, stated because the mandate asks for them to be explicit:
#: weights are **safetensors** (not pickle: memory-mappable, and it refuses to write a
#: file it cannot read back); optimizer/scheduler/scaler/RNG are **torch pickle**, which
#: is what their ``state_dict()`` round-trips through; state and metadata are **JSON**.
ARTIFACTS: tuple[Artifact, ...] = (
    Artifact("model.safetensors", CORE, 64, (0.5, 6.0),
             "model weights; safetensors, fp32 unless the run says otherwise"),
    Artifact("training_state.json", CORE, 2, None, "step, epoch, data position, history"),
    Artifact("metadata.json", CORE, 2, None, "what this checkpoint is, self-contained"),
    Artifact(COMPLETE_MARKER, CORE, 1, None, "written last; carries a line of text"),
    Artifact("optimizer.pt", RESUME, 64, (0.3, 24.0),
             "optimizer moments; without it the checkpoint is inference-only"),
    Artifact("scheduler.pt", OPTIONAL, 64, None, "LR schedule position"),
    Artifact("scaler.pt", OPTIONAL, 64, None, "AMP loss scale; absent on a CPU/fp32 run"),
    Artifact("rng.pt", OPTIONAL, 64, None, "python/torch/numpy/cuda generators"),
    Artifact("config.json", OPTIONAL, 2, None, "the experiment config that produced this"),
    Artifact(MANIFEST_FILENAME, OPTIONAL, 2, None, "sizes and digests of everything above"),
)

ARTIFACTS_BY_NAME: dict[str, Artifact] = {a.name: a for a in ARTIFACTS}

#: Present in every checkpoint, regardless of intent.
CORE_FILES: tuple[str, ...] = tuple(a.name for a in ARTIFACTS if a.role == CORE)
#: Additionally required to continue training.
RESUME_FILES: tuple[str, ...] = tuple(a.name for a in ARTIFACTS if a.role == RESUME)
#: The full set a resumable checkpoint is expected to carry.
RESUMABLE_FILES: tuple[str, ...] = CORE_FILES + RESUME_FILES

#: Read size for digesting. Large enough that hashing a 1 GB file is not syscall-bound,
#: small enough not to matter on a Drive mount.
_HASH_CHUNK = 1024 * 1024


def sha256_file(path: str | Path, *, chunk: int = _HASH_CHUNK) -> str:
    """Digest a file without holding it in memory. Checkpoints are gigabytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------------------
# the manifest
# ----------------------------------------------------------------------------------


def build_manifest(
    directory: str | Path,
    *,
    step: int,
    parameter_count: int | None = None,
    names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Record size and SHA-256 for every artifact present in ``directory``.

    Written into the checkpoint at save time. It is the only thing that makes a
    *truncated* or *rewritten* file detectable: existence checks cannot, and a size check
    alone cannot tell 360 MB of the right weights from 360 MB of the wrong ones.

    ``COMPLETE`` is deliberately excluded — it is written after the manifest, so it
    cannot describe itself.
    """
    root = Path(directory)
    wanted = names or tuple(a.name for a in ARTIFACTS if a.name != COMPLETE_MARKER)
    files: dict[str, Any] = {}
    for name in wanted:
        candidate = root / name
        if name == MANIFEST_FILENAME or not candidate.is_file():
            continue
        files[name] = {
            "size_bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }
    return {
        "manifest_version": MANIFEST_VERSION,
        "step": step,
        "parameter_count": parameter_count,
        "files": files,
    }


def read_manifest(directory: str | Path) -> dict[str, Any] | None:
    """The recorded manifest, or ``None`` when there is none or it is unreadable."""
    path = Path(directory) / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and isinstance(payload.get("files"), dict) else None


# ----------------------------------------------------------------------------------
# the result
# ----------------------------------------------------------------------------------


@dataclass
class FileCheck:
    """One artifact, as found on disk against what was expected of it."""

    name: str
    present: bool = False
    size_bytes: int | None = None
    expected_size: int | None = None
    sha256: str | None = None
    expected_sha256: str | None = None
    role: str = OPTIONAL
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "present": self.present, "role": self.role,
            "size_bytes": self.size_bytes, "expected_size": self.expected_size,
            "sha256": self.sha256, "expected_sha256": self.expected_sha256,
            "ok": self.ok, "problems": self.problems,
        }


@dataclass
class CheckpointValidation:
    """Whether one checkpoint is usable, and precisely why not.

    Every failure category the mandate names is its own field. A caller deciding what to
    do next needs to distinguish "the weights were deleted" from "the digest moved" from
    "it will not deserialize" — those have different causes and different remedies.
    """

    path: str
    level: str = STRUCTURE
    step: int | None = None
    valid: bool = False
    #: One sentence naming the first disqualifying problem. ``None`` when valid.
    invalid_reason: str | None = None

    missing_files: list[str] = field(default_factory=list)
    zero_length_files: list[str] = field(default_factory=list)
    implausible_sizes: list[dict[str, Any]] = field(default_factory=list)
    size_mismatches: list[dict[str, Any]] = field(default_factory=list)
    checksum_mismatches: list[dict[str, Any]] = field(default_factory=list)
    load_failure: str | None = None
    architecture_mismatch: str | None = None
    resume_failure: str | None = None

    manifest_present: bool = False
    marker_present: bool = False
    metadata_complete: bool = False
    parameter_count: int | None = None
    files: list[FileCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Whether this checkpoint can continue training, as opposed to merely being loadable
    #: for inference. False whenever optimizer state is missing or unusable.
    resumable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "level": self.level, "step": self.step,
            "valid": self.valid, "resumable": self.resumable,
            "invalid_reason": self.invalid_reason,
            "missing_files": self.missing_files,
            "zero_length_files": self.zero_length_files,
            "implausible_sizes": self.implausible_sizes,
            "size_mismatches": self.size_mismatches,
            "checksum_mismatches": self.checksum_mismatches,
            "load_failure": self.load_failure,
            "architecture_mismatch": self.architecture_mismatch,
            "resume_failure": self.resume_failure,
            "manifest_present": self.manifest_present,
            "marker_present": self.marker_present,
            "metadata_complete": self.metadata_complete,
            "parameter_count": self.parameter_count,
            "warnings": self.warnings,
            "files": [f.to_dict() for f in self.files],
        }

    def summary(self) -> str:
        """One line, for a list of checkpoints."""
        name = Path(self.path).name
        if self.valid:
            note = "" if self.resumable else " (inference only — no optimizer state)"
            return f"{name}  VALID{note}"
        return f"{name}  INVALID — {self.invalid_reason}"

    def render(self) -> str:
        lines = [
            f"checkpoint : {self.path}",
            f"step       : {self.step if self.step is not None else 'unknown'}",
            f"level      : {self.level}",
            f"verdict    : {'CHECKPOINT VALID' if self.valid else 'CHECKPOINT INVALID'}",
        ]
        if self.valid and not self.resumable:
            lines.append("             loadable for inference; NOT resumable")
        if self.invalid_reason:
            lines.append(f"reason     : {self.invalid_reason}")
        if not self.manifest_present:
            lines.append(
                f"note       : no {MANIFEST_FILENAME}; sizes and digests could not be "
                f"verified against a record"
            )

        for label, entries in (
            ("missing", self.missing_files),
            ("zero-length", self.zero_length_files),
        ):
            if entries:
                lines.append(f"{label:<11}: {', '.join(entries)}")
        for label, entries in (
            ("implausible size", self.implausible_sizes),
            ("size mismatch", self.size_mismatches),
            ("checksum mismatch", self.checksum_mismatches),
        ):
            for entry in entries:
                lines.append(f"{label:<11}: {entry['name']} — {entry['detail']}")
        for label, value in (
            ("load failure", self.load_failure),
            ("architecture", self.architecture_mismatch),
            ("resume failure", self.resume_failure),
        ):
            if value:
                lines.append(f"{label:<11}: {value}")

        if self.files:
            lines.append("")
            lines.append(f"  {'file':<26}{'size':>14}  status")
            for check in self.files:
                size = f"{check.size_bytes:,}" if check.size_bytes is not None else "-"
                status = "PASS" if check.ok else "; ".join(check.problems)
                lines.append(f"  {check.name:<26}{size:>14}  {status}")
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------------

#: A loaded tensor total differing from the recorded parameter count by more than this
#: fraction is an architecture mismatch rather than a tied-embedding bookkeeping
#: difference. Tying moves the count by one embedding table — under 0.2% at 94M with a
#: 256-byte vocabulary, and a few percent even for large vocabularies.
ARCHITECTURE_TOLERANCE = 0.10


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _expected_files(
    metadata: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    *,
    require_resumable: bool,
) -> dict[str, str]:
    """Which artifacts this particular checkpoint must contain, and why.

    Three sources, unioned:

    * the **core** set, always;
    * the **resume** set, when the checkpoint is expected to continue training;
    * anything the checkpoint *says about itself* — ``metadata.contents`` lists every
      file that was written, and the manifest lists every file that was digested.

    The third source is what turns this into a deletion detector, and it works on
    checkpoints written long before this module existed: a directory whose own metadata
    claims a ``model.safetensors`` it no longer has is not a checkpoint missing an
    optional extra, it is a checkpoint that has been damaged.
    """
    expected: dict[str, str] = {name: "core artifact" for name in CORE_FILES}
    if require_resumable:
        for name in RESUME_FILES:
            expected[name] = "needed to resume training"

    for name in (metadata or {}).get("contents", []) or []:
        if isinstance(name, str) and name not in expected and name in ARTIFACTS_BY_NAME:
            expected[name] = "listed in this checkpoint's own metadata"
    for name in (manifest or {}).get("files", {}) or {}:
        if isinstance(name, str) and name not in expected:
            expected[name] = "recorded in checkpoint_manifest.json"
    return expected


def _check_size_plausibility(
    check: FileCheck, artifact: Artifact | None, parameter_count: int | None
) -> list[dict[str, Any]]:
    """Reject sizes that are physically impossible, without hard-coding a right answer.

    Serialization varies — dtype, tied embeddings, optimizer choice, torch version — so
    there is no exact expected size. There is, however, an obviously-impossible one: a
    94.5M-parameter model does not serialize to 50 bytes, and an optimizer state is never
    empty. Two floors, both deliberately far below any real value.
    """
    if artifact is None or check.size_bytes is None:
        return []
    problems: list[dict[str, Any]] = []
    if check.size_bytes < artifact.min_bytes:
        detail = (
            f"{check.size_bytes:,} bytes is below the {artifact.min_bytes:,}-byte floor "
            f"for {check.name} — this is a deleted or truncated file, not a small one"
        )
        check.problems.append(f"implausibly small ({check.size_bytes:,} bytes)")
        problems.append({"name": check.name, "detail": detail,
                         "size_bytes": check.size_bytes, "floor": artifact.min_bytes})
        return problems

    if artifact.bytes_per_param and parameter_count:
        low, high = artifact.bytes_per_param
        floor = int(parameter_count * low)
        ceiling = int(parameter_count * high) + SERIALIZATION_OVERHEAD_BYTES
        if check.size_bytes < floor:
            detail = (
                f"{check.size_bytes:,} bytes for {parameter_count:,} parameters is "
                f"under {low} bytes/parameter (floor {floor:,}) — too small to hold them"
            )
            check.problems.append(f"too small for {parameter_count:,} parameters")
            problems.append({"name": check.name, "detail": detail,
                             "size_bytes": check.size_bytes, "floor": floor})
        elif check.size_bytes > ceiling:
            detail = (
                f"{check.size_bytes:,} bytes for {parameter_count:,} parameters exceeds "
                f"{high} bytes/parameter (ceiling {ceiling:,}) — wrong file for this model?"
            )
            check.problems.append("larger than this parameter count can explain")
            problems.append({"name": check.name, "detail": detail,
                             "size_bytes": check.size_bytes, "ceiling": ceiling})
    return problems


def validate_checkpoint_dir(
    path: str | Path,
    *,
    level: str = STRUCTURE,
    require_resumable: bool = True,
    expected_parameter_count: int | None = None,
) -> CheckpointValidation:
    """Decide whether one directory is a checkpoint that can actually be used.

    The only definition of checkpoint validity in this project. Trainer, backup, restore,
    status reporting and the validation script all call this — a second, slightly
    different notion of "complete" living somewhere else is exactly how a hollow
    checkpoint came to be advertised as resumable.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown validation level {level!r}; known: {LEVELS}")

    directory = Path(path)
    result = CheckpointValidation(path=str(directory), level=level)

    if not directory.is_dir():
        result.invalid_reason = "not a directory"
        return result

    metadata = _read_json(directory / "metadata.json")
    manifest = read_manifest(directory)
    result.manifest_present = manifest is not None
    result.marker_present = (directory / COMPLETE_MARKER).is_file()
    if metadata:
        result.metadata_complete = bool(metadata.get("complete"))
        if isinstance(metadata.get("step"), int):
            result.step = metadata["step"]
        if isinstance(metadata.get("parameter_count"), int):
            result.parameter_count = metadata["parameter_count"]
    if result.step is None and directory.name.startswith("step_"):
        tail = directory.name.rsplit("_", 1)[-1]
        result.step = int(tail) if tail.isdigit() else None
    if result.parameter_count is None:
        result.parameter_count = expected_parameter_count

    expected = _expected_files(metadata, manifest, require_resumable=require_resumable)
    recorded = (manifest or {}).get("files", {}) or {}

    for name in sorted(expected):
        artifact = ARTIFACTS_BY_NAME.get(name)
        check = FileCheck(name=name, role=artifact.role if artifact else OPTIONAL)
        candidate = directory / name
        if not candidate.is_file():
            check.problems.append(f"MISSING ({expected[name]})")
            result.missing_files.append(name)
            result.files.append(check)
            continue

        check.present = True
        check.size_bytes = candidate.stat().st_size
        if check.size_bytes == 0:
            check.problems.append("ZERO LENGTH")
            result.zero_length_files.append(name)
            result.files.append(check)
            continue

        result.implausible_sizes.extend(
            _check_size_plausibility(check, artifact, result.parameter_count)
        )

        record = recorded.get(name)
        if isinstance(record, dict):
            expected_size = record.get("size_bytes")
            check.expected_size = expected_size if isinstance(expected_size, int) else None
            check.expected_sha256 = record.get("sha256")
            if check.expected_size is not None and check.size_bytes != check.expected_size:
                detail = (
                    f"{check.size_bytes:,} bytes on disk, manifest recorded "
                    f"{check.expected_size:,} — the file changed after it was written"
                )
                check.problems.append("SIZE MISMATCH")
                result.size_mismatches.append({
                    "name": name, "detail": detail,
                    "size_bytes": check.size_bytes, "expected_size": check.expected_size,
                })
        result.files.append(check)

    # --- manifest level: digests -------------------------------------------------
    if level in (MANIFEST, LOAD) and manifest is not None:
        for check in result.files:
            record = recorded.get(check.name)
            if not check.present or not isinstance(record, dict):
                continue
            if any(p.startswith(("MISSING", "ZERO", "SIZE MISMATCH")) for p in check.problems):
                continue  # already disqualified; hashing it proves nothing further
            try:
                check.sha256 = sha256_file(directory / check.name)
            except OSError as exc:
                check.problems.append(f"unreadable: {type(exc).__name__}")
                result.checksum_mismatches.append(
                    {"name": check.name, "detail": f"could not be read: {exc}"}
                )
                continue
            if check.expected_sha256 and check.sha256 != check.expected_sha256:
                detail = (
                    f"digest {check.sha256[:16]} does not match the recorded "
                    f"{str(check.expected_sha256)[:16]} — the bytes are not the ones "
                    f"that were written"
                )
                check.problems.append("CHECKSUM MISMATCH")
                result.checksum_mismatches.append({
                    "name": check.name, "detail": detail,
                    "sha256": check.sha256, "expected_sha256": check.expected_sha256,
                })
    elif level in (MANIFEST, LOAD):
        result.warnings.append(
            f"{level} verification requested but this checkpoint has no "
            f"{MANIFEST_FILENAME}; only structural checks were possible"
        )

    _finalise(result, require_resumable=require_resumable)
    if result.valid and level == LOAD:
        _verify_by_loading(
            result, directory,
            require_resumable=require_resumable,
            expected_parameter_count=expected_parameter_count,
        )
    return result


def _finalise(result: CheckpointValidation, *, require_resumable: bool) -> None:
    """Turn the collected problems into a verdict and one naming sentence.

    Ordered by how badly wrong the checkpoint is, so ``invalid_reason`` names the thing
    worth fixing rather than the first thing noticed.
    """
    if not result.marker_present:
        result.invalid_reason = f"no {COMPLETE_MARKER} marker — the write never finished"
        return
    if result.missing_files:
        result.invalid_reason = (
            f"missing {', '.join(result.missing_files)}"
            + ("" if result.metadata_complete is False else
               " — the checkpoint's own metadata says these were written")
        )
        return
    if result.zero_length_files:
        result.invalid_reason = f"zero-length {', '.join(result.zero_length_files)}"
        return
    if result.implausible_sizes:
        first = result.implausible_sizes[0]
        result.invalid_reason = f"{first['name']}: {first['detail']}"
        return
    if result.size_mismatches:
        first = result.size_mismatches[0]
        result.invalid_reason = f"{first['name']}: {first['detail']}"
        return
    if result.checksum_mismatches:
        first = result.checksum_mismatches[0]
        result.invalid_reason = f"{first['name']}: {first['detail']}"
        return
    if not result.metadata_complete:
        result.invalid_reason = "metadata.json does not record this checkpoint as complete"
        return

    result.valid = True
    result.invalid_reason = None
    optimizer = next((f for f in result.files if f.name == "optimizer.pt"), None)
    result.resumable = bool(optimizer and optimizer.present and optimizer.ok)
    if require_resumable and not result.resumable:
        result.valid = False
        result.invalid_reason = "optimizer state is missing or unusable — cannot resume"


def _verify_by_loading(
    result: CheckpointValidation,
    directory: Path,
    *,
    require_resumable: bool,
    expected_parameter_count: int | None,
) -> None:
    """Deserialize the weights and the optimizer state. The strongest available check.

    Runs only after the structural checks pass, because loading a file that is already
    known to be missing or truncated tells nobody anything new — and because a full load
    reads every byte of a multi-gigabyte checkpoint.

    Torch and safetensors are imported here rather than at module scope so structure- and
    manifest-level validation works on a machine that has neither.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        result.warnings.append("safetensors is not installed; the load check was skipped")
        return

    total_parameters = 0
    try:
        with safe_open(str(directory / "model.safetensors"), framework="pt") as handle:
            keys = list(handle.keys())
            if not keys:
                result.valid = False
                result.load_failure = "model.safetensors contains no tensors"
                result.invalid_reason = result.load_failure
                return
            for key in keys:
                # Materialising each tensor is what actually proves the declared byte
                # ranges are backed by real data. A header parse alone would pass a file
                # truncated after the header.
                total_parameters += int(handle.get_tensor(key).numel())
    except Exception as exc:  # noqa: BLE001 - the failure is the result
        result.valid = False
        result.load_failure = f"model.safetensors will not load: {type(exc).__name__}: {exc}"
        result.invalid_reason = result.load_failure
        return

    reference = expected_parameter_count or result.parameter_count
    if reference:
        drift = abs(total_parameters - reference) / reference
        if drift > ARCHITECTURE_TOLERANCE:
            result.valid = False
            result.architecture_mismatch = (
                f"the checkpoint holds {total_parameters:,} parameters but this "
                f"architecture expects {reference:,} ({drift:.1%} apart) — these weights "
                f"are not for this model"
            )
            result.invalid_reason = result.architecture_mismatch
            return

    optimizer_file = directory / "optimizer.pt"
    if optimizer_file.is_file():
        try:
            import torch

            state = torch.load(optimizer_file, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - the failure is the result
            result.resumable = False
            result.resume_failure = (
                f"optimizer.pt will not load: {type(exc).__name__}: {exc}"
            )
            if require_resumable:
                result.valid = False
                result.invalid_reason = result.resume_failure
            return
        if not isinstance(state, dict) or "param_groups" not in state:
            result.resumable = False
            result.resume_failure = (
                "optimizer.pt loaded but is not an optimizer state dict "
                "(no param_groups) — training cannot be continued from it"
            )
            if require_resumable:
                result.valid = False
                result.invalid_reason = result.resume_failure
    elif require_resumable:
        result.resumable = False
        result.resume_failure = "optimizer.pt is absent"
        result.valid = False
        result.invalid_reason = result.resume_failure


# ----------------------------------------------------------------------------------
# a directory of checkpoints
# ----------------------------------------------------------------------------------


def checkpoint_directories(root: str | Path) -> list[Path]:
    """Every ``step_NNNNNN`` directory, oldest first, valid or not.

    Staging directories are excluded by name: they start with a dot and end in
    ``.incomplete``, and a complete checkpoint is never named that way.
    """
    directory = Path(root)
    if not directory.is_dir():
        return []
    found = [
        child for child in directory.iterdir()
        if child.is_dir()
        and child.name.startswith("step_")
        and not child.name.endswith(".incomplete")
    ]
    return sorted(found, key=lambda p: p.name)


def validate_checkpoint_root(
    root: str | Path,
    *,
    level: str = STRUCTURE,
    require_resumable: bool = True,
    expected_parameter_count: int | None = None,
) -> list[CheckpointValidation]:
    """Validate every checkpoint under ``root``, oldest first.

    ``structure`` by default: a status report scans every checkpoint, and hashing tens of
    gigabytes to answer "where did this run get to?" would make the answer too expensive
    to ask for. Escalate deliberately.
    """
    return [
        validate_checkpoint_dir(
            candidate, level=level, require_resumable=require_resumable,
            expected_parameter_count=expected_parameter_count,
        )
        for candidate in checkpoint_directories(root)
    ]


def newest_valid(validations: list[CheckpointValidation]) -> CheckpointValidation | None:
    """The newest checkpoint that passes. Ordering is by step, not by mtime.

    mtime would be wrong twice over: a copy to Drive rewrites it, and a restore rewrites
    it again, so the newest file is routinely the oldest checkpoint.
    """
    valid = [v for v in validations if v.valid]
    if not valid:
        return None
    return max(valid, key=lambda v: (v.step if v.step is not None else -1, v.path))


@dataclass
class LatestResolution:
    """What ``latest`` actually resolves to right now — not what it once meant.

    ``latest.json`` records a claim made at write time: ``"complete": true``. That claim
    is durable and the checkpoint it describes is not. Reading the pointer without
    re-verifying its target is how a directory that lost its weights kept being named as
    the place to resume from.
    """

    root: str
    pointer: dict[str, Any] | None = None
    pointer_step: int | None = None
    pointer_path: str | None = None
    pointer_valid: bool = False
    pointer_invalid_reason: str | None = None
    resolved: str | None = None
    resolved_step: int | None = None
    fell_back: bool = False
    checked: list[CheckpointValidation] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.resolved is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "pointer": self.pointer,
            "pointer_step": self.pointer_step,
            "pointer_path": self.pointer_path,
            "pointer_valid": self.pointer_valid,
            "pointer_invalid_reason": self.pointer_invalid_reason,
            "resolved": self.resolved,
            "resolved_step": self.resolved_step,
            "fell_back": self.fell_back,
            "usable": self.usable,
            "checkpoints": [v.summary() for v in self.checked],
        }

    def render(self) -> str:
        lines: list[str] = []
        if self.pointer_path and not self.pointer_valid:
            lines.append(f"latest checkpoint {self.pointer_path} is invalid:")
            lines.append(f"  {self.pointer_invalid_reason}")
            lines.append("")
        if self.fell_back and self.resolved:
            lines.append(f"falling back to {Path(self.resolved).name}")
            lines.append("")
        if self.resolved:
            lines.append(f"resumable at step {self.resolved_step}")
        else:
            lines.append("no valid checkpoint — nothing here can be resumed from")
        return "\n".join(lines)

    def render_inventory(self, *, label: str = "checkpoints found") -> str:
        """Every checkpoint with its verdict — the answer a fresh session needs.

        Counts **verified resumable** checkpoints. Counting directories is what made a
        run look recoverable when three of its checkpoints were hollow.
        """
        valid = [v for v in self.checked if v.valid]
        lines = [f"{label}: {len(self.checked)} ({len(valid)} verified resumable)", ""]
        for validation in self.checked:
            lines.append(f"  {validation.summary()}")
        lines.append("")
        if self.resolved:
            lines.append(f"newest valid checkpoint: {Path(self.resolved).name}")
        else:
            lines.append("newest valid checkpoint: NONE")
        return "\n".join(lines)


def resolve_latest(
    root: str | Path,
    *,
    level: str = STRUCTURE,
    require_resumable: bool = True,
    expected_parameter_count: int | None = None,
    validations: list[CheckpointValidation] | None = None,
) -> LatestResolution:
    """Resolve ``latest`` by verifying now, and report any fallback rather than hiding it.

    The pointer is a hint, never an authority. It is read, its target is validated *at
    this moment*, and if that fails the newest checkpoint that does validate is used
    instead — with the substitution stated, because a silent fallback tells the user
    their run is fine when their newest checkpoint is not.
    """
    directory = Path(root)
    result = LatestResolution(root=str(directory))

    pointer_file = directory / "latest.json"
    if pointer_file.is_file():
        result.pointer = _read_json(pointer_file)
    if result.pointer:
        step = result.pointer.get("step")
        result.pointer_step = step if isinstance(step, int) else None
        name = result.pointer.get("path")
        result.pointer_path = str(name) if name else None

    result.checked = validations if validations is not None else validate_checkpoint_root(
        directory, level=level, require_resumable=require_resumable,
        expected_parameter_count=expected_parameter_count,
    )
    by_name = {Path(v.path).name: v for v in result.checked}

    if result.pointer_path:
        target = by_name.get(result.pointer_path)
        if target is None:
            result.pointer_invalid_reason = (
                f"{result.pointer_path} is named by latest.json but is not present"
            )
        elif target.valid:
            result.pointer_valid = True
            result.resolved, result.resolved_step = target.path, target.step
        else:
            result.pointer_invalid_reason = target.invalid_reason

    if result.resolved is None:
        best = newest_valid(result.checked)
        if best is not None:
            result.resolved, result.resolved_step = best.path, best.step
            result.fell_back = bool(result.pointer_path) and not result.pointer_valid
    return result
