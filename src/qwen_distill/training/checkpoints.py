"""Checkpoints that survive a Colab disconnect.

The Level-2 run reached ~step 500 on a T4 — no OOM, ~2100 tokens/s, validation BPB down
to 1.279 — and then the runtime disconnected and took the ephemeral filesystem with it.
Nothing about the model was wrong. Everything about the *persistence* was.

Two failures had to be fixed, and they are different problems:

**1. A checkpoint did not contain enough to resume.** It held weights, optimizer state
and a step counter. It did not hold the LR scheduler, the AMP GradScaler, any RNG state,
or the data position — so "resuming" restarted the one-cycle schedule from its warmup,
reset the loss scale, and silently rewound the data to epoch 0. That is not a resume; it
is a differently-initialised new run wearing the old run's step number.

**2. A checkpoint was written in place.** `torch.save` straight into
`checkpoints/step_000500/` means a disconnect halfway through leaves a directory that
exists, looks plausible, and cannot be loaded — and it is the newest one, so it is the
one you would reach for.

The invariant this module enforces:

    A crash may lose the step currently executing. It must never invalidate the last
    checkpoint that completed.

Achieved by writing every checkpoint into `.step_NNNNNN.incomplete/`, fsyncing, checking
the required files are present, then `os.replace`-ing the directory into place — atomic
on POSIX — and only then updating `latest.json`. A reader that trusts only `latest.json`
and the `COMPLETE` marker can never be handed a partial write.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Marker file written last inside a checkpoint directory. Its presence means every
#: other file was written and fsynced first.
COMPLETE_MARKER = "COMPLETE"

#: Prefix for a checkpoint still being written. Directories starting with a dot are
#: skipped by the discovery logic and by the Drive backup, and are removed on restart.
INCOMPLETE_PREFIX = "."
INCOMPLETE_SUFFIX = ".incomplete"

#: Files a checkpoint must contain to be considered loadable. Anything optional (a
#: scaler on a CPU run, a scheduler for a constant LR) is absent rather than empty.
REQUIRED_FILES = ("model.safetensors", "optimizer.pt", "training_state.json", "metadata.json")

#: Everything a checkpoint may contain, in the order it is written.
CHECKPOINT_FILES = (
    "model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "rng.pt",
    "training_state.json",
    "config.json",
    "metadata.json",
)


def step_dirname(step: int) -> str:
    """Zero-padded so lexical order matches numeric order in any file listing."""
    return f"step_{step:06d}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def config_sha256(config: dict[str, Any]) -> str:
    """A stable digest of a config, so a checkpoint can be matched to what produced it."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON so a reader never observes a half-written file.

    `latest.json` is read by a process that may start seconds after this one died. A
    plain write can be interrupted between `open` (which truncates) and the final byte,
    leaving a pointer that parses as nothing. Writing to a temporary file in the same
    directory and renaming makes the swap atomic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    """Force the directory entry itself to disk, not just the files inside it."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # not all filesystems support directory fsync; the rename is still atomic
    finally:
        os.close(fd)


@dataclass
class CheckpointMetadata:
    """What a checkpoint is, readable without the session that produced it.

    Deliberately self-contained: months later, or on a different machine, the only thing
    available may be the directory itself.
    """

    step: int
    created_at: str = field(default_factory=_utc_now)
    complete: bool = False
    parameter_count: int | None = None
    architecture_sha256: str | None = None
    config_sha256: str | None = None
    git_commit: str | None = None
    precision: str | None = None
    optimizer: str | None = None
    sequence_length: int | None = None
    batch_size: int | None = None
    effective_batch_size: int | None = None
    gradient_checkpointing: bool | None = None
    tokens_seen: int | None = None
    training_loss: float | None = None
    validation_loss: float | None = None
    validation_bits_per_byte: float | None = None
    hostname: str | None = None
    platform: str | None = None
    gpu_name: str | None = None
    torch_version: str | None = None
    transformers_version: str | None = None
    contents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        known.setdefault("step", 0)
        return cls(**known)


def _runtime_metadata() -> dict[str, Any]:
    """Where this checkpoint was produced. Best-effort; absence is never an error."""
    info: dict[str, Any] = {
        "hostname": platform.node() or None,
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except (ImportError, AttributeError, RuntimeError):
        pass
    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    return info


def _is_tied(model: Any, name: str) -> bool:
    """Whether a "missing" weight is really an alias of one that was loaded.

    Tied embeddings appear as missing because `save_model` writes the storage once. That
    is correct and expected; a genuinely absent weight is not.
    """
    tied = getattr(model.config, "tie_word_embeddings", False) if hasattr(model, "config") else False
    return bool(tied) and name.endswith(("lm_head.weight", "embed_tokens.weight"))


def is_complete(path: str | Path) -> bool:
    """Whether a directory is a checkpoint that can actually be loaded.

    Requires the marker *and* every required file: the marker alone would trust a write
    that was interrupted after the marker but before an fsync completed, and the files
    alone would trust a directory mid-write.
    """
    directory = Path(path)
    if not directory.is_dir():
        return False
    if not (directory / COMPLETE_MARKER).is_file():
        return False
    if not all((directory / name).is_file() for name in REQUIRED_FILES):
        return False
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(metadata.get("complete"))


def list_checkpoints(root: str | Path) -> list[Path]:
    """Every complete checkpoint under ``root``, oldest first."""
    directory = Path(root)
    if not directory.is_dir():
        return []
    found = [
        child for child in directory.iterdir()
        if child.is_dir() and child.name.startswith("step_") and is_complete(child)
    ]
    return sorted(found, key=lambda p: p.name)


def cleanup_incomplete(root: str | Path) -> list[str]:
    """Remove leftover partial writes from a previous process that was killed.

    Safe by construction: only directories carrying the incomplete marker in their name
    are touched, and a complete checkpoint is never named that way.
    """
    directory = Path(root)
    if not directory.is_dir():
        return []
    removed: list[str] = []
    for child in directory.iterdir():
        if child.is_dir() and child.name.endswith(INCOMPLETE_SUFFIX):
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child.name)
    return removed


def read_latest_pointer(root: str | Path) -> dict[str, Any] | None:
    """The recorded newest checkpoint, or ``None`` if there is no usable pointer."""
    pointer = Path(root) / "latest.json"
    if not pointer.is_file():
        return None
    try:
        return json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def resolve_checkpoint(root: str | Path, reference: str | Path) -> Path | None:
    """Turn ``latest``, a step number, or a path into a verified checkpoint directory.

    Returns ``None`` rather than a best guess when nothing valid matches: resuming from
    a directory that is not a complete checkpoint is worse than refusing to start.
    """
    directory = Path(root)
    token = str(reference)

    if token == "latest":
        pointer = read_latest_pointer(directory)
        if pointer and pointer.get("complete"):
            candidate = directory / str(pointer.get("path", ""))
            if is_complete(candidate):
                return candidate
        # The pointer is missing, stale or points at a checkpoint that did not survive.
        # Fall back to the newest directory that verifies on its own terms.
        existing = list_checkpoints(directory)
        return existing[-1] if existing else None

    if token.isdigit():
        candidate = directory / step_dirname(int(token))
        return candidate if is_complete(candidate) else None

    candidate = Path(token)
    if is_complete(candidate):
        return candidate
    nested = directory / token
    return nested if is_complete(nested) else None


def save_checkpoint(
    directory: str | Path,
    step: int,
    *,
    model: Any,
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
    training_state: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
    metadata: CheckpointMetadata | None = None,
) -> Path:
    """Write a complete, resumable checkpoint atomically.

    Everything lands in ``.step_NNNNNN.incomplete/`` first and is renamed into place only
    once every required file is written and fsynced. An interrupted call therefore leaves
    a directory that discovery ignores and startup deletes — never a checkpoint that
    looks loadable and is not.
    """
    import torch
    from safetensors.torch import save_model

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    final = root / step_dirname(step)
    staging = root / f"{INCOMPLETE_PREFIX}{step_dirname(step)}{INCOMPLETE_SUFFIX}"

    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    written: list[str] = []

    def dump(name: str, payload: Any) -> None:
        target = staging / name
        with open(target, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        written.append(name)

    try:
        # safetensors for weights: no pickle, memory-mappable, and it refuses to write a
        # file it cannot read back.
        #
        # `save_model` rather than `save_file` because this architecture ties
        # `lm_head.weight` to `model.embed_tokens.weight`, and safetensors rejects two
        # names pointing at one storage. save_model drops the duplicate and load_model
        # restores it from the module's own tie — writing the tensor twice would work
        # but would inflate every checkpoint by a whole embedding table.
        save_model(model, str(staging / "model.safetensors"),
                   metadata={"step": str(step), "format": "pt"})
        written.append("model.safetensors")

        if optimizer is not None:
            dump("optimizer.pt", optimizer.state_dict())
        if scheduler is not None:
            dump("scheduler.pt", scheduler.state_dict())
        if scaler is not None:
            dump("scaler.pt", scaler.state_dict())
        if rng_state is not None:
            dump("rng.pt", rng_state)

        state_payload = dict(training_state or {})
        state_payload["step"] = step
        atomic_write_json(staging / "training_state.json", state_payload)
        written.append("training_state.json")

        if config is not None:
            atomic_write_json(staging / "config.json", config)
            written.append("config.json")

        record = metadata or CheckpointMetadata(step=step)
        record.step = step
        record.contents = sorted(written) + ["metadata.json", COMPLETE_MARKER]
        for key, value in _runtime_metadata().items():
            if getattr(record, key, None) is None:
                setattr(record, key, value)
        if record.git_commit is None:
            record.git_commit = _git_commit()
        if config is not None and record.config_sha256 is None:
            record.config_sha256 = config_sha256(config)
        record.complete = True
        atomic_write_json(staging / "metadata.json", record.to_dict())

        missing = [name for name in REQUIRED_FILES if not (staging / name).is_file()]
        if missing:
            raise OSError(f"checkpoint is missing required files: {missing}")

        # The marker goes last, after everything it vouches for is durable.
        marker = staging / COMPLETE_MARKER
        with open(marker, "w", encoding="utf-8") as stream:
            stream.write(f"step {step} written {_utc_now()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(staging)

        # Atomic on POSIX: the checkpoint either exists entirely or not at all.
        shutil.rmtree(final, ignore_errors=True)
        os.replace(staging, final)
        _fsync_directory(root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Only now is it safe to advertise it.
    update_latest_pointer(root, final, step)
    return final


def update_latest_pointer(root: str | Path, checkpoint: Path, step: int) -> None:
    """Advertise a checkpoint as the newest — only if it verifies first."""
    directory = Path(root)
    if not is_complete(checkpoint):
        raise ValueError(
            f"refusing to point `latest` at {checkpoint}, which is not a complete checkpoint"
        )
    atomic_write_json(directory / "latest.json", {
        "step": step,
        "path": checkpoint.name,
        "created_at": _utc_now(),
        "complete": True,
    })


def load_checkpoint(
    path: str | Path,
    *,
    model: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Restore every component a checkpoint carries, refusing an incomplete one.

    Returns the training state and metadata so the caller can restore step, data
    position and RNG. Components absent from the checkpoint are left untouched and named
    in the ``restored`` list, so a caller can tell what actually came back rather than
    assuming.
    """
    import torch

    directory = Path(path)
    if not is_complete(directory):
        raise ValueError(
            f"{directory} is not a complete checkpoint. Resuming from a partial write "
            "would silently produce a different run; refusing."
        )

    restored: list[str] = []
    if model is not None:
        from safetensors.torch import load_model

        # strict=False tolerates the tied duplicate that save_model omitted; the module
        # re-establishes the tie itself, so the weight is present either way.
        missing, unexpected = load_model(
            model, str(directory / "model.safetensors"), strict=False, device=map_location
        )
        if strict and unexpected:
            raise ValueError(f"checkpoint has tensors this model does not: {unexpected[:5]}")
        untied = [name for name in missing if not _is_tied(model, name)]
        if strict and untied:
            raise ValueError(f"checkpoint is missing weights this model needs: {untied[:5]}")
        restored.append("model")

    def maybe_load(name: str, target: Any, label: str) -> None:
        source = directory / name
        if target is not None and source.is_file():
            target.load_state_dict(
                torch.load(source, map_location=map_location, weights_only=False)
            )
            restored.append(label)

    maybe_load("optimizer.pt", optimizer, "optimizer")
    maybe_load("scheduler.pt", scheduler, "scheduler")
    maybe_load("scaler.pt", scaler, "scaler")

    training_state = json.loads((directory / "training_state.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

    rng_state = None
    rng_file = directory / "rng.pt"
    if rng_file.is_file():
        rng_state = torch.load(rng_file, map_location="cpu", weights_only=False)
        restored.append("rng")

    return {
        "path": str(directory),
        "step": training_state.get("step", metadata.get("step", 0)),
        "training_state": training_state,
        "metadata": metadata,
        "rng_state": rng_state,
        "restored": restored,
    }


# --- RNG -------------------------------------------------------------------
def capture_rng_state() -> dict[str, Any]:
    """Every generator the training loop draws from.

    Without these, a resumed run takes different dropout masks and a different data
    shuffle from the one it claims to continue — the numbers diverge and the divergence
    looks like a bug in the model.
    """
    import random

    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy

        state["numpy"] = numpy.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> list[str]:
    """Put the generators back. Returns which ones were actually restored."""
    if not state:
        return []
    import random

    import torch

    restored: list[str] = []
    if "python" in state:
        random.setstate(state["python"])
        restored.append("python")
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
        restored.append("torch")
    if "numpy" in state:
        try:
            import numpy

            numpy.random.set_state(state["numpy"])
            restored.append("numpy")
        except ImportError:
            pass
    # CUDA state only transfers onto a machine with at least as many devices.
    if "cuda" in state and torch.cuda.is_available():
        try:
            if len(state["cuda"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(state["cuda"])
                restored.append("cuda")
        except (RuntimeError, ValueError):
            pass
    return restored
