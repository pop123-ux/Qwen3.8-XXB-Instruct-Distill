"""Gradient-scale calibration for Run014: a gradient-matched normalised residual-delta control.

Run011 (normalised residual-delta KD) finished far worse than Run008 (raw residual-delta KD).
That gap has two candidate explanations: residual-update magnitude carries useful information,
or the two objectives simply produce different gradient scales. Run014 separates them by giving
the normalised objective one fixed multiplier, estimated before training from the ratio of raw
to normalised gradient norms at the identical seed-1 QLoRA initialisation.

Everything here is deliberately free of model loading, so each property Run014 relies on can be
tested on a tiny network: which sequences are sampled, how they are hashed, which parameters a
gradient norm counts, how the multiplier is estimated, that it scales the hidden gradient exactly
once, and that a manifest with mismatched provenance is refused.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

METHOD_VERSION = "run014-gradcal-1.0"

#: Frozen before any gradient was measured. Evenly spaced across the 433 training sequences;
#: 432 is the last training index, so no validation sequence can be selected.
CALIBRATION_INDICES: tuple[int, ...] = (0, 144, 288, 432)

PACKED_CORPUS_SHA256 = "e11ca38bb099fc89c2f74e96f5d2f1209def6a16f6a8432d4e9972acd50c100d"
EXPECTED_N_TRAIN = 433
EXPECTED_N_VALIDATION = 22
EXPECTED_SEQUENCE_LENGTH = 1536
TOKEN_BYTE_FORMAT = "uint32-little-endian"

#: The per-sample RNG seed shared by the raw and the normalised measurement of one sequence, so
#: both see identical LoRA-dropout masks and differ only in the objective.
RNG_SEED_STRIDE = 1_000_003


class CalibrationError(RuntimeError):
    """A calibration input or manifest does not match what Run014 preregistered."""


def sequence_token_sha256(tokens: Sequence[int]) -> str:
    """SHA-256 of a packed sequence, each token id as an unsigned 32-bit little-endian integer."""
    digest = hashlib.sha256()
    for token in tokens:
        value = int(token)
        if not 0 <= value <= 0xFFFFFFFF:
            raise CalibrationError(f"token id {value} does not fit {TOKEN_BYTE_FORMAT}")
        digest.update(value.to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def calibration_seed(training_seed: int, index: int) -> int:
    """The RNG seed used for both measurements of training sequence ``index``."""
    return training_seed * RNG_SEED_STRIDE + index


def select_calibration_samples(
    train_sequences: Sequence[Sequence[int]],
    *,
    indices: Sequence[int] = CALIBRATION_INDICES,
    n_train: int = EXPECTED_N_TRAIN,
    sequence_length: int = EXPECTED_SEQUENCE_LENGTH,
) -> list[dict[str, Any]]:
    """Return ``[{index, tokens, sha256}]`` for the frozen indices of the TRAINING split.

    Refuses rather than adapting: a training split of the wrong size means the packing changed,
    and an index outside it would silently draw a validation sequence.
    """
    if len(train_sequences) != n_train:
        raise CalibrationError(
            f"training split has {len(train_sequences)} sequences; Run014 preregistered {n_train}"
        )
    samples = []
    for index in indices:
        if not 0 <= index < len(train_sequences):
            raise CalibrationError(f"index {index} is not a training index (n_train={n_train})")
        tokens = list(train_sequences[index])
        if len(tokens) != sequence_length:
            raise CalibrationError(
                f"training sequence {index} has {len(tokens)} tokens, expected {sequence_length}"
            )
        samples.append({"index": index, "tokens": tokens,
                        "sha256": sequence_token_sha256(tokens)})
    return samples


def check_frozen_hashes(samples: Sequence[dict[str, Any]], frozen: dict[int, str]) -> None:
    """Refuse unless every sample's hash equals the preregistered one, for exactly those indices."""
    got = {int(s["index"]): s["sha256"] for s in samples}
    want = {int(k): v for k, v in frozen.items()}
    if set(got) != set(want):
        raise CalibrationError(f"sample indices {sorted(got)} != preregistered {sorted(want)}")
    bad = {i: (got[i], want[i]) for i in want if got[i] != want[i]}
    if bad:
        raise CalibrationError(f"calibration sample hashes do not match the preregistration: {bad}")


def trainable_parameters(model_or_params: Any) -> list[Any]:
    params = model_or_params.parameters() if hasattr(model_or_params, "parameters") \
        else model_or_params
    return [p for p in params if p.requires_grad]


def trainable_grad_norm(model_or_params: Any) -> float:
    """Global L2 norm of ``.grad`` over parameters with ``requires_grad`` only.

    A frozen parameter is excluded even if something left a gradient on it, because Run014's
    quantity is the scale of the update the optimiser would actually receive.
    """
    total = 0.0
    for param in trainable_parameters(model_or_params):
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum())
    return math.sqrt(total)


def trainable_state_digest(model: Any) -> str:
    """SHA-256 over every trainable parameter's name, shape, dtype and bytes."""
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        tensor = param.detach().float().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def median_multiplier(
    raw_norms: Sequence[float], normalised_norms: Sequence[float]
) -> tuple[list[float], float]:
    """``ratio_i = raw_i / normalised_i`` and their median, which is the Run014 multiplier.

    For an even count the median is the mean of the two middle ratios (``statistics.median``);
    Run014 preregisters that definition explicitly.
    """
    if len(raw_norms) != len(normalised_norms) or not raw_norms:
        raise CalibrationError("raw and normalised norms must be non-empty and equal in length")
    ratios = []
    for raw, norm in zip(raw_norms, normalised_norms, strict=True):
        if not (math.isfinite(raw) and raw > 0):
            raise CalibrationError(f"raw gradient norm {raw!r} is not finite and positive")
        if not (math.isfinite(norm) and norm > 0):
            raise CalibrationError(f"normalised gradient norm {norm!r} is not finite and positive")
        ratio = raw / norm
        if not (math.isfinite(ratio) and ratio > 0):
            raise CalibrationError(f"ratio {ratio!r} is not finite and positive")
        ratios.append(ratio)
    multiplier = statistics.median(ratios)
    if not (math.isfinite(multiplier) and multiplier > 0):
        raise CalibrationError(f"median multiplier {multiplier!r} is not finite and positive")
    return ratios, multiplier


@contextmanager
def scaled_hidden_loss(trainer_module: Any, multiplier: float) -> Iterator[dict[str, int]]:
    """Scale the chunked hidden objective's GRADIENT by ``multiplier``, exactly once.

    ``behavioral_loss_chunked`` takes each chunk's gradient internally, multiplying the chunk
    loss by ``loss_scale`` immediately before its ``backward``. The value it returns is built
    from detached floats, so multiplying that return value would change a log line and nothing
    that is optimised. The multiplier therefore rides in ``loss_scale``.

    Composes with ``run004_behavioral_kd.patch_trainer_for_delta``: apply that patch first, then
    this one. Yields a counter of wrapped calls so a run can assert the patch was exercised.
    """
    if not (isinstance(multiplier, (int, float)) and math.isfinite(multiplier) and multiplier > 0):
        raise CalibrationError(f"multiplier {multiplier!r} must be finite and positive")
    original = trainer_module.behavioral_loss_chunked
    calls = {"n": 0}

    def scaled(*args: Any, **kwargs: Any) -> Any:
        kwargs["loss_scale"] = kwargs.get("loss_scale", 1.0) * multiplier
        calls["n"] += 1
        return original(*args, **kwargs)

    trainer_module.behavioral_loss_chunked = scaled
    try:
        yield calls
    finally:
        trainer_module.behavioral_loss_chunked = original


#: Fields a Run014 calibration manifest must carry, each compared against what the launcher
#: independently expects. Any mismatch refuses the run.
PROVENANCE_FIELDS: tuple[str, ...] = (
    "experiment_id", "method_version", "calibration_indices", "token_hashes",
    "packed_corpus_sha256", "teacher_model", "teacher_revision", "student_id", "seed",
    "lora_rank", "lora_alpha", "lora_dropout", "trainable_parameters",
    "preregistration_sha256", "git_commit",
)


def validate_manifest(manifest: dict[str, Any], expected: dict[str, Any]) -> float:
    """Refuse on any provenance mismatch; return the verified multiplier.

    Also recomputes the median from the recorded norms, so a manifest whose multiplier was edited
    by hand, or whose norms were altered, is refused rather than trusted.
    """
    missing = [f for f in PROVENANCE_FIELDS if f not in manifest]
    if missing:
        raise CalibrationError(f"calibration manifest is missing fields: {missing}")
    for field in PROVENANCE_FIELDS:
        if field not in expected:
            continue
        got, want = manifest[field], expected[field]
        if field == "calibration_indices":
            got, want = [int(x) for x in got], [int(x) for x in want]
        if field == "token_hashes":
            got = {int(k): v for k, v in got.items()}
            want = {int(k): v for k, v in want.items()}
        if got != want:
            raise CalibrationError(f"calibration manifest {field} mismatch: {got!r} != {want!r}")
    samples = manifest.get("samples")
    if not samples:
        raise CalibrationError("calibration manifest has no per-sample measurements")
    raw = [float(s["raw_grad_norm"]) for s in samples]
    norm = [float(s["normalised_grad_norm"]) for s in samples]
    _, recomputed = median_multiplier(raw, norm)
    recorded = float(manifest.get("multiplier", float("nan")))
    if not math.isclose(recorded, recomputed, rel_tol=1e-12, abs_tol=0.0):
        raise CalibrationError(
            f"recorded multiplier {recorded!r} != median recomputed from norms {recomputed!r}"
        )
    if manifest.get("parameters_mutated") is not False:
        raise CalibrationError("calibration manifest does not certify parameters_mutated=False")
    return recomputed
