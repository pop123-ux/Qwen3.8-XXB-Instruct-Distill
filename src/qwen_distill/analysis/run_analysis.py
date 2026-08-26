"""What a training run left behind, read back honestly.

Level 2 finished 2000/2000 steps at validation BPB 1.270 and generated ``"and and and"``.
Its logs claimed 139,256 tok/s. Both numbers were in the record; neither was questioned
until someone looked. This module is the looking.

It reads only files a finished (or running) experiment already writes —
``metrics.jsonl``, ``progress/latest.json``, ``summary.json``, ``checkpoints/`` — and
never loads a model, touches a GPU, or writes into the run directory. It is safe to point
at a run that is still going.

Four things it refuses to do, because each one is a mistake this project has already made:

**It does not collapse throughput into one number.** Tokens per second is ambiguous until
you say *over what*. Three scopes are reported separately and labelled:

===============  ===============================================  ==========================
scope            definition                                       answers
===============  ===============================================  ==========================
step-level       Δtokens / Δtime between consecutive log records   is it slowing down *now*?
interval         the same, over a rolling window of records        is the trend real or noise?
run-wide         all tokens / all seconds, every session           what did the experiment cost?
===============  ===============================================  ==========================

"step-level" is the finest resolution the log supports, and that is ``log_every`` steps
apart, not one step. It is named for the scope it reports, and :attr:`ThroughputAnalysis`
carries ``steps_per_record`` so nobody reads it as per-step.

**It does not trust the logged rate.** Every record carries cumulative ``tokens_seen`` and
cumulative ``elapsed_s``, so the correct run-wide rate is recoverable independently of
whatever the trainer printed at the time. The two are compared, and a disagreement is
reported as a finding rather than silently corrected. That is how the Level-2 bug is
caught in a log written by code that still has it.

**It does not call a flat curve convergence.** A loss that stops moving because the model
converged and a loss that stops moving because the corpus ran out are the same shape. The
plateau analysis reports *when* improvement stopped and *how much of the run came after*,
and reads ``epoch`` to say whether the data was exhausted — it does not offer a verdict on
which cause applies.

**It does not require matplotlib.** Plots are written if it imports, and text curves are
rendered either way, so the analysis works on a bare CPU box.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..training.throughput import recompute_from_history

#: A wall-clock gap this many times larger than the training time that elapsed between two
#: records means the process was not running in between — a resume, not a slow step.
SESSION_GAP_FACTOR = 3.0
#: ...but only once the absolute gap is big enough to be worth calling. Two records 4 s of
#: training apart and 20 s of wall-clock apart is scheduling noise, not a session boundary.
SESSION_GAP_MIN_SECONDS = 120.0
#: A logged rate differing from the recomputed one by more than this is reported.
THROUGHPUT_DISAGREEMENT_TOLERANCE = 0.05
#: "Improvement stopped here" threshold: the first step within this fraction of the best
#: value ever reached. 1% is the figure the Level-2 report used.
PLATEAU_TOLERANCE = 0.01


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# ----------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------


@dataclass
class RunFiles:
    """Where the artefacts of one run live, and which of them exist.

    A run directory is accepted in whatever state it is in. A run killed before its first
    checkpoint has metrics and no checkpoints; a run restored from Drive may have
    checkpoints and a truncated metrics file. Missing is reported, never fatal.
    """

    root: Path
    metrics: Path | None = None
    latest_progress: Path | None = None
    summary: Path | None = None
    checkpoints_dir: Path | None = None
    config: Path | None = None

    @classmethod
    def discover(cls, root: str | Path) -> RunFiles:
        root = Path(root)
        def _if_file(path: Path) -> Path | None:
            return path if path.is_file() else None
        checkpoints = root / "checkpoints"
        return cls(
            root=root,
            metrics=_if_file(root / "metrics.jsonl"),
            latest_progress=_if_file(root / "progress" / "latest.json"),
            summary=_if_file(root / "summary.json"),
            checkpoints_dir=checkpoints if checkpoints.is_dir() else None,
            config=_if_file(root / "config.json") or _if_file(root / "resolved_config.json"),
        )

    def missing(self) -> list[str]:
        absent = []
        for name in ("metrics", "latest_progress", "summary", "checkpoints_dir"):
            if getattr(self, name) is None:
                absent.append(name)
        return absent


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Every complete record. A truncated final line is skipped, not fatal.

    A run that is still training is being appended to as this reads, so the last line may
    be half-written. That costs one record.
    """
    path = Path(path)
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


@dataclass
class RunRecords:
    """Log records split by what they measure.

    Training and validation records live in the same file and have different shapes. A
    validation record carries no ``tokens_seen``, so including it in a throughput series
    would produce a step of zero tokens over some seconds — a fabricated slowdown.
    """

    training: list[dict[str, Any]] = field(default_factory=list)
    validation: list[dict[str, Any]] = field(default_factory=list)
    raw: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> RunRecords:
        training, validation = [], []
        for record in records:
            if "validation_loss" in record:
                validation.append(record)
            if "loss" in record and record.get("tokens_seen") is not None:
                training.append(record)
        training.sort(key=lambda r: (r.get("step") or 0))
        validation.sort(key=lambda r: (r.get("step") or 0))
        return cls(training=training, validation=validation, raw=list(records))

    @property
    def last_step(self) -> int | None:
        steps = [r.get("step") for r in self.raw if isinstance(r.get("step"), int)]
        return max(steps) if steps else None


# ----------------------------------------------------------------------------------
# sessions
# ----------------------------------------------------------------------------------


@dataclass
class Segment:
    """One contiguous stretch of training by a single process.

    Segments matter because they are exactly the boundary the Level-2 throughput bug
    straddled: tokens accumulate across them, the session clock does not.
    """

    index: int
    first_step: int
    last_step: int
    first_elapsed_s: float
    last_elapsed_s: float
    tokens_at_start: int
    tokens_at_end: int
    n_records: int
    boundary_reason: str

    @property
    def tokens(self) -> int:
        return max(0, self.tokens_at_end - self.tokens_at_start)

    @property
    def seconds(self) -> float:
        return max(0.0, self.last_elapsed_s - self.first_elapsed_s)

    @property
    def tokens_per_second(self) -> float | None:
        """This segment only. Not comparable to a run-wide figure, and not meant to be."""
        return round(self.tokens / self.seconds, 1) if self.seconds > 0 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "steps": self.last_step - self.first_step,
            "tokens": self.tokens,
            "seconds": round(self.seconds, 1),
            "segment_tokens_per_second": self.tokens_per_second,
            "n_records": self.n_records,
            "boundary_reason": self.boundary_reason,
        }


def detect_segments(training: list[dict[str, Any]]) -> list[Segment]:
    """Split the training records wherever the process restarted.

    Three signals, in order of confidence:

    ``step regression``
        The step number went backwards. A resume from a checkpoint older than the last
        log record; unambiguous.
    ``elapsed regression``
        Cumulative elapsed time went backwards. Only possible across a restart.
    ``wall-clock gap``
        The timestamps say far more time passed than the elapsed counter did. The process
        was down in between. Requires timestamps, which older logs may not carry, so this
        is the weakest of the three and is deliberately conservative — see
        :data:`SESSION_GAP_FACTOR` and :data:`SESSION_GAP_MIN_SECONDS`.

    A run that never restarted yields exactly one segment. That is the common case and it
    is not a special case.
    """
    if not training:
        return []

    boundaries: list[tuple[int, str]] = [(0, "run start")]
    for i in range(1, len(training)):
        previous, current = training[i - 1], training[i]
        step_now, step_before = current.get("step"), previous.get("step")
        if isinstance(step_now, int) and isinstance(step_before, int) and step_now <= step_before:
            boundaries.append((i, "step regression"))
            continue
        elapsed_now = _as_float(current.get("elapsed_s"))
        elapsed_before = _as_float(previous.get("elapsed_s"))
        if elapsed_now is not None and elapsed_before is not None and elapsed_now < elapsed_before:
            boundaries.append((i, "elapsed regression"))
            continue
        stamp_now = _parse_timestamp(current.get("timestamp"))
        stamp_before = _parse_timestamp(previous.get("timestamp"))
        if stamp_now is not None and stamp_before is not None:
            wall_gap = (stamp_now - stamp_before).total_seconds()
            trained = (elapsed_now - elapsed_before) if (
                elapsed_now is not None and elapsed_before is not None
            ) else 0.0
            idle = wall_gap - trained
            if idle > SESSION_GAP_MIN_SECONDS and wall_gap > SESSION_GAP_FACTOR * max(trained, 1.0):
                boundaries.append((i, f"wall-clock gap of {idle / 60:.0f} min"))

    segments: list[Segment] = []
    for n, (start, reason) in enumerate(boundaries):
        end = boundaries[n + 1][0] - 1 if n + 1 < len(boundaries) else len(training) - 1
        head, tail = training[start], training[end]
        segments.append(
            Segment(
                index=n,
                first_step=int(head.get("step") or 0),
                last_step=int(tail.get("step") or 0),
                first_elapsed_s=_as_float(head.get("elapsed_s")) or 0.0,
                last_elapsed_s=_as_float(tail.get("elapsed_s")) or 0.0,
                tokens_at_start=int(head.get("tokens_seen") or 0),
                tokens_at_end=int(tail.get("tokens_seen") or 0),
                n_records=end - start + 1,
                boundary_reason=reason,
            )
        )
    return segments


# ----------------------------------------------------------------------------------
# throughput
# ----------------------------------------------------------------------------------


@dataclass
class ThroughputAnalysis:
    """Tokens against time, at three scopes, plus an audit of what was logged.

    The scopes are kept in separate fields for the same reason
    :mod:`qwen_distill.training.throughput` keeps them in separate keys: collapsing any
    two of them is the bug.
    """

    #: Δtokens / Δtime between consecutive log records. The finest resolution available.
    step_level: list[dict[str, Any]] = field(default_factory=list)
    #: The same quantity smoothed over a rolling window of records.
    interval: list[dict[str, Any]] = field(default_factory=list)
    #: How many optimizer steps one record actually spans. ``step_level`` is per *record*,
    #: and this is the number that stops it being read as per *step*.
    steps_per_record: int | None = None
    window: int = 4

    total_tokens: int = 0
    total_seconds: float = 0.0
    #: Every token over every second, across every session. The headline figure.
    run_wide_tokens_per_second: float | None = None

    segments: list[Segment] = field(default_factory=list)
    #: Per-record disagreement between the logged rate and the rate recomputed from the
    #: cumulative counters in that same record.
    logged_vs_recomputed: list[dict[str, Any]] = field(default_factory=list)
    max_disagreement_ratio: float | None = None
    findings: list[str] = field(default_factory=list)

    @property
    def resumed(self) -> bool:
        return len(self.segments) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps_per_record": self.steps_per_record,
            "window_records": self.window,
            "total_tokens": self.total_tokens,
            "total_seconds": round(self.total_seconds, 1),
            "run_wide_tokens_per_second": self.run_wide_tokens_per_second,
            "scope_note": (
                "run_wide_tokens_per_second spans every session. step_level and interval "
                "are per log record, which is steps_per_record optimizer steps apart."
            ),
            "n_segments": len(self.segments),
            "segments": [s.to_dict() for s in self.segments],
            "step_level": self.step_level,
            "interval": self.interval,
            "max_disagreement_ratio": self.max_disagreement_ratio,
            "logged_vs_recomputed": self.logged_vs_recomputed,
            "findings": self.findings,
        }


def _rolling(series: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    """Rate over the last ``window`` records, computed from summed deltas.

    Summed, not averaged: the mean of per-record rates weights a fast short record the
    same as a slow long one. Total tokens over total seconds is the rate over the window.
    """
    out: list[dict[str, Any]] = []
    for i in range(len(series)):
        chunk = series[max(0, i - window + 1) : i + 1]
        tokens = sum(int(c["delta_tokens"]) for c in chunk)
        seconds = sum(float(c["delta_seconds"]) for c in chunk)
        out.append(
            {
                "step": series[i]["step"],
                "records": len(chunk),
                "tokens_per_second": round(tokens / seconds, 1) if seconds > 0 else None,
            }
        )
    return out


def analyse_throughput(
    records: RunRecords, *, window: int = 4
) -> ThroughputAnalysis:
    """Recompute every rate from the cumulative counters, then audit what was logged.

    Nothing here trusts ``tokens_per_second`` as written. Each record carries cumulative
    ``tokens_seen`` and cumulative ``elapsed_s``; those two are sufficient, and they are
    the two fields the Level-2 bug did *not* corrupt. Rates are derived from them and the
    logged value is compared against the result.
    """
    analysis = ThroughputAnalysis(window=window)
    training = records.training
    if not training:
        analysis.findings.append("no training records with token counts — nothing to measure")
        return analysis

    analysis.segments = detect_segments(training)

    steps = [r.get("step") for r in training if isinstance(r.get("step"), int)]
    gaps = sorted({b - a for a, b in zip(steps, steps[1:], strict=False) if b > a})
    if gaps:
        analysis.steps_per_record = gaps[len(gaps) // 2]

    # Run-wide: the largest cumulative reading of each counter. Deliberately not summed
    # over segments — the counters are already cumulative across sessions, and adding
    # them again would double-count every resumed token.
    analysis.total_tokens = max(int(r.get("tokens_seen") or 0) for r in training)
    analysis.total_seconds = max(_as_float(r.get("elapsed_s")) or 0.0 for r in training)
    if analysis.total_seconds > 0:
        analysis.run_wide_tokens_per_second = round(
            analysis.total_tokens / analysis.total_seconds, 1
        )

    segment_start_steps = {s.first_step for s in analysis.segments}
    for previous, current in zip(training, training[1:], strict=False):
        step = current.get("step")
        delta_tokens = int(current.get("tokens_seen") or 0) - int(previous.get("tokens_seen") or 0)
        delta_seconds = (_as_float(current.get("elapsed_s")) or 0.0) - (
            _as_float(previous.get("elapsed_s")) or 0.0
        )
        # A pair that straddles a restart measures nothing: the counters were restored
        # from a checkpoint that may predate the last record. Kept in the series with a
        # null rate rather than dropped, so the gap is visible.
        crosses_boundary = step in segment_start_steps
        usable = delta_seconds > 0 and delta_tokens >= 0 and not crosses_boundary
        analysis.step_level.append(
            {
                "step": step,
                "delta_tokens": delta_tokens if usable else 0,
                "delta_seconds": round(delta_seconds, 3) if usable else 0.0,
                "tokens_per_second": (
                    round(delta_tokens / delta_seconds, 1) if usable else None
                ),
                "crosses_session_boundary": crosses_boundary,
            }
        )

    analysis.interval = _rolling(analysis.step_level, window)

    # The audit. `recompute_from_history` already knows how to recover correct rates from
    # the cumulative counters, including in logs written by the buggy code, so it is
    # reused here rather than reimplemented.
    for row in recompute_from_history(training):
        logged = _as_float(row.get("logged_tokens_per_second"))
        correct = _as_float(row.get("tokens_per_second"))
        if logged is None or correct is None or correct <= 0:
            continue
        ratio = logged / correct
        if abs(ratio - 1.0) > THROUGHPUT_DISAGREEMENT_TOLERANCE:
            analysis.logged_vs_recomputed.append(
                {
                    "step": row.get("step"),
                    "logged_tokens_per_second": logged,
                    "recomputed_tokens_per_second": correct,
                    "ratio": round(ratio, 2),
                }
            )
    if analysis.logged_vs_recomputed:
        analysis.max_disagreement_ratio = round(
            max(abs(r["ratio"]) for r in analysis.logged_vs_recomputed), 2
        )
        worst = max(analysis.logged_vs_recomputed, key=lambda r: abs(r["ratio"]))
        analysis.findings.append(
            f"{len(analysis.logged_vs_recomputed)} record(s) logged a run-wide rate "
            f"disagreeing with their own cumulative counters, by up to "
            f"{analysis.max_disagreement_ratio}x (step {worst['step']}: logged "
            f"{worst['logged_tokens_per_second']:.0f}, actually "
            f"{worst['recomputed_tokens_per_second']:.0f} tok/s). "
            f"Repair with: python scripts/recompute_throughput.py <metrics.jsonl>"
        )
    if analysis.resumed:
        analysis.findings.append(
            f"{len(analysis.segments)} sessions detected; run-wide figures span all of "
            f"them, per-segment figures do not"
        )
    return analysis


# ----------------------------------------------------------------------------------
# curves and plateaus
# ----------------------------------------------------------------------------------


@dataclass
class Curve:
    """One metric over steps, with the byte-level baseline where one exists."""

    name: str
    steps: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    #: What an untrained model scores. 8.0 for bits-per-byte on a 256-symbol vocabulary:
    #: a uniform distribution over bytes. Without it, "1.27" has no scale.
    baseline: float | None = None

    def __len__(self) -> int:
        return len(self.values)

    @property
    def best(self) -> tuple[int, float] | None:
        if not self.values:
            return None
        index = min(range(len(self.values)), key=lambda i: self.values[i])
        return self.steps[index], self.values[index]

    @property
    def final(self) -> tuple[int, float] | None:
        return (self.steps[-1], self.values[-1]) if self.values else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_points": len(self.values),
            "baseline": self.baseline,
            "first": {"step": self.steps[0], "value": self.values[0]} if self.values else None,
            "best": ({"step": self.best[0], "value": self.best[1]} if self.best else None),
            "final": ({"step": self.final[0], "value": self.final[1]} if self.final else None),
            "steps": self.steps,
            "values": self.values,
        }


def extract_curve(
    records: list[dict[str, Any]], key: str, *, name: str, baseline: float | None = None
) -> Curve:
    curve = Curve(name=name, baseline=baseline)
    for record in records:
        value = _as_float(record.get(key))
        step = record.get("step")
        if value is None or not isinstance(step, int):
            continue
        curve.steps.append(step)
        curve.values.append(value)
    return curve


def build_curves(records: RunRecords) -> dict[str, Curve]:
    """The four curves a byte-level run produces, whichever of them exist.

    A run on a non-text corpus logs no ``bits_per_byte``; the curve is then empty rather
    than absent, so downstream code does not branch on presence.
    """
    return {
        "train_loss": extract_curve(records.training, "loss", name="train loss"),
        "train_bpb": extract_curve(
            records.training, "bits_per_byte", name="train bits/byte", baseline=8.0
        ),
        "validation_loss": extract_curve(
            records.validation, "validation_loss", name="validation loss"
        ),
        "validation_bpb": extract_curve(
            records.validation, "validation_bits_per_byte",
            name="validation bits/byte", baseline=8.0,
        ),
    }


@dataclass
class PlateauAnalysis:
    """When improvement stopped, and how much of the run came after.

    Level 2's validation BPB reached 1.279 at step 400 and 1.270 at step 2000: 80% of the
    run bought under 1% of the improvement. That is a fact about the experiment worth
    surfacing on every run, because it is the difference between "trained for 2000 steps"
    and "needed 2000 steps".

    This deliberately stops at the measurement. A curve flattens because the model
    converged, because the corpus was exhausted, because the learning rate decayed to
    nothing, or because the objective was already saturated — and the curve looks the same
    in all four cases. :attr:`epochs_seen` is reported alongside precisely so the corpus
    explanation can be checked, not so it can be assumed.
    """

    metric: str
    n_points: int = 0
    best_step: int | None = None
    best_value: float | None = None
    plateau_step: int | None = None
    plateau_value: float | None = None
    tolerance: float = PLATEAU_TOLERANCE
    final_step: int | None = None
    improvement_after_plateau: float | None = None
    fraction_of_run_after_plateau: float | None = None
    epochs_seen: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "n_points": self.n_points,
            "tolerance": self.tolerance,
            "best": {"step": self.best_step, "value": self.best_value},
            "plateau": {"step": self.plateau_step, "value": self.plateau_value},
            "final_step": self.final_step,
            "improvement_after_plateau": self.improvement_after_plateau,
            "fraction_of_run_after_plateau": self.fraction_of_run_after_plateau,
            "epochs_seen": self.epochs_seen,
            "notes": self.notes,
            "interpretation_note": (
                "A plateau says improvement stopped. It does not say why. Convergence, an "
                "exhausted corpus and a decayed learning rate produce the same shape."
            ),
        }


def analyse_plateau(
    curve: Curve, *, tolerance: float = PLATEAU_TOLERANCE, epochs_seen: float | None = None
) -> PlateauAnalysis:
    """Find the first point within ``tolerance`` of the best value ever reached.

    "First within 1% of the best" rather than "first non-improving point", because a noisy
    curve produces a non-improving point early and often, and the question being asked is
    when the run stopped *mattering*, not when it stopped being monotone.

    Note this is 1% of the *value*, not 1% of the improvement — the sense in which the
    Level-2 report said 1.279 to 1.270 was "under 1%".
    """
    analysis = PlateauAnalysis(
        metric=curve.name, n_points=len(curve), tolerance=tolerance, epochs_seen=epochs_seen
    )
    if len(curve) < 2:
        analysis.notes.append("fewer than two points — no plateau can be identified")
        return analysis

    best_step, best_value = curve.best  # type: ignore[misc]
    analysis.best_step, analysis.best_value = best_step, best_value
    analysis.final_step = curve.steps[-1]

    first_value = curve.values[0]
    total_improvement = first_value - best_value
    # Within ``tolerance`` **of the best value**, which is the sense the Level-2 report
    # used: 1.279 to 1.270 is 0.7% of 1.270, so step 400 is the plateau and the
    # remaining 1600 steps are the finding. Measuring against the improvement *span*
    # instead would call 1.279 an 19%-remaining point and put the plateau at the last
    # step of every run, which reports nothing.
    #
    # Loss and bits-per-byte are both positive with a meaningful zero, so a relative
    # threshold is well defined. A metric that reaches zero or goes negative falls back
    # to the span, where relative-to-value would be degenerate.
    if best_value > 0:
        threshold = best_value * (1.0 + tolerance)
    else:
        threshold = best_value + abs(total_improvement) * tolerance

    for step, value in zip(curve.steps, curve.values, strict=True):
        if value <= threshold:
            analysis.plateau_step, analysis.plateau_value = step, value
            break

    if analysis.plateau_step is None or analysis.final_step is None:
        return analysis

    analysis.improvement_after_plateau = round(
        (analysis.plateau_value or 0.0) - best_value, 6
    )
    span = analysis.final_step - curve.steps[0]
    if span > 0:
        analysis.fraction_of_run_after_plateau = round(
            (analysis.final_step - analysis.plateau_step) / span, 3
        )

    fraction = analysis.fraction_of_run_after_plateau
    if fraction is not None and fraction >= 0.5 and total_improvement > 0:
        analysis.notes.append(
            f"{fraction:.0%} of the run happened after {curve.name} was already within "
            f"{tolerance:.0%} of its best value"
        )
    if epochs_seen is not None and epochs_seen >= 2.0:
        analysis.notes.append(
            f"the corpus was consumed {epochs_seen:.1f} times — a flat curve here may be "
            f"data exhaustion rather than convergence"
        )
    return analysis


# ----------------------------------------------------------------------------------
# checkpoints
# ----------------------------------------------------------------------------------


@dataclass
class CheckpointRecord:
    step: int
    path: str
    complete: bool
    reason: str | None = None
    created_at: str | None = None
    missing_files: list[str] = field(default_factory=list)
    #: Why the validator rejected this checkpoint, when it did. ``None`` when valid.
    invalid_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step, "path": self.path, "complete": self.complete,
            "reason": self.reason, "created_at": self.created_at,
            "missing_files": self.missing_files,
            "invalid_reason": self.invalid_reason,
        }


@dataclass
class CheckpointTimeline:
    """Which checkpoints exist, which are resumable, and where the run can restart from.

    ``complete`` is read from the marker the atomic write publishes last, not inferred
    from the directory existing. A ``.incomplete`` staging directory left by a killed
    process is reported as what it is: evidence of a crash, and not a resume point.
    """

    checkpoints: list[CheckpointRecord] = field(default_factory=list)
    incomplete_staging: list[str] = field(default_factory=list)
    latest_pointer_step: int | None = None
    findings: list[str] = field(default_factory=list)

    @property
    def complete(self) -> list[CheckpointRecord]:
        return [c for c in self.checkpoints if c.complete]

    @property
    def resumable_step(self) -> int | None:
        steps = [c.step for c in self.complete]
        return max(steps) if steps else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_checkpoints": len(self.checkpoints),
            "n_complete": len(self.complete),
            "resumable_step": self.resumable_step,
            "latest_pointer_step": self.latest_pointer_step,
            "incomplete_staging": self.incomplete_staging,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "findings": self.findings,
        }


def build_checkpoint_timeline(checkpoints_dir: Path | None) -> CheckpointTimeline:
    """Which checkpoints exist and which can actually be resumed from.

    Validity comes from
    :func:`qwen_distill.training.checkpoint_validation.validate_checkpoint_dir` — the one
    definition the trainer, the backup and the restore all use. This module used to check
    the required filenames itself, which meant a checkpoint whose weights had been
    deleted was reported here as complete while the persistence layer disagreed.
    """
    from ..training.checkpoint_validation import validate_checkpoint_dir

    timeline = CheckpointTimeline()
    if checkpoints_dir is None or not Path(checkpoints_dir).is_dir():
        timeline.findings.append("no checkpoints/ directory — nothing to resume from")
        return timeline
    checkpoints_dir = Path(checkpoints_dir)

    for entry in sorted(checkpoints_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") and "incomplete" in entry.name:
            timeline.incomplete_staging.append(entry.name)
            continue
        if not entry.name.startswith("step_"):
            continue
        metadata: dict[str, Any] = {}
        metadata_file = entry / "metadata.json"
        if metadata_file.is_file():
            try:
                loaded = json.loads(metadata_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    metadata = loaded
            except (OSError, json.JSONDecodeError):
                metadata = {}
        try:
            step = int(metadata.get("step", entry.name.split("_", 1)[1]))
        except (ValueError, IndexError):
            continue
        validation = validate_checkpoint_dir(entry)
        timeline.checkpoints.append(
            CheckpointRecord(
                step=step,
                path=str(entry),
                complete=validation.valid,
                reason=metadata.get("reason"),
                created_at=metadata.get("created_at") or metadata.get("timestamp"),
                missing_files=validation.missing_files,
                invalid_reason=validation.invalid_reason,
            )
        )

    timeline.checkpoints.sort(key=lambda c: c.step)

    pointer = checkpoints_dir / "latest.json"
    if pointer.is_file():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                value = payload.get("step")
                timeline.latest_pointer_step = int(value) if value is not None else None
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            timeline.findings.append("checkpoints/latest.json is present but unreadable")

    if timeline.incomplete_staging:
        timeline.findings.append(
            f"{len(timeline.incomplete_staging)} incomplete staging directory(ies) left "
            f"behind — a process died mid-write; these are not resume points"
        )
    broken = [c for c in timeline.checkpoints if not c.complete]
    if broken:
        timeline.findings.append(
            f"{len(broken)} checkpoint(s) cannot be resumed from: "
            + "; ".join(f"step {c.step} ({c.invalid_reason})" for c in broken[:5])
        )
    if (
        timeline.latest_pointer_step is not None
        and timeline.resumable_step is not None
        and timeline.latest_pointer_step != timeline.resumable_step
    ):
        timeline.findings.append(
            f"latest.json points at step {timeline.latest_pointer_step} but the newest "
            f"complete checkpoint is step {timeline.resumable_step}"
        )
    return timeline


# ----------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------


def render_text_curve(curve: Curve, *, width: int = 62, height: int = 12) -> str:
    """A curve as text, so the analysis is readable over SSH and inside a git diff.

    Not a substitute for a plot — it is what there is when matplotlib is not installed,
    which on a fresh Colab CPU runtime is the normal case.
    """
    if len(curve) < 2:
        return f"{curve.name}: {len(curve)} point(s) — nothing to plot"

    lo, hi = min(curve.values), max(curve.values)
    if not math.isfinite(lo) or not math.isfinite(hi):
        return f"{curve.name}: non-finite values"
    if hi - lo < 1e-12:
        return f"{curve.name}: flat at {lo:.4f} across {len(curve)} points"

    # Resample onto the plot width by taking the minimum in each column: a downsample by
    # striding would hide a spike between two sampled points.
    columns: list[float] = []
    for c in range(width):
        start = int(c * len(curve) / width)
        end = max(start + 1, int((c + 1) * len(curve) / width))
        columns.append(min(curve.values[start:end]))

    grid = [[" "] * width for _ in range(height)]
    for c, value in enumerate(columns):
        row = int((hi - value) / (hi - lo) * (height - 1))
        grid[min(height - 1, max(0, row))][c] = "*"

    lines = [f"{curve.name}  (step {curve.steps[0]} .. {curve.steps[-1]})"]
    for r, row in enumerate(grid):
        label = hi if r == 0 else (lo if r == height - 1 else None)
        prefix = f"{label:9.4f} |" if label is not None else " " * 9 + " |"
        lines.append(prefix + "".join(row))
    lines.append(" " * 10 + "+" + "-" * width)
    if curve.baseline is not None:
        lines.append(f"{' ' * 10}baseline (untrained): {curve.baseline:.3f}")
    return "\n".join(lines)


def write_plots(analysis: RunAnalysis, directory: str | Path) -> list[str]:
    """Write PNG plots if matplotlib is importable. Absence is not an error.

    Returns the files written — empty when matplotlib is missing, which the caller reports
    rather than treating as a failure. A plotting library is not a dependency of being
    able to read your own logs.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def _save(figure: Any, name: str) -> None:
        path = directory / name
        figure.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(figure)
        written.append(str(path))

    losses = [c for c in (analysis.curves["train_loss"], analysis.curves["validation_loss"]) if len(c)]
    if losses:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        for curve in losses:
            axis.plot(curve.steps, curve.values, label=curve.name, marker="." if len(curve) < 60 else None)
        axis.set_xlabel("step")
        axis.set_ylabel("loss")
        axis.legend()
        axis.grid(alpha=0.3)
        axis.set_title(f"{analysis.name} — loss")
        _save(figure, "loss.png")

    bpbs = [c for c in (analysis.curves["train_bpb"], analysis.curves["validation_bpb"]) if len(c)]
    if bpbs:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        for curve in bpbs:
            axis.plot(curve.steps, curve.values, label=curve.name, marker="." if len(curve) < 60 else None)
        axis.axhline(8.0, linestyle="--", color="grey", label="uniform byte baseline (8.0)")
        axis.set_xlabel("step")
        axis.set_ylabel("bits per byte")
        axis.legend()
        axis.grid(alpha=0.3)
        axis.set_title(f"{analysis.name} — bits per byte")
        _save(figure, "bits_per_byte.png")

    step_level = [(r["step"], r["tokens_per_second"]) for r in analysis.throughput.step_level
                  if r.get("tokens_per_second") is not None]
    interval = [(r["step"], r["tokens_per_second"]) for r in analysis.throughput.interval
                if r.get("tokens_per_second") is not None]
    if step_level:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.plot(*zip(*step_level, strict=True), alpha=0.35, label=f"per record ({analysis.throughput.steps_per_record} steps)")
        if interval:
            axis.plot(*zip(*interval, strict=True), label=f"interval ({analysis.throughput.window}-record window)")
        if analysis.throughput.run_wide_tokens_per_second:
            axis.axhline(
                analysis.throughput.run_wide_tokens_per_second, linestyle="--", color="black",
                label=f"run-wide ({analysis.throughput.run_wide_tokens_per_second:.0f} tok/s)",
            )
        for segment in analysis.throughput.segments[1:]:
            axis.axvline(segment.first_step, color="red", alpha=0.4, linestyle=":")
        axis.set_xlabel("step")
        axis.set_ylabel("tokens / second")
        axis.legend()
        axis.grid(alpha=0.3)
        axis.set_title(f"{analysis.name} — throughput (three scopes)")
        _save(figure, "throughput.png")

    return written


# ----------------------------------------------------------------------------------
# the whole run
# ----------------------------------------------------------------------------------

#: Reaching ``max_steps`` means the loop exited, and nothing else. See
#: ``docs/experiments/POST_RUN_CHECKLIST.md``.
COMPLETION_CAVEAT = (
    "Reaching max_steps means the training loop ran to its configured end. It does not "
    "mean the experiment is complete, that the model is any good, or that the result is "
    "publishable. See docs/experiments/POST_RUN_CHECKLIST.md."
)


@dataclass
class RunAnalysis:
    """Everything recoverable about one run, with each claim kept at its own confidence.

    Nothing here is a verdict on model quality. Loss curves, throughput and checkpoint
    integrity are all measurements of the *training process*; Level 2 scored well on every
    one of them and produced ``"and and and"``. Quality requires
    :mod:`qwen_distill.training.sanity` and an evaluation, which are separate tools
    because they answer a separate question.
    """

    name: str
    files: RunFiles
    records: RunRecords
    curves: dict[str, Curve]
    throughput: ThroughputAnalysis
    plateau: PlateauAnalysis
    checkpoints: CheckpointTimeline
    summary: dict[str, Any] = field(default_factory=dict)
    max_steps: int | None = None
    steps_completed: int | None = None
    epochs_seen: float | None = None
    findings: list[str] = field(default_factory=list)

    @property
    def reached_max_steps(self) -> bool:
        return (
            self.max_steps is not None
            and self.steps_completed is not None
            and self.steps_completed >= self.max_steps
        )

    @property
    def loop_status(self) -> str:
        """The state of the training *loop*, never a judgement about the experiment."""
        if self.steps_completed is None:
            return "UNKNOWN — no step recorded"
        if self.max_steps is None:
            return f"ran to step {self.steps_completed} (no max_steps found)"
        if self.reached_max_steps:
            return f"loop reached max_steps ({self.steps_completed}/{self.max_steps})"
        return (
            f"loop stopped early at {self.steps_completed}/{self.max_steps} "
            f"({self.steps_completed / self.max_steps:.0%})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run_directory": str(self.files.root),
            "missing_artefacts": self.files.missing(),
            "loop_status": self.loop_status,
            "reached_max_steps": self.reached_max_steps,
            "completion_caveat": COMPLETION_CAVEAT,
            "steps_completed": self.steps_completed,
            "max_steps": self.max_steps,
            "epochs_seen": self.epochs_seen,
            "n_training_records": len(self.records.training),
            "n_validation_records": len(self.records.validation),
            "curves": {name: curve.to_dict() for name, curve in self.curves.items()},
            "throughput": self.throughput.to_dict(),
            "plateau": self.plateau.to_dict(),
            "checkpoints": self.checkpoints.to_dict(),
            "findings": self.findings,
            "quality_note": (
                "None of these measurements establish model quality. A run can score well "
                "on every one of them and generate degenerate text — Level 2 did. Run "
                "scripts/sanity_generate.py against a checkpoint before claiming anything "
                "about capability."
            ),
        }

    def render(self, *, plots: bool = True) -> str:
        rule = "=" * 78
        lines = [rule, f"RUN ANALYSIS — {self.name}", rule, "", f"  directory     : {self.files.root}"]
        missing = self.files.missing()
        if missing:
            lines.append(f"  missing       : {', '.join(missing)}")
        lines += [
            f"  loop status   : {self.loop_status}",
            f"  records       : {len(self.records.training)} training, "
            f"{len(self.records.validation)} validation",
        ]
        if self.epochs_seen is not None:
            lines.append(f"  epochs seen   : {self.epochs_seen:.2f}")

        lines += ["", "-" * 78, "THROUGHPUT — three scopes, never collapsed", "-" * 78]
        throughput = self.throughput
        run_wide = throughput.run_wide_tokens_per_second
        lines += [
            f"  run-wide      : {run_wide:>10,.1f} tok/s   "
            f"({throughput.total_tokens:,} tokens / {throughput.total_seconds:,.0f} s, "
            f"{len(throughput.segments)} session(s))" if run_wide else "  run-wide      : unavailable",
        ]
        recent = [r["tokens_per_second"] for r in throughput.step_level
                  if r.get("tokens_per_second") is not None]
        if recent:
            lines += [
                f"  per record    : {min(recent):>10,.1f} .. {max(recent):,.1f} tok/s   "
                f"(each record spans {throughput.steps_per_record} steps)",
                f"  latest record : {recent[-1]:>10,.1f} tok/s",
            ]
        windowed = [r["tokens_per_second"] for r in throughput.interval
                    if r.get("tokens_per_second") is not None]
        if windowed:
            lines.append(
                f"  latest window : {windowed[-1]:>10,.1f} tok/s   "
                f"({throughput.window}-record rolling)"
            )
        if throughput.resumed:
            lines.append("")
            lines.append("  sessions:")
            for segment in throughput.segments:
                rate = segment.tokens_per_second
                rate_text = f"{rate:,.1f} tok/s" if rate else "n/a"
                lines.append(
                    f"    [{segment.index}] steps {segment.first_step:>6} -> "
                    f"{segment.last_step:<6}  {segment.tokens:>12,} tok  "
                    f"{segment.seconds:>8,.0f} s  {rate_text:>14}   ({segment.boundary_reason})"
                )

        lines += ["", "-" * 78, "CURVES", "-" * 78]
        for key in ("train_loss", "train_bpb", "validation_loss", "validation_bpb"):
            curve = self.curves[key]
            if not len(curve):
                continue
            first, best, final = curve.values[0], curve.best, curve.final
            lines.append(
                f"  {curve.name:<22} first {first:8.4f}   best {best[1]:8.4f} @ {best[0]:<6}"
                f"   final {final[1]:8.4f} @ {final[0]}"
            )
            if curve.baseline is not None:
                lines.append(
                    f"  {'':<22} baseline (untrained) {curve.baseline:.3f}"
                )
        if plots:
            for key in ("validation_bpb", "validation_loss", "train_loss"):
                curve = self.curves[key]
                if len(curve) >= 2:
                    lines += ["", render_text_curve(curve)]
                    break

        lines += ["", "-" * 78, "PLATEAU", "-" * 78]
        plateau = self.plateau
        if plateau.plateau_step is None:
            lines.append(f"  {plateau.metric}: no plateau identified ({plateau.n_points} points)")
        else:
            lines += [
                f"  metric        : {plateau.metric}",
                f"  best          : {plateau.best_value:.4f} at step {plateau.best_step}",
                f"  within {plateau.tolerance:.0%} of best by step {plateau.plateau_step} "
                f"({plateau.plateau_value:.4f})",
            ]
            if plateau.fraction_of_run_after_plateau is not None:
                lines.append(
                    f"  after that    : {plateau.fraction_of_run_after_plateau:.0%} of the "
                    f"run remained, buying {plateau.improvement_after_plateau:.4f}"
                )
        for note in plateau.notes:
            lines.append(f"  ! {note}")
        lines.append("  A plateau says improvement stopped, not why.")

        lines += ["", "-" * 78, "CHECKPOINTS", "-" * 78]
        timeline = self.checkpoints
        lines.append(
            f"  {len(timeline.complete)} complete of {len(timeline.checkpoints)}; "
            f"newest resumable: "
            + (f"step {timeline.resumable_step}" if timeline.resumable_step is not None else "none")
        )
        for record in timeline.checkpoints[-8:]:
            mark = "ok " if record.complete else "BAD"
            lines.append(
                f"    [{mark}] step {record.step:>6}  {record.reason or '-':<12} "
                f"{record.created_at or ''}"
            )

        if self.findings:
            lines += ["", "-" * 78, "FINDINGS", "-" * 78]
            for finding in self.findings:
                lines.append(f"  ! {finding}")

        lines += [
            "", "-" * 78,
            "WHAT THIS DOES NOT ESTABLISH",
            "-" * 78,
            "  Everything above measures the training process, not the model. Level 2",
            "  scored 1.270 bits/byte here and generated \"and and and\". Run",
            "  scripts/sanity_generate.py on a checkpoint before claiming capability.",
            "",
            f"  {COMPLETION_CAVEAT}",
            rule,
        ]
        return "\n".join(lines)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {}
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _dig(payload: dict[str, Any], *keys: str) -> Any:
    """First matching key at the top level or one level down.

    ``summary.json`` has grown nested sections across phases, and an analysis that only
    looks at the top level silently reports ``max_steps`` as unknown for every run written
    before or after the shape it was written against.
    """
    for key in keys:
        if key in payload:
            return payload[key]
    for value in payload.values():
        if isinstance(value, dict):
            for key in keys:
                if key in value:
                    return value[key]
    return None


def analyse_run(
    run_directory: str | Path, *, window: int = 4, name: str | None = None
) -> RunAnalysis:
    """Read one run directory and report what is in it.

    Read-only, CPU-only, and safe against a run that is still training: the last line of
    ``metrics.jsonl`` may be half-written, and that costs one record.
    """
    files = RunFiles.discover(run_directory)
    records = RunRecords.from_records(read_jsonl(files.metrics) if files.metrics else [])
    summary = _read_json(files.summary)
    latest = _read_json(files.latest_progress)

    epochs = [
        _as_float(r.get("epoch")) for r in records.training if _as_float(r.get("epoch")) is not None
    ]
    epochs_seen = max(epochs) if epochs else None

    curves = build_curves(records)
    throughput = analyse_throughput(records, window=window)

    # Prefer validation BPB, then validation loss, then the training curve: a plateau in
    # training loss says the least, so it is the last resort rather than the default.
    for key in ("validation_bpb", "validation_loss", "train_bpb", "train_loss"):
        if len(curves[key]) >= 2:
            plateau = analyse_plateau(curves[key], epochs_seen=epochs_seen)
            break
    else:
        plateau = analyse_plateau(curves["train_loss"], epochs_seen=epochs_seen)

    timeline = build_checkpoint_timeline(files.checkpoints_dir)

    max_steps = _dig(summary, "max_steps")
    steps_completed = records.last_step
    for candidate in (latest.get("step"), _dig(summary, "steps_completed", "step")):
        if isinstance(candidate, int):
            steps_completed = max(steps_completed or 0, candidate)

    analysis = RunAnalysis(
        name=name or _dig(summary, "name", "experiment") or files.root.name,
        files=files, records=records, curves=curves, throughput=throughput,
        plateau=plateau, checkpoints=timeline, summary=summary,
        max_steps=max_steps if isinstance(max_steps, int) else None,
        steps_completed=steps_completed, epochs_seen=epochs_seen,
    )

    analysis.findings.extend(throughput.findings)
    analysis.findings.extend(timeline.findings)
    analysis.findings.extend(plateau.notes)
    if not records.training:
        analysis.findings.append("no training records — check that metrics.jsonl exists")
    if not records.validation:
        analysis.findings.append(
            "no validation records — the run reports only training loss, which cannot "
            "distinguish learning from memorisation"
        )
    if analysis.reached_max_steps:
        analysis.findings.append(COMPLETION_CAVEAT)
    if timeline.resumable_step is not None and steps_completed is not None:
        behind = steps_completed - timeline.resumable_step
        if behind > 0:
            analysis.findings.append(
                f"the newest complete checkpoint is {behind} steps behind the newest log "
                f"record — that much work would be lost to a crash right now"
            )
    return analysis
