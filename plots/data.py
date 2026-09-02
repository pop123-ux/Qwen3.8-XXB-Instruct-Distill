"""Normalised access to run artifacts: the layer between raw JSONL and any figure.

The pipeline the figures follow is::

    raw run artifacts  ->  this module  ->  figure-specific selection  ->  matplotlib

so a figure never parses a file itself and never learns a directory layout. Two properties
matter more than convenience:

**Trajectories come from ``metrics.jsonl``, not from the summary.** A 128-step run has 128
points; reducing it to the summary's first/final pair because the summary is easier to read
would throw away the shape, which is the part a training figure exists to show. The summary
and the ledger stay the provenance and index layer.

**Comparability is derived, not asserted.** :func:`matched_arms` admits a run into a
cross-objective comparison only when its protocol — sequence length, steps, batch, seed,
optimiser, precision, PEFT geometry, schedule, corpus and teacher revision — equals the
reference arm's. The objective is excluded from that comparison because the objective is
the experimental variable. A run that does not match is reported with the fields that
differ rather than quietly plotted next to one that does, which is how Run 001 (1024
tokens, mixed KD, 50 steps) stays out of a Run 002 comparison without anybody remembering
to exclude it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from common import ROOT, MissingData

EXPERIMENTS = ROOT / "experiments"
LEDGER = EXPERIMENTS / "ledger.jsonl"

#: How the ladder's objectives are named in a figure legend.
OBJECTIVE_LABELS: dict[str, str] = {
    "sft": "CE only",
    "logit_kd": "logit KD",
    "layer_kd": "layer / intermediate KD",
    "behavioral_kd": "behavioural / state KD",
    "mixed_kd": "CE + logit KD",
}

#: The order objectives appear in every comparison figure, so colours stay stable as arms
#: arrive. An objective absent from a figure leaves a gap rather than shifting the rest.
OBJECTIVE_ORDER: tuple[str, ...] = ("sft", "logit_kd", "layer_kd", "behavioral_kd")

#: Config fields whose equality defines "the same experiment apart from the objective".
PROTOCOL_FIELDS: tuple[tuple[str, str], ...] = (
    ("data", "max_sequence_length"),
    ("training", "max_steps"),
    ("training", "batch_size"),
    ("training", "gradient_accumulation_steps"),
    ("training", "seed"),
    ("training", "optimizer"),
    ("training", "precision"),
    ("training", "strategy"),
    ("training", "lora_rank"),
    ("training", "lora_alpha"),
    ("training", "lora_dropout"),
    ("training", "learning_rate"),
    ("training", "weight_decay"),
    ("training", "warmup_steps"),
    ("training", "scheduler"),
    ("training", "gradient_checkpointing"),
)

CALIBRATION = "calibration"
TRAINING = "training"


class MissingMetric(MissingData):
    """A run exists but never logged the requested field."""

    def __init__(self, experiment_id: str, field: str, available: list[str]) -> None:
        super().__init__(
            f"{field!r} in {experiment_id}",
            f"the run logged {', '.join(sorted(available)) or 'nothing'}; "
            f"a figure over {field!r} needs a run that records it",
        )


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------
@dataclass
class RunArtifacts:
    """Everything one experiment directory says, parsed once."""

    experiment_id: str
    root: Path
    summary: dict[str, Any]
    records: list[dict[str, Any]]

    # -- record views ------------------------------------------------------
    @property
    def steps(self) -> list[dict[str, Any]]:
        """Per-optimizer-step records, in step order. The trajectory."""
        rows = [r for r in self.records if r.get("status") == "completed_step"]
        return sorted(rows, key=lambda r: r["step"])

    @property
    def validations(self) -> list[dict[str, Any]]:
        """Actual validation observations. There are as many as the run performed."""
        rows = [r for r in self.records if r.get("status") == "validated"]
        return sorted(rows, key=lambda r: r["step"])

    @property
    def logged_fields(self) -> set[str]:
        fields: set[str] = set()
        for row in self.steps:
            fields |= {k for k, v in row.items() if v is not None}
        return fields

    # -- identity ----------------------------------------------------------
    @property
    def config(self) -> dict[str, Any]:
        return self.summary.get("config") or {}

    @property
    def objective(self) -> str:
        return (self.summary.get("objective")
                or self.config.get("training", {}).get("objective") or "unknown")

    @property
    def label(self) -> str:
        return OBJECTIVE_LABELS.get(self.objective, self.objective)

    @property
    def sequence_length(self) -> int | None:
        return self.config.get("data", {}).get("max_sequence_length")

    @property
    def n_logged_steps(self) -> int:
        """Step *records* in the trajectory. ``log_every`` may exceed 1, so this is the
        number of plotted points, not the number of optimizer steps."""
        return len(self.steps)

    @property
    def last_step(self) -> int | None:
        """The highest optimizer step reached, taken from the record rather than the
        summary. The two differ in ``kd_run_001``, where ``summary.json`` is the one-step
        smoke and ``metrics.jsonl`` holds the 50-step pilot. The record wins."""
        return self.steps[-1]["step"] if self.steps else None

    @property
    def tokens_seen(self) -> int | None:
        return self.steps[-1].get("tokens_seen") if self.steps else None

    @property
    def data_commit(self) -> str | None:
        for row in self.steps or self.records:
            if row.get("git_commit"):
                return row["git_commit"]
        return self.summary.get("git_commit")

    @property
    def run_class(self) -> str:
        """``calibration`` for a single-step memory probe, ``training`` otherwise."""
        return CALIBRATION if self.n_logged_steps <= 1 else TRAINING

    @property
    def memory(self) -> dict[str, Any]:
        return self.summary.get("memory") or {}

    @property
    def teacher_revision(self) -> str | None:
        return (self.config.get("teacher") or {}).get("revision")

    @property
    def corpus_sha(self) -> str | None:
        return (self.summary.get("corpus") or {}).get("sha256")

    def protocol(self) -> dict[str, Any]:
        """The fields that must be equal for two runs to be a controlled comparison."""
        config = self.config
        values = {f"{section}.{field}": (config.get(section) or {}).get(field)
                  for section, field in PROTOCOL_FIELDS}
        values["corpus.sha256"] = self.corpus_sha
        values["teacher.revision"] = self.teacher_revision
        return values

    def protocol_diff(self, other: RunArtifacts) -> dict[str, tuple[Any, Any]]:
        mine, theirs = self.protocol(), other.protocol()
        return {k: (mine[k], theirs[k]) for k in mine if mine[k] != theirs[k]}

    # -- series ------------------------------------------------------------
    def series(self, field: str, *, x: str = "tokens_seen") -> tuple[list[Any], list[Any]]:
        """``(xs, ys)`` over every step record carrying ``field``.

        Raises :class:`MissingMetric` rather than returning an empty pair, so a figure
        cannot silently draw nothing and look like a flat result.
        """
        points = [(r.get(x), r[field]) for r in self.steps
                  if r.get(field) is not None and r.get(x) is not None]
        if not points:
            raise MissingMetric(self.experiment_id, field, sorted(self.logged_fields))
        return [p[0] for p in points], [p[1] for p in points]

    def validation_series(self, *, x: str = "tokens_seen") -> tuple[list[Any], list[float]]:
        """Validation observations only — never interpolated onto the step grid.

        ``metrics.jsonl`` records a validation with its step but no token count, so the
        token axis is recovered from the step record at the same step. A validation whose
        step has no matching step record is dropped rather than placed by guesswork.
        """
        observations = self.validations
        if not observations:
            raise MissingMetric(self.experiment_id, "validation_loss", sorted(self.logged_fields))
        if x == "step":
            return [r["step"] for r in observations], [r["validation_loss"] for r in observations]
        by_step = {r["step"]: r.get("tokens_seen") for r in self.steps}
        pairs = [(by_step[r["step"]], r["validation_loss"])
                 for r in observations if by_step.get(r["step"]) is not None]
        if not pairs:
            raise MissingMetric(self.experiment_id, "validation_loss@tokens",
                                sorted(self.logged_fields))
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def source_paths(self) -> tuple[str, ...]:
        return (str((self.root / "metrics.jsonl").relative_to(ROOT)),
                str((self.root / "summary.json").relative_to(ROOT)))


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_run(experiment_id: str, experiments: Path | None = None) -> RunArtifacts:
    """Load one experiment directory, or fail naming what would produce it."""
    base = Path(experiments) if experiments else EXPERIMENTS
    return _load(base / experiment_id)


@cache
def _load(root: Path) -> RunArtifacts:
    """Cached by resolved directory, so pointing the loader at a fixture tree in a test
    cannot return a run parsed out of the real ``experiments/``."""
    experiment_id = root.name
    metrics = root / "metrics.jsonl"
    if not metrics.exists():
        raise MissingData(
            f"experiment {experiment_id!r} (no {metrics.relative_to(ROOT) if root.is_relative_to(ROOT) else metrics})",
            f"run the experiment; scripts/kd_run.py writes experiments/{experiment_id}/",
        )
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return RunArtifacts(experiment_id, root, summary, _read_jsonl(metrics))


def discover_runs(experiments: Path | None = None) -> list[RunArtifacts]:
    """Every experiment directory carrying a metrics trajectory, id-sorted."""
    base = Path(experiments) if experiments else EXPERIMENTS
    if not base.exists():
        return []
    found = []
    for directory in sorted(base.iterdir()):
        if (directory / "metrics.jsonl").exists():
            found.append(load_run(directory.name, base))
    return found


@dataclass
class ArmSet:
    """The runs admitted to one controlled comparison, and why the rest were not."""

    reference: RunArtifacts
    arms: dict[str, RunArtifacts]
    excluded: list[tuple[str, str]]

    def ordered(self) -> list[tuple[str, RunArtifacts]]:
        return [(o, self.arms[o]) for o in OBJECTIVE_ORDER if o in self.arms]

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    def sources(self) -> tuple[str, ...]:
        paths: list[str] = []
        for _, run in self.ordered():
            paths.extend(run.source_paths())
        return tuple(paths)

    def experiments(self) -> tuple[str, ...]:
        return tuple(run.experiment_id for _, run in self.ordered())


def matched_arms(reference: str = "run002_logit_kd", *,
                 experiments: Path | None = None) -> ArmSet:
    """Runs that differ from ``reference`` only in their distillation objective.

    The reference itself is always admitted. Every other training run is compared field by
    field against :data:`PROTOCOL_FIELDS`; the first mismatch is the recorded reason. Two
    runs with the same objective would make the comparison ambiguous, so the later one is
    excluded and said so.
    """
    ref = load_run(reference, experiments)
    arms: dict[str, RunArtifacts] = {ref.objective: ref}
    excluded: list[tuple[str, str]] = []
    for run in discover_runs(experiments):
        if run.experiment_id == ref.experiment_id:
            continue
        if run.run_class == CALIBRATION:
            excluded.append((run.experiment_id,
                             f"{run.n_logged_steps}-step calibration, not an arm"))
            continue
        diff = run.protocol_diff(ref)
        if diff:
            detail = "; ".join(f"{k}: {v[0]!r} vs reference {v[1]!r}"
                               for k, v in sorted(diff.items()))
            excluded.append((run.experiment_id, f"protocol differs — {detail}"))
            continue
        if run.objective in arms:
            excluded.append((run.experiment_id,
                             f"a second {run.objective!r} arm; {arms[run.objective].experiment_id} "
                             f"is already the one plotted"))
            continue
        arms[run.objective] = run
    return ArmSet(ref, arms, excluded)


def require_arms(armset: ArmSet, minimum: int, what: str) -> ArmSet:
    """Refuse to draw a comparison that has nothing to compare."""
    if armset.n_arms < minimum:
        have = ", ".join(f"{o} ({r.experiment_id})" for o, r in armset.ordered())
        missing = [o for o in OBJECTIVE_ORDER if o not in armset.arms]
        raise MissingData(
            f"{what} — {armset.n_arms} matched arm(s): {have}",
            f"run the remaining arm(s) at the reference protocol "
            f"({', '.join(missing)}) so the comparison is controlled; "
            f"the reference is {armset.reference.experiment_id}",
        )
    return armset


# ---------------------------------------------------------------------------
# the ledger, as index and provenance
# ---------------------------------------------------------------------------
def ledger_entries(kind: str | None = None, arm: str | None = None) -> list[dict[str, Any]]:
    """Ledger rows, filtered. The ledger indexes and attests; it is not a trajectory."""
    if not LEDGER.exists():
        return []
    rows = _read_jsonl(LEDGER)
    if kind:
        rows = [r for r in rows if r.get("kind") == kind]
    if arm:
        rows = [r for r in rows if r.get("arm") == arm]
    retracted = {r.get("supersedes") for r in rows if r.get("kind") == "retraction"}
    return [r for r in rows if r.get("id") not in retracted]
