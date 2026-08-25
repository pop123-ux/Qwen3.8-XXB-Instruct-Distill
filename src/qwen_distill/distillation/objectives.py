"""What the student is actually trained against — and what is not implemented yet.

The rule this module exists to enforce: **never call something KD that is SFT.** A
knowledge-distillation run that silently trains on teacher text and reports itself as
logit KD would make the project's central comparison — does KD beat SFT? — meaningless,
and the error would be invisible in every artifact.

So each objective declares its own availability. Selecting an unavailable one raises,
with what is missing and why. Nothing degrades quietly.

Status in this phase:

* ``sft`` — **implemented**. Trains on the teacher's response text.
* ``logit_kd`` — **NOT IMPLEMENTED**. Needs stored teacher logits, which no dataset has
  yet: full distributions over a ~248k vocabulary are prohibitive to store, so top-k is
  the intended first step and the storage format is not settled.
* ``mixed_kd`` — **NOT IMPLEMENTED**. Requires ``logit_kd``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Objective names this project recognises.
SFT = "sft"
LOGIT_KD = "logit_kd"
MIXED_KD = "mixed_kd"

IMPLEMENTED = "IMPLEMENTED"
NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class ObjectiveUnavailable(NotImplementedError):
    """A named objective exists in the schema but cannot run."""


@dataclass(frozen=True)
class ObjectiveSpec:
    """One training objective: what it needs, and whether it can run at all."""

    name: str
    status: str
    description: str
    #: Record fields an example must carry for this objective to be trainable.
    required_fields: tuple[str, ...] = ()
    blocking_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.status == IMPLEMENTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "status": self.status, "available": self.available,
            "description": self.description,
            "required_fields": list(self.required_fields),
            "blocking_reason": self.blocking_reason,
        }


OBJECTIVES: dict[str, ObjectiveSpec] = {
    SFT: ObjectiveSpec(
        name=SFT,
        status=IMPLEMENTED,
        description=(
            "supervised fine-tuning on the teacher's response text: the student learns "
            "prompt -> teacher_response"
        ),
        required_fields=("prompt", "teacher_answer"),
    ),
    LOGIT_KD: ObjectiveSpec(
        name=LOGIT_KD,
        status=NOT_IMPLEMENTED,
        description=(
            "token-level knowledge distillation against stored teacher logits: the "
            "student matches the teacher's output distribution rather than its samples"
        ),
        required_fields=("teacher_top_logits", "teacher_logits_path"),
        blocking_reason=(
            "no teacher logits exist yet, and the storage format is unsettled. Full "
            "distributions over a ~248k vocabulary are prohibitive to store per token, "
            "so top-k is the intended first step — that decision has not been made or "
            "measured. This raises rather than falling back to SFT: a KD run that is "
            "secretly SFT would invalidate the comparison the project exists to make."
        ),
    ),
    MIXED_KD: ObjectiveSpec(
        name=MIXED_KD,
        status=NOT_IMPLEMENTED,
        description="weighted combination of SFT and logit KD",
        required_fields=("teacher_top_logits",),
        blocking_reason=f"requires {LOGIT_KD!r}, which is not implemented",
    ),
}


@dataclass
class ObjectiveConfig:
    """A requested objective and its hyperparameters.

    KD parameters are ``None`` by default rather than carrying invented values, so a
    config cannot look tuned when nothing has been chosen.
    """

    type: str = SFT
    kd_temperature: float | None = None
    kd_alpha: float | None = None
    #: Whether SFT trains on the teacher's reasoning trace as well as its answer. A real
    #: experimental variable: it changes what the student learns to spend tokens on.
    include_reasoning_in_target: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def spec(self) -> ObjectiveSpec:
        if self.type not in OBJECTIVES:
            raise ObjectiveUnavailable(
                f"unknown objective {self.type!r}. Known: {', '.join(sorted(OBJECTIVES))}"
            )
        return OBJECTIVES[self.type]

    def validate(self) -> list[str]:
        """Problems with this configuration; empty means it can be requested."""
        problems: list[str] = []
        try:
            spec = self.spec()
        except ObjectiveUnavailable as exc:
            return [str(exc)]
        if not spec.available:
            problems.append(f"{self.type}: {spec.blocking_reason}")
        if self.type == SFT and (self.kd_temperature or self.kd_alpha):
            problems.append(
                "kd_temperature/kd_alpha are set but the objective is sft, where they "
                "do nothing — this usually means the objective was not switched"
            )
        if self.type in (LOGIT_KD, MIXED_KD):
            if self.kd_temperature is not None and self.kd_temperature <= 0:
                problems.append("kd_temperature must be positive")
            if self.kd_alpha is not None and not 0.0 <= self.kd_alpha <= 1.0:
                problems.append("kd_alpha must be between 0 and 1")
        return problems

    def require_available(self) -> ObjectiveSpec:
        """Return the spec, or raise explaining exactly why it cannot run."""
        spec = self.spec()
        if not spec.available:
            raise ObjectiveUnavailable(
                f"objective {self.type!r} is {spec.status}: {spec.blocking_reason}"
            )
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "kd_temperature": self.kd_temperature,
            "kd_alpha": self.kd_alpha,
            "include_reasoning_in_target": self.include_reasoning_in_target,
            "status": self.spec().status if self.type in OBJECTIVES else "UNKNOWN",
            "metadata": self.metadata,
        }


def check_dataset_supports(config: ObjectiveConfig, dataset: Any) -> list[str]:
    """Whether a loaded dataset can actually serve this objective.

    Catches the case where the objective is implemented but the data cannot feed it —
    asking for KD against a dataset with no logits, for instance.
    """
    problems = config.validate()
    if config.type in (LOGIT_KD, MIXED_KD):
        with_targets = getattr(dataset.stats, "n_with_kd_targets", 0)
        if not with_targets:
            problems.append(
                "no record in this dataset carries teacher logits, so KD has nothing to "
                "match against"
            )
    if config.type == SFT and not len(dataset):
        problems.append("dataset is empty")
    return problems


def describe_objectives() -> str:
    """A human-readable status table. Used by --list-objectives and the docs."""
    lines = [f"  {'objective':<12}{'status':<18}notes"]
    for name in (SFT, LOGIT_KD, MIXED_KD):
        spec = OBJECTIVES[name]
        lines.append(f"  {name:<12}{spec.status:<18}{spec.description}")
        if spec.blocking_reason:
            lines.append(f"  {'':<30}BLOCKED: {spec.blocking_reason}")
    return "\n".join(lines)
