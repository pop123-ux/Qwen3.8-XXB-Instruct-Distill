"""What the student is actually trained against — and what is not implemented yet.

The rule this module exists to enforce: **never call something KD that is SFT.** A
knowledge-distillation run that silently trains on teacher text and reports itself as
logit KD would make the project's central comparison — does KD beat SFT? — meaningless,
and the error would be invisible in every artifact.

So each objective declares its own availability. Selecting an unavailable one raises,
with what is missing and why. Nothing degrades quietly.

Status:

* ``sft`` — **implemented**. Trains on the teacher's response text.
* ``logit_kd`` — **implemented** (:mod:`qwen_distill.distillation.kd_loss`). Matches the
  teacher's distribution. The distribution has to come from somewhere, and that is a
  separate axis: ``signal_source="online"`` runs a resident teacher, ``"dataset"`` reads
  stored top-k logits. Only the online source is wired today; requesting the other says
  so rather than falling back.
* ``mixed_kd`` — **implemented**. ``kd_weight`` of the KD term, the rest cross-entropy.
  ``kd_weight=0`` reduces exactly to SFT through the same code path, which is what makes
  it usable as the control.

The rule at the top still holds, and now has teeth in a second place: an objective is only
available when the *signal source* it needs is, so "KD" cannot silently become SFT because
no teacher was configured.
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
        status=IMPLEMENTED,
        description=(
            "token-level knowledge distillation: the student matches the teacher's output "
            "distribution rather than its samples"
        ),
        required_fields=(),
    ),
    MIXED_KD: ObjectiveSpec(
        name=MIXED_KD,
        status=IMPLEMENTED,
        description="weighted combination of cross-entropy and logit KD",
        required_fields=(),
    ),
}

#: Where the teacher distribution comes from, and whether that path exists yet.
ONLINE = "online"
DATASET = "dataset"
SIGNAL_SOURCES: dict[str, str | None] = {
    ONLINE: None,
    DATASET: (
        "stored teacher logits are not readable yet: the loss and the capture format are "
        "ready (top-k logits plus the full-vocabulary logsumexp), but the on-disk corpus "
        "layout is deliberately unchosen until a real run reports its tail mass at a "
        "candidate k. Use signal_source='online' with a resident teacher."
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
    #: Where the teacher distribution comes from. Ignored when the objective is SFT.
    #: Defaults to the *unimplemented* source deliberately: this axis is the one that can
    #: turn KD into SFT without anything looking wrong, so a config that does not say
    #: where its teacher comes from is refused rather than guessed at.
    signal_source: str = DATASET
    #: How the mass outside the stored top-k is treated. ``bucket`` is exact and needs the
    #: teacher's full-vocabulary logsumexp; ``renormalize`` discards that mass, which is a
    #: different objective rather than a cheaper version of the same one.
    kd_tail: str = "bucket"
    #: Teacher truncation. ``None`` keeps the full distribution, which is exact but holds
    #: a (batch, positions, vocab) tensor.
    kd_top_k: int | None = 64
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
            if self.signal_source not in SIGNAL_SOURCES:
                problems.append(
                    f"unknown signal_source {self.signal_source!r}; known: "
                    f"{', '.join(sorted(SIGNAL_SOURCES))}"
                )
            elif SIGNAL_SOURCES[self.signal_source]:
                # Names the objective as well as the source: a reader seeing only
                # "dataset: ..." cannot tell which objective was refused, and the whole
                # point of the refusal is that the requested objective did not run.
                problems.append(
                    f"{self.type} with signal_source={self.signal_source!r}: "
                    f"{SIGNAL_SOURCES[self.signal_source]}"
                )
            if self.kd_tail not in ("bucket", "renormalize"):
                problems.append(
                    f"unknown kd_tail {self.kd_tail!r}; known: 'bucket', 'renormalize'"
                )
            if self.kd_top_k is not None and self.kd_top_k < 1:
                problems.append("kd_top_k must be at least 1, or None for the full distribution")
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
            "signal_source": self.signal_source,
            "kd_tail": self.kd_tail,
            "kd_top_k": self.kd_top_k,
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
    if config.type in (LOGIT_KD, MIXED_KD) and config.signal_source == DATASET:
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
    lines.append(f"\n  {'source':<12}{'status':<18}where the teacher distribution comes from")
    for source, blocked in SIGNAL_SOURCES.items():
        lines.append(f"  {source:<12}{(NOT_IMPLEMENTED if blocked else IMPLEMENTED):<18}"
                     f"{blocked or 'a resident teacher answers every batch'}")
    return "\n".join(lines)
