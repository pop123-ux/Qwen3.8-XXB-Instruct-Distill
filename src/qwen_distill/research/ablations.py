"""The ablation matrix: what each arm claims, and what result would refute it.

Two families, each a controlled comparison rather than a sweep.

**A — layer matching.** A 2x2 factorial on two independent switches: whether conventional
*pointwise* hidden-state matching is on, and whether *behavioural* delta matching is on.
Factorial rather than a ladder, because a ladder cannot separate "delta helps" from "more
supervision helps"; A2 and A3 use the same amount of supervision and differ only in kind.

    ================  ==================  ==================
    arm               pointwise matching  behavioural delta
    ================  ==================  ==================
    A1 (control)      on                  off
    A2                off                 off
    A3                off                 on
    A4                on                  on
    ================  ==================  ==================

Attention matching is a real loss term and is deliberately *not* a third factor here: adding
it would need eight cells to stay factorial, and four arms that each change one thing is
worth more than eight that are under-powered. It is available for a follow-up once the A
matrix has an answer.

**B — context specialisation.** Defined in :mod:`qwen_distill.research.context`; the four
curricula share a token budget where it matters, so ordering can be separated from exposure.

Every arm carries a ``prediction`` and a ``falsified_if``. The second field is the one that
makes this a research plan instead of a marketing plan: it is written before any run, and
it names the observation that would make the arm's claim false. An arm whose ``falsified_if``
cannot be checked with the metrics the project actually collects is not a real hypothesis and
should not be here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..distillation.behavioral import (
    CE,
    HIDDEN_DELTA,
    HIDDEN_POINTWISE,
    LOGIT_KD,
    ROUTER_BALANCE,
    CompositeLossConfig,
)
from .context import CURRICULA, ContextCurriculum


@dataclass(frozen=True)
class Arm:
    """One cell of the matrix."""

    arm: str
    family: str
    name: str
    question: str
    prediction: str
    falsified_if: str
    #: Loss weights for family A; empty for family B, which varies data rather than loss.
    loss_weights: dict[str, float] = field(default_factory=dict)
    #: Context curriculum for family B; family A arms all use the shared default.
    context_arm: str = ""
    is_control: bool = False
    #: Only A4 deliberately runs both hidden-matching terms; see CompositeLossConfig.
    combined_hidden: bool = False

    def loss_config(self, **overrides: Any) -> CompositeLossConfig:
        weights = dict(self.loss_weights or DEFAULT_LOSS_WEIGHTS)
        overrides.setdefault("allow_combined_hidden", self.combined_hidden)
        config = CompositeLossConfig(weights=weights, **overrides)
        config.validate()
        return config

    def curriculum(self) -> ContextCurriculum:
        return CURRICULA[self.context_arm or DEFAULT_CONTEXT_ARM]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm, "family": self.family, "name": self.name,
            "is_control": self.is_control, "question": self.question,
            "prediction": self.prediction, "falsified_if": self.falsified_if,
            "loss_weights": dict(sorted((self.loss_weights or DEFAULT_LOSS_WEIGHTS).items())),
            "context_arm": self.context_arm or DEFAULT_CONTEXT_ARM,
            "combined_hidden": self.combined_hidden,
        }


#: Terms every A arm keeps fixed, so the only difference between arms is the matching switch.
#: The router balance term is not optional: without it the load-balancing loss the
#: architecture ships with is off, and every arm would be measuring expert collapse instead
#: of what it meant to measure.
BASE_WEIGHTS: dict[str, float] = {CE: 0.1, LOGIT_KD: 1.0, ROUTER_BALANCE: 1.0}
DEFAULT_LOSS_WEIGHTS: dict[str, float] = BASE_WEIGHTS | {HIDDEN_DELTA: 0.5}
DEFAULT_CONTEXT_ARM = "B2"


ARMS: dict[str, Arm] = {
    "A1": Arm(
        arm="A1", family="layer_matching", name="pointwise_layer_matching", is_control=True,
        question="Does conventional hidden-state layer matching help when 16 of 64 layers "
                 "are removed?",
        prediction="Beats logits-only (A2) on short context, because it supplies dense "
                   "per-layer supervision that logits alone do not.",
        falsified_if="A1 does not beat A2 on any benchmark, which would mean pointwise "
                     "matching contributes nothing over the logit signal at this "
                     "compression ratio.",
        loss_weights=BASE_WEIGHTS | {HIDDEN_POINTWISE: 0.5},
    ),
    "A2": Arm(
        arm="A2", family="layer_matching", name="logits_only",
        question="How much of the result is attributable to any intermediate supervision at "
                 "all?",
        prediction="Weakest of the four, and the honest floor every other arm must clear.",
        falsified_if="A2 matches or beats A3 and A4, which would mean intermediate "
                     "supervision is unnecessary and the paper's premise is wrong. This "
                     "outcome is reportable and would be reported.",
        loss_weights=dict(BASE_WEIGHTS),
    ),
    "A3": Arm(
        arm="A3", family="layer_matching", name="behavioural_delta",
        question="Is matching a layer's residual *contribution* better than matching its "
                 "position, at equal supervision cost?",
        prediction="Beats A1 by more at deeper layers than at shallow ones, because the "
                   "removed layers' work is attributed to a student layer instead of being "
                   "dropped. This is the paper's central claim.",
        falsified_if="A3 does not beat A1 on the aggregate benchmark score, or beats it only "
                     "within the seed-to-seed variance measured across the repeated control "
                     "runs. Either outcome refutes the central claim and must be stated as "
                     "such rather than reframed.",
        loss_weights=BASE_WEIGHTS | {HIDDEN_DELTA: 0.5},
    ),
    "A4": Arm(
        arm="A4", family="layer_matching", name="pointwise_plus_delta",
        question="Do positional and behavioural supervision compose, or does one subsume the "
                 "other? This is the interaction cell of the factorial: without it, A1 and A3 "
                 "can be ranked but their relationship cannot be characterised.",
        prediction="Best of the four, but by less than A3's margin over A1 — if delta "
                   "supervision already captures what pointwise supervision provides, the "
                   "gains should overlap rather than add.",
        falsified_if="A4 is worse than A3, which would mean the two objectives conflict and "
                     "that pointwise matching is actively harmful once behaviour is matched. "
                     "A4 beating A3 by A3's full margin over A1 would instead mean the two "
                     "are independent and the 'beyond layer matching' framing is too strong.",
        loss_weights=BASE_WEIGHTS | {HIDDEN_POINTWISE: 0.5, HIDDEN_DELTA: 0.5},
        combined_hidden=True,
    ),
    "B1": Arm(
        arm="B1", family="context_specialisation", name="short_only", is_control=True,
        question="What does the context curve look like with conventional 4K distillation?",
        prediction="Degrades sharply past 8K; effective context well below the declared "
                   "262K window.",
        falsified_if="B1's curve holds to 128K, which would mean the architecture's context "
                     "capability is inherited from the teacher and distillation length is "
                     "irrelevant — making the whole B family unnecessary.",
        context_arm="B1",
    ),
    "B2": Arm(
        arm="B2", family="context_specialisation", name="progressive_lengthening",
        question="Does training length in increasing order move the degradation knee?",
        prediction="Effective context at 90% retention extends past B1's by at least one "
                   "octave.",
        falsified_if="B2's effective context equals B1's, or its short-context score drops "
                     "more than the long-context score gains.",
        context_arm="B2",
    ),
    "B3": Arm(
        arm="B3", family="context_specialisation", name="length_balanced_mixture",
        question="Is it the ordering that matters, or only the exposure? B3 has B2's exact "
                 "token budget per length, interleaved instead of staged.",
        prediction="Matches B2 on long context with less short-context regression, because "
                   "short data is never stale.",
        falsified_if="B3 and B2 differ by more than the measured seed variance at every "
                     "length, which would mean ordering matters independently of exposure and "
                     "the mixture framing is wrong.",
        context_arm="B3",
    ),
    "B4": Arm(
        arm="B4", family="context_specialisation", name="long_weighted",
        question="What is the exchange rate between long-context capability and short-context "
                 "quality?",
        prediction="Best long-context curve and a measurable short-context regression. The "
                   "point is to price the trade, not to win the arm.",
        falsified_if="B4 shows no short-context regression, which would mean long-context "
                     "training is free and the other arms are under-training long context.",
        context_arm="B4",
    ),
}

FAMILIES = ("layer_matching", "context_specialisation")


def arms(family: str | None = None) -> list[Arm]:
    if family is None:
        return list(ARMS.values())
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; have {list(FAMILIES)}")
    return [a for a in ARMS.values() if a.family == family]


def control(family: str) -> Arm:
    for arm in arms(family):
        if arm.is_control:
            return arm
    raise ValueError(f"{family} has no control arm")


def matrix() -> dict[str, Any]:
    """The whole plan as one serialisable object, for the ledger and the paper's appendix."""
    return {
        "families": {
            family: {
                "control": control(family).arm,
                "arms": [a.to_dict() for a in arms(family)],
            }
            for family in FAMILIES
        },
        "design_note": (
            "Family A is a 2x2 factorial over {pointwise matching on/off} x {behavioural "
            "delta on/off}: A1 pointwise-only, A2 neither, A3 delta-only, A4 both. A2 and A3 "
            "carry the same number of loss terms, so a difference between them is attributable "
            "to the kind of supervision rather than its quantity. Family B varies data, not "
            "loss; B2 and B3 share a token budget exactly, isolating ordering from exposure."
        ),
        "comparisons": [
            {"name": "does intermediate supervision help at all", "baseline": "A2",
             "candidate": "A1"},
            {"name": "is behaviour better than position", "baseline": "A1", "candidate": "A3"},
            {"name": "do they compose", "baseline": "A3", "candidate": "A4"},
            {"name": "does context specialisation move the knee", "baseline": "B1",
             "candidate": "B2"},
            {"name": "ordering or exposure", "baseline": "B2", "candidate": "B3"},
            {"name": "the long-context exchange rate", "baseline": "B1", "candidate": "B4"},
        ],
        "confounds_controlled": [
            "identical initialisation for every arm within a family (same seed, same "
            "transfer plan), so a difference is caused by the arm and not by the draw",
            "identical token budget across A arms; identical per-length token budget between "
            "B2 and B3",
            "the router balance term is on in every arm, so no arm is silently measuring "
            "expert collapse",
        ],
        "not_controlled": [
            "seed-to-seed variance is unmeasured until the control arm is run more than once; "
            "until then no margin should be called significant",
        ],
    }


def save_matrix(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matrix(), indent=2) + "\n", encoding="utf-8")
    return path
