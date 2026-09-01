"""Context-length specialisation during distillation.

The question this component exists to answer is narrow enough to be falsifiable:

    **Does the distribution of sequence lengths seen during distillation change *where* the
    student's context-performance curve breaks, independently of the architecture's
    nominal context window?**

The nominal window is a configuration field — 262,144 — and it is free. It says nothing
about whether the model can use those positions. The measurable quantity is the
*context-performance curve*: accuracy on a held-out probe as a function of input length.
Its shape has a knee. This component is about moving that knee.

Why the hybrid layout makes the question interesting
----------------------------------------------------
The student is 36 DeltaNet layers and 12 full-attention layers. Those two mixers fail
differently as length grows:

* the 12 attention layers keep an exact record of every token, at a KV cost linear in
  length — they can look back arbitrarily far, and pay for it;
* the 36 DeltaNet layers keep a fixed-size recurrent state whose cost does not grow at all
  — they cannot look back arbitrarily far, and do not pay for it.

So a hybrid model's effective context is not one number: it is whatever the 12 attention
layers can still resolve, plus whatever the fixed DeltaNet state managed to carry forward.
The second term is learned, and it is learned from whatever lengths the training data
contained. That is the mechanism by which a *training-time* choice could move a
*deployment-time* capability, and it is the specific claim under test.

What would refute it
--------------------
If the curves produced by :data:`CURRICULA` are indistinguishable — if a model distilled
only on 4K sequences degrades at the same length as one distilled on a length-balanced
mixture — then context specialisation does nothing here and should be reported as
ineffective. :func:`compare_curves` computes the comparison; it is not written to favour
either outcome.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

#: The declared window. Every regime below is a subdivision of it.
MAX_CONTEXT = 262_144


# ---------------------------------------------------------------------------
# regimes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContextRegime:
    """A band of sequence lengths that fails for one reason rather than several.

    Bands are chosen so that a result inside one is attributable. "Long context is worse"
    is not a finding; "retrieval survives to 128K but multi-hop reasoning breaks at 32K"
    is, and it needs bands that separate the two.
    """

    name: str
    min_tokens: int
    max_tokens: int
    #: What changes about the *model's* situation in this band.
    mechanism: str
    #: A capability that can be probed at this length and not below it.
    probe: str

    def contains(self, n_tokens: int) -> bool:
        return self.min_tokens <= n_tokens < self.max_tokens

    @property
    def midpoint(self) -> int:
        return (self.min_tokens + self.max_tokens) // 2


CONTEXT_REGIMES: tuple[ContextRegime, ...] = (
    ContextRegime(
        "local", 0, 2_048,
        mechanism="Everything fits in the attention layers' immediate span and inside the "
                  "DeltaNet convolution's effective reach. Nothing is being compressed yet, "
                  "so this band measures raw capability, not context handling.",
        probe="instruction following, single-turn reasoning, short code completion",
    ),
    ContextRegime(
        "short", 2_048, 8_192,
        mechanism="The working range of ordinary chat and single-document tasks. The "
                  "DeltaNet state begins to be a summary rather than a record.",
        probe="multi-turn dialogue coherence, single-file code edits",
    ),
    ContextRegime(
        "medium", 8_192, 32_768,
        mechanism="KV cache becomes a material fraction of the 16 GB budget, and the "
                  "recurrent state must now discard. Which information it discards is "
                  "learned, and is the first place specialisation could matter.",
        probe="single-document QA, long-file comprehension, needle retrieval",
    ),
    ContextRegime(
        "long", 32_768, 131_072,
        mechanism="Beyond any plausible pretraining length for most of the corpus. The "
                  "12 attention layers still hold everything; the 36 DeltaNet layers are "
                  "operating far outside the regime they were fitted on.",
        probe="multi-document synthesis, repository-scale code reasoning, multi-hop retrieval",
    ),
    ContextRegime(
        "ultra", 131_072, MAX_CONTEXT + 1,
        mechanism="The declared maximum. RoPE is at the edge of its extrapolation and the "
                  "KV cache dominates the memory budget outright. Capability here has to be "
                  "demonstrated, never assumed from the config field.",
        probe="full-window retrieval, position-invariance of retrieval accuracy",
    ),
)

REGIME_NAMES = tuple(r.name for r in CONTEXT_REGIMES)


def regime_for(n_tokens: int) -> ContextRegime:
    for regime in CONTEXT_REGIMES:
        if regime.contains(n_tokens):
            return regime
    raise ValueError(f"{n_tokens} tokens is beyond the declared window of {MAX_CONTEXT}")


#: Evaluation lengths, one per octave. Powers of two keep the curve legible on a log axis
#: and make the KV-cache cost double from point to point, so capability and memory can be
#: read off the same x-axis.
CURVE_LENGTHS: tuple[int, ...] = (2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144)


# ---------------------------------------------------------------------------
# curricula
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CurriculumStage:
    """One phase of training at one sequence length."""

    sequence_length: int
    #: Share of total training steps. Stages must sum to 1.
    fraction: float
    rationale: str

    @property
    def regime(self) -> str:
        return regime_for(self.sequence_length).name


@dataclass(frozen=True)
class ContextCurriculum:
    """A named sequence-length policy — one arm of the context ablation.

    ``interleave`` decides whether the stages run in order (a curriculum in the usual sense)
    or are sampled throughout (a length-balanced mixture). They are different hypotheses:
    the first says the model needs to learn short before long, the second says it needs to
    see long often enough not to forget it.
    """

    name: str
    stages: tuple[CurriculumStage, ...]
    interleave: bool
    hypothesis: str
    arm: str = ""

    def __post_init__(self) -> None:
        total = sum(s.fraction for s in self.stages)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"{self.name}: stage fractions sum to {total}, not 1.0")
        if not self.stages:
            raise ValueError(f"{self.name}: a curriculum needs at least one stage")

    @property
    def max_length(self) -> int:
        return max(s.sequence_length for s in self.stages)

    @property
    def regimes_covered(self) -> list[str]:
        seen = {s.regime for s in self.stages}
        return [name for name in REGIME_NAMES if name in seen]

    def token_share(self) -> dict[int, float]:
        """Share of *tokens* — not steps — at each length.

        These differ, and the difference is the point: a stage that is 10% of steps at 128K
        is 10% of steps but a much larger share of tokens than a 10% stage at 4K. Reporting
        step fractions alone would overstate how little long-context data a schedule uses.
        """
        weighted = {s.sequence_length: s.fraction * s.sequence_length for s in self.stages}
        total = sum(weighted.values())
        return {length: value / total for length, value in sorted(weighted.items())}

    def schedule(self, total_steps: int) -> list[dict[str, Any]]:
        """Concrete per-step assignment of sequence lengths.

        Sequential mode gives contiguous step ranges; interleaved mode gives the same totals
        with the stages cycled, so both arms train on identical step counts per length and
        differ only in the order.
        """
        counts = [max(1, round(s.fraction * total_steps)) for s in self.stages]
        # Absorb rounding drift into the largest stage so the totals match exactly.
        drift = total_steps - sum(counts)
        counts[counts.index(max(counts))] += drift
        if not self.interleave:
            plan, start = [], 0
            for stage, n in zip(self.stages, counts, strict=True):
                plan.append({"start_step": start, "end_step": start + n,
                             "sequence_length": stage.sequence_length,
                             "regime": stage.regime, "steps": n})
                start += n
            return plan
        remaining = list(counts)
        plan, step = [], 0
        while step < total_steps:
            for i, stage in enumerate(self.stages):
                if not remaining[i]:
                    continue
                take = min(remaining[i], max(1, counts[i] // 10))
                plan.append({"start_step": step, "end_step": step + take,
                             "sequence_length": stage.sequence_length,
                             "regime": stage.regime, "steps": take})
                remaining[i] -= take
                step += take
        return plan

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arm": self.arm,
            "interleave": self.interleave,
            "hypothesis": self.hypothesis,
            "max_length": self.max_length,
            "regimes_covered": self.regimes_covered,
            "stages": [asdict(s) | {"regime": s.regime} for s in self.stages],
            "token_share": {str(k): v for k, v in self.token_share().items()},
        }


def _stage(length: int, fraction: float, why: str) -> CurriculumStage:
    return CurriculumStage(sequence_length=length, fraction=fraction, rationale=why)


#: The four arms of the context-specialisation ablation. B1 is the control; each of the
#: others changes exactly one thing relative to it, so a difference is attributable.
CURRICULA: dict[str, ContextCurriculum] = {
    "B0": ContextCurriculum(
        name="uniform_token_mixture",
        arm="B0",
        interleave=True,
        hypothesis="The neutral mixture: an equal share of *tokens* at every length, which "
                   "is not the same as an equal share of steps and is the reference point "
                   "the other arms are read against. Predicts a curve that degrades "
                   "gracefully rather than falling off a knee, because no length is rare.",
        stages=(
            # Step fractions are 1/length, normalised, so the *token* shares come out equal.
            # A step at 262,144 carries 64x the tokens of one at 4,096, so it gets 1/64 of
            # the steps: 64/85, 16/85, 4/85, 1/85. Splitting the steps evenly instead would
            # put 57% of the tokens at the longest length and would not be a uniform mixture
            # at all — which is the confusion this arm exists to avoid.
            _stage(4_096, 64 / 85, "equal token share, so 64x the steps of the longest stage"),
            _stage(16_384, 16 / 85, "equal token share"),
            _stage(65_536, 4 / 85, "equal token share"),
            _stage(262_144, 1 / 85, "equal token share"),
        ),
    ),
    "B1": ContextCurriculum(
        name="short_only",
        arm="B1",
        interleave=False,
        hypothesis="Control. Distilling only at 4K reproduces conventional practice. If the "
                   "other arms do not beat this curve beyond 8K, context specialisation is "
                   "not doing anything and should be reported as ineffective.",
        stages=(_stage(4_096, 1.0, "the conventional distillation length"),),
    ),
    "B2": ContextCurriculum(
        name="progressive_lengthening",
        arm="B2",
        interleave=False,
        hypothesis="Length is a curriculum: the recurrent state learns what to keep at short "
                   "lengths first, then learns to keep it for longer. Predicts the knee moves "
                   "right relative to B1 without hurting short-context accuracy.",
        stages=(
            _stage(4_096, 0.40, "establish the behaviour to be preserved"),
            _stage(16_384, 0.30, "first length at which the DeltaNet state must discard"),
            _stage(65_536, 0.20, "beyond any plausible pretraining length"),
            _stage(262_144, 0.10, "the declared window, trained on rather than assumed"),
        ),
    ),
    "B3": ContextCurriculum(
        name="length_balanced_mixture",
        arm="B3",
        interleave=True,
        hypothesis="Order does not matter, exposure does: the same length distribution as B2 "
                   "but sampled throughout. Predicts a similar curve to B2 with less "
                   "short-context regression, because short data is never stale. B2 vs B3 "
                   "isolates ordering from exposure — they share a token budget exactly.",
        stages=(
            _stage(4_096, 0.40, "same token budget as B2, interleaved"),
            _stage(16_384, 0.30, "same token budget as B2, interleaved"),
            _stage(65_536, 0.20, "same token budget as B2, interleaved"),
            _stage(262_144, 0.10, "same token budget as B2, interleaved"),
        ),
    ),
    "B4": ContextCurriculum(
        name="long_weighted",
        arm="B4",
        interleave=True,
        hypothesis="Long context is worth paying for in short-context quality. Deliberately "
                   "over-weights the long regimes. Predicts the best curve beyond 32K and a "
                   "measurable short-context regression against B1 — which is the trade this "
                   "arm exists to price, not a failure.",
        stages=(
            _stage(4_096, 0.15, "minimum needed to retain instruction behaviour"),
            _stage(16_384, 0.20, ""),
            _stage(65_536, 0.35, "the regime the release is being optimised for"),
            _stage(262_144, 0.30, "full-window capability treated as a first-class target"),
        ),
    ),
}


CURRICULA["B5"] = ContextCurriculum(
    name="medium_weighted",
    arm="B5",
    interleave=True,
    hypothesis="Over-weights 8K-32K, the band where the DeltaNet state first has to discard "
               "and where most document-scale work actually happens. Predicts the best "
               "medium-context curve and, unlike B4, little short-context regression — the "
               "cheap arm to run if the long-context arms prove expensive.",
    stages=(
        # Weighted so 16K carries the largest *token* share, which needs a much larger step
        # share than intuition suggests: a 262,144 stage at 5% of steps would already be a
        # quarter of the tokens.
        _stage(4_096, 0.40, "retain instruction behaviour"),
        _stage(16_384, 0.50, "the band this arm is about; the largest token share"),
        _stage(65_536, 0.08, ""),
        _stage(262_144, 0.02, "enough exposure not to lose the window entirely"),
    ),
)


def curriculum(arm: str) -> ContextCurriculum:
    if arm not in CURRICULA:
        raise ValueError(f"unknown context arm {arm!r}; have {sorted(CURRICULA)}")
    return CURRICULA[arm]


# ---------------------------------------------------------------------------
# the result schema
# ---------------------------------------------------------------------------
MetricDirection = Literal["higher_is_better", "lower_is_better"]


@dataclass
class ContextPoint:
    """One measurement: a metric at one length."""

    sequence_length: int
    value: float
    n_samples: int
    #: Standard error, so "the curve dropped" can be distinguished from noise.
    stderr: float | None = None

    @property
    def regime(self) -> str:
        return regime_for(self.sequence_length).name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"regime": self.regime}


@dataclass
class ContextCurve:
    """A context-performance curve, with the derived quantities the paper reports.

    The headline number is :meth:`effective_context`: the longest length at which the model
    still retains a stated fraction of its own short-context score. It is defined relative
    to the model itself, so it is not inflated by a model that is simply better everywhere,
    and it is *not* the config's ``max_position_embeddings``.
    """

    model: str
    metric: str
    direction: MetricDirection
    points: list[ContextPoint] = field(default_factory=list)
    context_arm: str = ""
    layer_arm: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = sorted(self.points, key=lambda p: p.sequence_length)

    # -- derived quantities --------------------------------------------
    @property
    def baseline(self) -> float:
        """The model's own score at the shortest measured length."""
        if not self.points:
            raise ValueError(f"{self.model}: curve has no points")
        return self.points[0].value

    def _retained(self, value: float) -> float:
        """Fraction of the baseline retained, oriented so 1.0 is always "as good"."""
        base = self.baseline
        if self.direction == "higher_is_better":
            return value / base if base else float("nan")
        return base / value if value else float("inf")

    def retention(self) -> dict[int, float]:
        return {p.sequence_length: self._retained(p.value) for p in self.points}

    def effective_context(self, threshold: float = 0.90) -> int:
        """Longest length whose score, and every score before it, stays within ``threshold``.

        The "and every score before it" clause matters: a curve that dips at 32K and
        recovers at 64K has not earned a 64K claim, and taking the maximum passing length
        would award it one.
        """
        best = 0
        for point in self.points:
            if self._retained(point.value) < threshold:
                break
            best = point.sequence_length
        return best

    def degradation_onset(self, threshold: float = 0.90) -> int | None:
        """First length that falls below the threshold, or ``None`` if the curve holds."""
        for point in self.points:
            if self._retained(point.value) < threshold:
                return point.sequence_length
        return None

    def by_regime(self) -> dict[str, float]:
        """Mean score per regime — the coarse summary, for tables that cannot fit a curve."""
        buckets: dict[str, list[float]] = {}
        for point in self.points:
            buckets.setdefault(point.regime, []).append(point.value)
        return {name: sum(v) / len(v) for name, v in buckets.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "metric": self.metric,
            "direction": self.direction,
            "context_arm": self.context_arm,
            "layer_arm": self.layer_arm,
            "points": [p.to_dict() for p in self.points],
            "retention": {str(k): v for k, v in self.retention().items()},
            "effective_context_at_90pct": self.effective_context(0.90),
            "effective_context_at_95pct": self.effective_context(0.95),
            "degradation_onset_at_90pct": self.degradation_onset(0.90),
            "by_regime": self.by_regime(),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextCurve:
        return cls(
            model=data["model"], metric=data["metric"], direction=data["direction"],
            context_arm=data.get("context_arm", ""), layer_arm=data.get("layer_arm", ""),
            provenance=data.get("provenance", {}),
            points=[ContextPoint(sequence_length=p["sequence_length"], value=p["value"],
                                 n_samples=p["n_samples"], stderr=p.get("stderr"))
                    for p in data["points"]],
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def compare_curves(baseline: ContextCurve, candidate: ContextCurve,
                   *, threshold: float = 0.90) -> dict[str, Any]:
    """Score one arm against another, including the outcome that refutes the hypothesis.

    ``verdict`` is deliberately blunt. ``no_measurable_effect`` is a real result and is
    reported as one; it is the outcome that says context specialisation did not work.
    """
    if baseline.metric != candidate.metric:
        raise ValueError(f"cannot compare {baseline.metric!r} against {candidate.metric!r}")
    shared = sorted({p.sequence_length for p in baseline.points}
                    & {p.sequence_length for p in candidate.points})
    if not shared:
        raise ValueError("the two curves share no measured lengths")

    b = {p.sequence_length: p.value for p in baseline.points}
    c = {p.sequence_length: p.value for p in candidate.points}
    sign = 1.0 if baseline.direction == "higher_is_better" else -1.0
    deltas = {length: sign * (c[length] - b[length]) for length in shared}

    base_eff = baseline.effective_context(threshold)
    cand_eff = candidate.effective_context(threshold)
    short = [d for length, d in deltas.items() if length <= 8_192]
    long_ = [d for length, d in deltas.items() if length >= 32_768]

    if cand_eff > base_eff:
        verdict = "extends_effective_context"
    elif cand_eff < base_eff:
        verdict = "shortens_effective_context"
    elif long_ and max(long_) > 0 and all(d >= 0 for d in long_):
        verdict = "improves_long_context_without_extending_it"
    else:
        verdict = "no_measurable_effect"

    return {
        "metric": baseline.metric,
        "baseline": baseline.model,
        "candidate": candidate.model,
        "baseline_arm": baseline.context_arm,
        "candidate_arm": candidate.context_arm,
        "threshold": threshold,
        "delta_by_length": {str(k): v for k, v in sorted(deltas.items())},
        "effective_context": {"baseline": base_eff, "candidate": cand_eff,
                              "ratio": cand_eff / base_eff if base_eff else None},
        "short_context_delta": sum(short) / len(short) if short else None,
        "long_context_delta": sum(long_) / len(long_) if long_ else None,
        "verdict": verdict,
        "note": "A positive long_context_delta with a negative short_context_delta is the "
                "trade B4 predicts, not a failure; report both.",
    }
