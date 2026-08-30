"""What did the extra parameters actually buy, and what did they cost to serve?

The project's decision after Level 3 is not "is the bigger model better" — it is whether
the improvement is worth the memory, on a card that has 13.56 GiB and a target that has
10.76. Those are different questions and only the second one decides an architecture.

So this module puts two measured quantities next to each other:

* a **capability metric** that was actually measured — validation bits-per-byte on a
  shared held-out corpus, or generation repetition, or anything else the evaluation
  produced. Never a synthesised score.
* the **deployment cost** of the architecture that produced it, from
  :mod:`qwen_distill.analysis.deployment`.

Three rules it will not break:

**No universal capability score.** There is no single number that says how good a
language model is, and inventing one would let a bad architecture win by construction.
Comparisons are per-metric, and a metric that was not measured stays absent.

**No comparison across different validation corpora.** Bits-per-byte is only meaningful
against the bytes it was measured on. Level 2 scored 1.270 on procedural text and Level
2R scored 1.797 on English; the difference is the corpora, not the models. A step whose
two runs disagree on the validation corpus is refused, not annotated.

**No verdict of "bigger is better".** The conclusion is drawn from the numbers, and
"2.5x the parameters for 1% of the loss" is a real answer that argues against scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Metrics where a LOWER value is an improvement, and how each is compared. Anything not
#: named here is reported without a direction rather than guessed at.
LOWER_IS_BETTER: dict[str, str] = {
    "validation_bits_per_byte": "held-out bits per byte — the primary capability metric",
    "validation_loss": "held-out cross-entropy in nats",
    "train_bits_per_byte": "training bits per byte; NOT a capability measure on its own",
    "mean_repeated_3gram": "share of 3-word windows that repeat an earlier one",
}

#: Improvement at or below this, relative to the baseline value, is not worth a
#: deployment cost — it is inside the range two seeds of the same architecture could
#: differ by. This project has never measured that range (no seed has been repeated), so
#: the threshold is a stated judgment and is reported as one.
MATERIAL_RELATIVE_IMPROVEMENT = 0.02


@dataclass
class MetricStep:
    """One metric, moving from a baseline architecture to a candidate."""

    metric: str
    baseline_value: float | None = None
    candidate_value: float | None = None
    comparable: bool = True
    incomparable_reason: str | None = None

    @property
    def measured(self) -> bool:
        return self.baseline_value is not None and self.candidate_value is not None

    @property
    def absolute_change(self) -> float | None:
        if not (self.measured and self.comparable):
            return None
        return round(self.candidate_value - self.baseline_value, 6)

    @property
    def relative_improvement(self) -> float | None:
        """Fraction of the baseline value recovered. Positive means better."""
        if not (self.measured and self.comparable) or not self.baseline_value:
            return None
        if self.metric not in LOWER_IS_BETTER:
            return None
        return round(
            (self.baseline_value - self.candidate_value) / abs(self.baseline_value), 6
        )

    @property
    def material(self) -> bool | None:
        relative = self.relative_improvement
        return None if relative is None else relative >= MATERIAL_RELATIVE_IMPROVEMENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": self.baseline_value,
            "candidate": self.candidate_value,
            "measured": self.measured,
            "comparable": self.comparable,
            "incomparable_reason": self.incomparable_reason,
            "absolute_change": self.absolute_change,
            "relative_improvement": self.relative_improvement,
            "material": self.material,
            "lower_is_better": self.metric in LOWER_IS_BETTER,
        }


@dataclass
class ScalingStep:
    """One rung to the next: what changed, what it bought, what it costs to serve."""

    baseline: str
    candidate: str
    baseline_parameters: int | None = None
    candidate_parameters: int | None = None
    baseline_inference_gib: float | None = None
    candidate_inference_gib: float | None = None
    baseline_corpus: str | None = None
    candidate_corpus: str | None = None
    metrics: list[MetricStep] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)

    @property
    def parameter_ratio(self) -> float | None:
        if not (self.baseline_parameters and self.candidate_parameters):
            return None
        return round(self.candidate_parameters / self.baseline_parameters, 4)

    @property
    def inference_memory_ratio(self) -> float | None:
        if not (self.baseline_inference_gib and self.candidate_inference_gib):
            return None
        return round(self.candidate_inference_gib / self.baseline_inference_gib, 4)

    @property
    def primary(self) -> MetricStep | None:
        """The metric a decision should rest on, when it was measured comparably."""
        return next(
            (m for m in self.metrics
             if m.metric == "validation_bits_per_byte" and m.measured and m.comparable),
            None,
        )

    def improvement_per_doubling(self) -> float | None:
        """Relative improvement per doubling of parameters.

        Normalises steps of different sizes so a 2.5x step and a 2x step can be set side
        by side. It is a *description of two points*, not a scaling law — two points
        cannot distinguish a power law from a straight line.
        """
        primary, ratio = self.primary, self.parameter_ratio
        if primary is None or not ratio or ratio <= 1:
            return None
        relative = primary.relative_improvement
        if relative is None:
            return None
        import math

        return round(relative / math.log2(ratio), 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "baseline_parameters": self.baseline_parameters,
            "candidate_parameters": self.candidate_parameters,
            "parameter_ratio": self.parameter_ratio,
            "baseline_inference_gib": self.baseline_inference_gib,
            "candidate_inference_gib": self.candidate_inference_gib,
            "inference_memory_ratio": self.inference_memory_ratio,
            "improvement_per_parameter_doubling": self.improvement_per_doubling(),
            "metrics": [m.to_dict() for m in self.metrics],
            "findings": self.findings,
            "material_threshold": MATERIAL_RELATIVE_IMPROVEMENT,
            "threshold_status": (
                "a stated judgment, not a measured noise floor — this project has never "
                "repeated a seed, so the run-to-run variance is unknown"
            ),
        }


def _corpus_identity(facts: Any) -> str | None:
    """What a run validated on, as an identity string. ``None`` when unrecorded."""
    corpus = getattr(facts, "corpus", None)
    if corpus is None:
        return None
    return getattr(corpus, "validation_sha256", "") or getattr(corpus, "name", "") or None


def build_step(
    baseline: Any,
    candidate: Any,
    *,
    baseline_inference_gib: float | None = None,
    candidate_inference_gib: float | None = None,
    metrics: tuple[str, ...] = tuple(LOWER_IS_BETTER),
) -> ScalingStep:
    """Compare two completed runs, refusing any comparison the data does not support.

    ``baseline`` and ``candidate`` are
    :class:`qwen_distill.analysis.compare.RunFacts`. Everything is taken from what those
    records actually contain: a metric absent from either side stays absent, and a metric
    measured on different validation corpora is marked incomparable rather than
    subtracted.
    """
    step = ScalingStep(
        baseline=baseline.name,
        candidate=candidate.name,
        baseline_parameters=baseline.get("parameters"),
        candidate_parameters=candidate.get("parameters"),
        baseline_inference_gib=baseline_inference_gib,
        candidate_inference_gib=candidate_inference_gib,
        baseline_corpus=_corpus_identity(baseline),
        candidate_corpus=_corpus_identity(candidate),
    )

    same_corpus = (
        step.baseline_corpus is not None
        and step.baseline_corpus == step.candidate_corpus
    )
    if not same_corpus:
        reason = (
            "the two runs did not record the same validation corpus identity, so a "
            "bits-per-byte difference between them is dominated by the data, not the "
            "architecture"
            if step.baseline_corpus != step.candidate_corpus
            else "neither run recorded which bytes it validated on"
        )
        step.findings.append(
            "validation metrics are NOT comparable: " + reason + ". Evaluate both "
            "checkpoints on one shared held-out corpus before drawing a conclusion."
        )

    for metric in metrics:
        entry = MetricStep(
            metric=metric,
            baseline_value=baseline.get(metric),
            candidate_value=candidate.get(metric),
        )
        # Corpus-dependent metrics are only comparable when the corpus matched.
        if metric.startswith(("validation_", "train_")) and not same_corpus:
            entry.comparable = False
            entry.incomparable_reason = (
                "measured against different (or unrecorded) validation bytes"
            )
        step.metrics.append(entry)

    primary = step.primary
    if primary is not None:
        relative, ratio = primary.relative_improvement, step.parameter_ratio
        if relative is not None and ratio:
            direction = "improved" if relative > 0 else "got worse"
            step.findings.append(
                f"validation bits/byte {direction} by {abs(relative):.2%} "
                f"({primary.baseline_value} -> {primary.candidate_value}) for "
                f"{ratio:.2f}x the parameters"
            )
            if relative <= 0:
                step.findings.append(
                    "the larger architecture is not better on the primary metric. That "
                    "is a real result and an argument against scaling further at this "
                    "token budget, not a reason to try a bigger one."
                )
            elif not primary.material:
                step.findings.append(
                    f"the improvement is below the {MATERIAL_RELATIVE_IMPROVEMENT:.0%} "
                    f"materiality threshold — diminishing returns. Another architectural "
                    f"strategy is likely to be worth more than more parameters."
                )
        memory_ratio = step.inference_memory_ratio
        if memory_ratio and relative is not None and relative > 0:
            step.findings.append(
                f"cost of that improvement: {memory_ratio:.2f}x inference memory "
                f"({step.baseline_inference_gib:.2f} -> {step.candidate_inference_gib:.2f} "
                f"GiB estimated)"
            )
    elif any(m.measured for m in step.metrics):
        step.findings.append(
            "the primary metric (validation bits/byte on a shared corpus) is not "
            "available, so no capability-per-memory conclusion can be drawn"
        )
    else:
        step.findings.append(
            "no metric was measured on both sides — nothing to compare yet"
        )
    return step


@dataclass
class ResearchSummary:
    """The architecture decision, laid out from what has actually been measured."""

    rungs: list[dict[str, Any]] = field(default_factory=list)
    steps: list[ScalingStep] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rungs": self.rungs,
            "steps": [s.to_dict() for s in self.steps],
            "open_questions": self.open_questions,
            "conclusion_policy": (
                "Conclusions are drawn from measured metrics and estimated deployment "
                "cost. 'Bigger is better' is never assumed; a small improvement for a "
                "large memory increase is reported as diminishing returns."
            ),
        }

    def render(self) -> str:
        rule = "=" * 78
        lines = [rule, "ARCHITECTURE RESEARCH SUMMARY", rule, ""]

        for rung in self.rungs:
            lines.append(f"  {rung['name']}")
            params = rung.get("parameters")
            lines.append(f"    parameters          : "
                         f"{format(params, ',') if params else 'UNKNOWN'}")
            for key, label in (
                ("validation_bits_per_byte", "validation BPB"),
                ("mean_repeated_3gram", "3-gram repetition"),
            ):
                value = rung.get(key)
                lines.append(f"    {label:<20}: "
                             f"{value if value is not None else 'UNKNOWN'}")
            inference = rung.get("inference_gib")
            lines.append(
                "    inference estimate  : "
                + (f"{inference:.2f} GiB" if inference is not None else "UNKNOWN")
            )
            for target in ("16 GB", "12 GB"):
                verdict = rung.get(f"status_{target.split()[0]}gb")
                context = rung.get(f"max_context_{target.split()[0]}gb")
                suffix = f" up to {context:,} ctx" if context else ""
                lines.append(f"    {target:<20}: {verdict or 'UNKNOWN'}{suffix}")
            lines.append(f"    status              : {rung.get('status', 'UNKNOWN')}")
            lines.append("")

        for step in self.steps:
            lines += ["-" * 78, f"STEP  {step.baseline}  ->  {step.candidate}", "-" * 78]
            ratio = step.parameter_ratio
            lines.append(f"  parameters      : {ratio:.2f}x" if ratio
                         else "  parameters      : UNKNOWN")
            primary = step.primary
            if primary is not None and primary.relative_improvement is not None:
                lines.append(f"  scaling gain    : "
                             f"{primary.relative_improvement:+.2%} on validation BPB")
                per_doubling = step.improvement_per_doubling()
                if per_doubling is not None:
                    lines.append(f"  per doubling    : {per_doubling:+.2%}")
            else:
                lines.append("  scaling gain    : NOT COMPARABLE or NOT MEASURED")
            memory = step.inference_memory_ratio
            lines.append(f"  memory increase : {memory - 1:+.1%}" if memory
                         else "  memory increase : UNKNOWN")
            lines.append("")
            lines.append("  conclusion:")
            for finding in step.findings:
                lines.append(f"    - {finding}")
            lines.append("")

        if self.open_questions:
            lines += ["-" * 78, "OPEN — not answerable from what has been measured",
                      "-" * 78]
            lines += [f"  ? {q}" for q in self.open_questions]
            lines.append("")

        lines += [
            "-" * 78,
            "  Capability figures are measured. Memory figures are ESTIMATED and are not",
            "  benchmarks. No universal capability score is computed, and no conclusion",
            "  assumes that a larger model is a better one.",
            rule,
        ]
        return "\n".join(lines)


def research_summary(
    rungs: list[dict[str, Any]], steps: list[ScalingStep],
    open_questions: list[str] | None = None,
) -> ResearchSummary:
    """Assemble the summary. Every value comes from a caller that measured it."""
    return ResearchSummary(
        rungs=rungs, steps=steps, open_questions=open_questions or []
    )
