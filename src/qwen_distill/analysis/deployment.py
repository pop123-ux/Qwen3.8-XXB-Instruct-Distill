"""Can this architecture actually be served on the card the project targets?

The destination is not a training run. It is a Qwen3.8-27B alternative that a person can
run locally on **16 GB of VRAM**, with a credible path to **12 GB**. Every architecture
decision has to be answerable against those two numbers before it costs GPU hours, and
the answer has to distinguish

    "it fits on 16 GB"

from

    "it fits on 16 GB at 4K context and not at 32K"

because for a long-context hybrid those are different models in practice.

**No new memory model is defined here.** Weights, KV cache, DeltaNet recurrent state,
conv state, activations and runtime overhead all come from
:mod:`qwen_distill.architecture.memory` via :func:`qwen_distill.diagnostics.fit.analyse_inference_fit`
— the same estimator the rest of the project uses. This module adds the two things the
research loop needs and that estimator does not have: **named hardware targets** that
cannot be silently swapped for a datacenter GPU, and a **per-context verdict** for each.

Everything here is an **estimate**. An analytical breakdown is not a benchmark, and the
verdicts say ``estimated`` in their own field names for that reason. Confirm a candidate
with ``scripts/benchmark_memory.py`` on the real card before committing hours to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..architecture.memory import GIB, QUANT_BYTES_PER_PARAM
from ..architecture.params import count_parameters
from ..architecture.presets import architecture_fields
from ..architecture.spec import HybridArchSpec
from ..diagnostics.fit import analyse_inference_fit

FIT = "FIT"
BORDERLINE = "BORDERLINE"
DOES_NOT_FIT = "DOES NOT FIT"


@dataclass(frozen=True)
class DeploymentTarget:
    """A card the project actually intends to run on.

    ``usable_gib`` is what a model may occupy, not what the box advertises. A "16 GB" T4
    reports 14.56 GiB — measured, Level 2 — and a desktop card is usually also driving a
    display. Planning against the marketing number is how a configuration that
    "obviously fits" OOMs.
    """

    name: str
    nominal_gb: int
    total_gib: float
    reserved_gib: float
    priority: str
    note: str

    @property
    def usable_gib(self) -> float:
        return round(self.total_gib - self.reserved_gib, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "nominal_gb": self.nominal_gb,
            "total_gib": self.total_gib, "reserved_gib": self.reserved_gib,
            "usable_gib": self.usable_gib, "priority": self.priority, "note": self.note,
        }


#: The two targets, and only these two by default. The project exists to reach consumer
#: hardware; a sweep that quietly reports "fits on an A100" would be answering a question
#: nobody asked. Larger cards can still be passed explicitly to any function here.
PRIMARY_TARGET = DeploymentTarget(
    name="16 GB", nominal_gb=16, total_gib=14.56, reserved_gib=1.0, priority="primary",
    note="Tesla T4 / RTX 4060 Ti 16GB. 14.56 GiB measured on the Level-2 T4.",
)
SECONDARY_TARGET = DeploymentTarget(
    name="12 GB", nominal_gb=12, total_gib=11.76, reserved_gib=1.0, priority="secondary",
    note="RTX 3060 12GB / 2060 12GB. Vendor spec converted to GiB; not measured here.",
)
TARGETS: tuple[DeploymentTarget, ...] = (PRIMARY_TARGET, SECONDARY_TARGET)

#: Context ladder, 4K to 256K. Long context is part of the project's goal, so the
#: cheapest way to be wrong about a candidate is to check one context and assume the
#: rest. The teacher declares 262,144.
CONTEXT_LADDER: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536, 131072, 262144)

#: The three deployment precisions the mandate asks about, named as an operator would.
#: The values behind them live in ``QUANT_BYTES_PER_PARAM``; nothing is redefined here.
PRECISIONS: dict[str, str] = {
    "fp16": "fp16",
    "int8": "int8",
    "4-bit": "q4_k_m",
}

#: Below this much headroom a fit is real but not safe: one background process, a
#: display, or allocator fragmentation takes it away.
BORDERLINE_MARGIN_GIB = 1.5

#: Width of the name column in the rendered sweep. Longer names are elided there and
#: kept in full in the JSON.
_NAME_WIDTH = 22


def _verdict(total_gib: float, usable_gib: float) -> str:
    if total_gib > usable_gib:
        return DOES_NOT_FIT
    return BORDERLINE if usable_gib - total_gib < BORDERLINE_MARGIN_GIB else FIT


@dataclass
class ContextFit:
    """One (target, precision, context) cell: what it costs and whether it fits."""

    context_length: int
    total_gib: float
    weights_gib: float
    kv_cache_gib: float
    state_gib: float
    activations_gib: float
    overhead_gib: float
    headroom_gib: float
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_length": self.context_length,
            "estimated_total_gib": round(self.total_gib, 3),
            "estimated_weights_gib": round(self.weights_gib, 3),
            "estimated_kv_cache_gib": round(self.kv_cache_gib, 3),
            "estimated_state_gib": round(self.state_gib, 3),
            "estimated_activations_gib": round(self.activations_gib, 3),
            "estimated_overhead_gib": round(self.overhead_gib, 3),
            "estimated_headroom_gib": round(self.headroom_gib, 3),
            "verdict": self.verdict,
        }


@dataclass
class TargetFeasibility:
    """One architecture at one precision against one card, across the context ladder."""

    target: DeploymentTarget
    precision: str
    quantization: str
    contexts: list[ContextFit] = field(default_factory=list)

    @property
    def max_fitting_context(self) -> int | None:
        """Longest context that is not ``DOES NOT FIT``.

        The single most useful number here. "Fits on 16 GB" without it is a claim that
        can be true at 4K and false at everything a long-context model is for.
        """
        usable = [c.context_length for c in self.contexts if c.verdict != DOES_NOT_FIT]
        return max(usable) if usable else None

    @property
    def max_comfortable_context(self) -> int | None:
        """Longest context that is ``FIT`` outright, with real headroom."""
        comfortable = [c.context_length for c in self.contexts if c.verdict == FIT]
        return max(comfortable) if comfortable else None

    @property
    def verdict(self) -> str:
        """The target's headline verdict, taken at the *shortest* context.

        If a model cannot be served at 4K it cannot be served at all; if it can, the
        interesting question becomes how far up the ladder it goes, which is what
        ``max_fitting_context`` answers.
        """
        return self.contexts[0].verdict if self.contexts else DOES_NOT_FIT

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "precision": self.precision,
            "quantization": self.quantization,
            "verdict_at_shortest_context": self.verdict,
            "max_fitting_context": self.max_fitting_context,
            "max_comfortable_context": self.max_comfortable_context,
            "contexts": [c.to_dict() for c in self.contexts],
        }


@dataclass
class DeploymentAssessment:
    """One architecture against every target and precision. All figures estimated."""

    name: str
    parameters: int
    non_embedding_parameters: int
    embedding_parameters: int
    architecture: dict[str, Any] = field(default_factory=dict)
    feasibility: list[TargetFeasibility] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def for_target(self, target_name: str, precision: str) -> TargetFeasibility | None:
        return next(
            (f for f in self.feasibility
             if f.target.name == target_name and f.precision == precision),
            None,
        )

    def best_precision_for(self, target_name: str) -> TargetFeasibility | None:
        """The highest-fidelity precision that fits this card at any context.

        Ordered by the precision list, which runs highest to lowest fidelity, so the
        first that fits is the best available rather than merely the smallest.
        """
        for precision in PRECISIONS:
            entry = self.for_target(target_name, precision)
            if entry and entry.max_fitting_context is not None:
                return entry
        return None

    def summary_row(self) -> dict[str, Any]:
        """One flat row per architecture, for a sweep table."""
        row: dict[str, Any] = {
            "name": self.name,
            "parameters": self.parameters,
            "non_embedding_parameters": self.non_embedding_parameters,
            "hidden_size": self.architecture.get("hidden_size"),
            "num_layers": self.architecture.get("num_layers"),
        }
        for target in (PRIMARY_TARGET, SECONDARY_TARGET):
            best = self.best_precision_for(target.name)
            key = f"{target.nominal_gb}gb"
            row[f"{key}_status"] = best.verdict if best else DOES_NOT_FIT
            row[f"{key}_precision"] = best.precision if best else None
            row[f"{key}_max_context"] = best.max_fitting_context if best else None
        return row

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "non_embedding_parameters": self.non_embedding_parameters,
            "embedding_parameters": self.embedding_parameters,
            "architecture": self.architecture,
            "feasibility": [f.to_dict() for f in self.feasibility],
            "notes": self.notes,
            "estimate_disclaimer": (
                "Every figure here is analytical. It is not a benchmark. Confirm a "
                "candidate with scripts/benchmark_memory.py on the real card before "
                "committing GPU hours."
            ),
        }


def assess(
    spec: HybridArchSpec,
    *,
    name: str | None = None,
    targets: tuple[DeploymentTarget, ...] = TARGETS,
    precisions: dict[str, str] | None = None,
    contexts: tuple[int, ...] = CONTEXT_LADDER,
) -> DeploymentAssessment:
    """Estimate one architecture against every target, precision and context.

    Delegates every byte to ``analyse_inference_fit``; this function decides *what to
    ask*, not what the answer is.
    """
    precisions = precisions or PRECISIONS
    params = count_parameters(spec)
    assessment = DeploymentAssessment(
        name=name or spec.name,
        parameters=params.total,
        non_embedding_parameters=params.total - params.embedding,
        embedding_parameters=params.embedding,
        architecture=architecture_fields(spec),
    )

    for target in targets:
        for precision, quantization in precisions.items():
            entry = TargetFeasibility(
                target=target, precision=precision, quantization=quantization
            )
            for context in contexts:
                if context > spec.max_position_embeddings:
                    # Beyond what the architecture declares it supports. Reporting a
                    # memory figure for it would describe a model that cannot run there.
                    continue
                fit = analyse_inference_fit(
                    spec, target.usable_gib,
                    quantization=quantization, context_length=context,
                    # The embedding is byte-level here (256 entries) so its precision is
                    # immaterial; left explicit rather than defaulted so a large-vocab
                    # candidate is not silently costed at the wrong precision.
                    embedding_quant=quantization,
                )
                entry.contexts.append(ContextFit(
                    context_length=context,
                    total_gib=fit.total_gib,
                    weights_gib=fit.weights_gib,
                    kv_cache_gib=fit.kv_cache_gib,
                    state_gib=fit.state_gib,
                    activations_gib=fit.activations_gib,
                    overhead_gib=fit.overhead_gib,
                    headroom_gib=target.usable_gib - fit.total_gib,
                    verdict=_verdict(fit.total_gib, target.usable_gib),
                ))
            assessment.feasibility.append(entry)

    declared = spec.max_position_embeddings
    if declared < max(contexts):
        assessment.notes.append(
            f"the architecture declares max_position_embeddings={declared:,}, so contexts "
            f"above that were not estimated — they would describe a model that cannot run "
            f"there without a rope/positional change"
        )
    weights_only = params.total * QUANT_BYTES_PER_PARAM["q4_k_m"] / GIB
    if weights_only > SECONDARY_TARGET.usable_gib:
        assessment.notes.append(
            f"4-bit weights alone are {weights_only:.2f} GiB, already over the "
            f"{SECONDARY_TARGET.name} budget before any cache or activation"
        )
    return assessment


@dataclass
class Sweep:
    """Several architectures assessed together, for choosing between them."""

    assessments: list[DeploymentAssessment] = field(default_factory=list)
    training_fits: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)

    def rows(self) -> list[dict[str, Any]]:
        return [a.summary_row() for a in self.assessments]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows(),
            "assessments": [a.to_dict() for a in self.assessments],
            "training_fits": self.training_fits,
            "findings": self.findings,
            "targets": [t.to_dict() for t in TARGETS],
            "estimate_disclaimer": (
                "Analytical estimates, not benchmarks. Nothing here was trained or run."
            ),
        }

    def render(self) -> str:
        rule = "=" * 100
        lines = [
            rule, "ARCHITECTURE SWEEP — inference feasibility on the deployment targets", rule, "",
            f"  primary   {PRIMARY_TARGET.name}: {PRIMARY_TARGET.usable_gib} GiB usable "
            f"({PRIMARY_TARGET.note})",
            f"  secondary {SECONDARY_TARGET.name}: {SECONDARY_TARGET.usable_gib} GiB usable "
            f"({SECONDARY_TARGET.note})",
            "",
            f"  {'architecture':<22}{'params':>15}{'train':>8}"
            f"{'16 GB':>14}{'prec':>7}{'max ctx':>10}"
            f"{'12 GB':>14}{'prec':>7}{'max ctx':>10}",
            "  " + "-" * 98,
        ]
        for assessment in self.assessments:
            row = assessment.summary_row()
            train_gib = self.training_fits.get(assessment.name, {}).get("total_gib")
            train = f"{train_gib:.1f}G" if train_gib else "-"
            # A derived name carries its whole diff ("level3+hidden_size=1280,..."),
            # which is what makes it useful and also far too wide for a column. The
            # table elides it; `to_dict()` keeps every name in full.
            label = row["name"]
            if len(label) > _NAME_WIDTH:
                label = label[: _NAME_WIDTH - 1] + "~"
            cells = [f"  {label:<22}", f"{row['parameters']:>15,}", f"{train:>8}"]
            for key in ("16gb", "12gb"):
                context = row[f"{key}_max_context"]
                cells += [
                    f"{row[f'{key}_status']:>14}",
                    f"{str(row[f'{key}_precision'] or '-'):>7}",
                    f"{(format(context, ',') if context else '-'):>10}",
                ]
            lines.append("".join(cells))
        elided = [a.name for a in self.assessments if len(a.name) > _NAME_WIDTH]
        if elided:
            lines += ["", "  names elided in the table above (full names in --json):"]
            lines += [f"    {name}" for name in elided]
        lines += [
            "",
            "  'train' is estimated TRAINING memory at the Level-2R recipe (seq 1024,",
            "  batch 4, gradient checkpointing on, fp16 AdamW) — a different question",
            "  from the inference columns, and usually the larger one at these sizes.",
            "  'prec' is the highest-fidelity precision that fits; 'max ctx' is the",
            "  longest context that does, capped by max_position_embeddings.",
        ]
        if self.findings:
            lines += ["", "-" * 100, "FINDINGS", "-" * 100]
            lines += [f"  ! {f}" for f in self.findings]
        lines += [
            "", "-" * 100,
            "  Analytical estimates, not benchmarks. Nothing here was trained or run.",
            "  Confirm a candidate with scripts/benchmark_memory.py on the real card.",
            rule,
        ]
        return "\n".join(lines)


def sweep(
    specs: dict[str, HybridArchSpec],
    *,
    targets: tuple[DeploymentTarget, ...] = TARGETS,
    contexts: tuple[int, ...] = CONTEXT_LADDER,
    include_training: bool = True,
) -> Sweep:
    """Assess several candidate architectures without training any of them.

    The point of the sweep is to make an architecture cheap to reject. A candidate that
    cannot be served on the primary target is not worth a training run, and finding that
    out costs milliseconds here against hours on a GPU.

    Training memory is included because the two constraints bind differently: at these
    sizes a model that serves comfortably in 1.5 GiB may still need 6 GiB to train, and
    a sweep that reported only inference would hide the binding one.
    """
    result = Sweep()
    for name, spec in specs.items():
        result.assessments.append(
            assess(spec, name=name, targets=targets, contexts=contexts)
        )
        if include_training:
            from ..diagnostics.fit import estimate_training_memory

            fit = estimate_training_memory(
                spec, PRIMARY_TARGET.usable_gib, strategy="full", optimizer="adamw",
                sequence_length=1024, batch_size=4, gradient_checkpointing=True,
                precision="fp16",
            )
            result.training_fits[name] = {
                "total_gib": round(fit.total_gib, 2),
                "verdict": fit.verdict,
                "headroom_gib": round(fit.headroom_gib, 2),
                "recipe": "seq 1024, batch 4, gradient checkpointing on, fp16 AdamW",
            }

    infeasible = [
        a.name for a in result.assessments
        if a.best_precision_for(PRIMARY_TARGET.name) is None
    ]
    if infeasible:
        result.findings.append(
            f"{len(infeasible)} candidate(s) cannot be served on the primary "
            f"{PRIMARY_TARGET.name} target at any precision or context: "
            + ", ".join(infeasible)
        )
    untrainable = [
        name for name, fit in result.training_fits.items()
        if fit["verdict"] == "NOT FEASIBLE"
    ]
    if untrainable:
        result.findings.append(
            f"{len(untrainable)} candidate(s) cannot be TRAINED on the primary target at "
            f"the Level-2R recipe, whatever their inference cost: " + ", ".join(untrainable)
        )
    capped = [
        a.name for a in result.assessments
        if a.architecture.get("context_length", 0) < max(contexts)
    ]
    if capped:
        result.findings.append(
            f"{len(capped)} candidate(s) declare a context shorter than the ladder's "
            f"{max(contexts):,}, so their long-context feasibility is UNKNOWN rather than "
            f"good: " + ", ".join(capped)
        )
    return result
