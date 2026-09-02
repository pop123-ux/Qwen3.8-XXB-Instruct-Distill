"""The figure registry: what each figure asks, what backs it, and whether it is real yet.

This is the index a reader consults before trusting a figure. Every entry says which
scientific question the figure answers, which research question it serves, which
experiments and artifacts it reads, which metric fields it plots, which script draws it,
what it writes, and — the field that matters most — its **status**:

``real``
    Every series is backed by an artifact this repository produced.
``partial``
    Some declared series exist and some do not. The figure draws what exists and says
    which series are absent; it never fills a gap.
``schematic``
    Deliberately a conceptual diagram, stamped as one. Never a result.
``unavailable``
    The experiment has not happened. The builder exits 2 naming what would produce it.

The status here is a *claim*, and ``tests/test_figure_registry.py`` checks it against what
the builders actually do: a figure declared ``real`` must render, and one declared
``unavailable`` must refuse. That is what stops the registry drifting into advertising.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common import ANALYTICAL, AUDITED, DESIGN, MEASURED, MIXED, PROFILES

REAL = "real"
PARTIAL = "partial"
SCHEMATIC = "schematic"
UNAVAILABLE = "unavailable"
STATUSES = (REAL, PARTIAL, SCHEMATIC, UNAVAILABLE)


@dataclass(frozen=True)
class FigureSpec:
    """One figure: one scientific question, one primary dependent variable."""

    id: str
    slug: str
    title: str
    #: The question the figure answers. If it needs "and", it is probably two figures.
    question: str
    #: Which research question(s) from the skill this serves.
    research_questions: tuple[str, ...]
    #: Experiment ids read. Empty when the figure is computed at plot time.
    experiments: tuple[str, ...]
    #: Repository-relative artifact paths, or ``module:function`` for computed figures.
    sources: tuple[str, ...]
    #: The metric field names plotted, by their name in the artifact.
    metrics: tuple[str, ...]
    #: ``module:function`` under ``plots/``.
    builder: str
    status: str
    value_kind: str
    profiles: tuple[str, ...] = ("paper", "readme")
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"{self.id}: unknown status {self.status!r}")
        unknown = set(self.profiles) - set(PROFILES)
        if unknown:
            raise ValueError(f"{self.id}: unknown profile(s) {sorted(unknown)}")

    @property
    def stem(self) -> str:
        return f"{self.id}_{self.slug}"

    def outputs(self) -> tuple[str, ...]:
        """Every file this figure writes, relative to ``plots/outputs``."""
        names = []
        for profile_name in self.profiles:
            profile = PROFILES[profile_name]
            for fmt in (*profile.formats, "json"):
                names.append(f"{profile_name}/{self.stem}.{fmt}")
        return tuple(names)


# ---------------------------------------------------------------------------
# the register
# ---------------------------------------------------------------------------
RUN002 = "run002_logit_kd"
RUN002_METRICS = f"experiments/{RUN002}/metrics.jsonl"
RUN002_SUMMARY = f"experiments/{RUN002}/summary.json"

FIGURES: tuple[FigureSpec, ...] = (
    # --- architecture ----------------------------------------------------
    FigureSpec(
        id="F01", slug="model_compression",
        title="Architecture and model compression",
        question="How does the student's structure differ from the teacher's, property by "
                 "property?",
        research_questions=("RQ1",), experiments=(),
        sources=("qwen_distill.architecture.moe_student:audit",
                 "qwen_distill.architecture.presets:get_spec('teacher')",
                 "qwen_distill.architecture.params:count_parameters"),
        metrics=("total_parameters", "active_parameters_per_token", "num_hidden_layers",
                 "num_experts", "num_experts_per_tok"),
        builder="figures.architecture:model_compression",
        status=REAL, value_kind=AUDITED,
        notes="Four panels, one architectural property each: parameters, depth by block "
              "type, attention/KV geometry, MoE sparsity. Counted from the frozen spec at "
              "plot time, so the figure cannot drift from the architecture.",
    ),
    FigureSpec(
        id="F02", slug="parameter_counts",
        title="Parameter counts: teacher, student, student active per token",
        question="How much smaller is the student, stored and per token?",
        research_questions=("RQ1", "RQ4"), experiments=(),
        sources=("qwen_distill.architecture.moe_student:audit",
                 "qwen_distill.architecture.presets:get_spec('teacher')"),
        metrics=("total_parameters", "active_parameters_per_token"),
        builder="figures.architecture:parameter_counts",
        status=REAL, value_kind=AUDITED,
        notes="Three bars, deliberately. Stored parameters set VRAM; active parameters set "
              "per-token compute; conflating them is the usual MoE reporting error.",
    ),
    FigureSpec(
        id="F16", slug="layer_mapping",
        title="Teacher to student layer mapping",
        question="Which teacher layers does each student layer correspond to, and which "
                 "are dropped?",
        research_questions=("RQ1",), experiments=(),
        sources=("qwen_distill.architecture.moe_init:map_layers",
                 "qwen_distill.distillation.behavioral:layer_spans"),
        metrics=("mapping", "removed_teacher_layers", "spans"),
        builder="figures.architecture:layer_mapping",
        status=REAL, value_kind=DESIGN,
        notes="The mapping the implementation actually uses, not an illustration. Layer "
              "KD supervises the mapped pairs; the behavioural objective supervises the "
              "spans, which is the difference the paper is about.",
    ),

    # --- Run 002 trajectories --------------------------------------------
    FigureSpec(
        id="F03", slug="run002_training_loss",
        title="Run 002 training loss",
        question="How did the pure logit-KD training objective evolve over the token budget?",
        research_questions=("RQ1",), experiments=(RUN002,),
        sources=(RUN002_METRICS,), metrics=("loss", "tokens_seen", "step"),
        builder="figures.run002:training_loss",
        status=REAL, value_kind=MEASURED,
        notes="All 128 per-step records. The summary's first/final pair is not used.",
    ),
    FigureSpec(
        id="F04", slug="run002_validation_loss",
        title="Run 002 validation loss",
        question="Did held-out loss fall over Run 002's token budget?",
        research_questions=("RQ1",), experiments=(RUN002,),
        sources=(RUN002_METRICS,), metrics=("validation_loss", "step"),
        builder="figures.run002:validation_loss",
        status=REAL, value_kind=MEASURED,
        notes="Four observations, at steps 32/64/96/128 — the run's actual eval schedule. "
              "Nothing is interpolated between them.",
    ),
    FigureSpec(
        id="F05", slug="run002_kd_loss",
        title="Run 002 KD loss",
        question="How did the teacher-imitation term itself evolve?",
        research_questions=("RQ1",), experiments=(RUN002,),
        sources=(RUN002_METRICS,), metrics=("kd_loss", "tokens_seen"),
        builder="figures.run002:kd_loss",
        status=REAL, value_kind=MEASURED,
        notes="Run 002 is pure logit KD (alpha=1.0), so kd_loss equals the training loss "
              "by construction; the figure says so rather than presenting two results.",
    ),
    FigureSpec(
        id="F06", slug="run002_top1_agreement",
        title="Run 002 teacher/student top-1 agreement",
        question="How often did the student's argmax match the teacher's during training?",
        research_questions=("RQ1",), experiments=(RUN002,),
        sources=(RUN002_METRICS,), metrics=("top1_agreement", "tokens_seen"),
        builder="figures.run002:top1_agreement",
        status=REAL, value_kind=MEASURED,
        notes="An imitation diagnostic on the training batches, not a capability measure. "
              "The axis label and the figure note both say so.",
    ),
    FigureSpec(
        id="F07", slug="run002_teacher_diagnostics",
        title="Run 002 teacher signal diagnostics",
        question="What did the teacher's distribution look like while it supervised Run 002?",
        research_questions=("RQ1",), experiments=(RUN002,),
        sources=(RUN002_METRICS,),
        metrics=("teacher_entropy", "teacher_tail_mass", "tokens_seen"),
        builder="figures.run002:teacher_diagnostics",
        status=REAL, value_kind=MEASURED,
        notes="Two panels: entropy and top-64 tail mass. Both describe the teacher, not "
              "the student, which is why they are kept off the loss figures. Measured at "
              "the KD temperature T=2.0, which flattens the distribution and inflates "
              "tail mass — stated on the figure.",
    ),
    FigureSpec(
        id="F08", slug="run002_training_memory",
        title="Run 002 training memory profile",
        question="How much A40 memory did the 1536-token QLoRA training step actually use?",
        research_questions=("RQ4",), experiments=(RUN002, "run002_calibration_1536",
                                                  "run002_calibration"),
        sources=(RUN002_SUMMARY, "experiments/run002_calibration_1536/summary.json",
                 "experiments/run002_calibration/summary.json"),
        metrics=("memory.peak_allocated_gib", "memory.peak_reserved_gib",
                 "memory.total_vram_gib", "memory.snapshots"),
        builder="figures.run002:training_memory",
        status=REAL, value_kind=MEASURED,
        notes="TRAINING memory on a 48 GB A40. It is not a deployment result and carries "
              "no bearing on the 16 GB target; the figure states this and the 42.0 GiB "
              "safety gate that stopped the 2048-token configuration.",
    ),

    # --- memory and context ----------------------------------------------
    FigureSpec(
        id="F09", slug="context_memory_accounting",
        title="Context-length memory accounting",
        question="As deployment context grows, where does the memory go?",
        research_questions=("RQ2", "RQ4"), experiments=(),
        sources=("qwen_distill.research.memory:account",
                 "qwen_distill.research.memory:build_table"),
        metrics=("weights", "quantisation_overhead", "kv_cache", "recurrent_state",
                 "conv_state", "activations", "runtime_overhead"),
        builder="figures.memory:context_memory_accounting",
        status=REAL, value_kind=ANALYTICAL,
        notes="Analytical, from the project's own memory model. No inference memory has "
              "been measured on hardware; the figure is labelled an estimate throughout.",
    ),
    FigureSpec(
        id="F22", slug="context_vs_memory",
        title="Context length against peak VRAM",
        question="Which deployment context lengths fit inside 16 GB, and at which "
                 "quantisation?",
        research_questions=("RQ2", "RQ4"),
        # The builder discovers every run carrying a memory profile; this list is what it
        # finds today and the sidecar carries the live set. A new arm appears on the
        # figure the moment its record lands, without an edit here.
        experiments=(RUN002, "run002_calibration_1536", "run002_calibration",
                     "run003_calibration_1536", "kd_run_001"),
        sources=("qwen_distill.research.memory:build_table",
                 "experiments/*/summary.json", RUN002_SUMMARY),
        metrics=("total_gib", "memory.peak_allocated_gib"),
        builder="figures.memory:context_vs_memory",
        status=REAL, value_kind=MIXED,
        notes="Two panels that are deliberately not one. Left: analytical inference VRAM "
              "against context, per quantisation, with the 16 GB boundary. Right: every "
              "measured A40 training peak, ordered by sequence length and labelled by "
              "objective — three runs share 1536 tokens, and the layer-KD calibration "
              "costs 2.13 GiB more than the logit-KD one at the same length. The two "
              "panels are different quantities on different hardware and are never drawn "
              "on one axis.",
    ),
    FigureSpec(
        id="F20", slug="training_context_distribution",
        title="Training context distribution by curriculum",
        question="What distribution of training sequence lengths does each curriculum arm "
                 "declare?",
        research_questions=("RQ2",), experiments=(),
        sources=("qwen_distill.research.context:CURRICULA",),
        metrics=("stage.sequence_length", "stage.fraction"),
        builder="figures.context:training_context_distribution",
        status=REAL, value_kind=DESIGN,
        notes="The declared experimental design, not an observation: these are the "
              "curricula the context study would run. No context arm has been trained.",
    ),

    # --- the controlled comparison, once the arms exist -------------------
    FigureSpec(
        id="F10", slug="matched_distillation_recovery",
        title="Matched distillation recovery",
        question="At a matched token budget, which distillation objective recovers the "
                 "most held-out performance?",
        research_questions=("RQ1",), experiments=(RUN002,),
        sources=(RUN002_METRICS,), metrics=("validation_loss", "top1_agreement", "tokens_seen"),
        builder="figures.comparison:matched_distillation_recovery",
        status=UNAVAILABLE, value_kind=MEASURED,
        notes="Needs at least two arms at the reference protocol. Only logit KD "
              "(Run 002) exists; Run 001 is excluded automatically because its protocol "
              "differs. Populates itself when Run 003 lands.",
    ),
    FigureSpec(
        id="F11", slug="training_loss_by_objective",
        title="Training loss by objective",
        question="Do the objectives differ in how their training loss falls?",
        research_questions=("RQ1",), experiments=(RUN002,),
        sources=(RUN002_METRICS,), metrics=("loss", "tokens_seen"),
        builder="figures.comparison:training_loss_by_objective",
        status=UNAVAILABLE, value_kind=MEASURED,
        notes="Training losses of different objectives are not on a common scale; the "
              "figure says so and exists to show shape, not to rank the arms.",
    ),
    FigureSpec(
        id="F12", slug="validation_loss_by_objective",
        title="Validation loss by objective",
        question="Which objective reaches the lower held-out loss at a matched budget?",
        research_questions=("RQ1",), experiments=(RUN002,),
        sources=(RUN002_METRICS,), metrics=("validation_loss", "tokens_seen"),
        builder="figures.comparison:validation_loss_by_objective",
        status=UNAVAILABLE, value_kind=MEASURED,
        notes="The one comparison that is on a common scale across arms: the same "
              "held-out cross-entropy, the same validation split.",
    ),
    FigureSpec(
        id="F13", slug="teacher_imitation_by_objective",
        title="Teacher imitation by objective",
        question="Which objective produces a student that imitates the teacher more closely?",
        research_questions=("RQ1",), experiments=(RUN002,),
        sources=(RUN002_METRICS,), metrics=("kd_loss", "top1_agreement", "tokens_seen"),
        builder="figures.comparison:teacher_imitation_by_objective",
        status=UNAVAILABLE, value_kind=MEASURED,
        notes="Two panels: KD loss and top-1 agreement. Split rather than twin-axed, "
              "because a twin axis invites reading a crossing that means nothing.",
    ),
    FigureSpec(
        id="F14", slug="convergence_efficiency",
        title="Convergence efficiency",
        question="How many training tokens does each objective need to reach a threshold "
                 "declared in advance?",
        research_questions=("RQ1",), experiments=(RUN002,),
        sources=(RUN002_METRICS,), metrics=("validation_loss", "top1_agreement", "tokens_seen"),
        builder="figures.comparison:convergence_efficiency",
        status=UNAVAILABLE, value_kind=MEASURED,
        notes="Thresholds are declared in figures/comparison.py as module constants and "
              "are fixed before any arm is read, so the figure cannot be tuned to its "
              "own result.",
    ),
    FigureSpec(
        id="F15", slug="behavior_state_alignment",
        title="Teacher/student behavioural alignment",
        question="Which of the teacher's internal signals does the student reproduce?",
        research_questions=("RQ1",), experiments=(),
        sources=("experiments/alignment.json",),
        metrics=("hidden_state_similarity", "deltanet_state_similarity",
                 "attention_similarity", "moe_reconstruction_error", "logit_similarity"),
        builder="figures.behavior:behavior_state_alignment",
        status=UNAVAILABLE, value_kind=MEASURED,
        notes="No alignment probe has been run against the real teacher. DeltaNet state "
              "similarity is additionally blocked at the method level — the shapes differ "
              "and the projection would be an untested modelling choice (see "
              "distillation/behavioral.py, DELTANET_STATE).",
    ),

    # --- MoE --------------------------------------------------------------
    FigureSpec(
        id="F17", slug="moe_expert_utilisation",
        title="MoE expert utilisation",
        question="Do the eight routed experts receive comparable shares of the tokens?",
        research_questions=("RQ1",), experiments=(),
        sources=("experiments/*/metrics.jsonl:expert_token_counts",),
        metrics=("expert_token_counts", "load_imbalance"),
        builder="figures.moe:expert_utilisation",
        status=UNAVAILABLE, value_kind=MEASURED,
        notes="No run logs routing statistics yet. The trainer would have to pass "
              "output_router_logits=True and record per-expert token counts per step.",
    ),
    FigureSpec(
        id="F18", slug="moe_routing_entropy",
        title="MoE routing entropy over training",
        question="Does the router's decision distribution sharpen or flatten as training "
                 "proceeds?",
        research_questions=("RQ1",), experiments=(),
        sources=("experiments/*/metrics.jsonl:routing_entropy",),
        metrics=("routing_entropy", "tokens_seen"),
        builder="figures.moe:routing_entropy",
        status=UNAVAILABLE, value_kind=MEASURED,
        notes="Separate from utilisation on purpose: entropy is a property of the routing "
              "distribution, utilisation is a property of the realised assignment.",
    ),
    FigureSpec(
        id="F19", slug="moe_dead_experts",
        title="Dead and near-dead experts over training",
        question="Does any expert stop receiving tokens as training proceeds?",
        research_questions=("RQ1",), experiments=(),
        sources=("experiments/*/metrics.jsonl:dead_experts",),
        metrics=("dead_experts", "near_dead_experts", "tokens_seen"),
        builder="figures.moe:dead_experts",
        status=UNAVAILABLE, value_kind=MEASURED,
        notes="Needs the same per-step routing record as F17 and F18.",
    ),

    # --- context specialisation -------------------------------------------
    FigureSpec(
        id="F21", slug="capability_vs_context",
        title="Evaluation capability against context length",
        question="Does the training-length mixture move the capability curve across "
                 "evaluation context lengths?",
        research_questions=("RQ2",), experiments=(),
        sources=("experiments/context_curves/*.json",),
        metrics=("sequence_length", "value"),
        builder="figures.context:capability_vs_context",
        status=UNAVAILABLE, value_kind=MEASURED,
        notes="One curve per curriculum. No context arm has been trained or evaluated.",
    ),
    FigureSpec(
        id="F23", slug="context_efficiency",
        title="Context efficiency",
        question="What quality does each context length buy for its memory and compute?",
        research_questions=("RQ2", "RQ4"), experiments=(),
        sources=("experiments/context_curves/*.json",
                 "qwen_distill.research.memory:build_table"),
        metrics=("value", "total_gib", "tokens_per_second"),
        builder="figures.context:context_efficiency",
        status=UNAVAILABLE, value_kind=MIXED,
        notes="Needs the F21 quality measurements before the memory axis means anything.",
    ),

    # --- deployment / systems ---------------------------------------------
    FigureSpec(
        id="F24", slug="deployment_frontier_16gb",
        title="16 GB deployment frontier",
        question="What quality is reachable inside the 16 GB budget, and by which "
                 "configuration?",
        research_questions=("RQ4",), experiments=(),
        sources=("experiments/deployment/*.json",
                 "qwen_distill.research.memory:build_table"),
        metrics=("peak_vram_gib", "quality"),
        builder="figures.systems:deployment_frontier_16gb",
        status=UNAVAILABLE, value_kind=MIXED,
        notes="The memory axis is available analytically; the quality axis has no "
              "measurement at all. An axes-only frame would be a frontier figure with no "
              "frontier, so this refuses instead.",
    ),
    FigureSpec(
        id="F25", slug="throughput_vs_context",
        title="Inference throughput against context length",
        question="How does generation throughput change as deployment context grows?",
        research_questions=("RQ4",), experiments=(),
        sources=("experiments/deployment/*.json",),
        metrics=("context_length", "tokens_per_second"),
        builder="figures.systems:throughput_vs_context",
        status=UNAVAILABLE, value_kind=MEASURED,
        notes="Training throughput exists for three sequence lengths but is a different "
              "quantity — teacher forward, QLoRA backward and optimizer step included — "
              "and substituting it here would be exactly the silent-metric-swap this "
              "system forbids.",
    ),
    FigureSpec(
        id="F26", slug="latency_vs_context",
        title="Inference latency against context length",
        question="How does time-to-first-token and per-token latency change with context?",
        research_questions=("RQ4",), experiments=(),
        sources=("experiments/deployment/*.json",),
        metrics=("context_length", "latency_ms"),
        builder="figures.systems:latency_vs_context",
        status=UNAVAILABLE, value_kind=MEASURED,
        notes="No inference benchmark has been run.",
    ),
    FigureSpec(
        id="F27", slug="quality_memory_throughput",
        title="Quality, memory and throughput trade-off",
        question="Which configurations are on the quality/memory/throughput Pareto front?",
        research_questions=("RQ4",), experiments=(),
        sources=("experiments/deployment/*.json",),
        metrics=("quality", "peak_vram_gib", "tokens_per_second"),
        builder="figures.systems:quality_memory_throughput",
        status=UNAVAILABLE, value_kind=MEASURED,
        notes="Drawn as a two-dimensional scatter with throughput encoded in marker size — "
              "never as a 3D surface — and only once the three measurements exist.",
    ),
)

BY_ID: dict[str, FigureSpec] = {f.id: f for f in FIGURES}


def get(figure_id: str) -> FigureSpec:
    try:
        return BY_ID[figure_id.upper()]
    except KeyError:
        raise KeyError(
            f"unknown figure {figure_id!r}; known: {', '.join(sorted(BY_ID))}"
        ) from None


def with_status(*statuses: str) -> tuple[FigureSpec, ...]:
    return tuple(f for f in FIGURES if f.status in statuses)


def check_integrity() -> list[str]:
    """Structural problems with the register itself. Empty means it is coherent."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_slugs: set[str] = set()
    for spec in FIGURES:
        if spec.id in seen_ids:
            problems.append(f"duplicate figure id {spec.id}")
        seen_ids.add(spec.id)
        if spec.slug in seen_slugs:
            problems.append(f"duplicate slug {spec.slug!r} ({spec.id})")
        seen_slugs.add(spec.slug)
        if ":" not in spec.builder:
            problems.append(f"{spec.id}: builder {spec.builder!r} is not module:function")
        for required, name in ((spec.question, "question"), (spec.title, "title"),
                               (spec.notes, "notes")):
            if not required.strip():
                problems.append(f"{spec.id}: empty {name}")
        if not spec.sources:
            problems.append(f"{spec.id}: no sources declared")
        if not spec.metrics:
            problems.append(f"{spec.id}: no source metric fields declared")
        if not spec.research_questions:
            problems.append(f"{spec.id}: no research-question linkage")
        if spec.value_kind not in (MEASURED, ANALYTICAL, AUDITED, DESIGN, MIXED):
            problems.append(f"{spec.id}: unknown value kind {spec.value_kind!r}")
    return problems


# ---------------------------------------------------------------------------
# documentation
# ---------------------------------------------------------------------------
_STATUS_MARK = {REAL: "**real**", PARTIAL: "partial", SCHEMATIC: "schematic",
                UNAVAILABLE: "unavailable"}


def render_markdown() -> str:
    """The registry as ``plots/REGISTRY.md``. Generated, never edited by hand."""
    counts = {s: len(with_status(s)) for s in STATUSES}
    lines = [
        "# Figure registry",
        "",
        "Generated by `python plots/make_figures.py --write-registry`. Do not edit by hand —",
        "`plots/registry.py` is the source, and `tests/test_figure_registry.py` checks that",
        "every status claimed here is what the builder actually does.",
        "",
        "| status | meaning | count |",
        "| --- | --- | ---: |",
        f"| **real** | every series is backed by an artifact this repository produced "
        f"| {counts[REAL]} |",
        f"| partial | some declared series exist, some do not; gaps are stated, never filled "
        f"| {counts[PARTIAL]} |",
        f"| schematic | a conceptual diagram, stamped as one; never a result "
        f"| {counts[SCHEMATIC]} |",
        f"| unavailable | the experiment has not happened; the builder exits 2 "
        f"| {counts[UNAVAILABLE]} |",
        "",
        "## Index",
        "",
        "| id | figure | status | values | RQ | experiments |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for spec in sorted(FIGURES, key=lambda f: f.id):
        experiments = ", ".join(f"`{e}`" for e in spec.experiments) or "—"
        lines.append(
            f"| {spec.id} | [{spec.title}](#{spec.id.lower()}-{spec.slug.replace('_', '-')}) "
            f"| {_STATUS_MARK[spec.status]} | {spec.value_kind} "
            f"| {', '.join(spec.research_questions)} | {experiments} |"
        )
    lines += ["", "## Figures", ""]
    for spec in sorted(FIGURES, key=lambda f: f.id):
        module, function = spec.builder.split(":")
        lines += [
            f"### {spec.id} — {spec.title}",
            "",
            f"**Question.** {spec.question}",
            "",
            f"- **status**: {_STATUS_MARK[spec.status]}",
            f"- **values**: {spec.value_kind}",
            f"- **research questions**: {', '.join(spec.research_questions)}",
            f"- **experiments**: {', '.join(f'`{e}`' for e in spec.experiments) or '—'}",
            "- **sources**: " + ", ".join(f"`{s}`" for s in spec.sources),
            "- **source metric fields**: " + ", ".join(f"`{m}`" for m in spec.metrics),
            f"- **script**: `plots/{module.replace('.', '/')}.py` → `{function}()`",
            "- **outputs**: " + ", ".join(f"`plots/outputs/{o}`" for o in spec.outputs()),
            "",
            spec.notes,
            "",
        ]
    # No trailing hard-break spaces and no blank line at EOF: `git diff --check` is part of
    # the pre-commit protocol, and a generated file that trips it every time trains people
    # to ignore it.
    return "\n".join(line.rstrip() for line in lines).rstrip("\n") + "\n"
