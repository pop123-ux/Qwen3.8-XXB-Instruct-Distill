"""Comparing two experiments without pretending they measured the same thing.

Level 2 reached validation bits-per-byte **1.270**. Level 2R will reach some other
number. Putting them side by side and subtracting is the single most tempting error
available in this project, and it would be wrong, because the two numbers are not
measurements of the same quantity:

* Level 2's validation text was a held-out tail of **procedurally generated** bytes —
  words drawn independently from a fixed Zipfian distribution. Its conditional entropy is
  low *by construction*: once the frequency table is learned there is nothing left to
  predict, which is why 1.270 coexisted with ``"and and and"``.
* Level 2R's validation text is **eight whole public-domain books** the model never saw.
  Real English carries syntax, semantics and long-range dependency. Its conditional
  entropy is higher, and no amount of learning drives it to Level 2's floor.

A model scoring 1.270 on the first and 2.1 on the second has not got worse. The axes are
different. Reporting ``+0.83 BPB`` would be a fabricated regression.

So this module does not annotate an incomparable metric — it **refuses to compute the
delta at all**. :attr:`MetricComparison.delta` is ``None`` whenever the metrics are not
commensurable, and the reason travels with it. A number that should not exist is not
safer for being labelled.

What *is* comparable across a corpus change is everything that measures the training
process rather than the data: throughput, memory, parameter count, stability. Level 2R
holds architecture, sequence length, batch sizes, optimizer, precision and schedule at
their Level-2 values precisely so those comparisons remain valid. That is what makes it a
controlled experiment, and this module checks that claim rather than assuming it.

The one comparison that answers the actual research question — *did it learn English?* —
is not a BPB delta at all. It is generation, and it is qualitative. See
:mod:`qwen_distill.training.sanity`.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The three verdicts. ``COMPARABLE_IF`` is not a soft ``COMPARABLE``: it means the
#: comparison is invalid as things stand and names the missing precondition.
COMPARABLE = "COMPARABLE"
NOT_COMPARABLE = "NOT_COMPARABLE"
COMPARABLE_IF = "COMPARABLE_IF"

#: Metric scopes. The distinction is the whole argument of this module: a corpus change
#: invalidates ``data`` comparisons and leaves ``process`` comparisons intact.
PROCESS = "process"        # measures the implementation and the hardware
DATA = "data"              # measures the model against a particular corpus
CAPABILITY = "capability"  # measures what the model can do


@dataclass(frozen=True)
class MetricRule:
    """Whether one metric survives a change of corpus, and why."""

    key: str
    label: str
    scope: str
    across_corpus_change: str
    reason: str
    #: What would have to be true for this metric to become comparable. Required for
    #: anything not already ``COMPARABLE`` — "you can't compare these" without a remedy
    #: is an obstacle, not an analysis.
    remedy: str = ""
    lower_is_better: bool = True
    unit: str = ""

    def __post_init__(self) -> None:
        if self.across_corpus_change != COMPARABLE and not self.remedy:
            raise ValueError(f"{self.key}: a non-comparable metric must name its remedy")


#: What would make two byte-level BPB numbers commensurable. Stated as a procedure
#: because "evaluate on the same data" is easy to say and easy to get wrong: the held-out
#: text must be identical bytes, the sequence length and batching must match (the loss is
#: a mean over sequences, so the segmentation changes it), and the precision must match
#: (fp16 autocast and fp32 do not agree to three decimals).
SHARED_BENCHMARK_PROTOCOL: tuple[str, ...] = (
    "Pick one held-out corpus. It must be disjoint from BOTH runs' training data — "
    "Level 2's procedural corpus and Level 2R's Gutenberg training split.",
    "Fix its SHA-256 and record it with the result. Two evaluations on 'the same books' "
    "that differ by a header are two different benchmarks.",
    "Segment it identically for both models: same sequence length, same stride, same "
    "batch size. Byte-level loss is a mean over sequences, so segmentation moves it.",
    "Evaluate both checkpoints under the same precision. fp16 autocast and fp32 do not "
    "agree to three decimals, and the difference is comparable to the effect being "
    "measured.",
    "Report both numbers with the corpus digest attached, never as a bare delta.",
    "Note what the comparison still cannot settle: two models trained on different data, "
    "scored on a third corpus, differ by training data AND by whatever that corpus "
    "happens to favour.",
)


#: The metric registry. Every quantity these experiments produce, classified once, so a
#: comparison cannot silently include something that does not survive the change.
CROSS_CORPUS_RULES: tuple[MetricRule, ...] = (
    MetricRule(
        key="validation_bits_per_byte", label="validation bits/byte", scope=DATA,
        across_corpus_change=NOT_COMPARABLE, unit="bits/byte",
        reason=(
            "measured on different held-out text with different intrinsic entropy. "
            "Procedural Zipfian bytes have low conditional entropy by construction; real "
            "English does not. The difference between the two numbers is dominated by the "
            "corpora, not by the models."
        ),
        remedy="evaluate both checkpoints on one shared held-out corpus — see SHARED_BENCHMARK_PROTOCOL",
    ),
    MetricRule(
        key="validation_loss", label="validation loss (nats)", scope=DATA,
        across_corpus_change=NOT_COMPARABLE, unit="nats",
        reason="the same quantity as validation bits/byte, before the change of base",
        remedy="evaluate both checkpoints on one shared held-out corpus",
    ),
    MetricRule(
        key="train_bits_per_byte", label="final train bits/byte", scope=DATA,
        across_corpus_change=NOT_COMPARABLE, unit="bits/byte",
        reason="measured on different training text; additionally reflects how many times each corpus was consumed",
        remedy="not recoverable — training loss is on training data by definition; use a shared held-out corpus instead",
    ),
    MetricRule(
        key="run_wide_tokens_per_second", label="run-wide throughput", scope=PROCESS,
        across_corpus_change=COMPARABLE, unit="tok/s", lower_is_better=False,
        reason=(
            "the same architecture at the same sequence length and batch size on the same "
            "GPU class. Byte-level tokenisation means a token is a byte in both runs, so "
            "the unit is identical. This measures the implementation, not the data."
        ),
    ),
    MetricRule(
        key="peak_vram_gib", label="peak VRAM", scope=PROCESS,
        across_corpus_change=COMPARABLE, unit="GiB",
        reason="activation and state footprint depend on shape, not on content",
    ),
    MetricRule(
        key="parameters", label="parameter count", scope=PROCESS,
        across_corpus_change=COMPARABLE, unit="params",
        reason="identical by construction — Level 2R changes only the corpus",
    ),
    MetricRule(
        key="steps_to_plateau", label="steps to plateau", scope=DATA,
        across_corpus_change=COMPARABLE_IF, unit="steps",
        reason=(
            "the plateau step is defined relative to each run's OWN best value, so the "
            "two are computed against different scales. The shapes can be described side "
            "by side; the ratio of the numbers means nothing."
        ),
        remedy="compare as a description of curve shape, never as a numeric ratio",
    ),
    MetricRule(
        key="epochs_seen", label="epochs over the corpus", scope=DATA,
        across_corpus_change=COMPARABLE_IF, unit="epochs",
        reason=(
            "a fact about each run's data budget, not about the models. Level 2 consumed "
            "8 MB many times over; Level 2R sees a far larger corpus."
        ),
        remedy="read as a property of each experiment, not as a comparison between them",
    ),
    MetricRule(
        key="degenerate_generation", label="generation degenerate?", scope=CAPABILITY,
        across_corpus_change=COMPARABLE, unit="",
        reason=(
            "the question the experiment exists to answer. 'Does it emit one token "
            "forever?' is corpus-independent, and it is the comparison that matters."
        ),
    ),
    MetricRule(
        key="training_stable", label="training stable (no NaN/divergence)", scope=PROCESS,
        across_corpus_change=COMPARABLE, unit="", lower_is_better=False,
        reason="a property of the optimisation, comparable across any data change",
    ),
)

RULES_BY_KEY: dict[str, MetricRule] = {rule.key: rule for rule in CROSS_CORPUS_RULES}


@dataclass
class CorpusDescriptor:
    """What a run was trained and validated on. The reason two BPBs differ."""

    name: str
    kind: str = "unknown"            # procedural | natural_language | unknown
    total_bytes: int | None = None
    train_sha256: str = ""
    validation_sha256: str = ""
    validation_split_rule: str = ""
    entropy_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "total_bytes": self.total_bytes,
            "train_sha256": self.train_sha256[:16], "validation_sha256": self.validation_sha256[:16],
            "validation_split_rule": self.validation_split_rule, "entropy_note": self.entropy_note,
        }

    @property
    def identity(self) -> str:
        """What makes this corpus a distinct benchmark. Two runs sharing this string
        validated on the same bytes; anything else did not."""
        return self.validation_sha256 or f"{self.name}:{self.kind}"


@dataclass
class RunFacts:
    """One run's numbers, normalised so two runs can be lined up.

    Every field is optional. A run that is still training has no final validation BPB,
    and the comparison must render that as UNKNOWN rather than as zero or as absent.
    """

    name: str
    corpus: CorpusDescriptor
    metrics: dict[str, Any] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    status: str = "unknown"
    source: str = ""

    def get(self, key: str) -> Any:
        return self.metrics.get(key)

    @classmethod
    def from_result_json(cls, path: str | Path) -> RunFacts:
        """Read a published ``RESULT.json``.

        Reads the shape ``experiments/runs/*/RESULT.json`` actually has, and treats every
        field as absent-until-found: a partially populated record must produce UNKNOWNs,
        never zeros.
        """
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        measurements = payload.get("training_measurements", {}) or {}
        configuration = payload.get("configuration", {}) or {}
        quality = payload.get("model_quality", {}) or {}

        degenerate: bool | None = None
        generations = quality.get("greedy_generations")
        if isinstance(generations, dict) and generations:
            degenerate = any(
                _looks_degenerate(str(text)) for text in generations.values()
            )
        elif "establishes_general_language_capability" in quality:
            degenerate = None  # a capability claim is not a degeneracy measurement

        metrics = {
            "validation_bits_per_byte": measurements.get("final_validation_bpb"),
            "validation_loss": measurements.get("final_validation_loss"),
            "train_bits_per_byte": measurements.get("final_train_bpb"),
            "run_wide_tokens_per_second": measurements.get("run_wide_tokens_per_second"),
            "parameters": configuration.get("parameters"),
            "steps_completed": measurements.get("steps_completed"),
            "degenerate_generation": degenerate,
            "training_stable": not payload.get("hardware", {}).get("cuda_oom", False),
        }
        corpus_name = str(configuration.get("corpus", "unknown"))
        return cls(
            name=str(payload.get("experiment", path.parent.name)),
            corpus=CorpusDescriptor(
                name=corpus_name,
                kind="procedural" if "procedural" in corpus_name.lower() else "unknown",
            ),
            metrics={k: v for k, v in metrics.items() if v is not None},
            configuration=configuration,
            status=str(payload.get("outcome", "unknown")),
            source=str(path),
        )

    @classmethod
    def from_analysis(cls, analysis: Any, *, corpus: CorpusDescriptor | None = None) -> RunFacts:
        """Read a :class:`~qwen_distill.analysis.run_analysis.RunAnalysis`.

        Typed loosely on purpose: importing ``RunAnalysis`` here would make the two
        modules mutually dependent for no benefit, and the only thing needed is its
        published shape.
        """
        curves = analysis.curves
        metrics: dict[str, Any] = {}
        for key, curve_name in (
            ("validation_bits_per_byte", "validation_bpb"),
            ("validation_loss", "validation_loss"),
            ("train_bits_per_byte", "train_bpb"),
        ):
            curve = curves.get(curve_name)
            if curve is not None and len(curve):
                metrics[key] = curve.final[1]
        if analysis.throughput.run_wide_tokens_per_second is not None:
            metrics["run_wide_tokens_per_second"] = analysis.throughput.run_wide_tokens_per_second
        if analysis.plateau.plateau_step is not None:
            metrics["steps_to_plateau"] = analysis.plateau.plateau_step
        if analysis.epochs_seen is not None:
            metrics["epochs_seen"] = analysis.epochs_seen
        if analysis.steps_completed is not None:
            metrics["steps_completed"] = analysis.steps_completed
        return cls(
            name=analysis.name,
            corpus=corpus or CorpusDescriptor(name="unknown"),
            metrics=metrics,
            configuration=analysis.summary.get("configuration", {}) or {},
            status=analysis.loop_status,
            source=str(analysis.files.root),
        )


def _looks_degenerate(text: str) -> bool:
    """Cheap check for the ``"and and and"`` failure, for reading published records.

    Uses the same top-token-share threshold as the real detector, so a record and a live
    generation are judged by one rule. The distinct-word *ratio* the live detector also
    applies is deliberately not used here: published records are abbreviated
    (``"and and and and ..."``), and an ellipsis is enough to push a four-word repetition
    over any ratio threshold.

    The real detector is :func:`qwen_distill.training.sanity.check_generation`, which
    runs against a model. This only has to recognise degeneracy a human already wrote
    down.
    """
    from ..training.sanity import MAX_TOP_TOKEN_SHARE

    words = [w for w in text.split() if w not in {"...", "…"}]
    if len(words) < 3:
        return False
    counts = Counter(words)
    return counts.most_common(1)[0][1] / len(words) > MAX_TOP_TOKEN_SHARE


@dataclass
class MetricComparison:
    """Two values and, only when the metric permits it, the difference between them."""

    rule: MetricRule
    left: Any = None
    right: Any = None
    #: ``None`` whenever the metric is not comparable. Not hidden, not annotated —
    #: absent. A number that should not exist is not made safe by a footnote.
    delta: float | None = None
    ratio: float | None = None
    note: str = ""

    @property
    def verdict(self) -> str:
        if self.left is None or self.right is None:
            return "UNKNOWN"
        return self.rule.across_corpus_change

    @property
    def both_present(self) -> bool:
        return self.left is not None and self.right is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.rule.key,
            "label": self.rule.label,
            "scope": self.rule.scope,
            "unit": self.rule.unit,
            "left": self.left,
            "right": self.right,
            "verdict": self.verdict,
            "delta": self.delta,
            "ratio": self.ratio,
            "reason": self.rule.reason,
            "remedy": self.rule.remedy,
            "note": self.note,
        }


#: Configuration keys Level 2R holds at their Level-2 values. If any of these differs,
#: the corpus is not the only variable and even the process comparisons are confounded.
CONTROLLED_KEYS: tuple[str, ...] = (
    "parameters", "layers", "layout", "sequence_length", "micro_batch_size",
    "gradient_accumulation_steps", "effective_batch", "precision", "optimizer",
    "gradient_checkpointing",
)


@dataclass
class RunComparison:
    """Two runs, lined up, with each comparison kept at its own validity."""

    left: RunFacts
    right: RunFacts
    metrics: list[MetricComparison] = field(default_factory=list)
    controlled: list[str] = field(default_factory=list)
    changed: list[dict[str, Any]] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)

    @property
    def same_benchmark(self) -> str:
        """Whether the two runs validated on identical bytes: ``YES``/``NO``/``UNKNOWN``.

        Tri-state on purpose. A missing validation digest is not evidence that the two
        corpora differ — it is evidence that nobody recorded which bytes were scored, and
        that is a different problem with the same consequence: the delta stays withheld.

        ``NO`` for Level 2 vs Level 2R once both digests exist, which is the entire point.
        """
        left_id, right_id = self.left.corpus.validation_sha256, self.right.corpus.validation_sha256
        if not left_id or not right_id:
            return "UNKNOWN"
        return "YES" if left_id == right_id else "NO"

    @property
    def comparable_benchmark(self) -> bool:
        """Only an affirmative match licenses a validation-metric delta."""
        return self.same_benchmark == "YES"

    @property
    def controlled_experiment(self) -> bool:
        """One variable changed, everything else held. Checked, not assumed."""
        return not self.changed

    def by_scope(self, scope: str) -> list[MetricComparison]:
        return [m for m in self.metrics if m.rule.scope == scope]

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": {"name": self.left.name, "status": self.left.status,
                     "corpus": self.left.corpus.to_dict(), "source": self.left.source},
            "right": {"name": self.right.name, "status": self.right.status,
                      "corpus": self.right.corpus.to_dict(), "source": self.right.source},
            "same_benchmark": self.same_benchmark,
            "comparable_benchmark": self.comparable_benchmark,
            "controlled_experiment": self.controlled_experiment,
            "controlled_at_identical_values": self.controlled,
            "uncontrolled_differences": self.changed,
            "metrics": [m.to_dict() for m in self.metrics],
            "shared_benchmark_protocol": list(SHARED_BENCHMARK_PROTOCOL),
            "findings": self.findings,
            "headline_refusal": (
                "No overall winner is declared. The runs trained on different corpora, so "
                "their validation bits-per-byte are not measurements of the same quantity "
                "and no delta between them is reported."
            ),
        }

    def render(self) -> str:
        rule = "=" * 78
        lines = [
            rule,
            "CROSS-EXPERIMENT COMPARISON",
            f"  {self.left.name}   vs   {self.right.name}",
            rule, "",
            f"  left  : {self.left.status}",
            f"          corpus {self.left.corpus.name} ({self.left.corpus.kind})",
            f"  right : {self.right.status}",
            f"          corpus {self.right.corpus.name} ({self.right.corpus.kind})",
            "",
            f"  same validation benchmark : {self.same_benchmark}",
            f"  one variable changed      : {'YES' if self.controlled_experiment else 'NO'}",
        ]
        if self.controlled:
            lines.append(f"  held identical            : {', '.join(self.controlled)}")
        for difference in self.changed:
            lines.append(
                f"  ! also differs            : {difference['key']} "
                f"({difference['left']!r} vs {difference['right']!r})"
            )

        for scope, title, blurb in (
            (PROCESS, "COMPARABLE — measures the implementation, not the data",
             "Same architecture, same shapes, same hardware class. These deltas are real."),
            (CAPABILITY, "COMPARABLE — the question the experiment exists to answer",
             "Qualitative, and the only comparison that speaks to language capability."),
            (DATA, "NOT COMPARABLE — measured against different corpora",
             "Both values are shown. No delta is computed, because none would mean anything."),
        ):
            entries = [m for m in self.by_scope(scope) if m.both_present]
            if not entries:
                continue
            lines += ["", "-" * 78, title, "-" * 78, f"  {blurb}", ""]
            for comparison in entries:
                lines += _render_metric(comparison)

        unknown = [m for m in self.metrics if not m.both_present and (m.left or m.right)]
        if unknown:
            lines += ["", "-" * 78, "UNKNOWN — one side has not reported this yet", "-" * 78]
            for comparison in unknown:
                known = "left" if comparison.left is not None else "right"
                value = comparison.left if comparison.left is not None else comparison.right
                lines.append(f"  {comparison.rule.label:<32} {known}={value}   other side: UNKNOWN")

        lines += ["", "-" * 78, "WHAT WOULD MAKE THE BPB NUMBERS COMPARABLE", "-" * 78]
        for n, step in enumerate(SHARED_BENCHMARK_PROTOCOL, 1):
            lines.append(f"  {n}. {step}")

        if self.findings:
            lines += ["", "-" * 78, "FINDINGS", "-" * 78]
            lines += [f"  ! {finding}" for finding in self.findings]

        lines += [
            "", "-" * 78,
            "  No overall winner is declared. Two runs on different corpora do not have",
            "  a better and a worse; they have two results.",
            rule,
        ]
        return "\n".join(lines)


def _render_metric(comparison: MetricComparison) -> list[str]:
    rule = comparison.rule
    left, right = comparison.left, comparison.right

    def _format(value: Any) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:,.4f}".rstrip("0").rstrip(".")
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    lines = [
        f"  {rule.label:<32} {_format(left):>16}   {_format(right):>16}   {rule.unit}".rstrip()
    ]
    if comparison.delta is not None:
        if comparison.delta == 0:
            direction = "identical"
        else:
            direction = (
                "better on the right"
                if (comparison.delta < 0) == rule.lower_is_better
                else "worse on the right"
            )
        lines.append(
            f"  {'':<32} {'delta':>16} {comparison.delta:>+16,.4f}   ({direction})"
        )
    elif rule.across_corpus_change == NOT_COMPARABLE:
        lines.append(f"  {'':<32} {'delta':>16} {'REFUSED':>16}   {rule.reason}")
        lines.append(f"  {'':<32} {'remedy':>16}   {rule.remedy}")
    elif rule.across_corpus_change == COMPARABLE_IF:
        lines.append(f"  {'':<32} {'delta':>16} {'WITHHELD':>16}   {rule.reason}")
        lines.append(f"  {'':<32} {'read as':>16}   {rule.remedy}")
    if comparison.note:
        lines.append(f"  {'':<32} note: {comparison.note}")
    return lines


def compare_runs(left: RunFacts, right: RunFacts) -> RunComparison:
    """Line two runs up, computing a delta only where one is meaningful.

    The refusal is structural. ``delta`` is left at ``None`` for any metric the registry
    marks ``NOT_COMPARABLE`` or ``COMPARABLE_IF``; there is no flag to override it,
    because the case for overriding it has never once been correct in this project.
    """
    comparison = RunComparison(left=left, right=right)

    for rule in CROSS_CORPUS_RULES:
        left_value, right_value = left.get(rule.key), right.get(rule.key)
        if left_value is None and right_value is None:
            continue
        entry = MetricComparison(rule=rule, left=left_value, right=right_value)
        if (
            rule.across_corpus_change == COMPARABLE
            and isinstance(left_value, (int, float))
            and isinstance(right_value, (int, float))
            and not isinstance(left_value, bool)
            and not isinstance(right_value, bool)
        ):
            entry.delta = round(float(right_value) - float(left_value), 4)
            if left_value:
                entry.ratio = round(float(right_value) / float(left_value), 4)
        comparison.metrics.append(entry)

    for key in CONTROLLED_KEYS:
        left_value, right_value = left.configuration.get(key), right.configuration.get(key)
        if left_value is None or right_value is None:
            continue
        if left_value == right_value:
            comparison.controlled.append(key)
        else:
            comparison.changed.append({"key": key, "left": left_value, "right": right_value})

    if comparison.changed:
        comparison.findings.append(
            f"{len(comparison.changed)} configuration key(s) differ besides the corpus — "
            f"this is not a controlled comparison, and even the process metrics are "
            f"confounded: " + ", ".join(d["key"] for d in comparison.changed)
        )
    if comparison.same_benchmark == "NO":
        comparison.findings.append(
            "the two runs validated on different bytes, so no validation-metric delta is "
            "reported. Follow SHARED_BENCHMARK_PROTOCOL to obtain a comparable number."
        )
    elif comparison.same_benchmark == "UNKNOWN":
        comparison.findings.append(
            "at least one run did not record the SHA-256 of the bytes it validated on, so "
            "whether the two BPB numbers are the same benchmark cannot be established. No "
            "validation-metric delta is reported: an unrecorded digest is a reason to "
            "withhold the comparison, not to assume it holds."
        )

    degenerate = next((m for m in comparison.metrics if m.rule.key == "degenerate_generation"), None)
    if degenerate is not None and degenerate.both_present:
        if degenerate.left and not degenerate.right:
            comparison.findings.append(
                "the right-hand run's generations are not degenerate where the left-hand "
                "run's were — this is the comparison that speaks to language capability, "
                "and it is qualitative"
            )
        elif degenerate.left and degenerate.right:
            comparison.findings.append(
                "BOTH runs generate degenerate text. Whatever the loss curves say, "
                "neither model has demonstrated language capability."
            )
    else:
        comparison.findings.append(
            "generation degeneracy has not been measured on both sides. Until it has, "
            "nothing here speaks to whether either model learned language — run "
            "scripts/sanity_generate.py on a checkpoint from each."
        )
    return comparison


def corpus_from_manifest(path: str | Path) -> CorpusDescriptor:
    """Build a descriptor from a ``corpus_manifest.json`` written by
    :mod:`qwen_distill.training.corpus`.

    The validation SHA-256 is the field that matters: it is what decides whether two runs
    scored themselves on the same bytes, and therefore whether their BPBs are the same
    benchmark at all.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CorpusDescriptor(
        name=str(payload.get("name", Path(path).parent.name)),
        kind="natural_language",
        total_bytes=payload.get("total_bytes"),
        train_sha256=str(payload.get("train_sha256", "")),
        validation_sha256=str(payload.get("validation_sha256", "")),
        validation_split_rule=str(payload.get("split_rule", "")),
        entropy_note="real prose: syntax, semantics and long-range dependency",
    )


def load_run_facts(path: str | Path, *, name: str | None = None) -> RunFacts:
    """Load whatever kind of run record is at ``path``.

    Accepts a published ``RESULT.json``, a run directory containing one, or a live run
    directory with ``metrics.jsonl``. A run that is still training loads fine and reports
    UNKNOWN for everything it has not measured yet.
    """
    path = Path(path)
    if path.is_file():
        facts = RunFacts.from_result_json(path)
        directory = path.parent
    elif (path / "RESULT.json").is_file():
        facts = RunFacts.from_result_json(path / "RESULT.json")
        directory = path
    else:
        from .run_analysis import analyse_run

        facts = RunFacts.from_analysis(analyse_run(path))
        directory = path

    for candidate in (
        directory / "corpus_manifest.json",
        directory / "data" / "corpus_manifest.json",
    ):
        if candidate.is_file():
            facts.corpus = corpus_from_manifest(candidate)
            break

    if name:
        facts.name = name
    return facts
