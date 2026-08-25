"""Teacher and student on exactly the same prompts, compared per example.

Aggregate averages hide the thing this project is trying to measure. "Student matches
teacher accuracy at 40% of the tokens" is a claim about a distribution, and a mean can
be produced by a model that is right on easy items and catastrophically verbose on hard
ones. So every pairing is stored per example, and the aggregates are computed from those
records rather than replacing them.

Two rules the metrics follow:

**Report raw measurements before ratios.** Accuracy and median reasoning tokens for each
model, side by side, is a statement anyone can check. A single "efficiency score" is
not, and is easy to make flattering.

**Every derived number carries its formula.** :data:`METRIC_DEFINITIONS` is the record
of what each one means, so a number in a report can be traced to how it was computed.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from .runner import GenerationResult

#: What each derived metric means. Anything reported must appear here.
METRIC_DEFINITIONS: dict[str, str] = {
    "accuracy": "correct / graded, where graded counts examples with a non-null verdict",
    "median_reasoning_tokens": "median of thinking_tokens over all examples, including zeros",
    "mean_reasoning_tokens": "arithmetic mean of thinking_tokens",
    "median_total_output_tokens": "median of thinking_tokens + answer_tokens",
    "median_answer_tokens": "median of answer_tokens",
    "reasoning_share": "sum(thinking_tokens) / sum(total output tokens); 0 when nothing was generated",
    "accuracy_per_1k_reasoning_tokens": (
        "accuracy / (mean reasoning tokens / 1000). Undefined — reported as null — when "
        "mean reasoning tokens is 0, because dividing by it is not a large number, it is "
        "a different question"
    ),
    "long_reasoning_fraction": (
        "fraction of examples whose thinking_tokens exceed the long_reasoning_threshold"
    ),
    "truncation_rate": "fraction of examples whose finish_reason is not 'stop'",
    "failure_rate": "fraction of examples that recorded an error",
    "median_latency_s": "median wall-clock seconds per generation",
    "tokens_per_second": "sum(total output tokens) / sum(latency); throughput, not per-example",
    "token_ratio": "student total output tokens / teacher total output tokens, summed",
    "accuracy_delta": "student accuracy - teacher accuracy, in percentage points",
}

#: Above this many thinking tokens, an example counts as "long reasoning". Arbitrary and
#: therefore configurable and always reported alongside the number it produces.
DEFAULT_LONG_REASONING_THRESHOLD = 512


@dataclass
class PairedRecord:
    """One prompt, answered by both models."""

    example_id: str
    prompt_sha256: str
    category: str = "unknown"
    difficulty: str = "unknown"
    teacher: dict[str, Any] = field(default_factory=dict)
    student: dict[str, Any] = field(default_factory=dict)

    def token_ratio(self) -> float | None:
        """Student output tokens as a fraction of teacher's, for this example."""
        teacher_tokens = self.teacher.get("total_output_tokens") or 0
        student_tokens = self.student.get("total_output_tokens") or 0
        return student_tokens / teacher_tokens if teacher_tokens else None

    def agreement(self) -> bool | None:
        """Whether both models were graded the same way. None if either is ungraded."""
        a, b = self.teacher.get("correct"), self.student.get("correct")
        return None if a is None or b is None else a == b

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["token_ratio"] = self.token_ratio()
        data["agreement"] = self.agreement()
        return data


def result_to_record(result: GenerationResult, *, model: str,
                     model_revision: str | None = None) -> dict[str, Any]:
    """One model's side of a pairing, in the evaluation data model's shape."""
    return {
        "model": model,
        "model_revision": model_revision,
        "reasoning_mode": result.reasoning_effort,
        "input_tokens": result.prompt_tokens,
        "reasoning_tokens": result.thinking_tokens,
        "answer_tokens": result.answer_tokens,
        "total_output_tokens": result.total_generated_tokens,
        "latency_seconds": result.latency_s,
        "tokens_per_second": result.tokens_per_second,
        "time_to_first_token_s": result.time_to_first_token_s,
        "output": result.answer_text,
        "reasoning": result.thinking_text,
        "correct": result.correct,
        "finish_reason": result.finish_reason,
        "error": result.error,
    }


def _cell(side: dict[str, Any], key: str, fmt: str) -> str:
    """Format one table cell, rendering a missing value as "-" rather than 0."""
    value = side.get(key)
    return "-" if value is None else fmt.format(value)


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarise_side(records: list[dict[str, Any]], *,
                   long_reasoning_threshold: int = DEFAULT_LONG_REASONING_THRESHOLD
                   ) -> dict[str, Any]:
    """Raw measurements for one model over a prompt set.

    Counts are reported alongside every rate, so a rate computed over three examples
    cannot be mistaken for one computed over three thousand.
    """
    if not records:
        return {"n": 0}

    graded = [r for r in records if r.get("correct") is not None]
    reasoning = [float(r.get("reasoning_tokens") or 0) for r in records]
    answers = [float(r.get("answer_tokens") or 0) for r in records]
    totals = [float(r.get("total_output_tokens") or 0) for r in records]
    latencies = [float(r.get("latency_seconds") or 0.0) for r in records]

    accuracy = (sum(1 for r in graded if r["correct"]) / len(graded)) if graded else None
    mean_reasoning = statistics.fmean(reasoning) if reasoning else 0.0
    total_output = sum(totals)
    total_latency = sum(latencies)

    return {
        "n": len(records),
        "n_graded": len(graded),
        "accuracy": accuracy,
        "median_reasoning_tokens": _median(reasoning),
        "mean_reasoning_tokens": mean_reasoning,
        "median_answer_tokens": _median(answers),
        "median_total_output_tokens": _median(totals),
        "total_output_tokens": total_output,
        "reasoning_share": (sum(reasoning) / total_output) if total_output else 0.0,
        # Undefined rather than infinite when nothing was reasoned: dividing accuracy by
        # zero reasoning is a different question, not a very good score.
        "accuracy_per_1k_reasoning_tokens": (
            accuracy / (mean_reasoning / 1000.0)
            if accuracy is not None and mean_reasoning > 0 else None
        ),
        "long_reasoning_fraction": (
            sum(1 for value in reasoning if value > long_reasoning_threshold) / len(reasoning)
        ),
        "long_reasoning_threshold": long_reasoning_threshold,
        "truncation_rate": (
            sum(1 for r in records
                if r.get("finish_reason") not in (None, "stop")) / len(records)
        ),
        "failure_rate": sum(1 for r in records if r.get("error")) / len(records),
        "median_latency_s": _median(latencies),
        "tokens_per_second": (total_output / total_latency) if total_latency else None,
    }


@dataclass
class PairedSummary:
    """Aggregates over a paired run, computed from the per-example records."""

    n_examples: int
    teacher: dict[str, Any]
    student: dict[str, Any]
    comparison: dict[str, Any]
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    metric_definitions: dict[str, str] = field(default_factory=lambda: dict(METRIC_DEFINITIONS))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        """Raw numbers side by side, before any ratio."""
        rows = [
            ("accuracy", "accuracy", "{:.1%}"),
            ("median reasoning tokens", "median_reasoning_tokens", "{:.0f}"),
            ("median answer tokens", "median_answer_tokens", "{:.0f}"),
            ("median total output", "median_total_output_tokens", "{:.0f}"),
            ("reasoning share", "reasoning_share", "{:.1%}"),
            ("long-reasoning fraction", "long_reasoning_fraction", "{:.1%}"),
            ("truncation rate", "truncation_rate", "{:.1%}"),
            ("failure rate", "failure_rate", "{:.1%}"),
            ("median latency (s)", "median_latency_s", "{:.2f}"),
        ]
        lines = [f"paired evaluation over {self.n_examples} example(s)", "",
                 f"  {'metric':<26}{'teacher':>14}{'student':>14}"]
        for label, key, fmt in rows:
            teacher = _cell(self.teacher, key, fmt)
            student = _cell(self.student, key, fmt)
            lines.append(f"  {label:<26}{teacher:>14}{student:>14}")

        lines.append("")
        delta = self.comparison.get("accuracy_delta")
        ratio = self.comparison.get("token_ratio")
        if delta is not None:
            lines.append(f"  accuracy delta   : {delta:+.1f} percentage points")
        if ratio is not None:
            lines.append(f"  token ratio      : {ratio:.2f}x "
                         "(student output tokens / teacher output tokens)")
        agreement = self.comparison.get("agreement_rate")
        if agreement is not None:
            lines.append(f"  agreement rate   : {agreement:.1%}")
        return "\n".join(lines)


def summarise_paired(
    records: list[PairedRecord],
    *,
    long_reasoning_threshold: int = DEFAULT_LONG_REASONING_THRESHOLD,
) -> PairedSummary:
    """Aggregate paired records without discarding what they contain."""
    teacher_side = [r.teacher for r in records if r.teacher]
    student_side = [r.student for r in records if r.student]
    teacher = summarise_side(teacher_side, long_reasoning_threshold=long_reasoning_threshold)
    student = summarise_side(student_side, long_reasoning_threshold=long_reasoning_threshold)

    agreements = [r.agreement() for r in records]
    graded_agreements = [a for a in agreements if a is not None]
    teacher_tokens = teacher.get("total_output_tokens") or 0
    student_tokens = student.get("total_output_tokens") or 0

    comparison = {
        "accuracy_delta": (
            (student["accuracy"] - teacher["accuracy"]) * 100
            if teacher.get("accuracy") is not None and student.get("accuracy") is not None
            else None
        ),
        "token_ratio": (student_tokens / teacher_tokens) if teacher_tokens else None,
        "reasoning_token_ratio": (
            student["mean_reasoning_tokens"] / teacher["mean_reasoning_tokens"]
            if teacher.get("mean_reasoning_tokens") else None
        ),
        "agreement_rate": (
            sum(1 for a in graded_agreements if a) / len(graded_agreements)
            if graded_agreements else None
        ),
        "n_agreement_graded": len(graded_agreements),
    }

    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({r.category for r in records}):
        subset = [r for r in records if r.category == category]
        by_category[category] = {
            "n": len(subset),
            "teacher": summarise_side([r.teacher for r in subset if r.teacher],
                                      long_reasoning_threshold=long_reasoning_threshold),
            "student": summarise_side([r.student for r in subset if r.student],
                                      long_reasoning_threshold=long_reasoning_threshold),
        }

    return PairedSummary(
        n_examples=len(records), teacher=teacher, student=student,
        comparison=comparison, by_category=by_category,
    )


def accuracy_at_token_budget(
    records: list[dict[str, Any]], budgets: tuple[int, ...] = (128, 256, 512, 1024, 2048)
) -> list[dict[str, Any]]:
    """Accuracy over the examples a model answered within each output-token budget.

    Two numbers per budget, because they answer different questions and one without the
    other misleads: ``coverage`` is the fraction of examples that fit in the budget, and
    ``accuracy_within_budget`` is accuracy over just those. A model that answers 10% of
    prompts within 128 tokens and gets them all right is not a 100% model.
    """
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        within = [r for r in records if (r.get("total_output_tokens") or 0) <= budget]
        graded = [r for r in within if r.get("correct") is not None]
        rows.append({
            "budget_tokens": budget,
            "n_within_budget": len(within),
            "coverage": len(within) / len(records) if records else 0.0,
            "accuracy_within_budget": (
                sum(1 for r in graded if r["correct"]) / len(graded) if graded else None
            ),
            # Accuracy over the WHOLE set, counting anything over budget as wrong: the
            # honest reading if the budget were a hard cap at serving time.
            "accuracy_if_budget_enforced": (
                sum(1 for r in graded if r["correct"]) / len(records) if records else None
            ),
        })
    return rows


def reasoning_sweep_table(by_mode: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """One row per reasoning mode: the table a reasoning-cost sweep produces.

    Ordered cheapest-first so a rising accuracy column and a rising token column can be
    read against each other directly.
    """
    from ..distillation.reasoning_modes import SUPPORTED_MODES

    order = {name: i for i, name in enumerate(SUPPORTED_MODES)}
    rows: list[dict[str, Any]] = []
    for mode in sorted(by_mode, key=lambda m: order.get(m, len(order))):
        summary = summarise_side(by_mode[mode])
        rows.append({
            "mode": mode,
            "n": summary.get("n", 0),
            "accuracy": summary.get("accuracy"),
            "median_reasoning_tokens": summary.get("median_reasoning_tokens"),
            "median_total_output_tokens": summary.get("median_total_output_tokens"),
            "median_latency_s": summary.get("median_latency_s"),
            "accuracy_per_1k_reasoning_tokens": summary.get("accuracy_per_1k_reasoning_tokens"),
        })
    return rows


def format_sweep_table(rows: list[dict[str, Any]]) -> str:
    lines = [f"  {'mode':<20}{'n':>5}{'accuracy':>10}{'reasoning':>11}"
             f"{'total':>9}{'latency':>9}"]
    for row in rows:
        lines.append(
            f"  {row['mode']:<20}{row['n']:>5}"
            f"{_cell(row, 'accuracy', '{:.1%}'):>10}"
            f"{_cell(row, 'median_reasoning_tokens', '{:.0f}'):>11}"
            f"{_cell(row, 'median_total_output_tokens', '{:.0f}'):>9}"
            f"{_cell(row, 'median_latency_s', '{:.2f}'):>9}"
        )
    return "\n".join(lines)
