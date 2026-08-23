"""Aggregate generation results into the metrics this project is judged on.

The reporting rule from ``docs/reasoning-efficiency.md``: a token reduction paired
with a hard-task accuracy drop is a capability regression, not an efficiency win.
:func:`summarise` therefore always breaks results out **by difficulty**, so the two
cannot be confused, and :func:`compare` reports the hard-stratum delta explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from .runner import GenerationResult
from .tasks import DIFFICULTY_ORDER


@dataclass
class StratumSummary:
    """Aggregates over one difficulty stratum (or the whole run)."""

    label: str
    n: int
    n_scored: int
    n_correct: int
    accuracy: float | None
    mean_thinking_tokens: float
    mean_answer_tokens: float
    mean_total_tokens: float
    mean_latency_s: float
    total_thinking_tokens: int

    @property
    def accuracy_per_1k_thinking_tokens(self) -> float | None:
        """Accuracy divided by thousands of reasoning tokens spent.

        A **descriptive project metric**, not a universal quality measure: it is a ratio
        of two quantities measured under one fixed configuration, and it is only
        comparable across runs of *our own* harness at identical settings. Never quote
        it against a number from another paper or harness.
        """
        if self.accuracy is None or self.mean_thinking_tokens <= 0:
            return None
        return self.accuracy / (self.mean_thinking_tokens / 1000.0)

    @property
    def accuracy_per_second(self) -> float | None:
        """Accuracy divided by mean wall-clock latency, in seconds.

        Descriptive, and hardware-dependent: it changes with GPU, batch size and
        backend, so it compares checkpoints only when everything else is held fixed.
        """
        if self.accuracy is None or self.mean_latency_s <= 0:
            return None
        return self.accuracy / self.mean_latency_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n": self.n,
            "n_scored": self.n_scored,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "mean_thinking_tokens": self.mean_thinking_tokens,
            "mean_answer_tokens": self.mean_answer_tokens,
            "mean_total_tokens": self.mean_total_tokens,
            "mean_latency_s": self.mean_latency_s,
            "total_thinking_tokens": self.total_thinking_tokens,
            "accuracy_per_1k_thinking_tokens": self.accuracy_per_1k_thinking_tokens,
            "accuracy_per_second": self.accuracy_per_second,
        }


def _summarise_group(label: str, results: list[GenerationResult]) -> StratumSummary:
    scored = [r for r in results if r.correct is not None]
    correct = sum(1 for r in scored if r.correct)
    return StratumSummary(
        label=label,
        n=len(results),
        n_scored=len(scored),
        n_correct=correct,
        accuracy=(correct / len(scored)) if scored else None,
        mean_thinking_tokens=mean([r.thinking_tokens for r in results]) if results else 0.0,
        mean_answer_tokens=mean([r.answer_tokens for r in results]) if results else 0.0,
        mean_total_tokens=mean([r.total_generated_tokens for r in results]) if results else 0.0,
        mean_latency_s=mean([r.latency_s for r in results]) if results else 0.0,
        total_thinking_tokens=sum(r.thinking_tokens for r in results),
    )


@dataclass
class RunSummary:
    """A whole evaluation run: overall plus per-difficulty and per-category."""

    overall: StratumSummary
    by_difficulty: dict[str, StratumSummary] = field(default_factory=dict)
    by_category: dict[str, StratumSummary] = field(default_factory=dict)
    n_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "by_difficulty": {k: v.to_dict() for k, v in self.by_difficulty.items()},
            "by_category": {k: v.to_dict() for k, v in self.by_category.items()},
            "n_errors": self.n_errors,
        }


def summarise(results: list[GenerationResult]) -> RunSummary:
    """Aggregate results overall, by difficulty and by category."""
    by_difficulty: dict[str, list[GenerationResult]] = {}
    by_category: dict[str, list[GenerationResult]] = {}
    for result in results:
        by_difficulty.setdefault(result.difficulty, []).append(result)
        by_category.setdefault(result.category, []).append(result)

    ordered = [d for d in DIFFICULTY_ORDER if d in by_difficulty]
    ordered += [d for d in by_difficulty if d not in DIFFICULTY_ORDER]

    return RunSummary(
        overall=_summarise_group("overall", results),
        by_difficulty={d: _summarise_group(d, by_difficulty[d]) for d in ordered},
        by_category={c: _summarise_group(c, by_category[c]) for c in sorted(by_category)},
        n_errors=sum(1 for r in results if r.error),
    )


def compare(reference: RunSummary, candidate: RunSummary) -> dict[str, Any]:
    """Compare two runs (typically teacher vs student).

    ``hard_stratum_accuracy_delta`` covers the ``hard`` and ``very_hard`` strata
    together. It is reported separately because it is the number that decides whether
    a token saving was real: a large negative value invalidates any efficiency claim,
    and ``efficiency_win`` encodes exactly that rule.
    """
    def ratio(a: float | None, b: float | None) -> float | None:
        if a is None or b is None or b == 0:
            return None
        return a / b

    hard_labels = ("hard", "very_hard")

    def hard_accuracy(summary: RunSummary) -> float | None:
        correct = sum(
            s.n_correct for label, s in summary.by_difficulty.items() if label in hard_labels
        )
        scored = sum(
            s.n_scored for label, s in summary.by_difficulty.items() if label in hard_labels
        )
        return (correct / scored) if scored else None

    ref_hard, cand_hard = hard_accuracy(reference), hard_accuracy(candidate)
    hard_delta = (
        None if ref_hard is None or cand_hard is None else cand_hard - ref_hard
    )
    thinking_ratio = ratio(
        candidate.overall.mean_thinking_tokens, reference.overall.mean_thinking_tokens
    )

    efficiency_win = None
    if hard_delta is not None and thinking_ratio is not None:
        # A saving only counts if hard-task accuracy did not meaningfully drop.
        efficiency_win = thinking_ratio < 1.0 and hard_delta >= -0.02

    return {
        "capability_retention": ratio(candidate.overall.accuracy, reference.overall.accuracy),
        "accuracy_delta": (
            None
            if candidate.overall.accuracy is None or reference.overall.accuracy is None
            else candidate.overall.accuracy - reference.overall.accuracy
        ),
        "thinking_token_ratio": thinking_ratio,
        "total_token_ratio": ratio(
            candidate.overall.mean_total_tokens, reference.overall.mean_total_tokens
        ),
        "latency_ratio": ratio(
            candidate.overall.mean_latency_s, reference.overall.mean_latency_s
        ),
        "reference_hard_accuracy": ref_hard,
        "candidate_hard_accuracy": cand_hard,
        "hard_stratum_accuracy_delta": hard_delta,
        "efficiency_win": efficiency_win,
        "efficiency_win_rule": (
            "thinking_token_ratio < 1.0 AND hard_stratum_accuracy_delta >= -0.02"
        ),
    }


def format_summary(summary: RunSummary) -> str:
    """Render a run summary as a fixed-width table."""
    lines = [
        f"{'stratum':<14}{'n':>4}{'acc':>8}{'think':>9}{'answer':>9}{'total':>9}{'lat(s)':>9}",
        "-" * 62,
    ]
    for label, stratum in summary.by_difficulty.items():
        acc = "-" if stratum.accuracy is None else f"{stratum.accuracy * 100:.1f}%"
        lines.append(
            f"{label:<14}{stratum.n:>4}{acc:>8}{stratum.mean_thinking_tokens:>9.0f}"
            f"{stratum.mean_answer_tokens:>9.0f}{stratum.mean_total_tokens:>9.0f}"
            f"{stratum.mean_latency_s:>9.2f}"
        )
    lines.append("-" * 62)
    overall = summary.overall
    acc = "-" if overall.accuracy is None else f"{overall.accuracy * 100:.1f}%"
    lines.append(
        f"{'OVERALL':<14}{overall.n:>4}{acc:>8}{overall.mean_thinking_tokens:>9.0f}"
        f"{overall.mean_answer_tokens:>9.0f}{overall.mean_total_tokens:>9.0f}"
        f"{overall.mean_latency_s:>9.2f}"
    )
    if summary.n_errors:
        lines.append(f"errors: {summary.n_errors}")
    return "\n".join(lines)
