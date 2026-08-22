"""Tests for task definitions, generation bookkeeping and metric aggregation."""

from __future__ import annotations

import pytest

from qwen_distill.evaluation.metrics import compare, format_summary, summarise
from qwen_distill.evaluation.runner import GenerationResult, split_thinking
from qwen_distill.evaluation.tasks import (
    DIFFICULTY_ORDER,
    contains,
    exact_match,
    final_number,
    long_context_suite,
    needle_in_haystack,
    reasoning_dev_set,
    tasks_by_difficulty,
)


# --- checkers ---------------------------------------------------------
def test_exact_match_normalises_whitespace_and_case():
    check = exact_match("Tokyo")
    assert check("tokyo") and check("  TOKYO . ") and not check("Kyoto")


def test_final_number_takes_the_last_number():
    """Models restate the question; the last number is the answer."""
    check = final_number(105)
    assert check("15 times 7 is 105")
    assert check("We compute 15 * 7. The answer is 105.")
    assert not check("105 is not it, the answer is 106")


def test_final_number_handles_thousands_separators_and_no_number():
    assert final_number(1234)("the total is 1,234")
    assert not final_number(5)("no digits here")


def test_contains_requires_every_substring():
    check = contains("def is_even", "%")
    assert check("def is_even(n):\n    return n % 2 == 0")
    assert not check("def is_even(n): return True")


# --- dev set ----------------------------------------------------------
def test_dev_set_covers_every_difficulty():
    grouped = tasks_by_difficulty(reasoning_dev_set())
    for difficulty in DIFFICULTY_ORDER:
        assert difficulty in grouped, difficulty


def test_dev_set_has_trivial_items():
    """Trivial items are what expose overthinking; the set is useless without them."""
    trivial = tasks_by_difficulty(reasoning_dev_set())["trivial"]
    assert len(trivial) >= 4


def test_dev_set_task_ids_are_unique():
    tasks = reasoning_dev_set()
    assert len({t.task_id for t in tasks}) == len(tasks)


def test_most_dev_tasks_are_mechanically_checkable():
    tasks = reasoning_dev_set()
    checkable = [t for t in tasks if t.checker is not None]
    assert len(checkable) / len(tasks) > 0.9


def test_task_score_survives_a_broken_checker():
    from qwen_distill.evaluation.tasks import Task

    def explode(_: str) -> bool:
        raise RuntimeError("boom")

    assert Task("t", "reasoning", "easy", "p", explode).score("x") is False


def test_unscored_task_returns_none():
    from qwen_distill.evaluation.tasks import Task

    assert Task("t", "reasoning", "easy", "p", None).score("x") is None


# --- long context -----------------------------------------------------
def test_needle_depth_moves_the_needle():
    early = needle_in_haystack(2000, depth=0.05)
    late = needle_in_haystack(2000, depth=0.95)
    assert early.prompt.index("access code for the vault is") < late.prompt.index(
        "access code for the vault is"
    )


def test_needle_is_answerable_and_rejects_wrong_answers():
    task = needle_in_haystack(500, depth=0.5)
    assert task.score("74619") is True
    assert task.score("12345") is False


def test_longer_context_produces_longer_prompt():
    assert len(needle_in_haystack(8000).prompt) > len(needle_in_haystack(1000).prompt)


def test_invalid_depth_rejected():
    with pytest.raises(ValueError, match="depth"):
        needle_in_haystack(1000, depth=1.5)


def test_long_context_suite_is_the_cross_product():
    suite = long_context_suite((1024, 4096), (0.1, 0.9))
    assert len(suite) == 4
    assert len({t.task_id for t in suite}) == 4


# --- thinking split ---------------------------------------------------
@pytest.mark.parametrize(
    "text,thinking,answer",
    [
        ("<think>reason</think>answer", "reason", "answer"),
        ("reason</think>answer", "reason", "answer"),      # template opened the tag
        ("<think>unterminated", "unterminated", ""),        # ran out of budget
        ("plain answer", "", "plain answer"),
        ("", "", ""),
    ],
)
def test_split_thinking(text, thinking, answer):
    assert split_thinking(text) == (thinking, answer)


# --- metrics ----------------------------------------------------------
def _result(task_id, difficulty, correct, thinking, answer=10, latency=1.0):
    return GenerationResult(
        task_id=task_id, category="reasoning", difficulty=difficulty,
        prompt_tokens=10, thinking_tokens=thinking, answer_tokens=answer,
        total_generated_tokens=thinking + answer, latency_s=latency,
        time_to_first_token_s=None, thinking_text="", answer_text="", correct=correct,
    )


def test_summarise_stratifies_by_difficulty():
    summary = summarise([
        _result("a", "trivial", True, 2),
        _result("b", "trivial", False, 4),
        _result("c", "hard", True, 900),
    ])
    assert summary.overall.n == 3
    assert summary.by_difficulty["trivial"].accuracy == 0.5
    assert summary.by_difficulty["hard"].accuracy == 1.0
    assert summary.by_difficulty["hard"].mean_thinking_tokens == 900


def test_summarise_orders_difficulty_ascending():
    summary = summarise([_result("a", "hard", True, 5), _result("b", "trivial", True, 5)])
    assert list(summary.by_difficulty) == ["trivial", "hard"]


def test_unscored_results_do_not_count_toward_accuracy():
    summary = summarise([_result("a", "easy", True, 1), _result("b", "easy", None, 1)])
    assert summary.overall.n == 2
    assert summary.overall.n_scored == 1
    assert summary.overall.accuracy == 1.0


def test_tokens_per_second():
    assert _result("a", "easy", True, 10, answer=10, latency=2.0).tokens_per_second == 10.0
    assert _result("a", "easy", True, 10, latency=0.0).tokens_per_second == 0.0


def test_compare_flags_efficiency_win():
    """Fewer thinking tokens with hard accuracy held: a genuine win."""
    teacher = summarise([_result("h1", "hard", True, 1000), _result("t1", "trivial", True, 500)])
    student = summarise([_result("h1", "hard", True, 400), _result("t1", "trivial", True, 20)])
    result = compare(teacher, student)
    assert result["thinking_token_ratio"] < 1.0
    assert result["hard_stratum_accuracy_delta"] == 0.0
    assert result["efficiency_win"] is True


def test_compare_rejects_token_saving_that_costs_hard_accuracy():
    """The central guard: this must NOT be reported as an efficiency win."""
    teacher = summarise([
        _result("h1", "hard", True, 1000), _result("h2", "very_hard", True, 1200),
        _result("t1", "trivial", True, 500),
    ])
    student = summarise([
        _result("h1", "hard", False, 30), _result("h2", "very_hard", False, 40),
        _result("t1", "trivial", True, 10),
    ])
    result = compare(teacher, student)
    assert result["thinking_token_ratio"] < 1.0        # it did use fewer tokens
    assert result["hard_stratum_accuracy_delta"] == -1.0
    assert result["efficiency_win"] is False           # but it is not a win


def test_compare_allows_small_hard_accuracy_noise():
    teacher = summarise([_result(f"h{i}", "hard", True, 1000) for i in range(100)])
    student = summarise(
        [_result("h0", "hard", False, 500)] + [_result(f"h{i}", "hard", True, 500) for i in range(1, 100)]
    )
    result = compare(teacher, student)
    assert result["hard_stratum_accuracy_delta"] == pytest.approx(-0.01)
    assert result["efficiency_win"] is True


def test_compare_handles_missing_strata():
    only_easy = summarise([_result("a", "easy", True, 5)])
    assert compare(only_easy, only_easy)["hard_stratum_accuracy_delta"] is None


def test_format_summary_renders_all_strata():
    text = format_summary(summarise([_result("a", "trivial", True, 1), _result("b", "hard", True, 900)]))
    assert "trivial" in text and "hard" in text and "OVERALL" in text
