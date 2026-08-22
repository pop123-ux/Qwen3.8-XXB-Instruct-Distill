"""Tests for reasoning-control measurement.

The headline capability under test: detecting that two reasoning settings render
*byte-identical* prompts, which proves one of them is a no-op by construction. The
fixture template deliberately reproduces the behaviour reported for Qwen3.8's
``medium`` so the detector is exercised against a known-positive case.
"""

from __future__ import annotations

from conftest import requires_stack

from qwen_distill.evaluation.metrics import summarise
from qwen_distill.evaluation.reasoning import ReasoningSweep, compare_rendered_prompts
from qwen_distill.evaluation.runner import GenerationResult


def _result(thinking: int, correct=True, difficulty="easy"):
    return GenerationResult(
        task_id="t", category="reasoning", difficulty=difficulty, prompt_tokens=10,
        thinking_tokens=thinking, answer_tokens=10, total_generated_tokens=thinking + 10,
        latency_s=1.0, time_to_first_token_s=None, thinking_text="", answer_text="",
        correct=correct,
    )


@requires_stack
def test_detects_settings_that_render_identical_prompts(tiny_tokenizer):
    comparison = compare_rendered_prompts(tiny_tokenizer)
    assert comparison.has_noop_settings
    identical = {frozenset(group) for group in comparison.identical_groups}
    assert any({"(default)", "medium"} <= group for group in identical)


@requires_stack
def test_distinct_settings_are_reported_as_distinct(tiny_tokenizer):
    comparison = compare_rendered_prompts(tiny_tokenizer)
    by_setting = {r.setting: r.sha256 for r in comparison.renderings}
    assert by_setting["xhigh"] != by_setting["low"]
    assert by_setting["xhigh"] != by_setting["(default)"]


@requires_stack
def test_disabling_thinking_changes_the_prompt(tiny_tokenizer):
    comparison = compare_rendered_prompts(tiny_tokenizer)
    by_setting = {r.setting: r.sha256 for r in comparison.renderings}
    assert by_setting["thinking_disabled"] != by_setting["(default)"]


@requires_stack
def test_renderings_are_deterministic(tiny_tokenizer):
    a = compare_rendered_prompts(tiny_tokenizer)
    b = compare_rendered_prompts(tiny_tokenizer)
    assert [r.sha256 for r in a.renderings] == [r.sha256 for r in b.renderings]


@requires_stack
def test_comparison_is_json_serialisable(tiny_tokenizer):
    import json

    json.dumps(compare_rendered_prompts(tiny_tokenizer).to_dict())


# --- sweep analysis (no model needed) ---------------------------------
def test_sweep_detects_indistinguishable_settings():
    sweep = ReasoningSweep()
    sweep.per_setting = {
        "(default)": summarise([_result(1000)]),
        "medium": summarise([_result(1000)]),      # identical: a no-op
        "low": summarise([_result(50)]),
    }
    pairs = {frozenset(p) for p in sweep.indistinguishable_settings()}
    assert frozenset({"(default)", "medium"}) in pairs
    assert frozenset({"(default)", "low"}) not in pairs


def test_sweep_token_ratios_are_relative_to_reference():
    sweep = ReasoningSweep()
    sweep.per_setting = {
        "xhigh": summarise([_result(1000)]),
        "low": summarise([_result(250)]),
    }
    ratios = sweep.token_ratios("xhigh")
    assert ratios["xhigh"] == 1.0
    assert ratios["low"] == 0.25


def test_sweep_token_ratios_handle_zero_reference():
    sweep = ReasoningSweep()
    sweep.per_setting = {"off": summarise([_result(0)]), "on": summarise([_result(100)])}
    assert all(v is None for v in sweep.token_ratios("off").values())


def test_zero_thinking_settings_are_indistinguishable_from_each_other():
    sweep = ReasoningSweep()
    sweep.per_setting = {"a": summarise([_result(0)]), "b": summarise([_result(0)])}
    assert sweep.indistinguishable_settings() == [("a", "b")]
