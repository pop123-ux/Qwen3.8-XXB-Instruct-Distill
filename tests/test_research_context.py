"""Context regimes, curricula and the context-performance curve schema."""
from __future__ import annotations

import json

import pytest

from qwen_distill.research.context import (
    CONTEXT_REGIMES,
    CURRICULA,
    CURVE_LENGTHS,
    MAX_CONTEXT,
    ContextCurriculum,
    ContextCurve,
    ContextPoint,
    CurriculumStage,
    compare_curves,
    curriculum,
    regime_for,
)


# ---------------------------------------------------------------------------
# regimes
# ---------------------------------------------------------------------------
def test_regimes_tile_the_whole_window_without_gaps_or_overlap():
    edges = [(r.min_tokens, r.max_tokens) for r in CONTEXT_REGIMES]
    assert edges[0][0] == 0
    assert edges[-1][1] == MAX_CONTEXT + 1
    for (_, end), (start, _) in zip(edges, edges[1:], strict=False):
        assert end == start, "regimes must be contiguous"


def test_every_length_on_the_curve_lands_in_exactly_one_regime():
    for length in CURVE_LENGTHS:
        matches = [r for r in CONTEXT_REGIMES if r.contains(length)]
        assert len(matches) == 1, f"{length} matched {len(matches)} regimes"


def test_lengths_beyond_the_window_are_refused_rather_than_clamped():
    with pytest.raises(ValueError, match="beyond the declared window"):
        regime_for(MAX_CONTEXT + 10)


def test_each_regime_says_what_changes_and_what_to_probe():
    """A band with no mechanism is a bucket, and a result in it is not attributable."""
    for regime in CONTEXT_REGIMES:
        assert len(regime.mechanism) > 40
        assert len(regime.probe) > 10


# ---------------------------------------------------------------------------
# curricula
# ---------------------------------------------------------------------------
def test_every_protocol_mixture_exists_and_only_the_control_is_short_only():
    assert sorted(CURRICULA) == ["B0", "B1", "B2", "B3", "B4", "B5"]
    assert CURRICULA["B1"].max_length == 4_096
    for arm in ("B0", "B2", "B3", "B4", "B5"):
        assert CURRICULA[arm].max_length == MAX_CONTEXT


def test_the_uniform_mixture_is_uniform_in_tokens_not_in_steps():
    """The distinction the arm exists to make. Equal *step* shares would put 57% of the
    tokens at the longest length, which is not a uniform mixture."""
    share = CURRICULA["B0"].token_share()
    assert len(share) == 4
    for value in share.values():
        assert value == pytest.approx(0.25, abs=1e-6)
    steps = {s.sequence_length: s.fraction for s in CURRICULA["B0"].stages}
    assert steps[4_096] > 10 * steps[MAX_CONTEXT], "uniform tokens needs unequal steps"


def test_each_weighted_mixture_actually_weights_its_own_band():
    """A mixture named for a band must carry the most tokens there; long sequences dominate
    a token budget so easily that this is easy to get wrong."""
    def heaviest(arm):
        share = CURRICULA[arm].token_share()
        return max(share, key=share.get)

    assert heaviest("B5") == 16_384, "medium_weighted must be heaviest in the medium band"
    assert heaviest("B4") == MAX_CONTEXT
    assert heaviest("B1") == 4_096


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to"):
        ContextCurriculum(
            name="broken", interleave=False, hypothesis="",
            stages=(CurriculumStage(4096, 0.5, ""), CurriculumStage(8192, 0.2, "")),
        )


def test_b2_and_b3_share_a_token_budget_exactly():
    """The comparison they exist to support is ordering against exposure, which is only a
    controlled comparison if the token budgets are identical."""
    assert CURRICULA["B2"].token_share() == CURRICULA["B3"].token_share()
    assert CURRICULA["B2"].interleave is False
    assert CURRICULA["B3"].interleave is True


def test_token_share_differs_from_step_share():
    """40% of *steps* at 4K is under 4% of *tokens*. Reporting step fractions alone would
    overstate how much short-context data these schedules actually contain."""
    c = CURRICULA["B2"]
    steps = {s.sequence_length: s.fraction for s in c.stages}
    tokens = c.token_share()
    assert steps[4_096] == pytest.approx(0.40)
    assert tokens[4_096] < 0.05
    assert tokens[MAX_CONTEXT] > 0.5
    assert sum(tokens.values()) == pytest.approx(1.0)


def test_b4_weights_long_context_more_heavily_than_b2():
    assert CURRICULA["B4"].token_share()[MAX_CONTEXT] > CURRICULA["B2"].token_share()[MAX_CONTEXT]


@pytest.mark.parametrize("arm", ["B1", "B2", "B3", "B4"])
def test_schedules_account_for_every_step(arm):
    plan = curriculum(arm).schedule(1_000)
    assert sum(block["steps"] for block in plan) == 1_000
    assert plan[0]["start_step"] == 0
    for a, b in zip(plan, plan[1:], strict=False):
        assert a["end_step"] == b["start_step"], "the schedule has a gap or an overlap"


def test_sequential_and_interleaved_schedules_have_the_same_totals_per_length():
    """B2 and B3 must differ only in order, so the per-length step counts must match."""
    def totals(arm):
        counts: dict[int, int] = {}
        for block in curriculum(arm).schedule(2_000):
            length = block["sequence_length"]
            counts[length] = counts.get(length, 0) + block["steps"]
        return counts

    assert totals("B2") == totals("B3")


def test_interleaved_schedule_revisits_short_data_late_in_training():
    """The mechanism B3 claims: short data is never stale."""
    plan = curriculum("B3").schedule(1_000)
    late = [b for b in plan if b["start_step"] > 800]
    assert any(b["sequence_length"] == 4_096 for b in late)
    sequential = curriculum("B2").schedule(1_000)
    late_seq = [b for b in sequential if b["start_step"] > 800]
    assert not any(b["sequence_length"] == 4_096 for b in late_seq)


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="unknown context arm"):
        curriculum("B9")


def test_curriculum_serialises():
    json.loads(json.dumps(curriculum("B4").to_dict()))


# ---------------------------------------------------------------------------
# the curve
# ---------------------------------------------------------------------------
def _curve(values, model="m", direction="higher_is_better", **kw):
    return ContextCurve(
        model=model, metric="needle_accuracy", direction=direction,
        points=[ContextPoint(length, value, 100) for length, value in values], **kw
    )


def test_effective_context_is_the_last_length_before_any_dip():
    """A curve that recovers after dipping has not earned the longer claim."""
    curve = _curve([(2048, 0.90), (8192, 0.88), (32768, 0.70), (131072, 0.89)])
    assert curve.effective_context(0.90) == 8_192
    assert curve.degradation_onset(0.90) == 32_768


def test_a_curve_that_holds_reports_no_onset():
    curve = _curve([(2048, 0.9), (8192, 0.9), (262144, 0.88)])
    assert curve.effective_context(0.90) == 262_144
    assert curve.degradation_onset(0.90) is None


def test_effective_context_is_relative_to_the_model_itself():
    """A uniformly weaker model with the same shape has the same effective context; the
    metric measures context handling, not raw capability."""
    strong = _curve([(2048, 0.90), (8192, 0.81), (32768, 0.45)])
    weak = _curve([(2048, 0.45), (8192, 0.405), (32768, 0.225)])
    assert strong.effective_context(0.90) == weak.effective_context(0.90)


def test_lower_is_better_metrics_are_oriented_correctly():
    """Perplexity rising with length is degradation, not improvement."""
    curve = _curve([(2048, 5.0), (8192, 5.2), (32768, 9.0)], direction="lower_is_better")
    assert curve.effective_context(0.90) == 8_192
    assert curve.degradation_onset(0.90) == 32_768


def test_points_are_sorted_regardless_of_insertion_order():
    curve = ContextCurve(model="m", metric="x", direction="higher_is_better", points=[
        ContextPoint(32768, 0.5, 10), ContextPoint(2048, 0.9, 10),
    ])
    assert [p.sequence_length for p in curve.points] == [2048, 32768]


def test_curve_round_trips_through_json(tmp_path):
    curve = _curve([(2048, 0.9), (32768, 0.6)], context_arm="B2", layer_arm="A3")
    path = curve.save(tmp_path / "curve.json")
    restored = ContextCurve.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert restored.to_dict() == curve.to_dict()
    assert restored.context_arm == "B2" and restored.layer_arm == "A3"


def test_by_regime_groups_points():
    curve = _curve([(2048, 0.9), (4096, 0.8), (65536, 0.5)])
    assert set(curve.by_regime()) == {"short", "long"}
    assert curve.by_regime()["short"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def test_a_longer_effective_context_is_reported_as_an_extension():
    base = _curve([(2048, 0.90), (8192, 0.85), (32768, 0.50)], model="B1", context_arm="B1")
    cand = _curve([(2048, 0.90), (8192, 0.88), (32768, 0.86)], model="B2", context_arm="B2")
    result = compare_curves(base, cand)
    assert result["verdict"] == "extends_effective_context"
    assert result["effective_context"]["candidate"] > result["effective_context"]["baseline"]
    assert result["long_context_delta"] > 0


def test_an_identical_curve_is_reported_as_no_effect():
    """The refuting outcome must be reachable and must be named plainly."""
    base = _curve([(2048, 0.9), (32768, 0.5)], model="B1")
    cand = _curve([(2048, 0.9), (32768, 0.5)], model="B2")
    assert compare_curves(base, cand)["verdict"] == "no_measurable_effect"


def test_a_regression_is_reported_as_a_regression():
    base = _curve([(2048, 0.9), (8192, 0.88), (32768, 0.85)], model="B1")
    cand = _curve([(2048, 0.9), (8192, 0.60), (32768, 0.55)], model="B4")
    result = compare_curves(base, cand)
    assert result["verdict"] == "shortens_effective_context"
    assert result["long_context_delta"] < 0


def test_the_long_for_short_trade_shows_up_in_both_numbers():
    """B4's predicted outcome: long context improves, short context regresses. Both must be
    visible, because reporting only the win would misprice the trade."""
    base = _curve([(2048, 0.90), (8192, 0.88), (65536, 0.40)], model="B1")
    cand = _curve([(2048, 0.84), (8192, 0.82), (65536, 0.70)], model="B4")
    result = compare_curves(base, cand)
    assert result["short_context_delta"] < 0
    assert result["long_context_delta"] > 0


def test_comparing_different_metrics_is_refused():
    base = _curve([(2048, 0.9)], model="a")
    other = ContextCurve(model="b", metric="perplexity", direction="lower_is_better",
                         points=[ContextPoint(2048, 5.0, 10)])
    with pytest.raises(ValueError, match="cannot compare"):
        compare_curves(base, other)


def test_comparing_disjoint_lengths_is_refused():
    base = _curve([(2048, 0.9)], model="a")
    cand = _curve([(32768, 0.9)], model="b")
    with pytest.raises(ValueError, match="share no measured lengths"):
        compare_curves(base, cand)
