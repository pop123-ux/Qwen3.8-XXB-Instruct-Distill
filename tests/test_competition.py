"""Tests for the competitive picture.

The failure guarded against here is a specific, tempting one: publishing "we beat
Qwen3.5-9B on MMLU-Pro" on the strength of a number nobody verified, produced under an
evaluation protocol nobody matched. It would look exactly like a result. So the assertions
below are mostly about what the module *refuses* to say.
"""

from __future__ import annotations

import json

import pytest

from qwen_distill.analysis.competition import (
    CORROBORATED,
    MEASURED,
    MISSING_CAPABILITIES,
    TARGET_BENCHMARKS,
    UNVERIFIED,
    VENDOR_REPORTED,
    Competitor,
    KVGeometry,
    Score,
    compare,
    envelope,
    gap_to_target,
    reference_field,
    scoreboard,
    verification_backlog,
    with_scores,
)

PROTOCOL = "0-shot CoT, greedy, harness v1"


def measured(benchmark: str, value: float, protocol: str = PROTOCOL) -> Score:
    return Score(benchmark, value, MEASURED, "experiments/runs/x", protocol=protocol)


def published(benchmark: str, value: float, protocol: str | None = PROTOCOL) -> Score:
    return Score(benchmark, value, VENDOR_REPORTED, "model card", protocol=protocol)


# --- what it refuses to say -----------------------------------------------
def test_beating_an_unverified_number_is_not_a_win():
    """The whole point. An aspiration is not a bar."""
    theirs = Score("mmlu_pro", 82.5, UNVERIFIED, "user-supplied", protocol=PROTOCOL)
    result = compare(measured("mmlu_pro", 90.0), theirs, benchmark="mmlu_pro")
    assert result.verdict == "INCOMPARABLE"
    assert result.margin is None
    assert any("verify it against the primary source" in r for r in result.reasons)


def test_our_own_score_must_be_measured_here():
    ours = Score("ifeval", 95.0, VENDOR_REPORTED, "a blog post", protocol=PROTOCOL)
    result = compare(ours, published("ifeval", 91.5), benchmark="ifeval")
    assert result.verdict == "INCOMPARABLE"
    assert any("needs a number this repository produced" in r for r in result.reasons)


def test_mismatched_protocols_are_not_compared():
    result = compare(
        measured("gpqa_diamond", 85.0, "5-shot"),
        published("gpqa_diamond", 81.7, "0-shot CoT"),
        benchmark="gpqa_diamond",
    )
    assert result.verdict == "INCOMPARABLE"
    assert any("protocols differ" in r for r in result.reasons)


def test_an_unrecorded_protocol_blocks_the_comparison():
    result = compare(
        measured("ifeval", 92.0), published("ifeval", 91.5, protocol=None), benchmark="ifeval"
    )
    assert result.verdict == "INCOMPARABLE"
    assert any("protocol is unrecorded" in r for r in result.reasons)


def test_a_missing_score_on_either_side_is_reported_not_defaulted():
    absent = compare(None, published("ifeval", 91.5), benchmark="ifeval")
    assert absent.verdict == "INCOMPARABLE"
    assert any("we have not measured" in r for r in absent.reasons)

    theirs = compare(measured("ifeval", 92.0), None, benchmark="ifeval")
    assert theirs.verdict == "INCOMPARABLE"
    assert any("no competitor score" in r for r in theirs.reasons)


# --- what it will say ------------------------------------------------------
@pytest.mark.parametrize(
    ("ours", "theirs", "verdict", "margin"),
    [(92.0, 91.5, "AHEAD", 0.5), (90.0, 91.5, "BEHIND", -1.5), (91.5, 91.5, "TIED", 0.0)],
)
def test_two_comparable_numbers_produce_a_verdict(ours, theirs, verdict, margin):
    result = compare(
        measured("ifeval", ours), published("ifeval", theirs), benchmark="ifeval"
    )
    assert result.verdict == verdict
    assert result.margin == pytest.approx(margin)


def test_the_scoreboard_covers_every_benchmark_either_side_records():
    competitor = with_scores(
        Competitor(name="rival"), [published("ifeval", 91.5), published("mmlu_pro", 82.5)]
    )
    board = scoreboard({"ifeval": measured("ifeval", 92.0), "bfcl_v4": measured("bfcl_v4", 70.0)},
                       competitor)
    assert {c.benchmark for c in board} == {"ifeval", "mmlu_pro", "bfcl_v4"}


def test_a_single_incomparable_benchmark_blocks_the_overall_verdict():
    """Winning four of five and being unable to compare the fifth is not winning."""
    competitor = with_scores(
        Competitor(name="rival"),
        [published("ifeval", 91.5), published("mmlu_pro", 82.5)],
    )
    summary = gap_to_target({"ifeval": measured("ifeval", 99.0)}, competitor)
    assert summary["verdict"] == "NOT YET COMPARABLE"
    assert summary["incomparable"] == ["mmlu_pro"]


# --- the deployment envelope ----------------------------------------------
def test_weight_memory_scales_with_the_quantisation():
    model = Competitor(name="m", parameters=9_000_000_000, parameters_provenance=MEASURED)
    assert model.weight_gib("bf16") > model.weight_gib("q4_k_m") > model.weight_gib("q3_k_m")


def test_an_unknown_kv_geometry_gives_no_cache_figure_and_no_verdict():
    """A partial sum that happens to be under budget is not a fit.

    Silently treating an unknown cache as zero is how a model that does not fit gets
    reported as fitting.
    """
    model = Competitor(name="m", parameters=9_000_000_000, parameters_provenance=MEASURED, kv=None)
    result = envelope(model, budget_gib=13.56)
    assert result.kv_gib is None
    assert result.verdict == "UNKNOWN"
    assert any("KV geometry unknown" in u for u in result.unknowns)


def test_a_known_geometry_produces_a_real_verdict():
    model = Competitor(
        name="m", parameters=9_000_000_000, parameters_provenance=MEASURED,
        kv=KVGeometry(layers=40, num_key_value_heads=8, head_dim=128, provenance=MEASURED),
    )
    assert envelope(model, budget_gib=13.56, context=8192).verdict in {"FITS", "TIGHT"}
    assert envelope(model, budget_gib=13.56, context=131072).verdict == "DOES NOT FIT"


def test_sliding_window_layers_cost_a_bounded_cache():
    """The difference that makes a 27B model plausible on a 16 GB card at long context."""
    uniform = KVGeometry(layers=62, num_key_value_heads=16, head_dim=128, provenance=MEASURED)
    interleaved = KVGeometry(
        layers=62, num_key_value_heads=16, head_dim=128, full_attention_layers=10,
        sliding_window_layers=52, sliding_window=1024, provenance=MEASURED,
    )
    assert interleaved.bytes_at(131072) < uniform.bytes_at(131072) / 4


def test_asking_beyond_a_models_context_is_flagged():
    model = Competitor(
        name="m", parameters=1_000_000_000, parameters_provenance=MEASURED,
        max_context=32768,
        kv=KVGeometry(layers=8, num_key_value_heads=2, head_dim=64, provenance=MEASURED),
    )
    result = envelope(model, budget_gib=13.56, context=131072)
    assert any("supports 32768" in u for u in result.unknowns)
    assert result.verdict == "UNKNOWN"


def test_an_moe_is_distinguished_from_a_dense_model_of_the_same_size():
    """Active parameters drive compute; total parameters are what has to fit."""
    moe = Competitor(name="moe", parameters=30_000_000_000, active_parameters=3_000_000_000)
    dense = Competitor(name="dense", parameters=30_000_000_000, active_parameters=30_000_000_000)
    assert moe.is_moe and not dense.is_moe
    assert moe.weight_gib("q4_k_m") == dense.weight_gib("q4_k_m")


# --- the field is honest about itself -------------------------------------
def test_no_registry_figure_claims_to_be_ours_or_read_from_a_model_card():
    """The registry may improve on UNVERIFIED, but it must never claim MEASURED.

    Nothing in it has been read from a primary source in this repository — the Hugging
    Face model card is blocked by this environment's egress proxy — so every figure is
    somewhere on the unverified-to-corroborated range and none of it is citable as ours.
    """
    for competitor in reference_field().values():
        assert competitor.parameters_provenance in (UNVERIFIED, CORROBORATED), competitor.name
        assert competitor.source, competitor.name
        assert "NOT" in competitor.source or "not checked" in competitor.source, competitor.name
        for score in competitor.scores.values():
            assert score.provenance in (UNVERIFIED, CORROBORATED), (
                f"{competitor.name}/{score.benchmark}"
            )
            assert not score.citable


def test_corroboration_records_what_was_and_was_not_confirmed():
    """Six of seven target scores were corroborated independently; LongBench v2 was not,
    and one benchmark came back with two conflicting figures. All three states are kept."""
    target = reference_field()["Qwen3.5-9B"]
    corroborated = [s for s in target.scores.values() if s.provenance == CORROBORATED]
    unverified = [s for s in target.scores.values() if s.provenance == UNVERIFIED]
    assert len(corroborated) == 6
    assert [s.benchmark for s in unverified] == ["longbench_v2"]
    assert "82.7" in target.scores["livecodebench_v6"].notes
    assert "egress proxy" in target.scores["mmlu_pro"].source


def test_the_primary_target_is_modelled_by_our_own_estimator():
    """It is in the teacher's architecture family, so it gets the hybrid-aware estimate
    rather than a parameter-count approximation — the same one we hold ourselves to."""
    target = reference_field()["Qwen3.5-9B"]
    assert target.spec is not None
    assert target.spec.full_attention_interval == 4
    assert target.spec.num_full_attention_layers == 8
    assert target.spec.head_dim == 256

    result = envelope(target, budget_gib=13.56, context=32768)
    assert result.verdict == "FITS"
    assert result.kv_gib is not None and result.kv_gib > 0


def test_the_inferred_vocabulary_is_the_one_that_reproduces_nine_billion():
    """The vocabulary is not stated in any source reached; it is inferred from the
    parameter count. This pins the arithmetic that justifies the inference."""
    from qwen_distill.analysis.competition import qwen35_9b_spec
    from qwen_distill.architecture.params import count_parameters

    total = count_parameters(qwen35_9b_spec()).total
    assert 8.9e9 < total < 9.0e9, f"{total:,} is not 9B, so the inference is wrong"


def test_the_targets_head_ratios_block_a_direct_transfer_from_the_teacher():
    """A real constraint, not a footnote: the teacher has 6 query heads per KV head and 3
    DeltaNet value heads per key head; this target has 4 and 2."""
    from qwen_distill.analysis.competition import qwen35_9b_spec

    target = qwen35_9b_spec()
    assert target.num_attention_heads // target.num_key_value_heads == 4
    assert target.linear_num_value_heads // target.linear_num_key_heads == 2
    assert "refused by materialize.py" in reference_field()["Qwen3.5-9B"].notes


def test_the_primary_target_carries_the_seven_named_benchmarks():
    target = reference_field()["Qwen3.5-9B"]
    assert set(target.scores) == set(TARGET_BENCHMARKS)
    assert target.scores["mmlu_pro"].value == 82.5
    assert target.scores["tau2_bench"].value == 79.1


def test_the_interleaved_competitor_carries_its_real_layer_split():
    """Gemma 3 is 5 sliding-window layers (window 1024) to 1 global, over 62 layers.

    Assuming uniform global attention would have overstated its cache roughly fourfold at
    long context — an error in our favour, which is when to be most careful. This asserts
    both the split and that it actually reduces the cache.
    """
    gemma = reference_field()["Gemma-3-27B"]
    assert gemma.kv is not None
    assert gemma.kv.full_attention_layers == 10
    assert gemma.kv.sliding_window_layers == 52
    assert gemma.kv.sliding_window == 1024

    uniform = KVGeometry(layers=62, num_key_value_heads=16, head_dim=128, provenance=MEASURED)
    assert gemma.kv.bytes_at(131072) < uniform.bytes_at(131072) / 3


def test_the_hybrid_competitor_pays_far_less_for_context_than_the_dense_one():
    """The measured reason a 9B hybrid beats a 14B dense model on this constraint.

    Qwen3-14B keeps a full KV cache on all 40 layers; Qwen3.5-9B keeps one on 8 of 32.
    At 32K that is the difference between fitting a 16 GB card and not.
    """
    field = reference_field()
    hybrid = envelope(field["Qwen3.5-9B"], budget_gib=13.56, context=32768)
    dense = envelope(field["Qwen3-14B"], budget_gib=13.56, context=32768)
    assert dense.kv_gib > hybrid.kv_gib * 4
    assert hybrid.verdict == "FITS"
    assert dense.verdict == "DOES NOT FIT"


def test_the_largest_competitor_does_not_fit_at_all():
    """Gemma-3-27B's weights alone exceed the budget at q4_k_m, so it is out of this
    field regardless of context. Recorded because 'fits under appropriate quantisation'
    is an assumption worth checking rather than repeating."""
    gemma = reference_field()["Gemma-3-27B"]
    for quant in ("q4_k_m", "q3_k_m"):
        assert envelope(gemma, budget_gib=13.56, quant=quant, context=8192).verdict == (
            "DOES NOT FIT"
        )


def test_no_target_benchmark_is_implemented_yet():
    """The objective is stated entirely in numbers this repository cannot yet produce.
    When that changes, this test changes with it — deliberately."""
    assert not any(b.implemented for b in TARGET_BENCHMARKS.values())


def test_the_multilingual_gap_is_recorded_rather_than_dropped():
    assert "multilingual" in MISSING_CAPABILITIES
    assert "unmeasured, not passing" in MISSING_CAPABILITIES["multilingual"]


def test_the_verification_backlog_names_what_is_still_open_and_nothing_else():
    """It must shrink as things are verified, and it must not report gaps the estimates
    do not actually have — a backlog that cries wolf gets ignored."""
    backlog = verification_backlog()
    joined = "\n".join(backlog)

    # Every unimplemented benchmark, and the one score still resting on a single source.
    for benchmark in TARGET_BENCHMARKS:
        assert benchmark in joined
    assert "longbench_v2" in joined

    # Fully corroborated competitors have dropped off.
    assert "Qwen3-14B" not in joined
    assert "Gemma-3-27B" not in joined
    # And a competitor modelled from a full spec is not reported as having unknown KV.
    assert "KV geometry unknown" not in joined


def test_the_backlog_still_blocks_a_strict_report():
    """No benchmark is implemented, so there is always something outstanding."""
    assert verification_backlog()


def test_everything_is_json_serialisable():
    field = reference_field()
    json.dumps([envelope(c, budget_gib=13.56).to_dict() for c in field.values()])
    json.dumps(gap_to_target({}, field["Qwen3.5-9B"]))
