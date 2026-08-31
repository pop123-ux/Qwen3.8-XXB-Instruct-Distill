"""Feasibility on the cards the project actually targets.

The destination is a 16 GB consumer GPU with a path to 12 GB. These tests pin the two
things that make the estimate useful rather than merely present: that the targets cannot
drift into datacenter hardware, and that "fits" is always qualified by the context length
it fits at.
"""

from __future__ import annotations

import pytest

from qwen_distill.analysis.deployment import (
    BORDERLINE,
    CONTEXT_LADDER,
    DOES_NOT_FIT,
    FIT,
    PRECISIONS,
    PRIMARY_TARGET,
    SECONDARY_TARGET,
    TARGETS,
    assess,
    sweep,
)
from qwen_distill.architecture.presets import derive, get_spec


def _long_context(name: str):
    """A preset re-declared for the full ladder. The students declare 4K, so measuring
    long-context behaviour requires saying so explicitly."""
    return derive(name, name=f"{name}_longctx", max_position_embeddings=262144)


# ----------------------------------------------------------------------------------
# the targets
# ----------------------------------------------------------------------------------


def test_the_two_targets_are_consumer_cards():
    assert [t.nominal_gb for t in TARGETS] == [16, 12]
    assert PRIMARY_TARGET.priority == "primary"
    assert SECONDARY_TARGET.priority == "secondary"


def test_nominal_gb_is_never_treated_as_usable():
    """A '16 GB' T4 reports 14.56 GiB and a desktop card is usually driving a display.
    Planning against the marketing number is how a configuration that obviously fits
    OOMs."""
    assert PRIMARY_TARGET.total_gib == 14.56
    assert PRIMARY_TARGET.usable_gib == pytest.approx(13.56)
    assert PRIMARY_TARGET.usable_gib < PRIMARY_TARGET.nominal_gb
    assert SECONDARY_TARGET.usable_gib < SECONDARY_TARGET.nominal_gb


def test_only_the_measured_target_claims_to_be_measured():
    assert "measured" in PRIMARY_TARGET.note
    assert "not measured" in SECONDARY_TARGET.note


def test_the_context_ladder_spans_4k_to_256k():
    assert CONTEXT_LADDER[0] == 4096
    assert CONTEXT_LADDER[-1] == 262144
    assert list(CONTEXT_LADDER) == sorted(CONTEXT_LADDER)


def test_the_three_deployment_precisions_are_offered():
    assert set(PRECISIONS) == {"fp16", "int8", "4-bit"}


# ----------------------------------------------------------------------------------
# assessment
# ----------------------------------------------------------------------------------


def test_the_teacher_does_not_fit_either_target():
    """The reference point. 27B at 4-bit is ~15.3 GiB of weights before any cache — if
    the estimator ever said this fits, nothing else it says could be trusted."""
    result = assess(get_spec("teacher"))
    for target in TARGETS:
        assert result.best_precision_for(target.name) is None
    assert result.summary_row()["16gb_status"] == DOES_NOT_FIT


def test_the_students_fit_both_targets():
    for name in ("level2r", "level3"):
        row = assess(get_spec(name), name=name).summary_row()
        assert row["16gb_status"] == FIT
        assert row["12gb_status"] == FIT


def test_a_short_declared_context_truncates_the_ladder_and_says_so():
    """Reporting a 256K memory figure for a model that declares 4K would describe a
    model that cannot run there."""
    result = assess(get_spec("level2r"))
    entry = result.for_target(PRIMARY_TARGET.name, "fp16")
    assert [c.context_length for c in entry.contexts] == [4096]
    assert any("max_position_embeddings" in n for n in result.notes)


def test_fitting_at_4k_is_not_fitting_at_256k():
    """The distinction the whole module exists for."""
    result = assess(_long_context("level3"))
    entry = result.for_target(PRIMARY_TARGET.name, "fp16")
    assert len(entry.contexts) == len(CONTEXT_LADDER)
    short, long = entry.contexts[0], entry.contexts[-1]
    assert long.total_gib > short.total_gib
    assert long.kv_cache_gib > short.kv_cache_gib
    assert entry.max_fitting_context is not None


def test_memory_is_broken_into_its_real_components():
    result = assess(_long_context("level3"))
    cell = result.for_target(PRIMARY_TARGET.name, "fp16").contexts[-1]
    parts = (cell.weights_gib, cell.kv_cache_gib, cell.state_gib,
             cell.activations_gib, cell.overhead_gib)
    assert all(p >= 0 for p in parts)
    assert cell.total_gib == pytest.approx(sum(parts), rel=1e-6)


def test_lower_precision_never_costs_more():
    result = assess(_long_context("level3"))
    totals = [
        result.for_target(PRIMARY_TARGET.name, p).contexts[0].total_gib
        for p in ("fp16", "int8", "4-bit")
    ]
    assert totals == sorted(totals, reverse=True)


def test_kv_cache_grows_with_context():
    result = assess(_long_context("level3"))
    kv = [c.kv_cache_gib for c in result.for_target(PRIMARY_TARGET.name, "fp16").contexts]
    assert kv == sorted(kv)
    assert kv[-1] > kv[0]


def test_a_model_too_large_for_12gb_is_reported_per_target():
    """The targets must be able to disagree — that is the point of having two."""
    big = derive("level3", name="big", hidden_size=4096, num_attention_heads=64,
                 num_key_value_heads=8, intermediate_size=13824,
                 linear_num_key_heads=24, linear_num_value_heads=72,
                 num_hidden_layers=40)
    result = assess(big)
    statuses = {t.name: result.best_precision_for(t.name) for t in TARGETS}
    assert statuses[SECONDARY_TARGET.name] is None or statuses[PRIMARY_TARGET.name] is not None


def test_every_reported_number_is_labelled_an_estimate():
    payload = assess(get_spec("level3")).to_dict()
    assert "not a benchmark" in payload["estimate_disclaimer"]
    cell = payload["feasibility"][0]["contexts"][0]
    assert all(k.startswith("estimated_") for k in cell if k.endswith("_gib"))


def test_borderline_sits_between_fit_and_does_not_fit():
    from qwen_distill.analysis.deployment import _verdict

    assert _verdict(1.0, 13.56) == FIT
    assert _verdict(13.0, 13.56) == BORDERLINE
    assert _verdict(20.0, 13.56) == DOES_NOT_FIT


# ----------------------------------------------------------------------------------
# sweep
# ----------------------------------------------------------------------------------


def test_sweep_rejects_an_unservable_candidate_without_training_it():
    result = sweep({"level3": get_spec("level3"), "teacher": get_spec("teacher")})
    assert any("cannot be served" in f and "teacher" in f for f in result.findings)
    assert any("cannot be TRAINED" in f for f in result.findings)


def test_sweep_reports_training_memory_separately_from_inference():
    """At these sizes training is the binding constraint and inference is not; a sweep
    that showed only one would hide it."""
    result = sweep({"level3": get_spec("level3")})
    training = result.training_fits["level3"]["total_gib"]
    inference = result.assessments[0].for_target(
        PRIMARY_TARGET.name, "fp16"
    ).contexts[0].total_gib
    assert training > inference


def test_sweep_flags_candidates_whose_long_context_is_unknown():
    result = sweep({"level2r": get_spec("level2r")})
    assert any("UNKNOWN rather than good" in f for f in result.findings)


def test_sweep_renders_without_column_collision():
    """`DOES NOT FIT` is twelve characters and used to overflow into the next column."""
    rendered = sweep({"tiny": get_spec("level2r"), "teacher": get_spec("teacher")}).render()
    for line in rendered.splitlines():
        assert "GDOES" not in line and "FITDOES" not in line
    assert "DOES NOT FIT" in rendered


def test_sweep_is_ordered_as_given():
    names = ["level3", "level2r", "prototype"]
    result = sweep({n: get_spec(n) for n in names})
    assert [a.name for a in result.assessments] == names
