"""Diminishing returns, and the comparisons the data does not support.

The decision after Level 3 is not "is the bigger model better" but "is the improvement
worth the memory on a 16 GB card". These tests pin the three refusals that keep that
question honest: no invented capability score, no comparison across corpora, and no
verdict that assumes scale wins.
"""

from __future__ import annotations

import pytest

from qwen_distill.analysis.compare import CorpusDescriptor, RunFacts
from qwen_distill.analysis.returns import (
    MATERIAL_RELATIVE_IMPROVEMENT,
    build_step,
    research_summary,
)

ENGLISH = CorpusDescriptor(name="level2r", kind="natural_language",
                           validation_sha256="d" * 64)
OTHER = CorpusDescriptor(name="other", kind="natural_language",
                         validation_sha256="e" * 64)


def _run(name, params, bpb=None, corpus=ENGLISH, **extra):
    metrics = {"parameters": params}
    if bpb is not None:
        metrics["validation_bits_per_byte"] = bpb
    metrics.update(extra)
    return RunFacts(name=name, corpus=corpus, metrics=metrics,
                    configuration={}, status="COMPLETE")


BASELINE = _run("level2r", 94_476_448, 1.797)


def _step(bpb, *, params=236_237_488, corpus=ENGLISH, gib=1.41):
    return build_step(
        BASELINE, _run("level3", params, bpb, corpus=corpus),
        baseline_inference_gib=1.12, candidate_inference_gib=gib,
    )


# ----------------------------------------------------------------------------------
# the three verdicts
# ----------------------------------------------------------------------------------


def test_a_large_improvement_is_reported_with_its_memory_cost():
    step = _step(1.55)
    assert step.primary.material
    assert step.primary.relative_improvement == pytest.approx(0.1374, abs=1e-3)
    assert step.parameter_ratio == pytest.approx(2.50, abs=0.01)
    assert step.inference_memory_ratio == pytest.approx(1.259, abs=1e-3)
    assert any("cost of that improvement" in f for f in step.findings)


def test_a_tiny_improvement_is_called_diminishing_returns():
    """2.5x the parameters for 0.4% of the loss argues against scaling, and the report
    has to say so rather than record an improvement and stop."""
    step = _step(1.79)
    assert step.primary.relative_improvement > 0
    assert not step.primary.material
    assert any("diminishing returns" in f for f in step.findings)
    assert any("another architectural strategy" in f.lower() for f in step.findings)


def test_a_regression_is_a_real_result_not_a_reason_to_scale_further():
    step = _step(1.85)
    assert step.primary.relative_improvement < 0
    assert any("not better on the primary metric" in f for f in step.findings)
    assert any("argument against scaling further" in f for f in step.findings)


def test_nothing_ever_concludes_that_bigger_is_better_by_default():
    for bpb in (1.55, 1.79, 1.85):
        joined = " ".join(_step(bpb).findings).lower()
        assert "bigger is better" not in joined


# ----------------------------------------------------------------------------------
# refusals
# ----------------------------------------------------------------------------------


def test_a_different_validation_corpus_is_refused_not_subtracted():
    """Level 2 scored 1.270 on procedural text and Level 2R 1.797 on English. The
    difference is the corpora. A step whose runs disagree on the corpus must not produce
    a delta at all."""
    step = _step(1.55, corpus=OTHER)
    bpb = next(m for m in step.metrics if m.metric == "validation_bits_per_byte")
    assert not bpb.comparable
    assert bpb.absolute_change is None
    assert bpb.relative_improvement is None
    assert step.primary is None
    assert any("NOT comparable" in f for f in step.findings)
    assert any("shared held-out corpus" in f for f in step.findings)


def test_an_unrecorded_corpus_is_also_refused():
    blank = CorpusDescriptor(name="", kind="unknown")
    step = build_step(BASELINE, _run("x", 236_237_488, 1.55, corpus=blank))
    assert step.primary is None
    assert any("NOT comparable" in f for f in step.findings)


def test_a_metric_measured_on_only_one_side_stays_absent():
    step = build_step(BASELINE, _run("level3", 236_237_488))
    bpb = next(m for m in step.metrics if m.metric == "validation_bits_per_byte")
    assert not bpb.measured
    assert bpb.absolute_change is None
    assert any("nothing to compare" in f or "not available" in f for f in step.findings)


def test_no_universal_capability_score_is_computed():
    step = _step(1.55)
    payload = step.to_dict()
    assert "capability_score" not in payload
    assert all("score" not in m["metric"] for m in payload["metrics"])


def test_the_materiality_threshold_is_labelled_a_judgment():
    payload = _step(1.55).to_dict()
    assert payload["material_threshold"] == MATERIAL_RELATIVE_IMPROVEMENT
    assert "never repeated a seed" in payload["threshold_status"]


# ----------------------------------------------------------------------------------
# normalisation
# ----------------------------------------------------------------------------------


def test_improvement_per_doubling_normalises_different_step_sizes():
    """A 2.5x step and a 2x step are not comparable raw; per-doubling they are."""
    small = _step(1.70, params=int(94_476_448 * 2))
    large = _step(1.70, params=int(94_476_448 * 4))
    assert small.improvement_per_doubling() > large.improvement_per_doubling()


def test_per_doubling_is_not_offered_as_a_scaling_law():
    from qwen_distill.analysis import returns

    assert "not a scaling law" in returns.ScalingStep.improvement_per_doubling.__doc__


def test_a_non_scaling_step_has_no_per_doubling_value():
    step = build_step(BASELINE, _run("same-size", 94_476_448, 1.70))
    assert step.parameter_ratio == 1.0
    assert step.improvement_per_doubling() is None


# ----------------------------------------------------------------------------------
# the summary
# ----------------------------------------------------------------------------------


def test_the_summary_renders_missing_values_as_unknown():
    summary = research_summary(
        rungs=[{"name": "level3", "parameters": None, "status": "RUNNING"}],
        steps=[], open_questions=["Level 3 has not finished"],
    )
    rendered = summary.render()
    assert "UNKNOWN" in rendered
    assert "Level 3 has not finished" in rendered
    assert "ESTIMATED and are not" in rendered


def test_the_summary_states_it_invents_no_capability_score():
    payload = research_summary([], []).to_dict()
    assert "never assumed" in payload["conclusion_policy"]
