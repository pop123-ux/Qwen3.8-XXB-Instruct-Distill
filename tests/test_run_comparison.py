"""The comparison must refuse the delta it exists to refuse.

The central test is ``test_bpb_delta_across_corpora_is_refused``. Everything else here
protects it: if a future change makes an incomparable metric quietly comparable, this
file fails.
"""

from __future__ import annotations

import json

import pytest

from qwen_distill.analysis.compare import (
    COMPARABLE,
    CROSS_CORPUS_RULES,
    NOT_COMPARABLE,
    SHARED_BENCHMARK_PROTOCOL,
    CorpusDescriptor,
    MetricRule,
    RunFacts,
    compare_runs,
    corpus_from_manifest,
    load_run_facts,
)

PROCEDURAL = CorpusDescriptor(
    name="procedural byte-level", kind="procedural", total_bytes=8_000_000,
    validation_sha256="a" * 64,
    validation_split_rule="contiguous 5% tail of one generated text",
)
ENGLISH = CorpusDescriptor(
    name="gutenberg public-domain english", kind="natural_language",
    total_bytes=60_000_000, validation_sha256="b" * 64,
    validation_split_rule="8 whole books held out at document level",
)


def _level2(**overrides):
    metrics = {
        "validation_bits_per_byte": 1.270, "validation_loss": 0.8806,
        "train_bits_per_byte": 1.258, "run_wide_tokens_per_second": 2089.2,
        "parameters": 94_480_000, "degenerate_generation": True, "training_stable": True,
        "steps_to_plateau": 400, "epochs_seen": 4.1,
    }
    metrics.update(overrides)
    return RunFacts(
        name="t4_level2_100m_ckpt", corpus=PROCEDURAL, metrics=metrics,
        configuration={"parameters": 94_480_000, "sequence_length": 1024,
                       "precision": "fp16 autocast", "optimizer": "adamw"},
        status="COMPLETE",
    )


def _level2r(**overrides):
    metrics = {
        "validation_bits_per_byte": 2.104, "validation_loss": 1.459,
        "train_bits_per_byte": 2.031, "run_wide_tokens_per_second": 2071.4,
        "parameters": 94_480_000, "degenerate_generation": False, "training_stable": True,
        "steps_to_plateau": 3400, "epochs_seen": 0.9,
    }
    metrics.update(overrides)
    return RunFacts(
        name="t4_level2r_100m_real_english", corpus=ENGLISH, metrics=metrics,
        configuration={"parameters": 94_480_000, "sequence_length": 1024,
                       "precision": "fp16 autocast", "optimizer": "adamw"},
        status="COMPLETE",
    )


# ----------------------------------------------------------------------------------
# the refusal
# ----------------------------------------------------------------------------------


def test_bpb_delta_across_corpora_is_refused():
    """1.270 and 2.104 are not measurements of the same quantity. No delta exists.

    Level 2's validation text was procedural bytes with a low entropy floor by
    construction; Level 2R's is real English. ``2.104 - 1.270 = +0.834`` would read as a
    regression and would be an artefact of the corpora.
    """
    comparison = compare_runs(_level2(), _level2r())
    bpb = next(m for m in comparison.metrics if m.rule.key == "validation_bits_per_byte")

    assert bpb.left == 1.270 and bpb.right == 2.104, "both values must still be shown"
    assert bpb.delta is None, "a delta was computed across incomparable corpora"
    assert bpb.ratio is None
    assert bpb.verdict == NOT_COMPARABLE
    assert bpb.rule.remedy


def test_no_delta_is_computed_for_any_data_scoped_metric():
    comparison = compare_runs(_level2(), _level2r())
    for entry in comparison.metrics:
        if entry.rule.across_corpus_change != COMPARABLE:
            assert entry.delta is None, f"{entry.rule.key} produced a delta it should not"


def test_process_metrics_still_compare():
    """A corpus change does not invalidate throughput: same shapes, same hardware."""
    comparison = compare_runs(_level2(), _level2r())
    throughput = next(
        m for m in comparison.metrics if m.rule.key == "run_wide_tokens_per_second"
    )
    assert throughput.verdict == COMPARABLE
    assert throughput.delta == pytest.approx(-17.8, abs=0.01)


def test_rendered_report_shows_both_values_and_says_refused():
    rendered = compare_runs(_level2(), _level2r()).render()
    assert "1.27" in rendered and "2.104" in rendered
    assert "REFUSED" in rendered
    assert "+0.834" not in rendered and "0.8340" not in rendered


def test_no_overall_winner_is_declared():
    payload = compare_runs(_level2(), _level2r()).to_dict()
    assert "No overall winner" in payload["headline_refusal"]
    assert "winner" not in compare_runs(_level2(), _level2r()).render().lower().replace(
        "no overall winner is declared", ""
    )


def test_shared_benchmark_protocol_is_offered():
    """Refusing a comparison without saying how to obtain a valid one is an obstacle,
    not an analysis."""
    payload = compare_runs(_level2(), _level2r()).to_dict()
    assert payload["shared_benchmark_protocol"] == list(SHARED_BENCHMARK_PROTOCOL)
    assert "SHARED_BENCHMARK_PROTOCOL" in compare_runs(_level2(), _level2r()).render() or \
        "shared held-out corpus" in compare_runs(_level2(), _level2r()).render()


# ----------------------------------------------------------------------------------
# benchmark identity
# ----------------------------------------------------------------------------------


def test_different_validation_digests_mean_different_benchmarks():
    assert compare_runs(_level2(), _level2r()).same_benchmark == "NO"


def test_identical_validation_digest_licenses_the_comparison():
    left, right = _level2(), _level2r()
    right.corpus = PROCEDURAL
    assert compare_runs(left, right).same_benchmark == "YES"
    assert compare_runs(left, right).comparable_benchmark


def test_missing_digest_is_unknown_not_no():
    """An unrecorded digest is a reason to withhold the comparison, not evidence about
    it either way."""
    left, right = _level2(), _level2r()
    right.corpus = CorpusDescriptor(name="unrecorded", kind="unknown")
    comparison = compare_runs(left, right)
    assert comparison.same_benchmark == "UNKNOWN"
    assert not comparison.comparable_benchmark
    assert any("cannot be established" in f for f in comparison.findings)


# ----------------------------------------------------------------------------------
# controlling the experiment
# ----------------------------------------------------------------------------------


def test_a_second_changed_variable_confounds_everything():
    """Level 2R's whole design is that only the corpus moves. If something else moved,
    even the throughput comparison is confounded and must say so."""
    right = _level2r()
    right.configuration["sequence_length"] = 2048
    comparison = compare_runs(_level2(), right)
    assert not comparison.controlled_experiment
    assert any("not a controlled comparison" in f for f in comparison.findings)
    assert comparison.changed[0]["key"] == "sequence_length"


def test_held_constant_keys_are_reported():
    comparison = compare_runs(_level2(), _level2r())
    assert comparison.controlled_experiment
    assert "sequence_length" in comparison.controlled
    assert "precision" in comparison.controlled


# ----------------------------------------------------------------------------------
# capability
# ----------------------------------------------------------------------------------


def test_generation_degeneracy_is_the_comparison_that_matters():
    comparison = compare_runs(_level2(), _level2r())
    assert any("speaks to language capability" in f for f in comparison.findings)


def test_both_degenerate_is_called_out():
    comparison = compare_runs(_level2(), _level2r(degenerate_generation=True))
    assert any("BOTH runs generate degenerate text" in f for f in comparison.findings)


def test_unmeasured_degeneracy_is_a_finding():
    right = _level2r()
    right.metrics.pop("degenerate_generation")
    comparison = compare_runs(_level2(), right)
    assert any("has not been measured on both sides" in f for f in comparison.findings)


# ----------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------


def test_reads_the_published_level2_result():
    facts = load_run_facts("experiments/runs/t4_level2_100m_ckpt_complete")
    assert facts.metrics["validation_bits_per_byte"] == 1.27
    assert facts.metrics["run_wide_tokens_per_second"] == 2089.2
    assert facts.metrics["parameters"] == 94_480_000
    # "and and and and ..." is degenerate, and the loader must say so.
    assert facts.metrics["degenerate_generation"] is True


def test_a_running_run_loads_with_unknowns(tmp_path):
    root = tmp_path / "running"
    root.mkdir()
    (root / "metrics.jsonl").write_text(
        json.dumps({"step": 75, "loss": 1.704, "elapsed_s": 600.0,
                    "tokens_seen": 1_228_800, "tokens_per_second": 2048.0,
                    "bits_per_byte": 2.458}) + "\n",
        encoding="utf-8",
    )
    facts = load_run_facts(root)
    assert facts.metrics.get("validation_bits_per_byte") is None
    comparison = compare_runs(_level2(), facts)
    assert "UNKNOWN" in comparison.render()


def test_corpus_manifest_supplies_the_validation_digest(tmp_path):
    manifest = tmp_path / "corpus_manifest.json"
    manifest.write_text(json.dumps({
        "name": "level2r", "total_bytes": 61_000_000,
        "train_sha256": "c" * 64, "validation_sha256": "d" * 64,
        "split_rule": "8 whole books held out",
    }), encoding="utf-8")
    descriptor = corpus_from_manifest(manifest)
    assert descriptor.validation_sha256 == "d" * 64
    assert descriptor.kind == "natural_language"


# ----------------------------------------------------------------------------------
# the registry itself
# ----------------------------------------------------------------------------------


def test_every_non_comparable_metric_names_a_remedy():
    for rule in CROSS_CORPUS_RULES:
        if rule.across_corpus_change != COMPARABLE:
            assert rule.remedy, f"{rule.key} refuses a comparison without offering one"


def test_a_rule_without_a_remedy_is_rejected_at_construction():
    with pytest.raises(ValueError, match="must name its remedy"):
        MetricRule(key="x", label="x", scope="data",
                   across_corpus_change=NOT_COMPARABLE, reason="because")


def test_validation_bpb_is_classified_not_comparable():
    """The one classification this project must never get wrong."""
    rule = next(r for r in CROSS_CORPUS_RULES if r.key == "validation_bits_per_byte")
    assert rule.across_corpus_change == NOT_COMPARABLE
    assert "entropy" in rule.reason
