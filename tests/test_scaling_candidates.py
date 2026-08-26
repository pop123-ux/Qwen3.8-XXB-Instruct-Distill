"""Scaling candidates, and the guards that keep the table honest.

``test_training_verdict_vocabulary_has_not_changed`` is the important one. ``fit.py``
carries two verdict vocabularies — ``FITS``/``TIGHT``/``DOES NOT FIT`` for inference and
``PLAUSIBLE``/``TIGHT``/``NOT FEASIBLE`` for training. Matching the wrong one made every
candidate report "no fit" on every device, with no error raised. This pins the string.
"""

from __future__ import annotations

import pytest

from qwen_distill.analysis.scaling import (
    ACCEPTABLE_VERDICT,
    DEVICE_BUDGETS,
    LEVEL2_CONFIG,
    LEVEL2_SPEC,
    LEVEL2_TOKENS_PER_SECOND,
    SCALE_CLASSES,
    TIGHT_VERDICT,
    TrainingConfig,
    build_candidates,
    build_matrix,
    default_sweep,
    evaluate_candidate,
    extrapolated_tokens_per_second,
    scaled_spec,
)
from qwen_distill.architecture.params import count_parameters
from qwen_distill.diagnostics.fit import estimate_training_memory

# ----------------------------------------------------------------------------------
# the guard
# ----------------------------------------------------------------------------------


def test_training_verdict_vocabulary_has_not_changed():
    """If fit.py renames its training verdicts, this fails instead of the table emptying."""
    fit = estimate_training_memory(
        LEVEL2_SPEC, 40.0, strategy="full", optimizer="adamw",
        sequence_length=512, batch_size=1, gradient_checkpointing=True, precision="fp16",
    )
    assert fit.verdict == ACCEPTABLE_VERDICT, (
        f"estimate_training_memory now returns {fit.verdict!r} for an obviously feasible "
        f"configuration; ACCEPTABLE_VERDICT is stale"
    )
    cramped = estimate_training_memory(
        LEVEL2_SPEC, 3.9, strategy="full", optimizer="adamw",
        sequence_length=1024, batch_size=4, gradient_checkpointing=True, precision="fp16",
    )
    assert cramped.verdict in {TIGHT_VERDICT, "NOT FEASIBLE"}


def test_no_second_estimator_is_defined():
    """Memory arithmetic must come from fit.py, not be reimplemented here."""
    import inspect

    from qwen_distill.analysis import scaling

    source = inspect.getsource(scaling)
    assert "estimate_training_memory" in source
    for invented in ("GIB = ", "bytes_per_param", "def _activation_bytes"):
        assert invented not in source, f"scaling.py appears to compute memory itself: {invented}"


# ----------------------------------------------------------------------------------
# the shape rule
# ----------------------------------------------------------------------------------


def test_the_rule_reproduces_level2():
    """The ladder starts from a measured rung. If the rule cannot rebuild Level 2, it is
    not the same family of architectures."""
    spec = LEVEL2_SPEC
    assert spec.hidden_size == 640
    assert spec.num_hidden_layers == 16
    assert spec.intermediate_size == 2176
    assert spec.num_attention_heads == 10
    assert spec.num_key_value_heads == 2
    assert spec.linear_num_key_heads == 4
    assert spec.linear_num_value_heads == 12
    assert spec.num_linear_attention_layers == 12
    assert spec.num_full_attention_layers == 4
    # Published as 94,480,000 (rounded); measured exactly here.
    assert count_parameters(spec).total == 94_476_448


def test_parameter_counts_are_measured_not_assumed():
    """The class labels are targets. The counts are outputs, and they do not land on
    round numbers — which is the point."""
    counts = {spec.name: count_parameters(spec).total for spec in build_candidates()}
    assert len(counts) == 2 * len(SCALE_CLASSES)
    for name, measured in counts.items():
        target = int(name.split("M_", 1)[0]) * 1_000_000
        assert measured % 1_000_000 != 0, f"{name} landed on a suspiciously round number"
        assert 0.85 <= measured / target <= 1.15, f"{name} is not in its class"


def test_each_class_is_bracketed_by_its_candidates():
    """One candidate wider and shallower, one narrower and deeper."""
    for scale in SCALE_CLASSES:
        specs = [
            scaled_spec(h, layers, name=f"{scale.label}_h{h}_L{layers}")
            for h, layers in scale.shapes
        ]
        assert len(specs) == 2
        wide, deep = sorted(specs, key=lambda s: s.num_hidden_layers)
        assert wide.hidden_size > deep.hidden_size
        assert wide.num_hidden_layers < deep.num_hidden_layers


def test_multi_query_collapse_is_rejected_not_fudged():
    """1088 hidden gives 17 attention heads — prime, so GQA would collapse to MQA. That
    is a different architecture, not a rounding difference."""
    with pytest.raises(ValueError, match="multi-query attention"):
        scaled_spec(1088, 16, name="bad")


def test_layer_count_must_keep_the_layout():
    """With interval 4, a layer count that is not a multiple of 4 leaves the final layers
    linear — a changed layout, not a scaled one."""
    with pytest.raises(ValueError, match="not a multiple of"):
        scaled_spec(1024, 18, name="bad")


def test_hidden_size_must_divide_by_head_dim():
    with pytest.raises(ValueError, match="not a multiple of head_dim"):
        scaled_spec(1000, 16, name="bad")


# ----------------------------------------------------------------------------------
# devices
# ----------------------------------------------------------------------------------


def test_nominal_gb_is_not_treated_as_gib():
    """A '16 GB' T4 reports 14.56 GiB. Planning against 16 is how a configuration that
    'obviously fits' OOMs."""
    t4 = next(d for d in DEVICE_BUDGETS if d.nominal_gb == 16)
    assert t4.total_gib == 14.56
    assert t4.total_gib < t4.nominal_gb
    assert t4.usable_gib == pytest.approx(13.56)
    assert "measured" in t4.source


def test_unmeasured_budgets_say_so():
    """Only the T4 figure was measured by this project. The rest must not claim to be."""
    for device in DEVICE_BUDGETS:
        if device.nominal_gb != 16:
            assert "not measured" in device.source


def test_all_four_budgets_are_present():
    assert sorted(d.nominal_gb for d in DEVICE_BUDGETS) == [12, 16, 24, 48]


# ----------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------


def test_tight_configurations_are_not_recommended():
    """A configuration with under 1 GiB spare dies the moment anything else touches the
    card. It must never be returned as 'the best that fits'."""
    for spec in build_candidates():
        for device in DEVICE_BUDGETS:
            result = evaluate_candidate(spec, device)
            if result.fit is not None and result.best is not None:
                assert result.fit.verdict == ACCEPTABLE_VERDICT
                assert result.fit.headroom_gib >= 1.0


def test_every_attempt_is_recorded_so_a_verdict_can_be_argued_with():
    result = evaluate_candidate(build_candidates()[0], DEVICE_BUDGETS[0])
    assert len(result.attempts) == len(default_sweep())
    assert all("verdict" in a and "total_gib" in a for a in result.attempts)


def test_binding_term_is_reported():
    """Shrinking the batch does nothing if the optimizer state is what does not fit."""
    result = evaluate_candidate(build_candidates()[0], DEVICE_BUDGETS[0])
    assert result.binding_term in {
        "base weights", "gradients", "optimizer state", "DeltaNet activations",
        "attention activations", "other activations", "logits + fp32 loss copies",
    }


def test_deltanet_domination_is_surfaced():
    """The term whose absence caused the Level-2 OOM dominates at these sizes, and the
    report has to say so rather than leaving it in a JSON field."""
    matrix = build_matrix()
    assert any("DeltaNet activations" in f for f in matrix.findings)
    assert "DeltaNet activations" in matrix.render()


def test_matrix_is_anchored_on_a_run_that_happened():
    """The one row that can be checked against reality."""
    matrix = build_matrix()
    assert matrix.anchor is not None
    assert matrix.anchor.sequence_length == LEVEL2_CONFIG.sequence_length
    assert matrix.anchor.batch_size == LEVEL2_CONFIG.batch_size
    # Level 2 completed 2000/2000 steps on this configuration without an OOM, so an
    # estimate calling it infeasible would be a falsified estimator.
    assert matrix.anchor.verdict == ACCEPTABLE_VERDICT
    assert "ANCHOR" in matrix.render()


def test_matrix_covers_every_candidate_and_device():
    matrix = build_matrix()
    assert len(matrix.fits) == 2 * len(SCALE_CLASSES) * len(DEVICE_BUDGETS)
    assert len(matrix.for_device("16 GB")) == 2 * len(SCALE_CLASSES)


def test_capped_sweep_is_disclosed():
    """'2048x8' at 48 GB means the sweep ran out, not the card."""
    matrix = build_matrix()
    assert any("sweep is capped" in f for f in matrix.findings)


def test_report_never_calls_an_estimate_a_measurement():
    rendered = build_matrix().render()
    assert "FEASIBILITY estimates, not measurements" in rendered
    assert "validated at ~100M only" in " ".join(build_matrix().findings)


def test_larger_models_need_more_memory_at_a_fixed_config():
    """Monotonicity — a sanity check on the whole pipeline."""
    config = TrainingConfig(sequence_length=1024, batch_size=4)
    device = next(d for d in DEVICE_BUDGETS if d.nominal_gb == 48)
    totals = []
    for spec in build_candidates():
        fit = estimate_training_memory(
            spec, device.usable_gib, strategy="full", optimizer=config.optimizer,
            sequence_length=config.sequence_length, batch_size=config.batch_size,
            gradient_checkpointing=True, precision=config.precision,
        )
        totals.append((count_parameters(spec).total, fit.total_gib))
    by_params = sorted(totals)
    assert [t for _, t in by_params] == sorted(t for _, t in by_params)


# ----------------------------------------------------------------------------------
# throughput extrapolation
# ----------------------------------------------------------------------------------


def test_throughput_extrapolation_is_labelled_unvalidated():
    """One measured point cannot establish a scaling law, and the output must say so
    every time rather than relying on the reader remembering."""
    estimate = extrapolated_tokens_per_second(build_candidates()[0])
    assert estimate["status"] == "UNVALIDATED EXTRAPOLATION"
    assert "ONE measured point" in estimate["caveat"]
    assert estimate["anchor"]["measured_tokens_per_second"] == LEVEL2_TOKENS_PER_SECOND
    assert estimate["anchor"]["hardware"] == "Tesla T4"


def test_bigger_models_extrapolate_slower():
    rates = [
        extrapolated_tokens_per_second(spec)["extrapolated_tokens_per_second"]
        for spec in build_candidates()
    ]
    assert rates == sorted(rates, reverse=True)
    assert all(rate < LEVEL2_TOKENS_PER_SECOND for rate in rates)


def test_level2_extrapolates_to_its_own_measurement():
    """The anchor must reproduce itself: ratio 1.0, rate 2,089.2."""
    estimate = extrapolated_tokens_per_second(LEVEL2_SPEC)
    assert estimate["flops_ratio_vs_level2"] == pytest.approx(1.0)
    assert estimate["extrapolated_tokens_per_second"] == pytest.approx(
        LEVEL2_TOKENS_PER_SECOND, rel=1e-3
    )
