"""Regression tests for the training memory estimator and its calibration.

Two things this suite exists to prevent, both previously real:

* **A single "calibration factor".** The old ~2.85 divided allocator *reserve* by
  *modelled tensors with overhead zeroed* — different quantities. Applying it as a
  multiplier would have baked the confusion into every future estimate. So the tests
  assert the two ratios stay separate and that corrections are named per term.

* **A right total made of wrong parts.** AMP holds fp32 weights and gradients while
  computing in fp16; pure-bf16 training holds bf16 weights with an fp32 master. Both
  land at 16 bytes/parameter for AdamW, so a total cannot distinguish them — but a
  measurement can, and would flag two components as wrong.

All CPU, no GPU, no network.
"""

from __future__ import annotations

import json

import pytest

from qwen_distill.architecture.spec import HybridArchSpec
from qwen_distill.diagnostics.calibrate import (
    ComponentResidual,
    calibrate_training_run,
)
from qwen_distill.diagnostics.fit import (
    GIB,
    LOSS_PATH_FP32_LOGIT_COPIES,
    OPTIMIZER_MOMENT_BYTES,
    PRECISION_SCHEMES,
    estimate_training_memory,
    resolve_precision_scheme,
)


def spec(**overrides) -> HybridArchSpec:
    base = dict(
        name="probe", hidden_size=512, num_hidden_layers=8, intermediate_size=1408,
        vocab_size=4096, num_attention_heads=8, num_key_value_heads=2, head_dim=64,
        linear_num_key_heads=4, linear_num_value_heads=12, linear_key_head_dim=64,
        linear_value_head_dim=64, full_attention_interval=4, max_position_embeddings=8192,
    )
    base.update(overrides)
    return HybridArchSpec(**base)


def estimate(**kwargs):
    defaults = dict(strategy="full", optimizer="adamw", sequence_length=1024, batch_size=1)
    defaults.update(kwargs)
    return estimate_training_memory(spec(), 40.0, **defaults)


# --- precision schemes ----------------------------------------------------
def test_amp_keeps_weights_and_gradients_in_fp32():
    """autocast casts the compute, not the parameters; modelling weights at 2 B/param
    would halve a term the probe measures directly."""
    scheme = PRECISION_SCHEMES["fp16"]
    assert scheme.weight_bytes == 4.0
    assert scheme.gradient_bytes == 4.0
    assert scheme.master_copy_bytes == 0.0
    assert scheme.activation_bytes == 2.0


def test_pure_bf16_holds_bf16_weights_with_an_fp32_master():
    scheme = PRECISION_SCHEMES["pure_bf16"]
    assert scheme.weight_bytes == 2.0
    assert scheme.gradient_bytes == 2.0
    assert scheme.master_copy_bytes == 4.0


def test_amp_and_pure_bf16_agree_on_the_total_but_not_the_components():
    """Exactly the case a single total cannot distinguish and a measurement can."""
    amp = estimate(precision="fp16")
    pure = estimate(precision="pure_bf16")

    per_param = lambda f: (  # noqa: E731
        f.base_weights_gib + f.gradients_gib + f.optimizer_state_gib
    ) * GIB / f.total_parameters
    assert per_param(amp) == pytest.approx(16.0, abs=0.01)
    assert per_param(pure) == pytest.approx(16.0, abs=0.01)

    assert amp.base_weights_gib == pytest.approx(2 * pure.base_weights_gib)
    assert amp.optimizer_state_gib < pure.optimizer_state_gib


def test_fp32_doubles_only_the_activations_that_autocast_actually_affects():
    """DeltaNet force-upcasts to fp32 internally, so its term does not move with the
    scheme. That is the whole reason fp16 bought so much less than expected."""
    fp32 = estimate(precision="fp32")
    bf16 = estimate(precision="bf16")

    assert fp32.non_attention_activations_gib == pytest.approx(
        2 * bf16.non_attention_activations_gib
    )
    assert fp32.attention_activations_gib == pytest.approx(
        2 * bf16.attention_activations_gib
    )
    assert fp32.deltanet_activations_gib == pytest.approx(bf16.deltanet_activations_gib)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("fp16", "amp_fp16"),
        ("bf16", "amp_bf16"),
        ("fp32", "fp32"),
        ("pure_bf16", "pure_bf16"),
        ("fp16 (CPU)", "amp_fp16"),
        ("fp32 (CPU)", "fp32"),
        ("4-bit base + bf16 compute", "amp_bf16"),
    ],
)
def test_decorated_precision_labels_still_resolve(label, expected):
    """Labels arrive from configs and run summaries carrying decoration."""
    assert resolve_precision_scheme(label).name == expected


def test_an_unrecognised_precision_falls_back_rather_than_raising():
    """A feasibility check must still produce an estimate for a label it has not seen."""
    assert resolve_precision_scheme("some-future-format").name == "amp_bf16"
    assert resolve_precision_scheme(None).name == "amp_bf16"


# --- component attribution ------------------------------------------------
def test_weights_are_exactly_parameters_times_the_scheme_byte_width():
    """Weights are derivable, not modelled; a mismatch here is arithmetic, not judgement."""
    fit = estimate(precision="fp16")
    assert fit.base_weights_gib == pytest.approx(fit.total_parameters * 4.0 / GIB)


def test_gradients_are_reported_separately_from_optimizer_state():
    """The probe measures them at different stages, so the estimate must too."""
    fit = estimate(precision="fp16", optimizer="adamw")
    assert fit.gradients_gib == pytest.approx(fit.trainable_parameters * 4.0 / GIB)
    assert fit.optimizer_state_gib == pytest.approx(fit.trainable_parameters * 8.0 / GIB)


def test_optimizer_state_excludes_the_gradient():
    """Folding the gradient in was what made the estimate uncheckable component-wise."""
    fit = estimate(optimizer="sgd_no_momentum", precision="fp16")
    assert fit.optimizer_state_gib == 0.0
    assert fit.gradients_gib > 0.0


@pytest.mark.parametrize("optimizer", sorted(OPTIMIZER_MOMENT_BYTES))
def test_every_optimizer_produces_a_consistent_breakdown(optimizer):
    fit = estimate(optimizer=optimizer, precision="fp16")
    assert fit.optimizer_state_gib == pytest.approx(
        fit.trainable_parameters * OPTIMIZER_MOMENT_BYTES[optimizer] / GIB
    )
    assert fit.predicted_allocated_gib == pytest.approx(
        fit.base_weights_gib + fit.gradients_gib + fit.optimizer_state_gib
        + fit.activations_gib + fit.logits_gib
    )


def test_8bit_moments_are_a_quarter_of_fp32_moments():
    eight = estimate(optimizer="adamw_8bit")
    full = estimate(optimizer="adamw")
    assert eight.optimizer_state_gib == pytest.approx(full.optimizer_state_gib / 4)
    assert eight.gradients_gib == pytest.approx(full.gradients_gib), "gradients are unaffected"


# --- the loss path --------------------------------------------------------
def test_the_loss_path_holds_the_logits_three_times_over():
    """Verified against transformers: ForCausalLMLoss upcasts with `logits.float()`,
    and cross_entropy retains a second fp32 log-softmax buffer for backward."""
    assert LOSS_PATH_FP32_LOGIT_COPIES == 2
    fit = estimate(precision="fp16", sequence_length=1024, batch_size=2)
    tokens = 1024 * 2
    expected = tokens * 4096 * (2.0 + 4.0 * 2) / GIB
    assert fit.logits_gib == pytest.approx(expected)


def test_the_logits_term_dominates_at_a_large_vocabulary():
    """With a 248k vocab this term is gigabytes; understating it by 2.5x causes an OOM."""
    byte_level = estimate_training_memory(spec(vocab_size=256), 40.0, sequence_length=1024)
    large = estimate_training_memory(spec(vocab_size=248_000), 40.0, sequence_length=1024)
    assert large.logits_gib > 100 * byte_level.logits_gib
    assert large.logits_gib > large.base_weights_gib


def test_logits_scale_with_tokens_not_with_parameters():
    single = estimate(batch_size=1)
    double = estimate(batch_size=2)
    assert double.logits_gib == pytest.approx(2 * single.logits_gib)
    assert double.base_weights_gib == pytest.approx(single.base_weights_gib)


# --- the two predicted quantities -----------------------------------------
def test_predicted_allocated_excludes_overhead_and_reserved_includes_it():
    """These are the estimator's counterparts to the probe's two measurements."""
    fit = estimate(runtime_overhead_gib=1.2)
    assert fit.predicted_reserved_gib == pytest.approx(fit.predicted_allocated_gib + 1.2)
    assert fit.predicted_reserved_gib == pytest.approx(fit.total_gib)


def test_the_feasibility_total_stays_the_conservative_one():
    """A fit verdict must be judged on what the process occupies, not on live tensors."""
    fit = estimate()
    assert fit.total_gib == fit.predicted_reserved_gib
    assert fit.total_gib > fit.predicted_allocated_gib


def test_component_estimate_uses_the_probes_own_keys():
    """So the two can be compared term by term without a translation table drifting."""
    from qwen_distill.training.memory_probe import derive_components

    keys = set(estimate().component_estimate())
    profile_keys = {
        "weights_gib", "activations_gib", "optimizer_state_gib",
        "peak_allocated_gib", "peak_reserved_gib",
    }
    assert profile_keys <= keys
    assert callable(derive_components)


# --- qlora ----------------------------------------------------------------
def test_qlora_freezes_a_4bit_base_regardless_of_compute_precision():
    fit = estimate(strategy="qlora", precision="fp16")
    assert fit.base_weights_gib < estimate(strategy="full", precision="fp16").base_weights_gib / 7
    assert fit.trainable_parameters < fit.total_parameters / 100


def test_qlora_gradients_cover_only_the_adapter():
    qlora = estimate(strategy="qlora")
    full = estimate(strategy="full")
    assert qlora.gradients_gib < full.gradients_gib / 100


# --- calibration: per-component, never a global multiplier -----------------
def write_summary(tmp_path, *, measured_activations=3.10, cuda=True):
    summary = {
        "experiment": "probe-run",
        "memory": {
            "cuda_available": cuda,
            "device_name": "Tesla T4",
            "components": {
                "weights_gib": 0.352,
                "activations_gib": measured_activations,
                "optimizer_state_gib": 0.704,
            },
            "peak_allocated_gib": 4.90,
            "peak_reserved_gib": 6.85,
        },
        "analytical_estimate": {
            "base_weights_gib": 0.352, "optimizer_state_gib": 0.704,
            "activations_gib": 1.91, "logits_gib": 0.02,
            "predicted_allocated_gib": 3.33, "predicted_reserved_gib": 4.53,
        },
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_calibration_isolates_the_single_wrong_term(tmp_path):
    """The point of per-component residuals: name what to fix, not what to multiply."""
    calibration = calibrate_training_run(write_summary(tmp_path))

    by_name = {r.component: r for r in calibration.residuals}
    assert by_name["weights"].verdict == "OK"
    assert by_name["optimizer_state"].verdict == "OK"
    assert by_name["activations+logits"].verdict == "UNDERESTIMATED"
    assert calibration.worst.component in ("activations+logits", "peak_allocated", "peak_reserved")


def test_calibration_corrections_name_terms_and_forbid_a_global_multiplier(tmp_path):
    corrections = " ".join(calibrate_training_run(write_summary(tmp_path)).corrections())
    assert "activations+logits" in corrections
    assert "not the total" in corrections


def test_an_accurate_run_produces_no_corrections(tmp_path):
    path = write_summary(tmp_path, measured_activations=1.93)
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["memory"]["peak_allocated_gib"] = 3.33
    summary["memory"]["peak_reserved_gib"] = 4.53
    path.write_text(json.dumps(summary), encoding="utf-8")

    corrections = calibrate_training_run(path).corrections()

    assert corrections == ["every modelled term is within 15% of measurement"]


def test_a_cpu_run_cannot_calibrate_a_vram_model(tmp_path):
    """A CPU run has no VRAM measurement; saying so beats a table of zeros."""
    calibration = calibrate_training_run(write_summary(tmp_path, cuda=False))
    assert calibration.residuals == []
    assert "no CUDA measurement" in calibration.error


def test_a_missing_summary_is_an_error_not_an_empty_pass(tmp_path):
    calibration = calibrate_training_run(tmp_path / "absent.json")
    assert calibration.error
    assert calibration.residuals == []


def test_calibration_never_compares_gradients(tmp_path):
    """The probe's backward delta is net of freed activations, so it is not the gross
    gradient size the estimator models. Comparing them repeats the 2.85 category error."""
    components = {r.component for r in calibrate_training_run(write_summary(tmp_path)).residuals}
    assert not any("gradient" in c for c in components)


def test_a_residual_with_no_estimate_is_not_modelled_rather_than_infinite():
    residual = ComponentResidual(component="x", estimated_gib=0.0, measured_gib=2.0)
    assert residual.ratio is None
    assert residual.verdict == "NOT MODELLED"


def test_residual_verdicts_have_a_tolerance_band():
    assert ComponentResidual("x", 1.0, 1.10).verdict == "OK"
    assert ComponentResidual("x", 1.0, 1.30).verdict == "UNDERESTIMATED"
    assert ComponentResidual("x", 1.0, 0.70).verdict == "OVERESTIMATED"
