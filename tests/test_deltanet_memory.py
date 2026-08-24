"""Regression tests for the DeltaNet activation term and OOM reporting.

The Level-2 T4 run OOMed at ~24.8 GiB against a 4.53 GiB estimate. The estimator had no
DeltaNet term at all — it treated all 16 layers as generic transformer blocks scaling
with hidden_size and intermediate_size. In reality the 12 Gated DeltaNet layers held 66%
of every retained activation, because `transformers`' pure-torch reference kernel
force-upcasts to fp32 and runs a 63-iteration loop that clones and retains O(chunk^2)
tensors per chunk.

These tests pin the corrected model against the measurements it was derived from, and
pin the property that made the failure invisible: an estimate that says PLAUSIBLE for a
configuration that cannot run.
"""

from __future__ import annotations

import json

import pytest
from conftest import HAS_STACK, requires_stack

from qwen_distill.architecture.spec import HybridArchSpec
from qwen_distill.diagnostics.calibrate import calibrate_training_run
from qwen_distill.diagnostics.fit import (
    DELTANET_CHUNK_SIZE,
    DELTANET_LOOP_SUM_I_SQUARED,
    DELTANET_MEASUREMENT,
    DELTANET_SAFETY_MARGIN,
    deltanet_activation_bytes,
    estimate_training_memory,
)
from qwen_distill.training.config import ExperimentConfig
from qwen_distill.training.memory_probe import (
    STAGES,
    MemoryProfile,
    MemorySnapshot,
    OOMRecord,
    derive_components,
    is_oom,
    record_oom,
)

LEVEL2 = "configs/experiments/t4_level2_100m.yaml"
LEVEL2_CKPT = "configs/experiments/t4_level2_100m_ckpt.yaml"


def spec(**overrides) -> HybridArchSpec:
    base = dict(
        name="probe", hidden_size=640, num_hidden_layers=16, intermediate_size=2176,
        vocab_size=256, num_attention_heads=10, num_key_value_heads=2, head_dim=64,
        linear_num_key_heads=4, linear_num_value_heads=12, linear_key_head_dim=64,
        linear_value_head_dim=64, full_attention_interval=4, max_position_embeddings=4096,
    )
    base.update(overrides)
    return HybridArchSpec(**base)


# --- the term that was missing --------------------------------------------
def test_the_sequential_loop_constant_is_the_exact_sum_of_squares():
    """sum(i^2 for i in 1..63): the clones the reference implementation's loop retains."""
    assert DELTANET_CHUNK_SIZE == 64
    assert sum(i * i for i in range(1, DELTANET_CHUNK_SIZE)) == DELTANET_LOOP_SUM_I_SQUARED
    assert DELTANET_LOOP_SUM_I_SQUARED == 85_344


def test_the_loop_term_alone_exceeds_a_whole_transformer_activation_budget():
    """64 KB per token per layer, before any other tensor — which is why a model that
    looked like it needed 1.91 GiB of activations actually needed 21."""
    heads = 12
    per_token = heads * DELTANET_LOOP_SUM_I_SQUARED * 4 / DELTANET_CHUNK_SIZE
    assert per_token == pytest.approx(64_008)


@pytest.mark.parametrize("case", DELTANET_MEASUREMENT["measurements"])
def test_the_model_matches_every_measured_configuration(case):
    """Six points spanning 4x in head count and 4x in head dimension. One point could be
    fitted by anything; six make it a model."""
    probe = spec(
        num_hidden_layers=4,
        linear_num_value_heads=case["v_heads"],
        linear_num_key_heads=max(1, case["v_heads"] // 3),
        linear_key_head_dim=case["head_dim"],
        linear_value_head_dim=case["head_dim"],
    )
    tokens = 1024
    predicted = deltanet_activation_bytes(probe, tokens) / (
        tokens * probe.num_linear_attention_layers
    )
    assert predicted == pytest.approx(case["bytes_per_token_per_layer"], rel=0.15)


def test_the_model_never_underestimates_a_measured_configuration():
    """A feasibility check that underestimates is the failure being fixed here."""
    for case in DELTANET_MEASUREMENT["measurements"]:
        probe = spec(
            num_hidden_layers=4,
            linear_num_value_heads=case["v_heads"],
            linear_num_key_heads=max(1, case["v_heads"] // 3),
            linear_key_head_dim=case["head_dim"],
            linear_value_head_dim=case["head_dim"],
        )
        predicted = deltanet_activation_bytes(probe, 1024) / (1024 * 3)
        assert predicted >= case["bytes_per_token_per_layer"], (
            f"underestimates at v_heads={case['v_heads']} head_dim={case['head_dim']}"
        )


def test_the_safety_margin_is_documented_and_applied():
    assert DELTANET_SAFETY_MARGIN > 1.0
    probe = spec()
    with_margin = deltanet_activation_bytes(probe, 1024)
    heads, dim = 12, 64
    raw = (heads * DELTANET_LOOP_SUM_I_SQUARED * 4 / DELTANET_CHUNK_SIZE
           + heads * 4 * (292.0 + 36.68 * dim))
    assert with_margin == pytest.approx(
        raw * DELTANET_SAFETY_MARGIN * 1024 * probe.num_linear_attention_layers, rel=1e-3
    )


def test_the_measurement_records_its_own_provenance():
    """A fitted number with no provenance cannot be refuted, only believed."""
    assert "saved_tensors_hooks" in DELTANET_MEASUREMENT["how"]
    assert "torch_chunk_gated_delta_rule" in DELTANET_MEASUREMENT["implementation"]
    assert DELTANET_MEASUREMENT["transformers_version"]
    assert len(DELTANET_MEASUREMENT["measurements"]) >= 6
    assert "fla" in DELTANET_MEASUREMENT["caveat"]


def test_a_model_with_no_deltanet_layers_has_no_deltanet_term():
    assert deltanet_activation_bytes(spec(full_attention_interval=1), 1024) == 0


def test_the_term_scales_linearly_with_tokens():
    probe = spec()
    assert deltanet_activation_bytes(probe, 2048) == pytest.approx(
        2 * deltanet_activation_bytes(probe, 1024), rel=1e-6
    )


def test_the_term_is_fp32_regardless_of_autocast():
    """torch_chunk_gated_delta_rule casts to float32 on entry, so fp16 does not help it."""
    fp16 = estimate_training_memory(spec(), 40.0, precision="fp16", sequence_length=1024,
                                    batch_size=1, gradient_checkpointing=False)
    fp32 = estimate_training_memory(spec(), 40.0, precision="fp32", sequence_length=1024,
                                    batch_size=1, gradient_checkpointing=False)
    assert fp16.deltanet_activations_gib == pytest.approx(fp32.deltanet_activations_gib)
    assert fp16.non_attention_activations_gib < fp32.non_attention_activations_gib


# --- the estimate must now reject the configuration that failed ------------
@requires_stack
def test_the_failed_level2_configuration_is_now_rejected():
    """The whole point: this exact config reported PLAUSIBLE at 4.53 GiB and then OOMed."""
    config = ExperimentConfig.load(LEVEL2)
    fit = estimate_training_memory(
        config.model.resolve_spec(), 13.56, strategy="full", optimizer="adamw",
        precision="fp16", sequence_length=1024, batch_size=8,
        gradient_checkpointing=False,
    )
    assert fit.verdict == "NOT FEASIBLE"
    assert fit.total_gib > 14.56, "must exceed the whole card, not merely the budget"
    assert fit.deltanet_activations_gib > fit.total_gib * 0.5, (
        "DeltaNet must be identified as the dominant term, not buried in a total"
    )


@requires_stack
def test_the_corrected_estimate_lands_near_the_measured_demand():
    """Measured demand was ~24.8 GiB: 23.1 GiB of activations extrapolated from the
    saved-tensor probe, plus 1.41 GiB of weights, gradients and optimizer state."""
    config = ExperimentConfig.load(LEVEL2)
    fit = estimate_training_memory(
        config.model.resolve_spec(), 13.56, strategy="full", optimizer="adamw",
        precision="fp16", sequence_length=1024, batch_size=8,
        gradient_checkpointing=False,
    )
    assert 20.0 < fit.total_gib < 30.0, f"estimated {fit.total_gib:.1f} GiB"


@requires_stack
def test_the_revised_configuration_is_feasible_and_keeps_the_experiment_intact():
    original = ExperimentConfig.load(LEVEL2)
    revised = ExperimentConfig.load(LEVEL2_CKPT)

    # Same experiment: architecture, sequence length and effective batch unchanged.
    assert revised.model.architecture == original.model.architecture
    assert revised.data.max_sequence_length == original.data.max_sequence_length
    effective = lambda c: c.training.batch_size * c.training.gradient_accumulation_steps  # noqa: E731
    assert effective(revised) == effective(original) == 16

    # Changed, deliberately.
    assert revised.training.gradient_checkpointing is True
    assert original.training.gradient_checkpointing is False

    fit = estimate_training_memory(
        revised.model.resolve_spec(), 13.56, strategy="full",
        optimizer=revised.training.optimizer, precision=revised.training.precision,
        sequence_length=revised.data.max_sequence_length,
        batch_size=revised.training.batch_size,
        gradient_checkpointing=True,
    )
    assert fit.feasible
    assert fit.total_gib < 13.56


@requires_stack
def test_the_failed_baseline_is_preserved_unchanged():
    """It is a real result — what 94.48M costs on a T4 without checkpointing."""
    config = ExperimentConfig.load(LEVEL2)
    assert config.training.batch_size == 8
    assert config.training.gradient_accumulation_steps == 2
    assert config.training.gradient_checkpointing is False


def test_the_failure_is_documented_as_an_artifact():
    from pathlib import Path

    directory = Path("experiments/runs/t4_level2_100m_oom_2026-08-24")
    record = json.loads((directory / "FAILURE.json").read_text(encoding="utf-8"))

    assert record["outcome"].startswith("FAILED")
    assert record["failure"]["steps_completed"] == 0
    assert record["estimate_at_time_of_run_gib"]["total"] == 4.53
    assert record["measured_demand_gib"]["total_demand_gib"] > 20
    assert "scaled_dot_product_attention" in record["failure"]["raised_in"]
    assert "NOT what caused it" in record["failure"]["note"]
    assert (directory / "README.md").is_file()


@requires_stack
def test_gradient_checkpointing_still_attributes_memory_to_deltanet():
    """Reporting the recompute as generic 'non-attention' would hide the dominant term."""
    fit = estimate_training_memory(
        spec(), 13.56, precision="fp16", sequence_length=1024, batch_size=4,
        gradient_checkpointing=True,
    )
    assert fit.deltanet_activations_gib > 0
    assert fit.deltanet_activations_gib > fit.non_attention_activations_gib


# --- OOM as data ----------------------------------------------------------
def test_the_new_measurement_stages_are_present_and_ordered():
    for stage in ("after_model_to_device", "after_input_allocation", "after_loss"):
        assert stage in STAGES
    assert STAGES.index("after_model_creation") < STAGES.index("after_model_to_device")
    assert STAGES.index("after_input_allocation") < STAGES.index("after_forward")
    assert STAGES.index("after_forward") < STAGES.index("after_loss")
    assert STAGES.index("after_loss") < STAGES.index("after_backward")


def test_the_input_batch_is_not_counted_as_activations():
    def snap(stage, allocated):
        return MemorySnapshot(stage=stage, allocated_gib=allocated, reserved_gib=allocated,
                              max_allocated_gib=allocated, max_reserved_gib=allocated)

    profile = MemoryProfile(cuda_available=True, snapshots=[
        snap("baseline", 0.3), snap("after_model_creation", 0.3),
        snap("after_model_to_device", 1.0), snap("after_input_allocation", 1.1),
        snap("after_forward", 4.0), snap("after_loss", 4.2),
    ])
    components = derive_components(profile)

    assert components["weights_gib"] == pytest.approx(0.7)
    assert components["input_batch_gib"] == pytest.approx(0.1)
    assert components["activations_gib"] == pytest.approx(2.9), "measured from the input stage"
    assert components["loss_path_gib"] == pytest.approx(0.2)


def test_an_oom_is_recognised_by_type_and_by_message():
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not is_oom(RuntimeError("shape mismatch"))
    assert not is_oom(ValueError("out of memory"))


def test_an_oom_record_states_a_lower_bound_not_a_point_estimate():
    """The allocation failed, so the requirement exceeds what was held. Reporting the
    held figure as 'the requirement' would understate it again."""
    record = OOMRecord(phase="forward pass", allocated_at_failure_gib=14.16,
                       estimated_total_gib=4.53, total_vram_gib=14.56)

    assert record.lower_bound_gib == 14.16
    payload = record.to_dict()
    assert payload["estimate_underestimated_by_at_least"] == pytest.approx(14.16 / 4.53)
    assert "AT LEAST" in record.render()


def test_recording_an_oom_without_cuda_does_not_fabricate_numbers():
    record = record_oom(MemoryProfile(cuda_available=False), "forward pass",
                        RuntimeError("CUDA out of memory"))
    assert record.allocated_at_failure_gib is None
    assert record.lower_bound_gib is None


@pytest.mark.skipif(not HAS_STACK, reason="requires torch and transformers")
def test_calibration_turns_an_oom_run_into_a_bound(tmp_path):
    summary = {
        "experiment": "t4_level2_100m", "outcome": "OOM",
        "oom": {"phase": "forward pass", "total_vram_gib": 14.56,
                "allocated_at_failure_gib": 14.16, "reserved_at_failure_gib": 14.42,
                "estimated_total_gib": 4.53, "configuration": {"batch_size": 8}},
        "memory": {"cuda_available": True, "device_name": "Tesla T4", "components": {},
                   "peak_allocated_gib": 14.16, "peak_reserved_gib": 14.42},
        "analytical_estimate": {"total_gib": 4.53, "predicted_allocated_gib": 3.33,
                                "predicted_reserved_gib": 4.53},
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    calibration = calibrate_training_run(path)

    assert calibration.oom is not None
    assert calibration.oom.phase == "forward pass"
    assert calibration.oom.underestimate_factor == pytest.approx(14.16 / 4.53)
    corrections = " ".join(calibration.corrections())
    assert "OOMed during the forward pass" in corrections
    assert "true requirement is higher" in corrections
    assert calibration.to_dict()["outcome"] == "OOM"


# --- the activation probe --------------------------------------------------
@requires_stack
def test_the_probe_identifies_deltanet_as_the_dominant_scope():
    """The measurement that found the bug, kept as a test so it cannot silently change."""
    from qwen_distill.training.activation_probe import probe_activations

    profile = probe_activations(spec(num_hidden_layers=4), batch_size=1, sequence_length=256)

    assert profile.error is None, profile.error
    dominant = profile.dominant()
    assert dominant.scope == "deltanet.mixer"
    assert dominant.bytes_retained > profile.total_bytes * 0.4
    # Thousands of retained tensors from one module is the signature of the loop.
    assert dominant.tensor_count > 100


@requires_stack
def test_gradient_checkpointing_collapses_retained_activations():
    """The measured 67x reduction that justifies the revised configuration."""
    from qwen_distill.training.activation_probe import probe_activations

    probe = spec(num_hidden_layers=4)
    without = probe_activations(probe, batch_size=1, sequence_length=256)
    with_ckpt = probe_activations(probe, batch_size=1, sequence_length=256,
                                  gradient_checkpointing=True)

    assert with_ckpt.error is None, with_ckpt.error
    assert with_ckpt.total_bytes < without.total_bytes / 10


@requires_stack
def test_the_scaling_study_extrapolates_to_a_batch_that_would_not_fit():
    """Ruling out a batch without ever allocating it is the check the failed run lacked."""
    from qwen_distill.training.activation_probe import scaling_study

    study = scaling_study(spec(num_hidden_layers=4), batch_sizes=(1, 2), sequence_length=256)

    assert study["available"]
    assert study["model"]["bytes_per_batch"] > 0
    extrapolated = study["extrapolated_gib"]
    assert extrapolated["8"] > extrapolated["4"] > extrapolated["2"]


def test_probe_reports_a_failure_rather_than_raising():
    from qwen_distill.training.activation_probe import ActivationProfile

    profile = ActivationProfile(batch_size=1, sequence_length=128, error="boom")
    assert profile.total_bytes == 0
    assert profile.dominant() is None
    assert "ERROR: boom" in profile.render()
