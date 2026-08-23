"""Tests for hardware diagnostics.

The whole suite must run without a GPU, so these exercise the CPU-only paths directly
and use synthetic VRAM figures for the tier/fit/recommendation logic. That is not a
compromise: CPU-only *is* the environment most contributors are in, and a diagnostics
tool that only works on the machines that need it least is useless.
"""

from __future__ import annotations

import pytest

from qwen_distill.architecture.spec import HybridArchSpec
from qwen_distill.diagnostics.devices import DeviceInfo, collect_devices, collect_system, cpu_device
from qwen_distill.diagnostics.fit import (
    analyse_inference_fit,
    estimate_training_memory,
    fit_matrix,
)
from qwen_distill.diagnostics.recommend import recommend
from qwen_distill.diagnostics.tiers import TIERS, classify, tier_for_devices

TEACHER = HybridArchSpec(name="Qwen3.8-27B")


def small_spec(name: str = "small") -> HybridArchSpec:
    return HybridArchSpec(
        name=name, hidden_size=2048, num_hidden_layers=24, intermediate_size=5632,
        vocab_size=248320, num_attention_heads=16, num_key_value_heads=4, head_dim=128,
        linear_num_key_heads=8, linear_num_value_heads=24, tie_word_embeddings=True,
    )


# --- detection never crashes -------------------------------------------
def test_collect_system_works_without_gpu():
    system = collect_system()
    assert system.os
    assert system.python_version
    assert isinstance(system.cuda_available, bool)


def test_collect_devices_returns_a_list_even_with_no_accelerator():
    assert isinstance(collect_devices(), list)


def test_cpu_device_is_always_available():
    device = cpu_device()
    assert device.vendor == "cpu"
    assert device.backend == "cpu"
    assert any("no accelerator" in n for n in device.notes)


def test_device_report_is_json_serialisable():
    import json

    json.dumps(cpu_device().to_dict())
    json.dumps(collect_system().to_dict())


# --- tiers ---------------------------------------------------------------
@pytest.mark.parametrize(
    "vram,level",
    [(None, 0), (0.0, 0), (8.0, 1), (12.0, 2), (16.0, 2), (24.0, 3),
     (32.0, 4), (48.0, 5), (80.0, 6), (192.0, 6)],
)
def test_tier_classification(vram, level):
    assert classify(vram).level == level


def test_tiers_are_ordered_and_contiguous():
    for lower, upper in zip(TIERS, TIERS[1:], strict=False):
        assert lower.level < upper.level
        assert lower.max_vram_gib == upper.min_vram_gib


def test_cpu_row_is_not_counted_as_vram():
    """A CPU-only laptop must be Tier 0, not Tier 2 because it has 16 GB of RAM."""
    assert tier_for_devices([cpu_device()]).level == 0


def test_tier_uses_the_largest_single_device_not_the_sum():
    """Two 8 GB cards are not a 16 GB card without model parallelism."""
    devices = [
        DeviceInfo(index=0, vendor="nvidia", name="A", backend="cuda", total_memory_gib=8.0),
        DeviceInfo(index=1, vendor="nvidia", name="B", backend="cuda", total_memory_gib=8.0),
    ]
    assert tier_for_devices(devices).level == 1


# --- inference fit --------------------------------------------------------
def test_teacher_does_not_fit_a_t4():
    for quant in ("bf16", "int8", "q6_k", "q5_k_m", "q4_k_m"):
        fit = analyse_inference_fit(TEACHER, 15.0, quantization=quant, context_length=8192)
        assert fit.verdict == "DOES NOT FIT", quant


def test_teacher_fits_a_large_gpu_at_4bit():
    fit = analyse_inference_fit(TEACHER, 79.0, quantization="q4_k_m", context_length=8192)
    assert fit.verdict == "FITS"


def test_fit_accounts_for_more_than_weights():
    fit = analyse_inference_fit(TEACHER, 15.0, quantization="q4_k_m", context_length=32768)
    assert fit.total_gib > fit.weights_gib
    assert fit.kv_cache_gib > 0
    assert fit.overhead_gib > 0


def test_tight_is_distinguished_from_fits():
    """Fitting with 200 MB spare is not the same as fitting."""
    spec = small_spec()
    exact = analyse_inference_fit(spec, 100.0, quantization="q4_k_m", context_length=4096)
    assert exact.verdict == "FITS"
    barely = analyse_inference_fit(
        spec, exact.total_gib + 0.5, quantization="q4_k_m", context_length=4096
    )
    assert barely.verdict == "TIGHT"


def test_fit_matrix_covers_the_grid():
    grid = fit_matrix(TEACHER, 15.0, quantizations=("bf16", "q4_k_m"), contexts=(8192, 32768))
    assert set(grid) == {"bf16", "q4_k_m"}
    assert set(grid["bf16"]) == {8192, 32768}


def test_longer_context_never_reduces_memory():
    grid = fit_matrix(TEACHER, 15.0, quantizations=("q4_k_m",), contexts=(8192, 32768, 131072))
    totals = [grid["q4_k_m"][c].total_gib for c in (8192, 32768, 131072)]
    assert totals == sorted(totals)


# --- training fit ---------------------------------------------------------
def test_full_training_of_the_teacher_is_impossible_on_consumer_hardware():
    fit = estimate_training_memory(TEACHER, 24.0, strategy="full", optimizer="adamw")
    assert not fit.feasible
    assert fit.total_gib > 200


def test_qlora_is_far_cheaper_than_full():
    full = estimate_training_memory(TEACHER, 15.0, strategy="full")
    qlora = estimate_training_memory(TEACHER, 15.0, strategy="qlora")
    assert qlora.total_gib < full.total_gib / 5


def test_8bit_optimizer_reduces_state():
    spec = small_spec()
    adamw = estimate_training_memory(spec, 15.0, strategy="full", optimizer="adamw")
    eight = estimate_training_memory(spec, 15.0, strategy="full", optimizer="adamw_8bit")
    assert eight.optimizer_state_gib < adamw.optimizer_state_gib


def test_gradient_checkpointing_reduces_activations():
    spec = small_spec()
    on = estimate_training_memory(spec, 15.0, sequence_length=2048, gradient_checkpointing=True)
    off = estimate_training_memory(spec, 15.0, sequence_length=2048, gradient_checkpointing=False)
    assert on.activations_gib < off.activations_gib


def test_infeasible_configurations_carry_ordered_suggestions():
    fit = estimate_training_memory(
        TEACHER, 15.0, strategy="full", optimizer="adamw",
        sequence_length=8192, batch_size=4, gradient_checkpointing=False,
    )
    assert not fit.feasible
    joined = " ".join(fit.suggestions)
    assert "gradient checkpointing" in joined
    assert "batch size" in joined
    assert "larger GPU" in fit.suggestions[-1]


def test_unknown_strategy_and_optimizer_are_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        estimate_training_memory(TEACHER, 15.0, strategy="magic")
    with pytest.raises(ValueError, match="unknown optimizer"):
        estimate_training_memory(TEACHER, 15.0, optimizer="magic")


# --- recommendations ------------------------------------------------------
def test_cpu_only_recommends_analysis_and_forbids_training():
    result = recommend(None, "no GPU")
    assert result.tier.level == 0
    assert any("unit tests" in item for item in result.good)
    assert any("training" in item for item in result.not_realistic)


def test_t4_can_train_small_students_but_not_the_teacher():
    result = recommend(16.0, "Tesla T4")
    assert result.tier.level == 2
    assert result.training, "a T4 should be able to train something"
    joined = " ".join(result.not_realistic)
    assert "Qwen3.8-27B bf16" in joined


def test_24gb_is_tier_3_and_more_capable_than_a_t4():
    t4 = recommend(16.0, "T4")
    big = recommend(24.0, "RTX 3090")
    assert big.tier.level == 3
    assert len(big.not_realistic) <= len(t4.not_realistic)


def test_48gb_is_tier_5():
    result = recommend(48.0, "A6000")
    assert result.tier.level == 5
    assert result.training


def test_no_hardware_can_full_train_a_27b_model():
    for vram in (16.0, 24.0, 48.0, 80.0):
        joined = " ".join(recommend(vram, "x").not_realistic)
        assert "full-parameter training of a 27B-class model" in joined, vram


def test_recommendations_are_json_serialisable():
    import json

    json.dumps(recommend(16.0, "Tesla T4").to_dict())
