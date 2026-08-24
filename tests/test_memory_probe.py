"""Tests for stage-by-stage memory instrumentation.

The bug this module exists to prevent is a reported number that is not a measurement.
Two shapes of that:

* **Zeros on a CPU-only machine.** ``0.0 GiB allocated`` is indistinguishable from a
  real measurement of nothing, so the absence must be explicit.
* **A conflated ratio.** The earlier ~2.85 "calibration" divided allocator *reserve* by
  modelled *tensor* memory — two different quantities. Applying it as a multiplier would
  bake that error into every future estimate, so the comparison must return both ratios
  separately and say what each one is for.

These tests run on CPU. Where CUDA behaviour is the subject, the snapshots are built
directly rather than by requiring a GPU, since the arithmetic — not the driver — is what
is being tested.
"""

from __future__ import annotations

import pytest

from qwen_distill.training.memory_probe import (
    GIB,
    STAGES,
    MemoryProfile,
    MemorySnapshot,
    compare_with_estimate,
    derive_components,
    new_profile,
    reset_peak,
    take,
)

HAS_CUDA = False
try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover - torch is optional for analysis-only installs
    pass


def snapshot(stage: str, allocated: float, reserved: float | None = None) -> MemorySnapshot:
    """A snapshot with plausible peaks, for testing the attribution arithmetic."""
    reserved = allocated * 1.2 if reserved is None else reserved
    return MemorySnapshot(
        stage=stage, allocated_gib=allocated, reserved_gib=reserved,
        max_allocated_gib=allocated, max_reserved_gib=reserved,
    )


def cuda_profile(*snapshots: MemorySnapshot) -> MemoryProfile:
    return MemoryProfile(
        cuda_available=True, device_name="Tesla T4", total_vram_gib=14.56,
        snapshots=list(snapshots),
    )


# --- absence is reported as absence, never as zero ------------------------
def test_cpu_only_profile_reports_unavailable_not_zero():
    """A CPU machine must not emit figures that read like GPU measurements."""
    profile = MemoryProfile(cuda_available=False)

    assert profile.peak_allocated_gib is None
    assert profile.peak_reserved_gib is None
    assert profile.components == {}
    assert derive_components(profile) == {}


def test_cpu_only_profile_explains_why_figures_are_missing():
    profile = MemoryProfile(cuda_available=False)
    profile.notes.append("no CUDA device")
    payload = profile.to_dict()

    assert payload["cuda_available"] is False
    assert payload["peak_allocated_gib"] is None
    assert payload["total_vram_gib"] is None
    assert payload["notes"]


def test_take_returns_none_without_cuda():
    profile = MemoryProfile(cuda_available=False)
    assert take(profile, "baseline") is None
    assert profile.snapshots == []


def test_reset_peak_is_a_no_op_without_cuda():
    reset_peak()  # must not raise on a CPU-only machine


@pytest.mark.skipif(HAS_CUDA, reason="asserts the CPU-only branch")
def test_new_profile_on_this_cpu_machine_records_the_reason():
    profile = new_profile()
    assert profile.cuda_available is False
    assert profile.device_name is None
    assert profile.notes, "the absence of instrumentation must be stated, not implied"
    assert "zero" in " ".join(profile.notes).lower()


def test_comparison_is_unavailable_rather_than_ratio_zero():
    """A ratio of 0.0 would look like 'the estimate was infinitely wrong'."""
    result = compare_with_estimate(MemoryProfile(cuda_available=False), 4.5)
    assert result == {"available": False}
    assert "ratio_allocated" not in result


def test_comparison_refuses_a_nonpositive_estimate():
    profile = cuda_profile(snapshot("peak_training", 8.0))
    assert compare_with_estimate(profile, 0.0)["available"] is False
    assert compare_with_estimate(profile, -1.0)["available"] is False


# --- attribution by difference --------------------------------------------
def test_components_are_attributed_by_difference_between_stages():
    profile = cuda_profile(
        snapshot("baseline", 0.30),
        snapshot("after_model_creation", 0.30 + 0.70),
        snapshot("after_optimizer_creation", 1.00),
        snapshot("after_forward", 1.00 + 2.50),
        snapshot("after_backward", 3.50 + 0.20),
        snapshot("after_optimizer_step", 3.70 + 1.40),
    )

    components = derive_components(profile)

    assert components["cuda_context_and_baseline_gib"] == pytest.approx(0.30)
    assert components["weights_gib"] == pytest.approx(0.70)
    assert components["activations_gib"] == pytest.approx(2.50)
    assert components["gradients_net_of_freed_activations_gib"] == pytest.approx(0.20)
    assert components["optimizer_state_gib"] == pytest.approx(1.40)


def test_optimizer_state_is_measured_after_the_step_because_adamw_is_lazy():
    """Snapshotting before the first step() would read zero and be reported as truth."""
    profile = cuda_profile(
        snapshot("after_optimizer_creation", 1.0),
        snapshot("after_backward", 1.2),
        snapshot("after_optimizer_step", 2.6),
    )
    components = derive_components(profile)
    assert components["optimizer_state_gib"] == pytest.approx(1.4)


def test_missing_stages_omit_their_term_rather_than_guessing_it():
    """A partial run must produce fewer numbers, not wrong ones."""
    profile = cuda_profile(snapshot("baseline", 0.3), snapshot("after_model_creation", 1.0))

    components = derive_components(profile)

    assert components["weights_gib"] == pytest.approx(0.7)
    assert "activations_gib" not in components
    assert "optimizer_state_gib" not in components
    assert "gradients_net_of_freed_activations_gib" not in components


def test_a_backward_pass_that_frees_more_than_it_allocates_is_reported_as_negative():
    """Backward frees activations as it consumes them; a negative net is real, not a bug."""
    profile = cuda_profile(snapshot("after_forward", 4.0), snapshot("after_backward", 2.5))
    components = derive_components(profile)
    assert components["gradients_net_of_freed_activations_gib"] == pytest.approx(-1.5)


def test_component_terms_never_go_negative_where_negative_is_impossible():
    """Weights cannot shrink; a negative there would be measurement noise, not signal."""
    profile = cuda_profile(snapshot("baseline", 1.0), snapshot("after_model_creation", 0.99))
    assert derive_components(profile)["weights_gib"] == 0.0


def test_peaks_are_the_maximum_across_every_snapshot():
    profile = cuda_profile(
        snapshot("baseline", 0.3, 0.4),
        snapshot("after_forward", 3.5, 4.9),
        snapshot("after_optimizer_step", 2.0, 4.9),
    )
    assert profile.peak_allocated_gib == pytest.approx(3.5)
    assert profile.peak_reserved_gib == pytest.approx(4.9)


def test_allocator_reserve_is_separated_from_live_tensors():
    """Reserve is what the process occupies; allocated is what the tensors need."""
    profile = cuda_profile(snapshot("peak_training", 4.0, 5.6))
    components = derive_components(profile)

    assert components["peak_allocated_gib"] == pytest.approx(4.0)
    assert components["peak_reserved_gib"] == pytest.approx(5.6)
    assert components["allocator_reserve_overhead_gib"] == pytest.approx(1.6)


# --- the two ratios must stay separate ------------------------------------
def test_comparison_returns_both_ratios_separately():
    """Conflating these two is precisely what produced the unusable 2.85 figure."""
    profile = cuda_profile(snapshot("peak_training", 4.0, 11.4))

    result = compare_with_estimate(profile, 4.0)

    assert result["available"] is True
    assert result["ratio_allocated"] == pytest.approx(1.0)
    assert result["ratio_reserved"] == pytest.approx(2.85)
    assert result["ratio_allocated"] != result["ratio_reserved"]


def test_comparison_says_what_each_ratio_is_for():
    """The interpretation travels with the number, so it cannot be reused blindly."""
    profile = cuda_profile(snapshot("peak_training", 4.0, 11.4))
    interpretation = compare_with_estimate(profile, 4.0)["interpretation"]

    assert "estimator" in interpretation
    assert "deployment" in interpretation
    assert "blind multiplier" in interpretation


def test_comparison_reports_the_measurements_alongside_the_ratios():
    """A bare ratio cannot be re-derived later; the inputs must be recorded with it."""
    profile = cuda_profile(snapshot("peak_training", 4.0, 5.6))
    result = compare_with_estimate(profile, 3.2)

    assert result["estimated_total_gib"] == pytest.approx(3.2)
    assert result["measured_peak_allocated_gib"] == pytest.approx(4.0)
    assert result["measured_peak_reserved_gib"] == pytest.approx(5.6)
    assert result["ratio_allocated"] == pytest.approx(4.0 / 3.2)
    assert result["ratio_reserved"] == pytest.approx(5.6 / 3.2)


# --- plumbing -------------------------------------------------------------
def test_stage_order_matches_the_order_a_training_step_executes_in():
    assert STAGES.index("baseline") < STAGES.index("after_model_creation")
    assert STAGES.index("after_model_creation") < STAGES.index("after_forward")
    assert STAGES.index("after_forward") < STAGES.index("after_backward")
    assert STAGES.index("after_backward") < STAGES.index("after_optimizer_step")


def test_gib_is_binary_not_decimal():
    assert GIB == 1024 ** 3


def test_snapshot_lookup_by_stage():
    profile = cuda_profile(snapshot("baseline", 0.3), snapshot("after_forward", 3.0))
    assert profile.snapshot("after_forward").allocated_gib == pytest.approx(3.0)
    assert profile.snapshot("never_taken") is None


def test_profile_serialises_to_json_safe_primitives():
    import json

    profile = cuda_profile(
        snapshot("baseline", 0.3),
        snapshot("after_model_creation", 1.0),
        snapshot("after_forward", 3.0),
    )
    derive_components(profile)

    payload = json.loads(json.dumps(profile.to_dict()))
    assert payload["device_name"] == "Tesla T4"
    assert [s["stage"] for s in payload["snapshots"]] == [
        "baseline", "after_model_creation", "after_forward",
    ]
    assert payload["components"]["activations_gib"] == pytest.approx(2.0)


@pytest.mark.skipif(not HAS_CUDA, reason="requires a CUDA device")
def test_real_cuda_snapshots_are_monotonic_in_peak():  # pragma: no cover - needs a GPU
    profile = new_profile()
    reset_peak()
    take(profile, "baseline")
    buffer = torch.zeros(64 * 1024 * 1024 // 4, device="cuda")  # 64 MiB
    take(profile, "after_model_creation")

    assert profile.components == {}
    components = derive_components(profile)
    assert components["weights_gib"] == pytest.approx(0.0625, abs=0.01)
    del buffer
