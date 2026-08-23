"""Tests for what precision a training run *actually* uses.

`AutoModelForCausalLM.from_config` always produces fp32 — it has no dtype argument to
honour — so a config declaring `precision: fp16` silently trained in fp32 at roughly
twice the modelled weight/gradient/optimizer memory, and without the tensor-core
speedup a T4 offers. That is a five-hour GPU window spent on the wrong run, and nothing
in the artifacts would have said so.

The fix is mixed precision applied to the *compute*: autocast plus GradScaler, fp32
master weights. These tests pin both halves — the resolution, and the fact that the
resolved value is what gets recorded and estimated against.
"""

from __future__ import annotations

import pytest
from conftest import HAS_STACK, requires_stack

from qwen_distill.diagnostics.fit import resolve_precision_scheme


@requires_stack
def test_fp32_is_never_silently_upgraded():
    from qwen_distill.training.trainer import resolve_precision

    assert resolve_precision("fp32", "cuda") == ("fp32", None)
    assert resolve_precision("fp32", "cpu") == ("fp32", None)


@requires_stack
def test_mixed_precision_falls_back_to_fp32_on_cpu_and_says_why():
    """Silently running fp16 on CPU would be slower, not faster."""
    from qwen_distill.training.trainer import resolve_precision

    effective, note = resolve_precision("fp16", "cpu")
    assert effective == "fp32"
    assert note and "CPU" in note


@requires_stack
def test_fp16_on_cuda_is_honoured():
    from qwen_distill.training.trainer import resolve_precision

    assert resolve_precision("fp16", "cuda") == ("fp16", None)


@requires_stack
def test_bf16_falls_back_to_fp16_when_the_device_cannot_do_bf16(monkeypatch):
    """Turing — the T4 — has no bf16. Failing at runtime is the worst outcome."""
    import torch

    from qwen_distill.training import trainer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "Tesla T4")

    effective, note = trainer.resolve_precision("bf16", "cuda")

    assert effective == "fp16"
    assert note and "bf16" in note


@requires_stack
def test_bf16_is_kept_on_a_device_that_supports_it(monkeypatch):
    import torch

    from qwen_distill.training import trainer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert trainer.resolve_precision("bf16", "cuda") == ("bf16", None)


# --- the estimate must follow the *effective* precision -------------------
def test_a_cpu_fallback_is_estimated_as_fp32_not_as_the_request():
    """Estimating fp16 activations for a run that will use fp32 halves a real term."""
    assert resolve_precision_scheme("fp32 (CPU)").activation_bytes == 4.0
    assert resolve_precision_scheme("fp16").activation_bytes == 2.0


@requires_stack
def test_the_cpu_feasibility_estimate_uses_the_resolved_precision():
    from qwen_distill.training.config import ExperimentConfig
    from qwen_distill.training.feasibility import check_feasibility

    config = ExperimentConfig.load("configs/experiments/t4_level2_100m.yaml")
    report = check_feasibility(config, config.model.resolve_spec(), available_gib=0.0)

    assert config.training.precision == "fp16"
    assert report.fit is not None
    assert report.fit.precision_scheme == "fp32", (
        "a CPU run falls back to fp32, so the estimate must model fp32 activations"
    )


# --- the run must record what really happened -----------------------------
@pytest.mark.skipif(not HAS_STACK, reason="requires torch and transformers")
def test_a_run_records_both_the_requested_and_the_effective_precision(tmp_path):
    """The artifact has to be readable months later without re-deriving the fallback."""
    import json

    from qwen_distill.training.config import ExperimentConfig
    from qwen_distill.training.trainer import train

    config = ExperimentConfig.load("configs/experiments/t4_level2_100m.yaml")
    config.model.architecture.update(
        hidden_size=64, num_hidden_layers=4, intermediate_size=128,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32,
    )
    config.model.architecture.pop("layer_types", None)
    config.data.max_sequence_length = 64
    config.data.procedural_bytes = 20_000
    config.training.max_steps = 2
    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 1
    config.training.log_every = 2
    config.training.eval_every = 2
    config.training.save_every = 2
    config.runtime.output_dir = str(tmp_path / "run")
    config.runtime.device = "cpu"

    assert train(config, config.model.resolve_spec()) == 0

    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["requested_precision"] == "fp16"
    assert summary["effective_precision"] == "fp32"
    assert "CPU" in summary["precision_note"]
    assert summary["analytical_estimate"]["precision_scheme"] == "fp32"
