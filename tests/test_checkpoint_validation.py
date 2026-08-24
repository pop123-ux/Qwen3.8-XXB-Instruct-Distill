"""Tests for checkpoint round-trip validation.

A checkpoint that saves without error but reloads into a *different* model is a silent
failure that invalidates every result produced after it. So the validator must do more
than confirm the file exists — and it must actually fail when a checkpoint is broken.

Both directions are tested: a good checkpoint passes, and each way a checkpoint can be
wrong is caught. Everything runs on CPU with a ~50k-parameter model.
"""

from __future__ import annotations

import json

import pytest
from conftest import HAS_STACK, requires_stack

from qwen_distill.training.validate_checkpoint import (
    DEFAULT_PROMPTS,
    CheckpointReport,
    CheckResult,
    ResumeReport,
    validate_checkpoint,
    validate_resume,
)

TINY_ARCHITECTURE = {
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "vocab_size": 256,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 32,
    "linear_num_key_heads": 1,
    "linear_num_value_heads": 2,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "full_attention_interval": 4,
    "tie_word_embeddings": True,
    "max_position_embeddings": 512,
}


def write_checkpoint(directory, *, step: int = 12, history: int = 3):
    """Build a tiny model, save it exactly as the trainer does, return the directory."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from qwen_distill.training.config import ExperimentConfig, ModelConfig

    config = ExperimentConfig(name="tiny")
    config.model = ModelConfig(architecture=dict(TINY_ARCHITECTURE))
    spec = config.model.resolve_spec()
    fields = {k: v for k, v in spec.to_hf_text_config().items() if k != "model_type"}

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model("qwen3_5_text", **fields))

    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, directory / "training_state.pt")
    (directory / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    (directory / "state.json").write_text(
        json.dumps({
            "step": step,
            "history": [{"step": i, "loss": 8.0 - i} for i in range(history)],
        }),
        encoding="utf-8",
    )
    return directory


@pytest.fixture(scope="module")
def good_checkpoint(tmp_path_factory):
    if not HAS_STACK:
        pytest.skip("requires torch and transformers")
    return write_checkpoint(tmp_path_factory.mktemp("valid") / "step-12")


# --- a sound checkpoint passes every check --------------------------------
@requires_stack
def test_a_sound_checkpoint_passes(good_checkpoint):
    report = validate_checkpoint(good_checkpoint, max_new_tokens=8)
    assert report.error is None, report.error
    assert report.passed, report.render()


@requires_stack
def test_all_four_independent_checks_actually_run(good_checkpoint):
    """Each catches a failure the others miss, so a missing check is a silent gap."""
    report = validate_checkpoint(good_checkpoint, max_new_tokens=8)
    names = " | ".join(c.name for c in report.checks)

    assert len(report.checks) == 4
    assert "state_dict loads" in names
    assert "bit-for-bit" in names
    assert "identical logits" in names
    assert "generation" in names


@requires_stack
def test_reloaded_logits_are_bit_identical_not_merely_close(good_checkpoint):
    """Approximately equal would hide a real weight mismatch; require exactly zero."""
    report = validate_checkpoint(good_checkpoint, max_new_tokens=8)
    logits = next(c for c in report.checks if "identical logits" in c.name)
    assert logits.data["max_abs_difference"] == 0.0


@requires_stack
def test_generations_are_recorded_for_every_prompt(good_checkpoint):
    report = validate_checkpoint(good_checkpoint, max_new_tokens=8)
    assert [g["prompt"] for g in report.generations] == list(DEFAULT_PROMPTS)
    assert all(isinstance(g["completion"], str) for g in report.generations)


@requires_stack
def test_custom_prompts_are_honoured(good_checkpoint):
    report = validate_checkpoint(good_checkpoint, prompts=("abc",), max_new_tokens=4)
    assert [g["prompt"] for g in report.generations] == ["abc"]


# --- broken checkpoints must fail -----------------------------------------
@requires_stack
def test_a_truncated_state_dict_is_caught(tmp_path):
    """The check has to be capable of failing, or passing means nothing."""
    import torch

    directory = write_checkpoint(tmp_path / "truncated")
    payload = torch.load(directory / "training_state.pt", weights_only=False)
    dropped = next(iter(payload["model"]))
    del payload["model"][dropped]
    torch.save(payload, directory / "training_state.pt")

    report = validate_checkpoint(directory, max_new_tokens=4)

    assert not report.passed
    loads = next(c for c in report.checks if "state_dict loads" in c.name)
    assert not loads.passed
    assert dropped in loads.data["missing"]


@requires_stack
def test_an_unexpected_tensor_is_caught(tmp_path):
    import torch

    directory = write_checkpoint(tmp_path / "extra")
    payload = torch.load(directory / "training_state.pt", weights_only=False)
    payload["model"]["model.not_a_real_parameter"] = torch.zeros(4)
    torch.save(payload, directory / "training_state.pt")

    report = validate_checkpoint(directory, max_new_tokens=4)

    loads = next(c for c in report.checks if "state_dict loads" in c.name)
    assert not loads.passed
    assert "model.not_a_real_parameter" in loads.data["unexpected"]


@requires_stack
def test_a_config_that_does_not_match_the_weights_is_caught(tmp_path):
    """Config drift is the failure a parameter-only comparison would miss."""
    directory = write_checkpoint(tmp_path / "drifted")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config["model"]["architecture"]["hidden_size"] = 128
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")

    report = validate_checkpoint(directory, max_new_tokens=4)

    assert not report.passed


def test_a_missing_checkpoint_reports_an_error_not_a_pass(tmp_path):
    report = validate_checkpoint(tmp_path / "absent")
    assert not report.passed
    assert report.error is not None
    assert "training_state.pt" in report.error


@requires_stack
def test_a_checkpoint_without_a_config_reports_an_error(tmp_path):
    directory = write_checkpoint(tmp_path / "no-config")
    (directory / "config.json").unlink()

    report = validate_checkpoint(directory)

    assert not report.passed
    assert "config.json" in report.error


@requires_stack
def test_a_corrupt_state_file_is_reported_rather_than_raised(tmp_path):
    """A validator that crashes gives no verdict; it must return a failing one."""
    directory = write_checkpoint(tmp_path / "corrupt")
    (directory / "training_state.pt").write_bytes(b"not a torch archive")

    report = validate_checkpoint(directory)

    assert not report.passed
    assert report.error


# --- resume ----------------------------------------------------------------
@requires_stack
def test_resume_restores_the_exact_step_and_history(good_checkpoint):
    """A wrong step silently corrupts the LR schedule and breaks reproducibility."""
    report = validate_resume(good_checkpoint)
    assert report.passed, report.to_dict()
    assert report.resumed_step == 12
    assert report.expected_step == 12
    assert report.history_preserved


@requires_stack
def test_resume_preserves_every_history_entry(tmp_path):
    directory = write_checkpoint(tmp_path / "history", step=7, history=5)
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    assert len(state["history"]) == 5

    report = validate_resume(directory)

    assert report.resumed_step == 7
    assert report.history_preserved
    assert report.passed


def test_resume_on_a_missing_state_file_fails(tmp_path):
    (tmp_path / "empty").mkdir()
    report = validate_resume(tmp_path / "empty")
    assert not report.passed
    assert "state.json" in report.error


def test_resume_on_malformed_json_fails_without_raising(tmp_path):
    directory = tmp_path / "malformed"
    directory.mkdir()
    (directory / "state.json").write_text("{not json", encoding="utf-8")

    report = validate_resume(directory)

    assert not report.passed
    assert report.error


# --- report plumbing -------------------------------------------------------
def test_a_report_with_no_checks_is_not_a_pass():
    """An empty report must never read as success; that is the exit-code-0 bug again."""
    assert not CheckpointReport(checkpoint="x").passed


def test_one_failing_check_fails_the_report():
    report = CheckpointReport(checkpoint="x", checks=[
        CheckResult(name="a", passed=True),
        CheckResult(name="b", passed=False),
    ])
    assert not report.passed


def test_an_error_fails_the_report_even_when_checks_passed():
    report = CheckpointReport(
        checkpoint="x",
        checks=[CheckResult(name="a", passed=True)],
        error="something went wrong afterwards",
    )
    assert not report.passed


def test_render_states_the_verdict_explicitly():
    report = CheckpointReport(checkpoint="x", checks=[CheckResult(name="a", passed=False)])
    rendered = report.render()
    assert "[FAIL] a" in rendered
    assert "VERDICT: FAIL" in rendered


def test_resume_report_requires_a_matching_step():
    assert not ResumeReport(checkpoint="x", resumed_step=5, expected_step=6,
                            history_preserved=True).passed
    assert ResumeReport(checkpoint="x", resumed_step=6, expected_step=6,
                        history_preserved=True).passed


def test_reports_serialise_to_json():
    payload = json.dumps(CheckpointReport(
        checkpoint="x", checks=[CheckResult(name="a", passed=True, detail="d")],
    ).to_dict())
    assert json.loads(payload)["passed"] is True
