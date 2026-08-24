"""End-to-end: train, lose the machine, come back, continue.

This is the scenario Phase 2B exists for. The Level-2 run reached ~step 500 on a T4 and
the Colab runtime disconnected, taking `/content` with it. The question these tests
answer is not "does the trainer run" but "if it stops, can it be picked back up, and is
the continued run the same run?"

The equivalence check is the strict one: train continuously to step N, versus train to
N/2, checkpoint, reload in a fresh process state, and continue to N. On CPU the weights
come out bit-identical. On GPU they need not — see the note on the equivalence test.

Everything here is CPU-only and runs in temporary directories.
"""

from __future__ import annotations

import json

import pytest
from conftest import HAS_STACK

from qwen_distill.training.checkpoints import (
    is_complete,
    list_checkpoints,
    read_latest_pointer,
    resolve_checkpoint,
    step_dirname,
)

pytestmark = pytest.mark.skipif(not HAS_STACK, reason="requires torch and transformers")

#: Small enough to train in seconds, but a real hybrid model: DeltaNet and full-attention
#: layers, tied embeddings, byte vocabulary — the same shapes Level 2 uses.
TINY = {
    "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
    "vocab_size": 256, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 32, "linear_num_key_heads": 1, "linear_num_value_heads": 2,
    "linear_key_head_dim": 32, "linear_value_head_dim": 32,
    "full_attention_interval": 4, "tie_word_embeddings": True,
    "max_position_embeddings": 512,
}


def make_config(output, *, max_steps=8, save_every=4, log_every=2, resume=None):
    from qwen_distill.training.config import ExperimentConfig, ModelConfig

    config = ExperimentConfig(name="lifecycle")
    config.model = ModelConfig(architecture=dict(TINY))
    config.data.text_corpus = True
    config.data.max_sequence_length = 64
    config.data.procedural_bytes = 20_000
    config.training.max_steps = max_steps
    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 1
    config.training.save_every = save_every
    config.training.log_every = log_every
    config.training.eval_every = max_steps
    config.training.precision = "fp32"
    config.training.strategy = "full"
    config.training.objective = "sft"
    config.training.gradient_checkpointing = True
    config.runtime.output_dir = str(output)
    config.runtime.device = "cpu"
    config.runtime.resume_from = resume
    return config


def run(config):
    from qwen_distill.training.trainer import train

    return train(config, config.model.resolve_spec())


# --- the artifact a short run must leave behind ----------------------------
def test_a_short_run_leaves_a_resumable_checkpoint(tmp_path):
    """A 20-step smoke test against save_every=500 previously produced no recoverable
    artifact at all. The final step must always checkpoint, aligned or not."""
    output = tmp_path / "run"
    assert run(make_config(output, max_steps=5, save_every=100)) == 0

    checkpoints = list_checkpoints(output / "checkpoints")
    assert [p.name for p in checkpoints] == ["step_000005"]
    assert is_complete(checkpoints[0])
    assert read_latest_pointer(output / "checkpoints")["step"] == 5


def test_the_final_step_is_not_checkpointed_twice(tmp_path):
    """When the last step does land on the interval, one directory, not two."""
    output = tmp_path / "run"
    assert run(make_config(output, max_steps=4, save_every=4)) == 0
    assert [p.name for p in list_checkpoints(output / "checkpoints")] == ["step_000004"]


def test_a_run_writes_progress_independently_of_checkpoints(tmp_path):
    """Progress is kilobytes and frequent; checkpoints are gigabytes and rare."""
    output = tmp_path / "run"
    assert run(make_config(output, max_steps=8, save_every=8, log_every=2)) == 0

    from qwen_distill.training.progress import ProgressWriter

    writer = ProgressWriter(output)
    steps = [r["step"] for r in writer.read_history() if r.get("status") == "completed_step"]
    assert steps == [2, 4, 6, 8], "one progress record per log interval"
    assert len(list_checkpoints(output / "checkpoints")) == 1, "one checkpoint only"
    assert writer.read_latest()["step"] == 8


def test_progress_records_carry_what_a_recovery_needs(tmp_path):
    output = tmp_path / "run"
    assert run(make_config(output, max_steps=4, save_every=4, log_every=2)) == 0

    from qwen_distill.training.progress import ProgressWriter

    latest = ProgressWriter(output).read_latest()
    for key in ("step", "timestamp", "status", "git_commit", "config_sha256"):
        assert key in latest, f"progress record is missing {key}"


# --- resume ----------------------------------------------------------------
def interrupt_at(tmp_path, step: int, *, max_steps: int = 8, save_every: int = 4):
    """A run that got to `step` of `max_steps` and then lost its machine.

    Produced by training the full run and keeping only the checkpoints up to `step`,
    which is exactly the on-disk state a Colab disconnect leaves behind — and it keeps
    max_steps fixed, so the LR schedule is the one the run actually intended.
    """
    import shutil

    source = tmp_path / "source"
    run(make_config(source, max_steps=max_steps, save_every=save_every))

    output = tmp_path / "run"
    (output / "checkpoints").mkdir(parents=True)
    shutil.copytree(
        source / "checkpoints" / step_dirname(step),
        output / "checkpoints" / step_dirname(step),
    )
    (output / "checkpoints" / "latest.json").write_text(
        json.dumps({"step": step, "path": step_dirname(step),
                    "created_at": "2026-01-01T00:00:00+00:00", "complete": True}),
        encoding="utf-8",
    )
    return output


def test_resume_continues_to_max_steps_without_repeating_work(tmp_path):
    """A run interrupted at step 4 of 8 must continue 5..8 — not restart at 0, and not
    stop at 4."""
    output = interrupt_at(tmp_path, 4, max_steps=8, save_every=4)

    assert run(make_config(output, max_steps=8, save_every=4, resume="latest")) == 0

    assert read_latest_pointer(output / "checkpoints")["step"] == 8
    assert [p.name for p in list_checkpoints(output / "checkpoints")] == [
        "step_000004", "step_000008",
    ]


def test_resuming_under_a_different_max_steps_is_refused(tmp_path):
    """OneCycleLR's shape depends on its total length, so its state cannot be moved onto
    a schedule of a different length. Failing clearly beats a confusing scheduler error
    thirty seconds in."""
    output = interrupt_at(tmp_path, 4, max_steps=8, save_every=4)
    assert run(make_config(output, max_steps=20, save_every=4, resume="latest")) == 2


def test_resume_restores_step_tokens_and_data_position(tmp_path):
    output = interrupt_at(tmp_path, 4, max_steps=8, save_every=4)
    first = json.loads(
        (output / "checkpoints" / step_dirname(4) / "training_state.json").read_text(
            encoding="utf-8"
        )
    )
    run(make_config(output, max_steps=8, save_every=4, resume="latest"))
    second = json.loads(
        (output / "checkpoints" / step_dirname(8) / "training_state.json").read_text(
            encoding="utf-8"
        )
    )

    assert second["step"] == 8
    assert second["tokens_seen"] == 2 * first["tokens_seen"], "tokens must accumulate"
    assert second["data_state"]["index"] > first["data_state"]["index"], (
        "the data stream must advance, not rewind to epoch 0"
    )


def test_resuming_from_nothing_fails_loudly_rather_than_starting_over(tmp_path):
    """Silently restarting at step 0 is how you lose 500 steps twice."""
    output = tmp_path / "run"
    output.mkdir()
    assert run(make_config(output, max_steps=4, resume="latest")) == 2
    assert not list_checkpoints(output / "checkpoints")


def test_resuming_from_an_incomplete_checkpoint_is_refused(tmp_path):
    output = tmp_path / "run"
    run(make_config(output, max_steps=4, save_every=4))
    # Corrupt it the way an interrupted write would.
    (output / "checkpoints" / step_dirname(4) / "COMPLETE").unlink()

    assert run(make_config(output, max_steps=8, save_every=4, resume="4")) == 2


def test_startup_clears_staging_left_by_a_killed_process(tmp_path):
    output = interrupt_at(tmp_path, 4, max_steps=8, save_every=4)
    staging = output / "checkpoints" / ".step_000008.incomplete"
    staging.mkdir()
    (staging / "model.safetensors").write_bytes(b"partial")

    assert run(make_config(output, max_steps=8, save_every=4, resume="latest")) == 0
    assert not staging.exists()


# --- the scientific equivalence check --------------------------------------
def test_resumed_training_matches_continuous_training(tmp_path):
    """Run A trains 0->8. Run B trains 0->4, checkpoints, reloads, continues 5->8.

    On CPU with a fixed seed this is bit-identical, and that is asserted. It is *not*
    guaranteed on GPU: cuDNN algorithm selection and atomic reductions make some kernels
    non-deterministic across processes, so a GPU resume should be expected to match
    closely rather than exactly.
    """
    import torch
    from safetensors.torch import load_file

    continuous = tmp_path / "A"
    assert run(make_config(continuous, max_steps=8, save_every=4)) == 0

    # Start B from A's own step-4 checkpoint, so both share one LR schedule. Building B
    # with max_steps=4 would give it a different OneCycle curve and prove nothing.
    interrupted = tmp_path / "B"
    (interrupted / "checkpoints").mkdir(parents=True)
    import shutil

    shutil.copytree(
        continuous / "checkpoints" / step_dirname(4),
        interrupted / "checkpoints" / step_dirname(4),
    )
    pointer = json.loads(
        (continuous / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    pointer.update(step=4, path=step_dirname(4))
    (interrupted / "checkpoints" / "latest.json").write_text(
        json.dumps(pointer), encoding="utf-8"
    )

    assert run(make_config(interrupted, max_steps=8, save_every=4, resume="latest")) == 0

    a = load_file(str(continuous / "checkpoints" / step_dirname(8) / "model.safetensors"))
    b = load_file(str(interrupted / "checkpoints" / step_dirname(8) / "model.safetensors"))
    assert set(a) == set(b)
    differing = [name for name in a if not torch.equal(a[name], b[name])]
    assert not differing, f"resumed weights differ from continuous: {differing[:5]}"


def test_the_scheduler_and_step_agree_after_a_resume(tmp_path):
    """A restarted LR schedule is the silent failure: the loss curve bends and nothing
    reports why."""
    import torch

    continuous = tmp_path / "A"
    run(make_config(continuous, max_steps=8, save_every=4))

    interrupted = tmp_path / "B"
    (interrupted / "checkpoints").mkdir(parents=True)
    import shutil

    shutil.copytree(
        continuous / "checkpoints" / step_dirname(4),
        interrupted / "checkpoints" / step_dirname(4),
    )
    pointer = json.loads(
        (continuous / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    pointer.update(step=4, path=step_dirname(4))
    (interrupted / "checkpoints" / "latest.json").write_text(
        json.dumps(pointer), encoding="utf-8"
    )
    run(make_config(interrupted, max_steps=8, save_every=4, resume="latest"))

    a = torch.load(
        continuous / "checkpoints" / step_dirname(8) / "scheduler.pt", weights_only=False
    )
    b = torch.load(
        interrupted / "checkpoints" / step_dirname(8) / "scheduler.pt", weights_only=False
    )
    assert a["last_epoch"] == b["last_epoch"] == 8
    assert a["_last_lr"] == b["_last_lr"]


# --- what a checkpoint says about itself -----------------------------------
def test_a_checkpoint_is_interpretable_without_the_session_that_wrote_it(tmp_path):
    output = tmp_path / "run"
    run(make_config(output, max_steps=4, save_every=4))
    metadata = json.loads(
        (output / "checkpoints" / step_dirname(4) / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    assert metadata["complete"] is True
    assert metadata["step"] == 4
    assert metadata["parameter_count"] > 0
    assert metadata["sequence_length"] == 64
    assert metadata["batch_size"] == 2
    assert metadata["effective_batch_size"] == 2
    assert metadata["gradient_checkpointing"] is True
    assert metadata["precision"] == "fp32"
    assert metadata["optimizer"] == "adamw_8bit"
    assert metadata["tokens_seen"] > 0
    assert metadata["config_sha256"]
    assert metadata["architecture_sha256"]
    assert metadata["torch_version"]
    assert "model.safetensors" in metadata["contents"]


def test_a_checkpoint_validates_and_generates(tmp_path):
    from qwen_distill.training.validate_checkpoint import (
        validate_checkpoint,
        validate_resume,
    )

    output = tmp_path / "run"
    run(make_config(output, max_steps=4, save_every=4))
    checkpoint = output / "checkpoints" / step_dirname(4)

    report = validate_checkpoint(checkpoint, prompts=("The ",), max_new_tokens=4)
    assert report.passed, report.render()
    assert validate_resume(checkpoint).passed


def test_the_run_summary_records_the_resume(tmp_path):
    output = interrupt_at(tmp_path, 4, max_steps=8, save_every=4)
    run(make_config(output, max_steps=8, save_every=4, resume="latest"))

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["outcome"] == "completed"
    assert summary["steps"] == 8


# --- discovery from a cold start -------------------------------------------
def test_status_reports_a_resumable_run_from_files_alone(tmp_path, capsys):
    from scripts_shim import report_status

    output = tmp_path / "run"
    run(make_config(output, max_steps=4, save_every=4))
    config = make_config(output, max_steps=8)

    assert report_status(config) == 0
    captured = capsys.readouterr().out
    assert "RESUMABLE: yes" in captured
    assert "step_000004" in captured
    assert "--resume latest" in captured


def test_status_on_an_unstarted_run_says_so(tmp_path, capsys):
    from scripts_shim import report_status

    config = make_config(tmp_path / "never-run", max_steps=8)
    assert report_status(config) == 0
    assert "has not been run" in capsys.readouterr().out


def test_resolve_finds_the_checkpoint_a_fresh_session_would_need(tmp_path):
    output = tmp_path / "run"
    run(make_config(output, max_steps=8, save_every=4))
    # A new process knows only the run directory.
    resolved = resolve_checkpoint(output / "checkpoints", "latest")
    assert resolved is not None
    assert resolved.name == "step_000008"
