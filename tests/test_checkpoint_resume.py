"""Tests for interruption-safe checkpointing and resume.

The Level-2 run reached ~step 500 on a T4 — no OOM, validation BPB down to 1.279 — and
then the Colab runtime disconnected and took the ephemeral filesystem with it. Nothing
about the model was wrong; the persistence was.

The invariant these tests defend:

    A crash may lose the step currently executing. It must never invalidate the last
    checkpoint that completed.

Two classes of bug are covered. **Incompleteness**: a checkpoint that saved weights and a
step counter but no scheduler, scaler, RNG or data position, so "resuming" restarted the
one-cycle schedule, reset the loss scale and rewound the data to epoch 0 — a different
run wearing the old run's step number. **Non-atomicity**: writing in place, so a kill
halfway through leaves a directory that exists, looks plausible, cannot be loaded, and is
the newest one.

All CPU. No GPU, no Drive, no network.
"""

from __future__ import annotations

import json

import pytest
from conftest import requires_stack

from qwen_distill.training.checkpoints import (
    COMPLETE_MARKER,
    REQUIRED_FILES,
    CheckpointMetadata,
    cleanup_incomplete,
    config_sha256,
    is_complete,
    list_checkpoints,
    read_latest_pointer,
    resolve_checkpoint,
    step_dirname,
    update_latest_pointer,
)
from qwen_distill.training.progress import ProgressWriter
from qwen_distill.training.text_data import ResumableBatchSampler


# --- naming and discovery --------------------------------------------------
def test_step_directories_sort_lexically_in_numeric_order():
    """Zero padding, so `ls` and any file listing agree with the step order."""
    names = [step_dirname(s) for s in (2, 10, 100, 2000)]
    assert names == ["step_000002", "step_000010", "step_000100", "step_002000"]
    assert names == sorted(names)


def make_checkpoint(root, step: int, *, complete: bool = True, files=REQUIRED_FILES):
    directory = root / step_dirname(step)
    directory.mkdir(parents=True, exist_ok=True)
    for name in files:
        if name == "metadata.json":
            (directory / name).write_text(
                json.dumps({"step": step, "complete": complete}), encoding="utf-8"
            )
        else:
            (directory / name).write_bytes(b"x")
    if complete:
        (directory / COMPLETE_MARKER).write_text(f"step {step}\n", encoding="utf-8")
    return directory


def test_a_checkpoint_needs_both_the_marker_and_every_required_file(tmp_path):
    """The marker alone would trust a write interrupted after it; the files alone would
    trust a directory mid-write."""
    assert is_complete(make_checkpoint(tmp_path, 100))
    assert not is_complete(make_checkpoint(tmp_path, 200, complete=False))
    assert not is_complete(
        make_checkpoint(tmp_path, 300, files=("model.safetensors", "metadata.json"))
    )


def test_a_metadata_file_claiming_incompleteness_is_not_complete(tmp_path):
    directory = make_checkpoint(tmp_path, 100)
    (directory / "metadata.json").write_text(
        json.dumps({"step": 100, "complete": False}), encoding="utf-8"
    )
    assert not is_complete(directory)


def test_unparseable_metadata_is_not_complete(tmp_path):
    directory = make_checkpoint(tmp_path, 100)
    (directory / "metadata.json").write_text("{truncated", encoding="utf-8")
    assert not is_complete(directory)


def test_listing_returns_only_complete_checkpoints_oldest_first(tmp_path):
    make_checkpoint(tmp_path, 100)
    make_checkpoint(tmp_path, 300)
    make_checkpoint(tmp_path, 200, complete=False)

    assert [p.name for p in list_checkpoints(tmp_path)] == ["step_000100", "step_000300"]


def test_a_missing_root_lists_nothing_rather_than_raising(tmp_path):
    assert list_checkpoints(tmp_path / "absent") == []


# --- the corruption scenario this exists for -------------------------------
def test_a_failed_write_at_step_200_leaves_step_100_resumable(tmp_path):
    """The single most important test for Colab: a kill during a checkpoint write must
    not cost the previous checkpoint."""
    make_checkpoint(tmp_path, 100)
    update_latest_pointer(tmp_path, tmp_path / step_dirname(100), 100)

    # Step 200 dies mid-write, leaving staging behind.
    staging = tmp_path / ".step_000200.incomplete"
    staging.mkdir()
    (staging / "model.safetensors").write_bytes(b"half a model")

    assert is_complete(tmp_path / step_dirname(100))
    assert read_latest_pointer(tmp_path)["step"] == 100
    assert resolve_checkpoint(tmp_path, "latest").name == "step_000100"
    assert not is_complete(staging)
    assert [p.name for p in list_checkpoints(tmp_path)] == ["step_000100"]


def test_a_partial_directory_named_like_a_checkpoint_is_never_resumable(tmp_path):
    """The dangerous shape: it exists, it is the newest, and it cannot be loaded."""
    make_checkpoint(tmp_path, 100)
    update_latest_pointer(tmp_path, tmp_path / step_dirname(100), 100)
    partial = tmp_path / step_dirname(200)
    partial.mkdir()
    (partial / "model.safetensors").write_bytes(b"half a model")

    assert not is_complete(partial)
    assert resolve_checkpoint(tmp_path, "latest").name == "step_000100"
    assert resolve_checkpoint(tmp_path, "200") is None


def test_startup_removes_leftover_staging_directories(tmp_path):
    make_checkpoint(tmp_path, 100)
    (tmp_path / ".step_000200.incomplete").mkdir()
    (tmp_path / ".step_000300.incomplete").mkdir()

    removed = cleanup_incomplete(tmp_path)

    assert sorted(removed) == [".step_000200.incomplete", ".step_000300.incomplete"]
    assert is_complete(tmp_path / step_dirname(100)), "a complete checkpoint is untouched"


def test_cleanup_never_touches_a_complete_checkpoint(tmp_path):
    make_checkpoint(tmp_path, 100)
    make_checkpoint(tmp_path, 200)
    assert cleanup_incomplete(tmp_path) == []
    assert len(list_checkpoints(tmp_path)) == 2


def test_the_latest_pointer_refuses_to_name_an_incomplete_checkpoint(tmp_path):
    partial = make_checkpoint(tmp_path, 200, complete=False)
    with pytest.raises(ValueError, match="not a complete checkpoint"):
        update_latest_pointer(tmp_path, partial, 200)


def test_a_stale_pointer_falls_back_to_a_checkpoint_that_verifies(tmp_path):
    """Drive copies can arrive out of order. A pointer naming something absent must not
    strand a checkpoint that is sitting right there."""
    make_checkpoint(tmp_path, 100)
    (tmp_path / "latest.json").write_text(
        json.dumps({"step": 999, "path": "step_000999", "complete": True}), encoding="utf-8"
    )
    assert resolve_checkpoint(tmp_path, "latest").name == "step_000100"


def test_a_corrupt_pointer_is_ignored_rather_than_fatal(tmp_path):
    make_checkpoint(tmp_path, 100)
    (tmp_path / "latest.json").write_text("{truncated", encoding="utf-8")
    assert read_latest_pointer(tmp_path) is None
    assert resolve_checkpoint(tmp_path, "latest").name == "step_000100"


def test_resolving_with_nothing_present_returns_none_rather_than_guessing(tmp_path):
    assert resolve_checkpoint(tmp_path, "latest") is None
    assert resolve_checkpoint(tmp_path, "400") is None
    assert resolve_checkpoint(tmp_path, "/nonexistent/path") is None


def test_a_checkpoint_can_be_named_by_step_number_or_path(tmp_path):
    directory = make_checkpoint(tmp_path, 400)
    assert resolve_checkpoint(tmp_path, "400") == directory
    assert resolve_checkpoint(tmp_path, str(directory)) == directory
    assert resolve_checkpoint(tmp_path, "step_000400") == directory


# --- atomic JSON -----------------------------------------------------------
def test_config_digests_are_stable_across_key_order():
    assert config_sha256({"a": 1, "b": 2}) == config_sha256({"b": 2, "a": 1})
    assert config_sha256({"a": 1}) != config_sha256({"a": 2})


def test_atomic_write_leaves_no_temporary_files(tmp_path):
    from qwen_distill.training.checkpoints import atomic_write_json

    atomic_write_json(tmp_path / "latest.json", {"step": 5})
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))["step"] == 5
    assert [p.name for p in tmp_path.iterdir()] == ["latest.json"]


# --- data position ---------------------------------------------------------
def test_the_batch_stream_resumes_to_an_identical_sequence():
    """Without this, a resumed run rewinds to epoch 0 and re-trains on seen data."""
    sequences = [[i] * 4 for i in range(50)]
    reference = ResumableBatchSampler(sequences, 4, seed=7)
    uninterrupted = [next(reference) for _ in range(30)]

    interrupted = ResumableBatchSampler(sequences, 4, seed=7)
    first = [next(interrupted) for _ in range(12)]
    saved = interrupted.state_dict()

    restored = ResumableBatchSampler(sequences, 4, seed=7)
    restored.load_state_dict(saved)

    assert first + [next(restored) for _ in range(18)] == uninterrupted
    assert restored.epoch > 0, "the test must cross an epoch boundary to be meaningful"


def test_the_sampler_refuses_a_position_from_a_different_corpus():
    """Resuming onto changed data would corrupt the run quietly."""
    sampler = ResumableBatchSampler([[1]] * 20, 2, seed=0)
    with pytest.raises(ValueError, match="data changed"):
        sampler.load_state_dict({"epoch": 0, "index": 0, "n_sequences": 999, "batch_size": 2})


def test_the_sampler_refuses_a_position_from_a_different_batch_size():
    sampler = ResumableBatchSampler([[1]] * 20, 2, seed=0)
    with pytest.raises(ValueError, match="batch indices do not carry"):
        sampler.load_state_dict({"epoch": 0, "index": 3, "n_sequences": 20, "batch_size": 8})


def test_the_sampler_reports_a_corpus_too_small_for_its_batch():
    sampler = ResumableBatchSampler([[1]] * 2, 8, seed=0)
    with pytest.raises(ValueError, match="cannot fill a batch"):
        next(sampler)


# --- progress records ------------------------------------------------------
def test_progress_records_are_appended_and_published(tmp_path):
    writer = ProgressWriter(tmp_path, git_commit="abc123", config_sha256="deadbeef")
    writer.write({"step": 25, "loss": 1.5})
    writer.write({"step": 50, "loss": 1.4})

    latest = writer.read_latest()
    assert latest["step"] == 50
    assert latest["status"] == "completed_step"
    assert latest["git_commit"] == "abc123"
    assert latest["config_sha256"] == "deadbeef"
    assert [r["step"] for r in writer.read_history()] == [25, 50]


def test_a_truncated_final_line_costs_one_record_not_the_history(tmp_path):
    """A kill mid-append leaves a partial line. The rest must still be readable."""
    writer = ProgressWriter(tmp_path)
    writer.write({"step": 25, "loss": 1.5})
    writer.write({"step": 50, "loss": 1.4})
    with open(writer.metrics_path, "a", encoding="utf-8") as stream:
        stream.write('{"step": 75, "loss": 1.3')  # killed mid-write

    assert [r["step"] for r in writer.read_history()] == [25, 50]


def test_progress_survives_without_any_prior_run(tmp_path):
    writer = ProgressWriter(tmp_path / "fresh")
    assert writer.read_latest() is None
    assert writer.read_history() == []


def test_progress_records_carry_no_model_weights(tmp_path):
    """The whole point of the split: these must stay kilobytes so they can be frequent."""
    writer = ProgressWriter(tmp_path)
    writer.write({"step": 25, "loss": 1.5, "tokens_seen": 100_000})
    assert writer.metrics_path.stat().st_size < 4096


# --- metadata --------------------------------------------------------------
def test_metadata_round_trips_and_tolerates_unknown_keys():
    metadata = CheckpointMetadata(step=400, tokens_seen=8_192_000, precision="fp16")
    payload = metadata.to_dict()
    payload["a_field_from_a_future_version"] = True

    restored = CheckpointMetadata.from_dict(payload)

    assert restored.step == 400
    assert restored.tokens_seen == 8_192_000
    assert restored.precision == "fp16"


def test_metadata_starts_incomplete():
    """`complete` is set by the writer as the last act, never by construction."""
    assert CheckpointMetadata(step=1).complete is False


# --- real save/load round trip ---------------------------------------------
@requires_stack
def test_a_saved_checkpoint_restores_every_component(tmp_path):
    import torch

    from qwen_distill.training.checkpoints import (
        capture_rng_state,
        load_checkpoint,
        restore_rng_state,
        save_checkpoint,
    )

    model = torch.nn.Linear(8, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=10)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    # Take real steps so the optimizer and scheduler hold non-trivial state.
    for _ in range(3):
        model(torch.randn(2, 8)).sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    path = save_checkpoint(
        tmp_path, 3, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        training_state={"step": 3, "tokens_seen": 96, "data_state": {"epoch": 0, "index": 7}},
        config={"name": "test"}, rng_state=capture_rng_state(),
        metadata=CheckpointMetadata(step=3, precision="fp32"),
    )
    assert is_complete(path)
    expected_lr = scheduler.get_last_lr()

    fresh_model = torch.nn.Linear(8, 4)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    fresh_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        fresh_optimizer, max_lr=1e-3, total_steps=10
    )
    fresh_scaler = torch.amp.GradScaler("cpu", enabled=False)

    loaded = load_checkpoint(
        path, model=fresh_model, optimizer=fresh_optimizer,
        scheduler=fresh_scheduler, scaler=fresh_scaler,
    )

    assert set(loaded["restored"]) == {"model", "optimizer", "scheduler", "scaler", "rng"}
    assert loaded["step"] == 3
    assert loaded["training_state"]["tokens_seen"] == 96
    assert loaded["training_state"]["data_state"] == {"epoch": 0, "index": 7}
    for original, restored in zip(model.parameters(), fresh_model.parameters(), strict=True):
        assert torch.equal(original, restored)
    assert fresh_scheduler.get_last_lr() == expected_lr, "the LR schedule must not restart"
    assert restore_rng_state(loaded["rng_state"])


@requires_stack
def test_loading_refuses_an_incomplete_checkpoint(tmp_path):
    import torch

    from qwen_distill.training.checkpoints import load_checkpoint

    partial = make_checkpoint(tmp_path, 100, complete=False)
    with pytest.raises(ValueError, match="not a complete checkpoint"):
        load_checkpoint(partial, model=torch.nn.Linear(4, 4))


@requires_stack
def test_an_interrupted_save_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    """Simulates the exact Colab failure: the process dies partway through a write."""
    import torch

    from qwen_distill.training import checkpoints as module

    model = torch.nn.Linear(8, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    good = module.save_checkpoint(
        tmp_path, 100, model=model, optimizer=optimizer, training_state={"step": 100},
    )
    assert is_complete(good)

    original_dump = torch.save

    def die_partway(payload, target, *args, **kwargs):
        # Fail while writing the optimizer, after the weights have landed.
        raise OSError("simulated Colab disconnect")

    monkeypatch.setattr(torch, "save", die_partway)
    with pytest.raises(OSError, match="simulated Colab disconnect"):
        module.save_checkpoint(
            tmp_path, 200, model=model, optimizer=optimizer, training_state={"step": 200},
        )
    monkeypatch.setattr(torch, "save", original_dump)

    assert is_complete(good), "the previous checkpoint must survive"
    assert read_latest_pointer(tmp_path)["step"] == 100
    assert resolve_checkpoint(tmp_path, "latest").name == "step_000100"
    assert not (tmp_path / step_dirname(200)).exists(), "no half-written directory remains"
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".incomplete")] == []
