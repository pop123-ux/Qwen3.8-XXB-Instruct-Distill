"""Tests for persisting checkpoints beyond the Colab runtime.

A Level-2 run reached **step 1925 of 2000** on a T4 — no OOM, ~2050 tokens/s — and lost
everything, because it ran with `persistent copy : off (local only)`. The checkpoints
were written correctly, atomically, to a filesystem that ceased to exist.

What these pin:

* an incomplete checkpoint is never copied, and never becomes the newest thing on Drive;
* a failed copy does not advance the persistent pointer, and is not reported as success;
* an unmounted Drive is caught **before training starts**, because `mkdir(parents=True)`
  on `/content/drive/MyDrive/...` silently succeeds when Drive is absent and turns
  "persisted" into "written to the disk that is about to disappear";
* the whole loop — persist, lose local disk, restore, resume — works.

All CPU, all in temporary directories. No Drive, no GPU, no network.
"""

from __future__ import annotations

import json

import pytest
from conftest import HAS_STACK, make_checkpoint_dir

from qwen_distill.training.checkpoints import (
    COMPLETE_MARKER,
    is_complete,
    list_checkpoints,
    read_latest_pointer,
    step_dirname,
)
from qwen_distill.training.persist import (
    MOUNT_ROOTS,
    PersistenceTarget,
    persist_checkpoint,
    persist_run_metadata,
    persistent_status,
    preflight,
    restore_run,
)


def make_checkpoint(root, step: int, *, complete: bool = True):
    """One shared builder — see ``conftest.make_checkpoint_dir``."""
    return make_checkpoint_dir(root, step, complete=complete)


# --- layout ----------------------------------------------------------------
def test_the_destination_is_the_run_directory_in_canonical_layout(tmp_path):
    """The persistent copy mirrors the local run exactly, so recovery is a plain copy."""
    target = PersistenceTarget(tmp_path / "qwen-distill" / "t4_level2_100m_ckpt")
    assert target.checkpoints == target.root / "checkpoints"
    assert target.progress == target.root / "progress"
    assert target.pointer == target.root / "checkpoints" / "latest.json"


def test_a_path_aimed_at_the_checkpoints_subdirectory_is_tolerated(tmp_path):
    """Nesting checkpoints/checkpoints/ would be a confusing way to punish an old path."""
    target = PersistenceTarget(tmp_path / "run" / "checkpoints")
    assert target.root == tmp_path / "run"
    assert target.checkpoints == tmp_path / "run" / "checkpoints"


# --- the unmounted-Drive trap ---------------------------------------------
def test_an_unmounted_drive_is_refused_rather_than_silently_written_to():
    """mkdir(parents=True) succeeds on an unmounted Drive path and creates an ordinary
    local directory. Every checkpoint would report "persisted" and then vanish."""
    result = preflight("/content/drive/MyDrive/qwen-distill/t4_level2_100m_ckpt")

    assert not result.usable
    assert result.mount_root == "/content/drive/MyDrive"
    assert "not mounted" in result.reason
    assert "drive.mount" in result.reason, "the message must say how to fix it"
    assert "UNUSABLE" in result.render()


@pytest.mark.parametrize("mount", MOUNT_ROOTS)
def test_every_known_mount_root_is_checked(mount):
    result = preflight(f"{mount}/qwen-distill/run")
    assert result.mount_root == mount
    assert not result.usable


def test_an_ordinary_writable_directory_passes_preflight(tmp_path):
    result = preflight(tmp_path / "drive" / "qwen-distill" / "run")
    assert result.usable
    assert result.reason is None
    assert (tmp_path / "drive" / "qwen-distill" / "run" / "checkpoints").is_dir()


def test_preflight_reports_what_is_already_there(tmp_path):
    """A fresh session should see prior progress before it trains a single step."""
    destination = tmp_path / "drive" / "run"
    make_checkpoint(destination / "checkpoints", 200)
    make_checkpoint(destination / "checkpoints", 400)

    result = preflight(destination)

    assert result.usable
    assert result.existing_checkpoints == [step_dirname(200), step_dirname(400)]
    assert "verified resumable" in result.render()


def test_preflight_leaves_no_probe_file_behind(tmp_path):
    destination = tmp_path / "drive" / "run"
    preflight(destination)
    assert list((destination / "checkpoints").iterdir()) == []


# --- requirement: an incomplete checkpoint is never persisted --------------
def test_an_incomplete_checkpoint_is_never_persisted(tmp_path):
    """A partial checkpoint on Drive is worse than none: it is the newest thing there,
    and recovery reaches for the newest."""
    local = tmp_path / "local" / "checkpoints"
    destination = tmp_path / "drive" / "run"
    partial = make_checkpoint(local, 400, complete=False)

    result = persist_checkpoint(partial, destination)
    assert not result.verified
    assert "refusing to persist" in result.failure
    assert "persisted ->" not in result.render()

    assert not (destination / "checkpoints" / step_dirname(400)).exists()
    assert read_latest_pointer(destination / "checkpoints") is None


def test_a_checkpoint_missing_its_marker_is_never_persisted(tmp_path):
    local = tmp_path / "local" / "checkpoints"
    destination = tmp_path / "drive" / "run"
    checkpoint = make_checkpoint(local, 400)
    (checkpoint / COMPLETE_MARKER).unlink()

    result = persist_checkpoint(checkpoint, destination)
    assert not result.verified
    assert "COMPLETE" in str(result.failure)
    assert not (destination / "checkpoints" / step_dirname(400)).exists()


# --- requirement: a failed copy does not advance the pointer ---------------
def test_a_failed_copy_leaves_the_previous_persisted_pointer_intact(tmp_path, monkeypatch):
    """The exact failure mode: step 400 copies, step 600's copy dies mid-flight. Drive
    must still name 400, and must not hold a partial 600."""
    import shutil

    local = tmp_path / "local" / "checkpoints"
    destination = tmp_path / "drive" / "run"

    good = make_checkpoint(local, 400)
    assert persist_checkpoint(good, destination).verified
    assert read_latest_pointer(destination / "checkpoints")["step"] == 400

    later = make_checkpoint(local, 600)

    def die(*args, **kwargs):
        raise OSError("simulated Drive failure")

    monkeypatch.setattr(shutil, "copyfileobj", die)
    result = persist_checkpoint(later, destination)
    monkeypatch.undo()

    assert not result.verified
    assert "simulated Drive failure" in result.failure
    assert not result.pointer_updated

    pointer = read_latest_pointer(destination / "checkpoints")
    assert pointer["step"] == 400, "the pointer must not advance past a failed copy"
    assert pointer["path"] == step_dirname(400)
    assert is_complete(destination / "checkpoints" / step_dirname(400))
    assert not (destination / "checkpoints" / step_dirname(600)).exists()

    # The staging directory is deliberately LEFT behind, named `.incomplete`. It is
    # skipped by every discovery path, so it cannot be resumed from, and it is the only
    # forensic evidence of what a failed copy actually managed to write. The next attempt
    # at this step clears it.
    staging = [p.name for p in (destination / "checkpoints").iterdir()
               if p.name.endswith(".incomplete")]
    assert staging == [f".{step_dirname(600)}.incomplete"]
    assert result.staging_left_behind
    assert list_checkpoints(destination / "checkpoints") == [
        destination / "checkpoints" / step_dirname(400)
    ]


def test_a_copy_that_arrives_corrupt_does_not_advance_the_pointer(tmp_path, monkeypatch):
    """Verification is of what landed, not of what we believe we sent — a truncated
    write on a full Drive must not be advertised as the newest good checkpoint."""
    import shutil

    local = tmp_path / "local" / "checkpoints"
    destination = tmp_path / "drive" / "run"
    persist_checkpoint(make_checkpoint(local, 400), destination)

    later = make_checkpoint(local, 600)
    real_copyfileobj = shutil.copyfileobj

    def truncate_the_weights(reader, writer, length=0):
        """Write the model file short, the way a full or flaky Drive does.

        This is the failure the old code could not see: every filename arrives, the
        marker arrives, and `is_file()` is True for all of them. Only comparing the
        destination's bytes against the source's catches it.
        """
        if writer.name.endswith("model.safetensors"):
            writer.write(reader.read(64))
            return None
        return real_copyfileobj(reader, writer, length)

    monkeypatch.setattr(shutil, "copyfileobj", truncate_the_weights)
    result = persist_checkpoint(later, destination)
    monkeypatch.undo()

    assert not result.verified
    assert not result.pointer_updated
    truncated = next(f for f in result.files if f.name == "model.safetensors")
    assert not truncated.ok
    assert truncated.size_bytes == 64
    assert "persisted ->" not in result.render()
    assert "latest pointer NOT updated" in result.render()

    assert read_latest_pointer(destination / "checkpoints")["step"] == 400
    assert not (destination / "checkpoints" / step_dirname(600)).exists()


def test_the_persistent_pointer_names_a_checkpoint_that_is_really_there(tmp_path):
    local = tmp_path / "local" / "checkpoints"
    destination = tmp_path / "drive" / "run"
    persist_checkpoint(make_checkpoint(local, 200), destination)

    pointer = read_latest_pointer(destination / "checkpoints")
    named = destination / "checkpoints" / pointer["path"]

    assert pointer["complete"] is True
    assert named.is_dir()
    assert is_complete(named)


def test_persisting_reads_the_step_from_the_copy_itself(tmp_path):
    """So the pointer is right even with no local pointer to mirror."""
    local = tmp_path / "local" / "checkpoints"
    destination = tmp_path / "drive" / "run"
    persist_checkpoint(make_checkpoint(local, 1800), destination)
    assert read_latest_pointer(destination / "checkpoints")["step"] == 1800


# --- run metadata ----------------------------------------------------------
def test_run_metadata_is_mirrored_so_a_checkpoint_stays_interpretable(tmp_path):
    run = tmp_path / "local"
    (run / "progress").mkdir(parents=True)
    (run / "metrics.jsonl").write_text('{"step": 200}\n', encoding="utf-8")
    (run / "summary.json").write_text('{"experiment": "x"}', encoding="utf-8")
    (run / "progress" / "latest.json").write_text('{"step": 200}', encoding="utf-8")
    destination = tmp_path / "drive" / "run"

    copied = persist_run_metadata(run, destination)

    assert "metrics.jsonl" in copied
    assert "summary.json" in copied
    assert "progress/latest.json" in copied
    assert (destination / "metrics.jsonl").is_file()
    assert (destination / "progress" / "latest.json").is_file()


# --- status and restore ----------------------------------------------------
def test_status_answers_from_persistent_storage_alone(tmp_path):
    """The fresh-Colab question: local disk is empty, what survived?"""
    destination = tmp_path / "drive" / "run"
    local = tmp_path / "local" / "checkpoints"
    persist_checkpoint(make_checkpoint(local, 200), destination)
    persist_checkpoint(make_checkpoint(local, 400), destination)
    (destination / "progress").mkdir(parents=True, exist_ok=True)
    (destination / "progress" / "latest.json").write_text(
        json.dumps({"step": 400, "validation_bits_per_byte": 1.279}), encoding="utf-8"
    )

    status = persistent_status(destination)

    assert status["exists"]
    assert status["checkpoints"] == [step_dirname(200), step_dirname(400)]
    assert status["resumable_step"] == 400
    assert status["latest_progress"]["validation_bits_per_byte"] == 1.279


def test_status_on_a_destination_that_is_not_there(tmp_path):
    status = persistent_status(tmp_path / "never-created")
    assert not status["exists"]
    assert status["checkpoints"] == []
    assert status["resumable_checkpoint"] is None


def test_restore_brings_back_only_complete_checkpoints(tmp_path):
    """A copy interrupted by a dying runtime must never become the thing resumed from."""
    destination = tmp_path / "drive" / "run"
    local_source = tmp_path / "src" / "checkpoints"
    persist_checkpoint(make_checkpoint(local_source, 200), destination)
    # A partial arrival on Drive, as a killed copy would leave.
    make_checkpoint(destination / "checkpoints", 400, complete=False)

    result = restore_run(destination, tmp_path / "fresh")

    assert result["restored"] == [step_dirname(200)]
    assert result["skipped_incomplete"] == [step_dirname(400)]
    assert result["pointer"]["step"] == 200
    assert is_complete(tmp_path / "fresh" / "checkpoints" / step_dirname(200))
    assert not (tmp_path / "fresh" / "checkpoints" / step_dirname(400)).exists()


def test_restore_rebuilds_the_local_pointer_from_what_arrived(tmp_path):
    """Copying the remote pointer would risk naming a checkpoint that failed to restore."""
    destination = tmp_path / "drive" / "run"
    source = tmp_path / "src" / "checkpoints"
    persist_checkpoint(make_checkpoint(source, 200), destination)
    persist_checkpoint(make_checkpoint(source, 400), destination)

    restore_run(destination, tmp_path / "fresh")

    pointer = read_latest_pointer(tmp_path / "fresh" / "checkpoints")
    assert pointer["step"] == 400
    assert is_complete(tmp_path / "fresh" / "checkpoints" / pointer["path"])


def test_restore_is_idempotent(tmp_path):
    destination = tmp_path / "drive" / "run"
    source = tmp_path / "src" / "checkpoints"
    persist_checkpoint(make_checkpoint(source, 200), destination)

    first = restore_run(destination, tmp_path / "fresh")
    second = restore_run(destination, tmp_path / "fresh")

    assert first["restored"] == [step_dirname(200)]
    assert second["restored"] == [], "an already-verified checkpoint is not re-copied"
    assert second["available_locally"] == [step_dirname(200)]


def test_restoring_from_nothing_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing to restore"):
        restore_run(tmp_path / "absent", tmp_path / "fresh")


# --- end to end ------------------------------------------------------------
pytestmark_stack = pytest.mark.skipif(not HAS_STACK, reason="requires torch and transformers")

TINY = {
    "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
    "vocab_size": 256, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 32, "linear_num_key_heads": 1, "linear_num_value_heads": 2,
    "linear_key_head_dim": 32, "linear_value_head_dim": 32,
    "full_attention_interval": 4, "tie_word_embeddings": True,
    "max_position_embeddings": 512,
}


def make_config(output, destination, *, max_steps, resume=None, save_every=10):
    from qwen_distill.training.config import ExperimentConfig, ModelConfig

    config = ExperimentConfig(name="persist")
    config.model = ModelConfig(architecture=dict(TINY))
    config.data.text_corpus = True
    config.data.max_sequence_length = 64
    config.data.procedural_bytes = 20_000
    config.training.max_steps = max_steps
    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 1
    config.training.save_every = save_every
    config.training.log_every = 5
    config.training.eval_every = max_steps
    config.training.precision = "fp32"
    config.training.strategy = "full"
    config.training.objective = "sft"
    config.training.gradient_checkpointing = True
    config.training.persistent_backup = str(destination) if destination else None
    config.runtime.output_dir = str(output)
    config.runtime.device = "cpu"
    config.runtime.resume_from = resume
    return config


def run(config):
    from qwen_distill.training.trainer import train

    return train(config, config.model.resolve_spec())


@pytestmark_stack
def test_persist_lose_local_restore_resume(tmp_path):
    """The whole point: train, lose the runtime, come back, continue from Drive."""
    import shutil

    local = tmp_path / "local"
    drive = tmp_path / "drive" / "qwen-distill" / "run"

    assert run(make_config(local, drive, max_steps=20)) == 0

    # The canonical layout arrived intact.
    assert is_complete(drive / "checkpoints" / step_dirname(20))
    assert read_latest_pointer(drive / "checkpoints")["step"] == 20
    assert (drive / "metrics.jsonl").is_file()
    assert (drive / "progress" / "latest.json").is_file()

    # The runtime dies.
    shutil.rmtree(local)
    assert not local.exists()

    # A fresh session can still answer where the run got to.
    status = persistent_status(drive)
    assert status["resumable_step"] == 20

    # Restore and continue.
    restored = restore_run(drive, local)
    assert step_dirname(20) in restored["restored"]
    assert run(make_config(local, drive, max_steps=40, resume="latest")) == 0

    assert read_latest_pointer(local / "checkpoints")["step"] == 40
    assert read_latest_pointer(drive / "checkpoints")["step"] == 40
    assert is_complete(drive / "checkpoints" / step_dirname(40))


@pytestmark_stack
def test_a_run_records_which_checkpoints_are_durable(tmp_path):
    """Reading summary.json later must not require trusting that persistence worked."""
    local = tmp_path / "local"
    drive = tmp_path / "drive" / "run"
    assert run(make_config(local, drive, max_steps=20)) == 0

    summary = json.loads((local / "summary.json").read_text(encoding="utf-8"))
    persistence = summary["persistence"]

    assert persistence["enabled"] is True
    assert persistence["destination"] == str(drive)
    assert persistence["failed"] == []
    assert persistence["all_checkpoints_persisted"] is True
    assert step_dirname(20) in persistence["persisted"]


@pytestmark_stack
def test_a_local_only_run_says_so_in_its_summary(tmp_path):
    """The 1925-step loss was a run that reported "off (local only)" and was believed."""
    local = tmp_path / "local"
    assert run(make_config(local, None, max_steps=10)) == 0

    persistence = json.loads(
        (local / "summary.json").read_text(encoding="utf-8")
    )["persistence"]

    assert persistence["enabled"] is False
    assert persistence["all_checkpoints_persisted"] is False


@pytestmark_stack
def test_training_refuses_to_start_when_persistence_cannot_work(tmp_path, monkeypatch):
    """Better to stop at step 0 than to train 1925 steps into a directory that is about
    to disappear while reporting that it is safe."""
    def unusable(_destination):
        from qwen_distill.training.persist import PreflightResult

        return PreflightResult(destination=tmp_path / "drive", usable=False,
                               reason="Drive is not mounted")

    monkeypatch.setattr("qwen_distill.training.persist.preflight", unusable)
    assert run(make_config(tmp_path / "local", tmp_path / "drive", max_steps=10)) == 2


@pytestmark_stack
def test_a_run_without_persistence_still_works(tmp_path):
    """Requirement 7: no Drive, no network, no mounted volume."""
    local = tmp_path / "local"
    assert run(make_config(local, None, max_steps=10)) == 0
    assert is_complete(local / "checkpoints" / step_dirname(10))
