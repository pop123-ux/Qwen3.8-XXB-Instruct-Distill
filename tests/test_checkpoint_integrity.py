"""The fourteen ways a checkpoint can be a lie, and the guarantee that catches each.

A Level-2R run reached ~step 800. Persistent storage held `step_000200`, `step_000400`
and `step_000600`, each carrying `COMPLETE`, `config.json`, `metadata.json`, `rng.pt`,
`scaler.pt`, `scheduler.pt` and `training_state.json` — and no `model.safetensors`, no
`optimizer.pt`. `latest.json` recorded `"complete": true`.

The cause is not recoverable after the fact and does not need to be. Manual deletion, an
interrupted copy, Drive sync, a full disk and a killed process all produce a directory
that the old check called complete, because the old check asked whether filenames existed
and `Path.is_file()` is `True` for a zero-byte file.

Each test below is named for a failure mode from the hardening mandate. Every one of them
passed silently before.
"""

from __future__ import annotations

import json
import shutil

import pytest
from conftest import PARAMETER_COUNT, make_checkpoint_dir

from qwen_distill.training.checkpoint_validation import (
    LOAD,
    MANIFEST,
    MANIFEST_FILENAME,
    STRUCTURE,
    build_manifest,
    read_manifest,
    resolve_latest,
    sha256_file,
    validate_checkpoint_dir,
    validate_checkpoint_root,
)
from qwen_distill.training.checkpoints import (
    COMPLETE_MARKER,
    atomic_write_json,
    is_complete,
    list_checkpoints,
)
from qwen_distill.training.persist import (
    persist_checkpoint,
    persistent_status,
    preflight,
    restore_run,
)


def _rewrite_manifest(checkpoint):
    """Recompute the manifest so a deliberate edit is not caught by the digest instead.

    Several tests below need to isolate one failure mode. Without this, editing a file
    trips the checksum check first and the test would pass for the wrong reason.
    """
    step = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))["step"]
    atomic_write_json(
        checkpoint / MANIFEST_FILENAME,
        build_manifest(checkpoint, step=step, parameter_count=PARAMETER_COUNT),
    )


# ----------------------------------------------------------------------------------
# A–H: the checkpoint itself
# ----------------------------------------------------------------------------------


def test_A_model_safetensors_missing(tmp_path):
    """The exact shape of the observed failure."""
    checkpoint = make_checkpoint_dir(tmp_path, 600)
    (checkpoint / "model.safetensors").unlink()

    validation = validate_checkpoint_dir(checkpoint)
    assert not validation.valid
    assert validation.missing_files == ["model.safetensors"]
    assert "model.safetensors" in validation.invalid_reason
    assert not is_complete(checkpoint)


def test_B_optimizer_pt_missing(tmp_path):
    """Without optimizer state a checkpoint may load, but it cannot continue training —
    and calling it resumable is how a resume silently restarts the moments from zero."""
    checkpoint = make_checkpoint_dir(tmp_path, 600)
    (checkpoint / "optimizer.pt").unlink()

    validation = validate_checkpoint_dir(checkpoint)
    assert not validation.valid
    assert not validation.resumable
    assert "optimizer.pt" in validation.invalid_reason

    # Explicitly asking for an inference-only checkpoint is a different question, and it
    # still reports that resuming is impossible.
    inference = validate_checkpoint_dir(checkpoint, require_resumable=False)
    assert not inference.valid, "the metadata says optimizer.pt was written; it is gone"


def test_C_model_safetensors_zero_bytes(tmp_path):
    """`Path.is_file()` is True for an empty file. That is the whole bug."""
    checkpoint = make_checkpoint_dir(tmp_path, 600)
    (checkpoint / "model.safetensors").write_bytes(b"")

    validation = validate_checkpoint_dir(checkpoint)
    assert not validation.valid
    assert validation.zero_length_files == ["model.safetensors"]
    assert (checkpoint / "model.safetensors").is_file(), "the old check would pass here"


def test_D_optimizer_pt_zero_bytes(tmp_path):
    checkpoint = make_checkpoint_dir(tmp_path, 600)
    (checkpoint / "optimizer.pt").write_bytes(b"")

    validation = validate_checkpoint_dir(checkpoint)
    assert not validation.valid
    assert validation.zero_length_files == ["optimizer.pt"]


def test_E_truncated_model_file(tmp_path):
    """50 bytes is not a small model. Caught by the size floor before any hashing."""
    checkpoint = make_checkpoint_dir(tmp_path, 600, manifest=False)
    (checkpoint / "model.safetensors").write_bytes(b"\x00" * 50)

    validation = validate_checkpoint_dir(checkpoint, level=STRUCTURE)
    assert not validation.valid
    assert validation.implausible_sizes
    assert "50 bytes" in validation.invalid_reason


def test_E2_truncated_model_file_above_the_format_floor(tmp_path):
    """Truncated to a size that clears the absolute floor, and is still impossible for
    the recorded parameter count. This is what the parameter band is for."""
    checkpoint = make_checkpoint_dir(tmp_path, 600, manifest=False)
    (checkpoint / "model.safetensors").write_bytes(b"\x00" * 4096)

    validation = validate_checkpoint_dir(checkpoint, level=STRUCTURE)
    assert not validation.valid
    assert "too small for" in " ".join(
        p for f in validation.files for p in f.problems
    )


def test_F_truncated_optimizer_file(tmp_path):
    checkpoint = make_checkpoint_dir(tmp_path, 600)
    (checkpoint / "optimizer.pt").write_bytes(b"\x00" * 128)

    validation = validate_checkpoint_dir(checkpoint)
    assert not validation.valid
    assert validation.size_mismatches or validation.implausible_sizes


def test_G_corrupted_checksum(tmp_path):
    """Same size, different bytes. No existence check and no size check can see this;
    only the recorded digest can."""
    checkpoint = make_checkpoint_dir(tmp_path, 600)
    weights = checkpoint / "model.safetensors"
    original_size = weights.stat().st_size
    weights.write_bytes(b"\xff" * original_size)
    assert weights.stat().st_size == original_size

    assert validate_checkpoint_dir(checkpoint, level=STRUCTURE).valid, (
        "structure level cannot see this — the file is present, non-empty and the right size"
    )
    validation = validate_checkpoint_dir(checkpoint, level=MANIFEST)
    assert not validation.valid
    assert validation.checksum_mismatches
    assert "sha256" in validation.invalid_reason.lower() or "digest" in validation.invalid_reason


def test_H_complete_marker_without_the_required_files(tmp_path):
    """The observed directory, reconstructed exactly: every small file plus COMPLETE,
    metadata claiming complete, and no weights."""
    checkpoint = make_checkpoint_dir(
        tmp_path, 600,
        files=("model.safetensors", "optimizer.pt", "scheduler.pt", "scaler.pt",
               "rng.pt", "training_state.json", "config.json", "metadata.json"),
    )
    (checkpoint / "model.safetensors").unlink()
    (checkpoint / "optimizer.pt").unlink()

    assert (checkpoint / COMPLETE_MARKER).is_file()
    assert json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))["complete"]

    validation = validate_checkpoint_dir(checkpoint)
    assert not validation.valid
    assert set(validation.missing_files) == {"model.safetensors", "optimizer.pt"}
    assert not is_complete(checkpoint)


def test_a_pre_manifest_checkpoint_still_detects_deletion(tmp_path):
    """The three broken checkpoints on Drive predate manifests. `metadata.contents` is
    the checkpoint's own statement of what it holds, and it is enough."""
    checkpoint = make_checkpoint_dir(tmp_path, 600, manifest=False)
    assert read_manifest(checkpoint) is None
    (checkpoint / "model.safetensors").unlink()

    validation = validate_checkpoint_dir(checkpoint)
    assert not validation.valid
    assert "model.safetensors" in validation.missing_files
    assert "own metadata" in validation.invalid_reason


def test_an_optional_file_that_was_never_written_is_not_a_failure(tmp_path):
    """A CPU run has no AMP scaler. Absence is only a failure when the checkpoint says
    the file was written — otherwise every fp32 run would be reported as damaged."""
    checkpoint = make_checkpoint_dir(
        tmp_path, 600,
        files=("model.safetensors", "optimizer.pt", "training_state.json", "metadata.json"),
    )
    assert not (checkpoint / "scaler.pt").exists()
    assert validate_checkpoint_dir(checkpoint).valid


def test_a_deleted_optional_file_is_a_failure(tmp_path):
    """...but one that WAS written and is now gone is a deletion."""
    checkpoint = make_checkpoint_dir(
        tmp_path, 600,
        files=("model.safetensors", "optimizer.pt", "scaler.pt",
               "training_state.json", "metadata.json"),
    )
    (checkpoint / "scaler.pt").unlink()
    validation = validate_checkpoint_dir(checkpoint)
    assert not validation.valid
    assert validation.missing_files == ["scaler.pt"]


def test_a_healthy_checkpoint_passes_every_level(tmp_path):
    """The guard against a validator so strict that nothing is ever valid."""
    checkpoint = make_checkpoint_dir(tmp_path, 600)
    for level in (STRUCTURE, MANIFEST):
        validation = validate_checkpoint_dir(checkpoint, level=level)
        assert validation.valid, validation.invalid_reason
        assert validation.resumable
    assert "CHECKPOINT VALID" in validate_checkpoint_dir(checkpoint).render()


# ----------------------------------------------------------------------------------
# I, L, N: the pointer and the fallback
# ----------------------------------------------------------------------------------


def test_I_latest_json_points_at_an_invalid_checkpoint(tmp_path):
    """`latest.json` records a claim made at write time. The claim outlives the files."""
    make_checkpoint_dir(tmp_path, 200)
    make_checkpoint_dir(tmp_path, 400)
    damaged = make_checkpoint_dir(tmp_path, 600)
    atomic_write_json(tmp_path / "latest.json",
                      {"step": 600, "path": "step_000600", "complete": True})
    (damaged / "model.safetensors").unlink()

    resolution = resolve_latest(tmp_path)
    assert resolution.pointer_path == "step_000600"
    assert not resolution.pointer_valid
    assert resolution.fell_back
    assert resolution.resolved_step == 400

    rendered = resolution.render()
    assert "step_000600 is invalid" in rendered
    assert "falling back to step_000400" in rendered
    assert "resumable at step 400" in rendered


def test_L_a_valid_checkpoint_followed_by_an_invalid_newer_one(tmp_path):
    """Recovery reaches for the newest, which is exactly why the newest must be checked."""
    make_checkpoint_dir(tmp_path, 200)
    newer = make_checkpoint_dir(tmp_path, 400)
    (newer / "optimizer.pt").write_bytes(b"")

    assert list_checkpoints(tmp_path) == [tmp_path / "step_000200"]
    assert resolve_latest(tmp_path).resolved_step == 200


def test_no_valid_checkpoint_is_reported_rather_than_guessed(tmp_path):
    damaged = make_checkpoint_dir(tmp_path, 200)
    (damaged / "model.safetensors").unlink()
    resolution = resolve_latest(tmp_path)
    assert resolution.resolved is None
    assert not resolution.usable
    assert "nothing here can be resumed from" in resolution.render()


def test_the_inventory_counts_verified_checkpoints_not_directories(tmp_path):
    """"persistent checkpoints found: 4" must mean four you can recover from."""
    make_checkpoint_dir(tmp_path, 200)
    make_checkpoint_dir(tmp_path, 400)
    broken = make_checkpoint_dir(tmp_path, 600)
    make_checkpoint_dir(tmp_path, 800)
    (broken / "model.safetensors").unlink()

    inventory = resolve_latest(tmp_path).render_inventory()
    assert "4 (3 verified resumable)" in inventory
    assert "step_000600  INVALID" in inventory
    assert "newest valid checkpoint: step_000800" in inventory


# ----------------------------------------------------------------------------------
# J, K, M: persistence
# ----------------------------------------------------------------------------------


def test_M_backup_refuses_a_damaged_source(tmp_path):
    """Copying a damaged checkpoint would put it where recovery looks first."""
    local = tmp_path / "local" / "checkpoints"
    destination = tmp_path / "drive" / "run"
    damaged = make_checkpoint_dir(local, 400)
    (damaged / "model.safetensors").unlink()

    result = persist_checkpoint(damaged, destination)
    assert not result.verified
    assert "refusing to persist" in result.failure
    assert not (destination / "checkpoints" / "step_000400").exists()
    assert "persisted ->" not in result.render()


def test_J_a_destination_holding_only_metadata_is_never_advertised(tmp_path):
    """The observed state, at the destination: every small file arrives, the large ones
    do not. The pointer must not move and the copy must not be promoted."""
    local = tmp_path / "local" / "checkpoints"
    destination = tmp_path / "drive" / "run"
    persist_checkpoint(make_checkpoint_dir(local, 200), destination)

    later = make_checkpoint_dir(local, 400)
    real_copyfileobj = shutil.copyfileobj

    def drop_the_large_files(reader, writer, length=0):
        if writer.name.endswith(("model.safetensors", "optimizer.pt")):
            return None          # arrives as a zero-byte file, exactly as observed
        return real_copyfileobj(reader, writer, length)

    shutil.copyfileobj = drop_the_large_files
    try:
        result = persist_checkpoint(later, destination)
    finally:
        shutil.copyfileobj = real_copyfileobj

    assert not result.verified
    assert not result.pointer_updated
    failed = {f.name for f in result.failed_files}
    assert failed == {"model.safetensors", "optimizer.pt"}
    assert not (destination / "checkpoints" / "step_000400").exists()

    status = persistent_status(destination)
    assert status["checkpoints"] == ["step_000200"]
    assert status["resumable_step"] == 200


def test_K_an_interrupted_copy_leaves_an_incomplete_directory(tmp_path):
    """Left deliberately, named `.incomplete`: it is skipped by every discovery path and
    it is the only evidence of what the failed copy managed to write."""
    local = tmp_path / "local" / "checkpoints"
    destination = tmp_path / "drive" / "run"
    real_copyfileobj = shutil.copyfileobj

    def die(reader, writer, length=0):
        if writer.name.endswith("optimizer.pt"):
            raise OSError("simulated Drive failure mid-copy")
        return real_copyfileobj(reader, writer, length)

    shutil.copyfileobj = die
    try:
        result = persist_checkpoint(make_checkpoint_dir(local, 400), destination)
    finally:
        shutil.copyfileobj = real_copyfileobj

    assert not result.verified
    assert result.staging_left_behind
    staging = destination / "checkpoints" / ".step_000400.incomplete"
    assert staging.is_dir()
    assert not (staging / COMPLETE_MARKER).exists(), (
        "the marker is written only after verification, so a failed staging directory "
        "can never look finished"
    )
    assert list_checkpoints(destination / "checkpoints") == []
    assert resolve_latest(destination / "checkpoints").resolved is None


def test_the_marker_is_written_at_the_destination_after_verification(tmp_path):
    """Not copied with the other files. A staging directory that fails verification must
    never carry a marker that would make it look complete."""
    local = tmp_path / "local" / "checkpoints"
    destination = tmp_path / "drive" / "run"
    result = persist_checkpoint(make_checkpoint_dir(local, 400), destination)

    assert result.verified
    marker = (destination / "checkpoints" / "step_000400" / COMPLETE_MARKER)
    assert "verified at destination" in marker.read_text(encoding="utf-8")


def test_a_verified_copy_is_byte_identical_to_its_source(tmp_path):
    """The guarantee `persisted ->` now stands for."""
    local = tmp_path / "local" / "checkpoints"
    destination = tmp_path / "drive" / "run"
    source = make_checkpoint_dir(local, 400)
    result = persist_checkpoint(source, destination)

    assert result.verified and result.pointer_updated
    copy = destination / "checkpoints" / "step_400".replace("400", "000400")
    for name in ("model.safetensors", "optimizer.pt", "metadata.json"):
        assert sha256_file(source / name) == sha256_file(copy / name)
    assert "persistent verification: PASS" in result.render()
    assert "persisted ->" in result.render()


def test_persistence_verification_can_be_escalated_to_a_real_load(tmp_path):
    """`--level load` deserializes at the destination. The fixture's synthetic weights
    are not a real safetensors file, so this must fail rather than pass by accident."""
    local = tmp_path / "local" / "checkpoints"
    destination = tmp_path / "drive" / "run"
    result = persist_checkpoint(make_checkpoint_dir(local, 400), destination,
                                verify_level=LOAD)
    assert not result.verified
    assert not result.pointer_updated
    assert not (destination / "checkpoints" / "step_000400").exists()


# ----------------------------------------------------------------------------------
# N + cross-session recovery
# ----------------------------------------------------------------------------------


def test_N_restore_falls_back_to_the_newest_valid_checkpoint(tmp_path):
    local = tmp_path / "local" / "checkpoints"
    destination = tmp_path / "drive" / "run"
    for step in (200, 400, 600):
        persist_checkpoint(make_checkpoint_dir(local, step), destination)

    # Deleted afterwards, on Drive, by whatever means.
    (destination / "checkpoints" / "step_000600" / "model.safetensors").unlink()

    fresh = tmp_path / "session-b"
    result = restore_run(destination, fresh)

    assert result["restored"] == ["step_000200", "step_000400"]
    assert [e["name"] for e in result["skipped"]] == ["step_000600"]
    assert result["pointer"]["step"] == 400
    assert not (fresh / "checkpoints" / "step_000600").exists()


def test_cross_session_recovery_session_a_then_session_b(tmp_path):
    """The Colab workflow: train, persist, lose the local disk, restore, resume."""
    session_a = tmp_path / "session-a"
    destination = tmp_path / "drive" / "run"

    for step in (200, 400):
        persist_checkpoint(make_checkpoint_dir(session_a / "checkpoints", step), destination)
    shutil.rmtree(session_a)          # the runtime recycles

    session_b = tmp_path / "session-b"
    result = restore_run(destination, session_b)
    assert result["restored"] == ["step_000200", "step_000400"]
    assert result["pointer"]["step"] == 400
    assert validate_checkpoint_dir(session_b / "checkpoints" / "step_000400").valid


def test_cross_session_recovery_when_the_newest_was_damaged_meanwhile(tmp_path):
    """Session B arrives to find the newest persisted checkpoint hollow."""
    session_a = tmp_path / "session-a"
    destination = tmp_path / "drive" / "run"
    for step in (200, 400):
        persist_checkpoint(make_checkpoint_dir(session_a / "checkpoints", step), destination)
    shutil.rmtree(session_a)
    (destination / "checkpoints" / "step_000400" / "optimizer.pt").write_bytes(b"")

    session_b = tmp_path / "session-b"
    result = restore_run(destination, session_b)
    assert result["restored"] == ["step_000200"]
    assert [e["name"] for e in result["skipped"]] == ["step_000400"]
    assert result["pointer"]["step"] == 200
    assert "1 (1 verified resumable)" in result["inventory"]


def test_startup_reports_storage_health_before_training(tmp_path):
    """Invalid checkpoints are named at startup, not discovered when a resume fails."""
    destination = tmp_path / "drive" / "run"
    checkpoints = destination / "checkpoints"
    make_checkpoint_dir(checkpoints, 200)
    damaged = make_checkpoint_dir(checkpoints, 400)
    (damaged / "model.safetensors").unlink()

    result = preflight(destination)
    assert result.usable
    assert result.existing_checkpoints == ["step_000200"]
    assert [e["name"] for e in result.invalid_checkpoints] == ["step_000400"]
    assert result.newest_valid == "step_000200"

    rendered = result.render()
    assert "2 checkpoint(s) there, 1 verified resumable" in rendered
    assert "step_000400 INVALID" in rendered


def test_persistent_status_names_every_invalid_checkpoint(tmp_path):
    destination = tmp_path / "drive" / "run"
    checkpoints = destination / "checkpoints"
    make_checkpoint_dir(checkpoints, 200)
    damaged = make_checkpoint_dir(checkpoints, 400)
    (damaged / "model.safetensors").unlink()
    atomic_write_json(checkpoints / "latest.json",
                      {"step": 400, "path": "step_000400", "complete": True})

    status = persistent_status(destination)
    assert status["checkpoints"] == ["step_000200"]
    assert status["all_checkpoints"] == ["step_000200", "step_000400"]
    assert status["invalid_checkpoints"][0]["name"] == "step_000400"
    assert not status["pointer_valid"]
    assert status["fell_back"]
    assert status["resumable_step"] == 200, "from the checkpoint that verified, not the pointer"


def test_validate_root_reports_every_checkpoint(tmp_path):
    make_checkpoint_dir(tmp_path, 200)
    broken = make_checkpoint_dir(tmp_path, 400)
    (broken / "optimizer.pt").unlink()

    validations = validate_checkpoint_root(tmp_path)
    assert [v.valid for v in validations] == [True, False]
    assert "INVALID" in validations[1].summary()


@pytest.mark.parametrize("level", [STRUCTURE, MANIFEST])
def test_an_empty_directory_is_never_valid(tmp_path, level):
    (tmp_path / "step_000200").mkdir()
    validation = validate_checkpoint_dir(tmp_path / "step_000200", level=level)
    assert not validation.valid
    assert "COMPLETE" in validation.invalid_reason
