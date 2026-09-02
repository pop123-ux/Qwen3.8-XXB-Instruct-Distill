"""The experiment record must be complete before the run, and honest after it."""

from __future__ import annotations

import json

import pytest

from qwen_distill.training.run_record import (
    ARCHIVED_FILES,
    RUN_ID,
    archive_to_repository,
    build_manifest,
    capture_dataset,
    capture_git,
    capture_teacher,
    capture_tokenizer,
    initialise_run,
    record_termination,
    verify_record,
    write_checksums,
)

REPOSITORY_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


@pytest.fixture
def run(tmp_path):
    manifest = build_manifest(
        repository=REPOSITORY_ROOT, config=None, config_path=None,
        teacher_directory=None, tokenizer_path=None, corpus_manifest=None,
        command="python scripts/train_student.py --config x.yaml",
    )
    root = tmp_path / "kd_run_001"
    initialise_run(root, manifest, command=manifest["command"])
    return root


def test_initialise_writes_every_record_before_the_run(run):
    for name in ("manifest.json", "git.txt", "environment.txt", "hardware.txt",
                 "command.txt", "README.md", "metrics.jsonl", "training.log",
                 "teacher_provenance.json", "tokenizer_provenance.json",
                 "dataset_provenance.json"):
        assert (run / name).is_file(), name
    for name in ("checkpoints", "artifacts", "final", "progress"):
        assert (run / name).is_dir(), name


def test_reinitialising_never_truncates_metrics_or_log(run):
    (run / "metrics.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
    (run / "training.log").write_text("step 1\n", encoding="utf-8")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    initialise_run(run, manifest, command="re-run")
    assert (run / "metrics.jsonl").read_text(encoding="utf-8") == '{"step": 1}\n'
    assert (run / "training.log").read_text(encoding="utf-8") == "step 1\n"


def test_environment_record_names_variables_but_never_their_values(tmp_path, monkeypatch):
    """A secret in the environment must reach the record as a name and nothing else.

    This record is pushed to a public repository, so a token that leaks into
    `environment.txt` leaks to the world.
    """
    secret = "hf_" + "S3CR3TT0KENVALUE" * 2
    monkeypatch.setenv("PERSISTENCE_TEST_TOKEN", secret)
    manifest = build_manifest(
        repository=REPOSITORY_ROOT, config=None, config_path=None,
        teacher_directory=None, tokenizer_path=None, corpus_manifest=None, command="x",
    )
    root = tmp_path / "secret_run"
    initialise_run(root, manifest, command="x")

    assert "PERSISTENCE_TEST_TOKEN" in manifest["environment"]["environment_variable_names"]
    for path in root.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8", errors="ignore"), path


def test_git_state_records_dirty_paths_not_just_a_flag():
    git = capture_git(REPOSITORY_ROOT)
    assert git["commit"] and len(git["commit"]) == 40
    assert git["branch"]
    assert git["dirty"] is bool(git["modified_paths"])


def test_missing_provenance_is_reported_not_invented():
    for capture, argument in ((capture_teacher, None), (capture_tokenizer, None),
                              (capture_dataset, None)):
        record = capture(argument)
        assert "status" in record and "absent" in record["status"]


def test_teacher_provenance_reads_the_revision_from_the_download_manifest(tmp_path):
    directory = tmp_path / "teacher"
    directory.mkdir()
    (directory / "teacher_download_manifest.json").write_text(
        json.dumps({"model": "Qwen/Qwen3.8-27B", "revision": "a" * 40,
                    "n_files": 96, "total_bytes": 1}), encoding="utf-8")
    (directory / "config.json").write_text("{}", encoding="utf-8")
    record = capture_teacher(directory)
    assert record["revision"] == "a" * 40
    assert record["model"] == "Qwen/Qwen3.8-27B"
    assert "config.json" in record["metadata_sha256"]


def test_dataset_provenance_carries_content_hashes(tmp_path):
    manifest = tmp_path / "corpus_manifest.json"
    manifest.write_text(json.dumps({
        "name": "level2r", "train_sha256": "b" * 64, "validation_sha256": "c" * 64,
        "train_bytes": 10, "documents": [{"title": "x"}] * 500,
    }), encoding="utf-8")
    record = capture_dataset(manifest)
    assert record["train_sha256"] == "b" * 64
    assert "documents" not in record  # the full list stays where it was generated


def test_checksums_cover_every_artefact_with_size_and_location(run):
    (run / "artifacts" / "plot.png").write_bytes(b"\x89PNG")
    target = write_checksums(run, external_locations={"artifacts/plot.png": "gh://record"})
    body = target.read_text(encoding="utf-8")
    assert "artifacts/plot.png" in body and "gh://record" in body
    entry = next(line for line in body.splitlines() if "artifacts/plot.png" in line)
    digest, size, relative, external = entry.split()
    assert len(digest) == 64 and size == "4" and external == "gh://record"


def test_a_failed_run_is_recorded_not_discarded(run):
    record_termination(run, status="oom", reason="CUDA OOM at step 1200",
                       exit_code=1, last_step=1200)
    payload = json.loads((run / "termination.json").read_text(encoding="utf-8"))
    assert payload["status"] == "oom" and payload["last_step"] == 1200
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "oom" and manifest["finished_at"]


def test_archive_copies_text_and_only_references_checkpoints(run, tmp_path):
    checkpoint = run / "checkpoints" / "step_000100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "COMPLETE").touch()
    (checkpoint / "model.safetensors").write_bytes(b"0" * 4096)
    (checkpoint / "metadata.json").write_text('{"step": 100}', encoding="utf-8")
    (run / "metrics.jsonl").write_text('{"step": 100}\n', encoding="utf-8")

    archive = tmp_path / "archive"
    index = archive_to_repository(run, archive)

    assert (archive / "manifest.json").is_file()
    assert (archive / "metrics.jsonl").is_file()
    assert not (archive / "model.safetensors").exists()
    referenced = {entry["name"] for entry in index["referenced_not_copied"]}
    assert "checkpoints/step_000100" in referenced
    entry = next(e for e in index["referenced_not_copied"]
                 if e["name"] == "checkpoints/step_000100")
    assert entry["complete"] is True and entry["metadata"] == {"step": 100}
    assert set(index["copied"][0]) >= {"name", "bytes", "sha256", "source"}


def test_archived_file_list_holds_no_binary_names():
    assert not [n for n in ARCHIVED_FILES if n.endswith((".safetensors", ".pt", ".bin"))]


def test_verification_fails_loudly_until_the_run_has_actually_produced_something(run):
    report = verify_record(run)
    assert report["verified"] is False
    assert "DO NOT TERMINATE POD" in report["status"]
    assert "metrics persisted" in report["failed"]
    assert "checkpoint persisted" in report["failed"]
    # What *is* knowable before the run must already pass.
    passed = {item["item"] for item in report["items"] if item["ok"]}
    assert {"manifest exists", "git SHA recorded", "branch recorded",
            "environment recorded", "hardware recorded"} <= passed


def test_verification_passes_only_on_a_complete_record(run):
    (run / "metrics.jsonl").write_text('{"step": 100}\n', encoding="utf-8")
    (run / "training.log").write_text("step 100\n", encoding="utf-8")
    checkpoint = run / "checkpoints" / "step_000100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "COMPLETE").touch()
    (checkpoint / "metadata.json").write_text('{"step": 100}', encoding="utf-8")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["teacher"]["revision"] = "d" * 40
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    record_termination(run, status="completed", reason="reached max_steps", exit_code=0)
    write_checksums(run)

    report = verify_record(run)
    assert report["verified"] is True, report["failed"]
    assert report["status"].endswith("VERIFIED")


def test_an_incomplete_checkpoint_does_not_satisfy_the_checklist(run):
    partial = run / "checkpoints" / "step_000200"
    partial.mkdir(parents=True)
    (partial / "model.safetensors").write_bytes(b"0")  # no COMPLETE marker
    report = verify_record(run)
    failed = set(report["failed"])
    assert "checkpoint persisted" in failed
    assert "checkpoint metadata preserved" in failed


def test_run_id_is_fixed_so_a_second_run_cannot_overwrite_this_one():
    assert RUN_ID == "kd_run_001"
