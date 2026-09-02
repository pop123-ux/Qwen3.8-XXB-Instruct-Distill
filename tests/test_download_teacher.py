"""The pinned teacher downloader — ``scripts/download_teacher.py``.

Nothing here touches the network. The 27B checkpoint is never fetched; what is tested is
the request validation, the destination guard, the completeness check and the manifest.

The property worth stating plainly: this script proves *files arrived*, not that they load
as the teacher. The tests assert that the manifest says so, because a manifest that read
like a verification would be exactly the false assurance the smoke-test gate exists to
prevent.
"""
from __future__ import annotations

import json

import pytest
from scripts_shim import load

PINNED = "0f9e8d7c6b5a49382716051423f6e5d4c3b2a190"


@pytest.fixture(scope="module")
def dl():
    return load("download_teacher")


def _checkpoint(directory, *, shards=("model-00001-of-00002.safetensors",
                                      "model-00002-of-00002.safetensors")):
    """A structurally complete stand-in: real layout, no real weights."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text('{"model_type": "qwen3_5"}', encoding="utf-8")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"w{i}": s for i, s in enumerate(shards)}}),
        encoding="utf-8",
    )
    for shard in shards:
        (directory / shard).write_bytes(b"\x00" * 128)
    return directory


# ---------------------------------------------------------------------------
# the revision gate, shared with the loader
# ---------------------------------------------------------------------------
def test_the_revision_is_required(dl):
    with pytest.raises(SystemExit) as exit_info:
        dl.parse_args(["--output", "/data/x"])
    assert exit_info.value.code == 2


@pytest.mark.parametrize("revision", ["main", "master", "latest", "HEAD"])
def test_a_moving_revision_is_refused(dl, revision, capsys, tmp_path):
    code = dl.main(["--revision", revision, "--output", "/data/x", "--dry-run"])
    assert code == 2
    assert "moving pointer" in capsys.readouterr().err


@pytest.mark.parametrize("revision", ["v1.0", "release", "abc12"])
def test_a_non_commit_revision_is_refused(dl, revision, capsys):
    code = dl.main(["--revision", revision, "--output", "/data/x", "--dry-run"])
    assert code == 2
    assert "not a commit SHA" in capsys.readouterr().err


def test_a_commit_sha_is_accepted(dl, capsys):
    assert dl.main(["--revision", PINNED, "--output", "/data/x", "--dry-run"]) == 0
    assert "dry run" in capsys.readouterr().out


def test_the_gate_is_the_loaders_own(dl):
    """Not a second implementation that could drift: the script builds a TeacherLoadPlan and
    asks it, so the downloader and the loader accept exactly the same revisions."""
    import inspect

    source = inspect.getsource(dl.main)
    assert "TeacherLoadPlan" in source and "plan.validate()" in source


# ---------------------------------------------------------------------------
# the destination guard
# ---------------------------------------------------------------------------
def test_a_destination_inside_the_repository_is_refused(dl, capsys):
    """54 GB written into src/ or tests/ would wreck the working tree."""
    code = dl.main(["--revision", PINNED, "--output", "src/weights", "--dry-run"])
    assert code == 2
    assert "inside the repository" in capsys.readouterr().err


def test_a_destination_outside_the_repository_is_allowed(dl, tmp_path):
    assert dl.inside_repository(tmp_path / "models") is False
    assert dl.main(["--revision", PINNED, "--output", str(tmp_path), "--dry-run"]) == 0


def test_the_default_output_is_outside_the_repository(dl):
    assert dl.inside_repository(dl.parse_args(["--revision", PINNED]).output) is False


# ---------------------------------------------------------------------------
# completeness
# ---------------------------------------------------------------------------
def test_a_complete_checkpoint_reports_no_problems(dl, tmp_path):
    assert dl.completeness(_checkpoint(tmp_path / "ckpt")) == []


def test_a_missing_config_is_caught(dl, tmp_path):
    directory = _checkpoint(tmp_path / "ckpt")
    (directory / "config.json").unlink()
    assert any("config.json" in p for p in dl.completeness(directory))


def test_a_missing_tokenizer_is_caught(dl, tmp_path):
    """The distillation path needs the teacher's own tokenizer; the vendored metadata has
    no tokenizer.json and cannot substitute."""
    directory = _checkpoint(tmp_path / "ckpt")
    (directory / "tokenizer.json").unlink()
    problems = dl.completeness(directory)
    assert any("tokenizer" in p for p in problems)


@pytest.mark.parametrize("name", ["tokenizer.json", "tokenizer_config.json",
                                  "tokenizer.model", "vocab.json"])
def test_any_recognised_tokenizer_layout_satisfies_the_check(dl, tmp_path, name):
    """Pinning one filename would reject a valid checkpoint packaged differently."""
    directory = _checkpoint(tmp_path / name)
    (directory / "tokenizer.json").unlink()
    (directory / name).write_text("{}", encoding="utf-8")
    assert dl.completeness(directory) == []


def test_a_shard_named_by_the_index_but_absent_is_caught(dl, tmp_path):
    """The failure a plain file count misses: the index promises shards that never arrived."""
    directory = _checkpoint(tmp_path / "ckpt")
    (directory / "model-00002-of-00002.safetensors").unlink()
    problems = dl.completeness(directory)
    assert any("missing" in p and "00002" in p for p in problems)


def test_a_zero_length_shard_is_caught(dl, tmp_path):
    directory = _checkpoint(tmp_path / "ckpt")
    (directory / "model-00001-of-00002.safetensors").write_bytes(b"")
    assert any("zero-length" in p for p in dl.completeness(directory))


def test_a_checkpoint_with_no_weights_at_all_is_caught(dl, tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert any("no weight files" in p for p in dl.completeness(directory))


def test_a_missing_directory_is_reported_not_crashed(dl, tmp_path):
    assert dl.completeness(tmp_path / "nope") != []


def test_verify_only_exits_nonzero_on_an_incomplete_checkpoint(dl, tmp_path, capsys):
    directory = _checkpoint(tmp_path / "ckpt")
    (directory / "config.json").unlink()
    code = dl.main(["--revision", PINNED, "--output", str(directory), "--verify-only"])
    assert code == 1
    assert "INCOMPLETE" in capsys.readouterr().err


def test_verify_only_succeeds_and_writes_a_manifest(dl, tmp_path):
    directory = _checkpoint(tmp_path / "ckpt")
    assert dl.main(["--revision", PINNED, "--output", str(directory), "--verify-only"]) == 0
    assert (directory / dl.MANIFEST_NAME).exists()


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------
def test_the_manifest_records_the_pin_and_the_inventory(dl, tmp_path):
    directory = _checkpoint(tmp_path / "ckpt")
    manifest = json.loads(
        dl.write_manifest(directory, "Qwen/Qwen3.8-27B", PINNED).read_text(encoding="utf-8")
    )
    assert manifest["revision"] == PINNED
    assert manifest["model"] == "Qwen/Qwen3.8-27B"
    assert manifest["n_files"] == len(manifest["files"]) == 5
    assert manifest["total_bytes"] > 0
    assert all(f["sha256"] for f in manifest["files"]), "small files should be checksummed"


def test_the_manifest_says_what_it_does_not_prove(dl, tmp_path):
    """A manifest that read like a verification would be the false assurance the
    missing-weight gate exists to prevent."""
    directory = _checkpoint(tmp_path / "ckpt")
    manifest = json.loads(
        dl.write_manifest(directory, "Qwen/Qwen3.8-27B", PINNED).read_text(encoding="utf-8")
    )
    assert "does_not_verify" in manifest
    assert "teacher_smoke_test" in manifest["does_not_verify"]
    assert "freshly-initialised" in manifest["does_not_verify"]


def test_the_manifest_excludes_itself_from_the_inventory(dl, tmp_path):
    directory = _checkpoint(tmp_path / "ckpt")
    dl.write_manifest(directory, "Qwen/Qwen3.8-27B", PINNED)
    second = json.loads(
        dl.write_manifest(directory, "Qwen/Qwen3.8-27B", PINNED).read_text(encoding="utf-8")
    )
    assert all(f["path"] != dl.MANIFEST_NAME for f in second["files"])
    assert second["n_files"] == 5


def test_large_files_are_recorded_by_size_rather_than_hashed(dl, tmp_path, monkeypatch):
    """Hashing 54 GB would cost more than the download; the policy is stated in the
    manifest rather than left implicit."""
    monkeypatch.setattr(dl, "CHECKSUM_MAX_BYTES", 64)
    directory = _checkpoint(tmp_path / "ckpt")
    manifest = json.loads(
        dl.write_manifest(directory, "Qwen/Qwen3.8-27B", PINNED).read_text(encoding="utf-8")
    )
    shards = [f for f in manifest["files"] if f["path"].endswith(".safetensors")]
    assert shards and all(f["sha256"] is None for f in shards)
    assert all(f["bytes"] == 128 for f in shards)
    assert "size only" in manifest["checksum_policy"]


def test_the_script_does_not_download_anything_it_was_not_asked_to(dl):
    """It has one job. A downloader that also loaded, materialised or trained would make
    'the download succeeded' mean something different every time."""
    import inspect

    source = inspect.getsource(dl)
    for forbidden in ("AutoModelForCausalLM", "materialise_student", "train("):
        assert forbidden not in source
