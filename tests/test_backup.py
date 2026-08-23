"""Tests for the Colab -> Drive backup utility.

The destination is the user's personal Google Drive, so the failure modes that matter
are not "did it copy the file" but "did it copy something it should never have copied"
and "did it delete something the user still wanted". Those are what is tested here.

Everything runs on CPU in temporary directories. Nothing here requires Google Drive,
a Colab runtime, or a network — the mount check is bypassed with ``--skip-mount-check``
and the "Drive" is an ordinary tmp_path.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "backup_colab_to_drive.py"


def _load_module():
    """Import the script by path; `scripts/` is not an importable package."""
    spec = importlib.util.spec_from_file_location("backup_colab_to_drive", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before executing: @dataclass resolves annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


backup = _load_module()


def make_source(root: Path) -> Path:
    """A miniature repository with the shapes the backup has to reason about."""
    source = root / "repo"
    (source / "src" / "qwen_distill").mkdir(parents=True)
    (source / "experiments" / "run-1").mkdir(parents=True)
    (source / ".git").mkdir()
    (source / "__pycache__").mkdir()

    (source / "README.md").write_text("readme", encoding="utf-8")
    (source / "src" / "qwen_distill" / "model.py").write_text("code", encoding="utf-8")
    (source / "experiments" / "run-1" / "summary.json").write_text("{}", encoding="utf-8")
    (source / ".git" / "config").write_text("[core]", encoding="utf-8")
    (source / "__pycache__" / "model.cpython-311.pyc").write_text("bytecode", encoding="utf-8")
    return source


# --- exclusion ------------------------------------------------------------
def test_git_and_caches_are_excluded(tmp_path):
    """Version control and bytecode caches are regenerable; Drive should not hold them."""
    source = make_source(tmp_path)
    plan = backup.build_plan(source, tmp_path / "drive")

    copied = {target.relative_to(tmp_path / "drive").as_posix() for _, target in plan.to_copy}
    assert "README.md" in copied
    assert "src/qwen_distill/model.py" in copied
    assert "experiments/run-1/summary.json" in copied
    assert not any(name.startswith(".git/") for name in copied)
    assert not any("__pycache__" in name for name in copied)
    assert not any(name.endswith(".pyc") for name in copied)


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.local",
        "server.pem",
        "private.key",
        "id_rsa",
        "id_ed25519",
        ".netrc",
        ".git-credentials",
        "hf_token.txt",
        "github_token.json",
        "aws_credentials.json",
        "secrets.yaml",
    ],
)
def test_credential_shaped_files_are_never_copied(tmp_path, name):
    """A token in the repo root must not end up in the user's Drive."""
    source = make_source(tmp_path)
    (source / name).write_text("SECRET-VALUE", encoding="utf-8")

    plan = backup.build_plan(source, tmp_path / "drive")

    assert name in plan.secrets_excluded
    assert all(path.name != name for path, _ in plan.to_copy)


def test_secrets_in_nested_directories_are_also_excluded(tmp_path):
    """Exclusion is by shape at any depth, not only at the root."""
    source = make_source(tmp_path)
    (source / ".ssh").mkdir()
    (source / ".ssh" / "known_hosts").write_text("host", encoding="utf-8")
    (source / "experiments" / "run-1" / ".env").write_text("HF_TOKEN=x", encoding="utf-8")

    plan = backup.build_plan(source, tmp_path / "drive")

    copied = {path.as_posix() for path, _ in plan.to_copy}
    assert not any(".ssh" in name for name in copied)
    assert not any(name.endswith(".env") for name in copied)


def test_secret_exclusion_survives_a_real_run(tmp_path):
    """End to end: after an actual copy, no secret exists at the destination."""
    source = make_source(tmp_path)
    destination = tmp_path / "drive"
    (source / ".env").write_text("HF_TOKEN=hunter2", encoding="utf-8")

    plan = backup.build_plan(source, destination)
    backup.execute(plan)

    assert (destination / "README.md").is_file()
    assert not (destination / ".env").exists()
    written = [p for p in destination.rglob("*") if p.is_file()]
    assert all("hunter2" not in p.read_text(encoding="utf-8") for p in written)


# --- symlinks -------------------------------------------------------------
@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privileges on Windows")
def test_symlinks_are_skipped_not_followed(tmp_path):
    """A symlink can point anywhere; following one could pull in an unrelated tree."""
    source = make_source(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("not part of the repo", encoding="utf-8")
    (source / "link.txt").symlink_to(outside / "private.txt")
    (source / "linkdir").symlink_to(outside, target_is_directory=True)

    plan = backup.build_plan(source, tmp_path / "drive")

    assert "link.txt" in plan.symlinks_skipped
    assert "linkdir" in plan.symlinks_skipped
    assert all("private.txt" not in path.as_posix() for path, _ in plan.to_copy)


# --- incremental behaviour ------------------------------------------------
def test_rerunning_copies_nothing_when_unchanged(tmp_path):
    """A backup is meant to be re-runnable without re-uploading the whole tree."""
    source = make_source(tmp_path)
    destination = tmp_path / "drive"

    first = backup.build_plan(source, destination)
    assert backup.execute(first) == len(first.to_copy) > 0

    second = backup.build_plan(source, destination)
    assert second.to_copy == []
    assert second.skipped_unchanged == len(first.to_copy)


def test_a_modified_file_is_copied_again(tmp_path):
    source = make_source(tmp_path)
    destination = tmp_path / "drive"
    backup.execute(backup.build_plan(source, destination))

    target = source / "README.md"
    target.write_text("readme, revised and longer", encoding="utf-8")
    os.utime(target, (10 ** 10, 10 ** 10))  # unambiguously newer

    plan = backup.build_plan(source, destination)
    assert [p.name for p, _ in plan.to_copy] == ["README.md"]


# --- deletion safety ------------------------------------------------------
def test_extraneous_files_are_not_even_listed_by_default(tmp_path):
    """Default behaviour must not so much as contemplate deleting."""
    source = make_source(tmp_path)
    destination = tmp_path / "drive"
    destination.mkdir()
    (destination / "irreplaceable.txt").write_text("user data", encoding="utf-8")

    plan = backup.build_plan(source, destination)
    assert plan.extraneous == []

    backup.execute(plan)
    assert (destination / "irreplaceable.txt").is_file()


def test_execute_deletes_only_when_explicitly_asked(tmp_path):
    source = make_source(tmp_path)
    destination = tmp_path / "drive"
    destination.mkdir()
    stale = destination / "stale.txt"
    stale.write_text("no longer in the source", encoding="utf-8")

    plan = backup.build_plan(source, destination, include_extraneous=True)
    assert stale in plan.extraneous

    backup.execute(plan, delete_extraneous=False)
    assert stale.is_file(), "deletion must not happen unless requested"

    backup.execute(plan, delete_extraneous=True)
    assert not stale.exists()


def test_delete_extraneous_refuses_without_yes(tmp_path):
    source = make_source(tmp_path)
    destination = tmp_path / "drive"
    destination.mkdir()
    stale = destination / "stale.txt"
    stale.write_text("still here", encoding="utf-8")

    code = backup.main([
        "--source", str(source), "--destination", str(destination),
        "--delete-extraneous", "--skip-mount-check",
    ])

    assert code == 2
    assert stale.is_file()


def test_dry_run_changes_nothing(tmp_path):
    source = make_source(tmp_path)
    destination = tmp_path / "drive"
    destination.mkdir()
    stale = destination / "stale.txt"
    stale.write_text("still here", encoding="utf-8")

    code = backup.main([
        "--source", str(source), "--destination", str(destination),
        "--delete-extraneous", "--yes", "--dry-run", "--skip-mount-check",
    ])

    assert code == 0
    assert stale.is_file(), "dry run must not delete"
    assert list(destination.iterdir()) == [stale], "dry run must not copy"


# --- CLI ------------------------------------------------------------------
def test_missing_source_exits_2(tmp_path):
    code = backup.main([
        "--source", str(tmp_path / "nope"), "--destination", str(tmp_path / "drive"),
        "--skip-mount-check",
    ])
    assert code == 2


def test_unmounted_drive_exits_2_with_mount_instructions(tmp_path, capsys, monkeypatch):
    """The common Colab mistake is forgetting to mount; the message must say how."""
    source = make_source(tmp_path)
    monkeypatch.setattr(backup, "drive_mounted", lambda *a, **k: False)

    code = backup.main(["--source", str(source), "--destination", str(tmp_path / "drive")])

    assert code == 2
    assert "drive.mount" in capsys.readouterr().err


def test_end_to_end_cli_run(tmp_path):
    source = make_source(tmp_path)
    destination = tmp_path / "drive"

    code = backup.main([
        "--source", str(source), "--destination", str(destination), "--skip-mount-check",
    ])

    assert code == 0
    assert (destination / "src" / "qwen_distill" / "model.py").read_text(encoding="utf-8") == "code"
    assert not (destination / ".git").exists()


def test_script_runs_as_a_subprocess(tmp_path):
    """The file must be executable as a script, not only importable."""
    source = make_source(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(source),
         "--destination", str(tmp_path / "drive"), "--skip-mount-check", "--dry-run"],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
