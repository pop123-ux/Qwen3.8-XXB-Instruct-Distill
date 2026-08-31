"""The research pilot — ``scripts/distill_pilot.py``.

Two properties are load-bearing and are the reason this file exists separately from
``test_chain_selftest.py``:

* **The pilot cannot run a student nobody specified.** It has no architecture arguments at
  all, so there is no way to reach an arbitrary geometry through the normal path.
* **A Hub load without an exact commit SHA fails before anything downloads.** A repo id
  does not name weights, and a pilot that silently took whatever ``main`` pointed at would
  produce a result that could never be reproduced or compared.
"""
from __future__ import annotations

import json

import pytest
from scripts_shim import load

pytest.importorskip("torch")
pytest.importorskip("transformers")

PINNED = "0f9e8d7c6b5a49382716051423f6e5d4c3b2a190"


@pytest.fixture(scope="module")
def pilot():
    return load("distill_pilot")


# ---------------------------------------------------------------------------
# no generic geometry can reach the normal path
# ---------------------------------------------------------------------------
GEOMETRY_FLAGS = ("--hidden", "--layers", "--ffn", "--kv-heads", "--dn-key-heads",
                  "--untie-embeddings", "--num-experts", "--expert-width", "--stand-in")


@pytest.mark.parametrize("flag", GEOMETRY_FLAGS)
def test_the_pilot_exposes_no_student_geometry(pilot, flag):
    """The guard against running an architecture nobody specified. argparse exits 2 on an
    unrecognised flag, which is the behaviour being asserted."""
    with pytest.raises(SystemExit) as exit_info:
        pilot.parse_args([flag, "8"])
    assert exit_info.value.code == 2


def test_the_help_text_names_the_canonical_student(pilot, capsys):
    with pytest.raises(SystemExit):
        pilot.parse_args(["--help"])
    text = capsys.readouterr().out
    assert "qwen38_19b_h5120_l48_moe" in text
    for flag in ("--hidden", "--layers", "--kv-heads"):
        assert flag not in text


def test_the_pilot_targets_the_frozen_student(pilot, tmp_path, capsys):
    """Not merely absent knobs: the run must actually report the canonical architecture."""
    code = pilot.main(["--dry-run", "--revision", PINNED, "--output", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "qwen38_19b_h5120_l48_moe" in out
    assert "13,008,505,728" in out
    assert "9,611,119,488" in out
    assert "8 routed (top-2) + 1 shared" in out


def test_the_dry_run_reports_the_sixteen_gb_verdict(pilot, tmp_path, capsys):
    """The deployment constraint is checked on every run, not only in a separate report."""
    pilot.main(["--dry-run", "--revision", PINNED, "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert "16 GB BUDGET" in out
    assert "FIT" in out


def test_the_dry_run_writes_nothing_but_the_record(pilot, tmp_path):
    pilot.main(["--dry-run", "--revision", PINNED, "--output", str(tmp_path)])
    assert not (tmp_path / "transferred").exists()
    record = json.loads((tmp_path / "pilot_record.json").read_text(encoding="utf-8"))
    assert record["student_id"] == "qwen38_19b_h5120_l48_moe"
    assert record["teacher_plan"]["revision"] == PINNED
    assert record["layer_mapping"]["block_types_preserved"] is True
    assert record["layer_mapping"]["n_removed"] == 16


# ---------------------------------------------------------------------------
# the revision gate
# ---------------------------------------------------------------------------
def test_a_hub_run_without_a_revision_is_refused(pilot, tmp_path, capsys):
    code = pilot.main(["--dry-run", "--output", str(tmp_path)])
    assert code == 2
    assert "--revision is required" in capsys.readouterr().err


@pytest.mark.parametrize("revision", ["main", "master", "latest", "HEAD"])
def test_a_moving_pointer_is_not_a_pin(pilot, tmp_path, capsys, revision):
    code = pilot.main(["--dry-run", "--revision", revision, "--output", str(tmp_path)])
    assert code == 2
    assert "moving pointer" in capsys.readouterr().err


def test_a_tag_is_refused_because_upstream_can_move_it(pilot, tmp_path, capsys):
    code = pilot.main(["--dry-run", "--revision", "v1.0", "--output", str(tmp_path)])
    assert code == 2
    assert "not a commit SHA" in capsys.readouterr().err


def test_a_local_checkpoint_may_omit_the_revision(pilot, tmp_path):
    """The bytes on disk are the pin; requiring a SHA there would block fixture runs."""
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    code = pilot.main(["--dry-run", "--teacher", str(teacher), "--output", str(tmp_path / "o")])
    assert code == 0


def test_the_supplied_sha_reaches_the_record(pilot, tmp_path):
    out = tmp_path / "o"
    pilot.main(["--dry-run", "--teacher", str(tmp_path), "--revision", PINNED,
                "--output", str(out)])
    record = json.loads((out / "pilot_record.json").read_text(encoding="utf-8"))
    assert record["teacher_plan"]["revision"] == PINNED


def test_a_hub_materialisation_says_what_is_missing_rather_than_downloading(pilot, tmp_path,
                                                                            capsys):
    """A real run needs the shards on disk. The refusal must name the fix, not attempt a
    54 GB download as a side effect of a pilot."""
    code = pilot.main(["--revision", PINNED, "--output", str(tmp_path)])
    assert code == 2
    assert "Download the pinned revision first" in capsys.readouterr().err


def test_a_missing_teacher_directory_is_reported(pilot, tmp_path, capsys):
    code = pilot.main(["--teacher", str(tmp_path / "nope"), "--output", str(tmp_path / "o")])
    assert code == 2
    assert "no config.json" in capsys.readouterr().err
