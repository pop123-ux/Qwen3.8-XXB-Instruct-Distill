"""Tests for the benchmark harness and the two CLI entry points.

The commitment the harness makes is not about which benchmarks to run — that is
deliberately undecided. It is that once a suite is published its prompts are immutable,
because a suite whose contents drift makes "the student improved" and "the questions got
easier" indistinguishable, with nothing in the results to say which happened.

CPU only. No teacher is loaded, no benchmark data is downloaded, nothing is generated.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qwen_distill.evaluation.benchmark import (
    CATEGORIES,
    BenchmarkItem,
    BenchmarkSuite,
    comparable,
    new_run,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def make_suite(name="probe", n=4) -> BenchmarkSuite:
    return BenchmarkSuite(
        name=name,
        items=[
            BenchmarkItem(id=f"q{i}", prompt=f"question {i}", category="math",
                          answers=(f"{i}",))
            for i in range(n)
        ],
        contamination_notes=["synthetic questions, generated in-repo; cannot be in any "
                             "training corpus"],
    )


# --- immutability ---------------------------------------------------------
def test_a_suites_digest_covers_its_prompts():
    a, b = make_suite(), make_suite()
    assert a.digest == b.digest
    b.items[1].prompt = "changed"
    assert a.digest != b.digest


def test_a_modified_suite_refuses_to_load(tmp_path):
    """The whole point: results across a changed suite are not comparable."""
    suite = make_suite()
    path = suite.write(tmp_path / "suite.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][0]["prompt"] = "a different question"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="modified since it was written"):
        BenchmarkSuite.read(path)


def test_an_unmodified_suite_round_trips(tmp_path):
    suite = make_suite()
    path = suite.write(tmp_path / "suite.json")
    reloaded = BenchmarkSuite.read(path)
    assert reloaded.digest == suite.digest
    assert [i.id for i in reloaded.items] == [i.id for i in suite.items]


def test_runs_over_different_suite_contents_are_not_comparable():
    original = make_suite()
    edited = make_suite()
    edited.items[0].prompt = "changed"

    a = new_run(original, model="teacher")
    b = new_run(edited, model="student")

    ok, reason = comparable(a, b)
    assert not ok
    assert "the questions changed" in reason


def test_runs_over_the_same_suite_are_comparable():
    suite = make_suite()
    ok, reason = comparable(new_run(suite, model="t"), new_run(suite, model="s"))
    assert ok and reason is None


def test_runs_over_different_suites_are_not_comparable():
    ok, reason = comparable(new_run(make_suite("a"), model="t"),
                            new_run(make_suite("b"), model="s"))
    assert not ok
    assert "different suites" in reason


# --- grading --------------------------------------------------------------
def test_an_item_without_answers_grades_as_none_not_wrong():
    """Otherwise accuracy silently measures how much of the suite has answer keys."""
    item = BenchmarkItem(id="q", prompt="open question")
    assert not item.gradable
    assert item.grade("anything") is None


def test_exact_match_normalises_whitespace_and_trailing_punctuation():
    item = BenchmarkItem(id="q", prompt="p", answers=("42",))
    assert item.grade("42") is True
    assert item.grade("  42 .") is True
    assert item.grade("43") is False


# --- validation -----------------------------------------------------------
def test_a_suite_without_contamination_notes_is_not_usable_evidence():
    suite = make_suite()
    suite.contamination_notes = []
    assert any("contamination" in p for p in suite.validate())


def test_duplicate_item_ids_are_reported():
    suite = make_suite()
    suite.items.append(BenchmarkItem(id="q0", prompt="dup", answers=("x",)))
    assert any("duplicate item id" in p for p in suite.validate())


def test_unknown_categories_are_reported():
    suite = make_suite()
    suite.items[0].category = "telepathy"
    assert any("unknown categories" in p for p in suite.validate())


def test_a_clean_suite_validates():
    assert make_suite().validate() == []


def test_the_declared_categories_cover_the_planned_areas():
    for expected in ("math", "reasoning", "coding", "long_context"):
        assert expected in CATEGORIES


# --- runs bind to what they measured --------------------------------------
def test_a_run_records_the_suite_digest_and_model_revision():
    suite = make_suite()
    run = new_run(suite, model="Qwen/Qwen3.8-27B", model_revision="abc123",
                  reasoning_mode="xhigh")
    payload = run.to_dict()
    assert payload["suite_digest"] == suite.digest
    assert payload["model_revision"] == "abc123"
    assert payload["manifest"]["reasoning_mode"] == "xhigh"


def test_a_run_with_no_results_summarises_to_nothing_rather_than_zero():
    assert new_run(make_suite(), model="m").summary() == {"n": 0}


def test_a_run_writes_and_reloads(tmp_path):
    run = new_run(make_suite(), model="m")
    run.results = [{"total_output_tokens": 10, "reasoning_tokens": 2,
                    "answer_tokens": 8, "correct": True, "latency_seconds": 0.1}]
    path = run.write(tmp_path / "run.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_results"] == 1
    assert payload["summary"]["accuracy"] == 1.0


# --- the CLIs -------------------------------------------------------------
def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        capture_output=True, text=True, timeout=300, cwd=ROOT,
    )


def test_generation_dry_run_loads_no_teacher(tmp_path):
    """A dry run must validate the plan without touching a 27B model."""
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "a", "prompt": "q"}) + "\n", encoding="utf-8")

    result = run_script("generate_teacher_data.py", "--input", str(prompts),
                        "--output", str(tmp_path / "out"), "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert not (tmp_path / "out").exists()


def test_generation_refuses_the_unimplemented_backend_without_a_traceback(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "a", "prompt": "q"}) + "\n", encoding="utf-8")

    result = run_script("generate_teacher_data.py", "--input", str(prompts),
                        "--output", str(tmp_path / "out"))

    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "not wired up yet" in result.stderr
    assert "--backend mock" in result.stderr, "it must say how to test the pipeline"


def test_generation_rejects_an_unsupported_reasoning_mode(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "a", "prompt": "q"}) + "\n", encoding="utf-8")

    result = run_script("generate_teacher_data.py", "--input", str(prompts),
                        "--output", str(tmp_path / "out"),
                        "--reasoning-mode", "high", "--dry-run")

    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_generation_with_the_mock_produces_a_verifiable_dataset(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("\n".join(
        json.dumps({"id": f"p{i}", "prompt": f"q{i}"}) for i in range(6)
    ), encoding="utf-8")
    output = tmp_path / "out"

    result = run_script("generate_teacher_data.py", "--input", str(prompts),
                        "--output", str(output), "--backend", "mock", "--shard-size", "4")

    assert result.returncode == 0, result.stderr
    assert "SYNTHETIC" in result.stdout
    status = run_script("generate_teacher_data.py", "--input", str(prompts),
                        "--output", str(output), "--status")
    assert "integrity : OK" in status.stdout


def test_the_missing_input_file_exits_2(tmp_path):
    result = run_script("generate_teacher_data.py", "--input", str(tmp_path / "nope"),
                        "--output", str(tmp_path / "out"), "--dry-run")
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_list_objectives_states_what_is_implemented():
    result = run_script("train_distilled_student.py", "--list-objectives")
    assert result.returncode == 0
    assert "IMPLEMENTED" in result.stdout
    assert "NOT_IMPLEMENTED" in result.stdout


def test_the_shipped_distillation_configs_load():
    from qwen_distill.training.config import ExperimentConfig

    for name in ("sft_smoke", "logit_kd_example"):
        config = ExperimentConfig.load(ROOT / "configs" / "distillation" / f"{name}.yaml")
        assert config.objective, f"{name}: the objective block must not be dropped"


def test_a_kd_config_is_not_silently_run_as_sft(tmp_path):
    """The failure that would make the project's central comparison meaningless."""
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("\n".join(
        json.dumps({"id": f"p{i}", "prompt": f"q{i}"}) for i in range(4)
    ), encoding="utf-8")
    dataset = tmp_path / "data"
    run_script("generate_teacher_data.py", "--input", str(prompts),
               "--output", str(dataset), "--backend", "mock")

    result = run_script("train_distilled_student.py", "--config",
                        str(ROOT / "configs/distillation/logit_kd_example.yaml"),
                        "--dataset", str(dataset))

    assert result.returncode == 2
    assert "CANNOT TRAIN" in result.stderr
    assert "logit_kd" in result.stderr
