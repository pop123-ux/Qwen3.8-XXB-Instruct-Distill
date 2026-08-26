"""The analyser must not repeat the mistakes it exists to catch.

Two of these tests are regression tests for specific published errors:
``test_catches_the_level2_throughput_bug`` reproduces the cumulative-tokens-over-session-
elapsed arithmetic that made a 2,090 tok/s run report 139,256, and
``test_reaching_max_steps_is_not_completion`` pins the refusal to call a finished loop a
finished experiment.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from qwen_distill.analysis import (
    Curve,
    RunRecords,
    analyse_plateau,
    analyse_run,
    analyse_throughput,
    build_checkpoint_timeline,
    detect_segments,
    read_jsonl,
    render_text_curve,
    write_plots,
)

TOKENS_PER_STEP = 16_384
TRUE_RATE = 2089.5
EPOCH = datetime(2026, 8, 24, 10, 0, tzinfo=timezone.utc)


def _train_record(step, elapsed, tokens, logged_rate, stamp, *, loss=1.0, epoch=1.0):
    return {
        "step": step, "loss": loss, "lr": 0.0006, "elapsed_s": round(elapsed, 2),
        "tokens_seen": tokens, "tokens_per_second": round(logged_rate, 1),
        "epoch": epoch, "bits_per_byte": round(loss / 0.6931, 4),
        "timestamp": stamp.isoformat(), "status": "completed_step",
    }


def _healthy_records(n_steps=400, every=25):
    records = []
    for step in range(every, n_steps + 1, every):
        tokens = step * TOKENS_PER_STEP
        elapsed = tokens / TRUE_RATE
        loss = 4.0 / (1.0 + step / 100.0) + 0.85
        records.append(
            _train_record(step, elapsed, tokens, tokens / elapsed,
                          EPOCH + timedelta(seconds=elapsed), loss=round(loss, 4))
        )
    return records


def _resumed_records_with_the_bug(resume_at=1600, last=2000, every=25):
    """Session 1 logs correctly; session 2 logs cumulative tokens / session elapsed."""
    records = _healthy_records(resume_at, every)
    resumed_seconds = resume_at * TOKENS_PER_STEP / TRUE_RATE
    wall = EPOCH + timedelta(seconds=resumed_seconds) + timedelta(minutes=47)
    for step in range(resume_at + every, last + 1, every):
        tokens = step * TOKENS_PER_STEP
        session_elapsed = (step - resume_at) * TOKENS_PER_STEP / TRUE_RATE
        records.append(
            _train_record(
                step, resumed_seconds + session_elapsed, tokens,
                tokens / session_elapsed,            # the bug, verbatim
                wall + timedelta(seconds=session_elapsed),
            )
        )
    return records


def _write_run(root, records, *, max_steps=2000, checkpoints=(), latest_pointer=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps({"name": "test_run", "configuration": {"max_steps": max_steps}}),
        encoding="utf-8",
    )
    if checkpoints:
        directory = root / "checkpoints"
        directory.mkdir(exist_ok=True)
        for step, complete in checkpoints:
            entry = directory / f"step_{step:06d}"
            entry.mkdir(exist_ok=True)
            for name in ("model.safetensors", "optimizer.pt", "training_state.json"):
                (entry / name).write_bytes(b"x")
            (entry / "metadata.json").write_text(
                json.dumps({"step": step, "complete": complete, "reason": "periodic"}),
                encoding="utf-8",
            )
        if latest_pointer is not None:
            (directory / "latest.json").write_text(
                json.dumps({"step": latest_pointer}), encoding="utf-8"
            )
    return root


# ----------------------------------------------------------------------------------
# the throughput bug
# ----------------------------------------------------------------------------------


def test_catches_the_level2_throughput_bug():
    """A log written by the buggy code must be flagged, not believed.

    The signature is a rate that decays as 1/n while the true rate is flat. The
    cumulative counters in the same record are untouched by the bug, so the correct rate
    is recoverable and the disagreement is a finding.
    """
    records = RunRecords.from_records(_resumed_records_with_the_bug())
    analysis = analyse_throughput(records)

    assert analysis.run_wide_tokens_per_second == pytest.approx(TRUE_RATE, rel=1e-3)
    assert analysis.logged_vs_recomputed, "the 65x overstatement was not detected"
    assert analysis.max_disagreement_ratio > 50
    first_bad = analysis.logged_vs_recomputed[0]
    assert first_bad["step"] == 1625
    assert first_bad["recomputed_tokens_per_second"] == pytest.approx(TRUE_RATE, rel=1e-3)
    assert any("recompute_throughput.py" in f for f in analysis.findings)


def test_healthy_log_produces_no_throughput_finding():
    """The audit must not cry wolf on a correct log."""
    analysis = analyse_throughput(RunRecords.from_records(_healthy_records()))
    assert analysis.logged_vs_recomputed == []
    assert analysis.max_disagreement_ratio is None
    assert analysis.run_wide_tokens_per_second == pytest.approx(TRUE_RATE, rel=1e-3)


def test_three_scopes_are_reported_separately():
    """The three rates must be distinct fields, not one number wearing three labels."""
    analysis = analyse_throughput(RunRecords.from_records(_healthy_records()))
    payload = analysis.to_dict()
    assert payload["run_wide_tokens_per_second"] is not None
    assert payload["step_level"] and payload["interval"]
    assert payload["steps_per_record"] == 25
    # A per-record rate is over 25 steps, and the report says so rather than implying
    # it is per step.
    assert "steps_per_record" in payload["scope_note"] or payload["steps_per_record"] == 25


def test_run_wide_does_not_double_count_resumed_tokens():
    """tokens_seen is already cumulative across sessions; summing segments would
    double-count every token trained before the resume."""
    analysis = analyse_throughput(RunRecords.from_records(_resumed_records_with_the_bug()))
    assert analysis.total_tokens == 2000 * TOKENS_PER_STEP
    assert sum(s.tokens for s in analysis.segments) < analysis.total_tokens


def test_records_straddling_a_resume_measure_nothing():
    """The pair spanning a restart is kept with a null rate, not silently dropped and not
    turned into a fabricated slowdown."""
    analysis = analyse_throughput(RunRecords.from_records(_resumed_records_with_the_bug()))
    crossing = [r for r in analysis.step_level if r["crosses_session_boundary"]]
    assert crossing, "no boundary was marked"
    assert all(r["tokens_per_second"] is None for r in crossing)


# ----------------------------------------------------------------------------------
# sessions
# ----------------------------------------------------------------------------------


def test_single_session_run_is_one_segment():
    segments = detect_segments(_healthy_records())
    assert len(segments) == 1
    assert segments[0].boundary_reason == "run start"


def test_wall_clock_gap_detects_a_resume():
    segments = detect_segments(_resumed_records_with_the_bug())
    assert len(segments) == 2
    assert segments[1].first_step == 1625
    assert "gap" in segments[1].boundary_reason


def test_step_regression_detects_a_resume_from_an_older_checkpoint():
    """Resuming from step 1400 after logging step 1600 sends the step number backwards.
    That is unambiguous and must not depend on timestamps being present."""
    records = _healthy_records(1600)
    for step in range(1425, 1601, 25):
        tokens = step * TOKENS_PER_STEP
        elapsed = tokens / TRUE_RATE
        record = _train_record(step, elapsed, tokens, TRUE_RATE, EPOCH)
        record.pop("timestamp")
        records.append(record)
    segments = detect_segments(records)
    assert len(segments) == 2
    assert segments[1].boundary_reason == "step regression"


def test_brief_pause_is_not_a_session_boundary():
    """A slow step, a checkpoint write or a scheduling hiccup must not read as a
    restart; the threshold is deliberately conservative."""
    records = _healthy_records(200)
    for record in records[3:]:
        stamp = datetime.fromisoformat(record["timestamp"]) + timedelta(seconds=30)
        record["timestamp"] = stamp.isoformat()
    assert len(detect_segments(records)) == 1


# ----------------------------------------------------------------------------------
# plateaus
# ----------------------------------------------------------------------------------


def test_plateau_reports_when_improvement_stopped():
    """Level 2's shape: most of the improvement early, most of the run after."""
    curve = Curve(name="validation bits/byte", baseline=8.0)
    curve.steps = [200, 400, 800, 1200, 1600, 2000]
    curve.values = [1.317, 1.279, 1.275, 1.273, 1.271, 1.270]
    analysis = analyse_plateau(curve)
    assert analysis.plateau_step == 400
    assert analysis.fraction_of_run_after_plateau == pytest.approx(0.889, abs=0.01)
    assert any("after" in note for note in analysis.notes)


def test_plateau_refuses_to_say_why():
    """Convergence and an exhausted corpus produce the same shape. The report must not
    pick one."""
    curve = Curve(name="validation bits/byte")
    curve.steps, curve.values = [100, 200, 300], [2.0, 1.5, 1.5]
    payload = analyse_plateau(curve, epochs_seen=4.1).to_dict()
    assert "does not say why" in payload["interpretation_note"]
    assert any("exhaustion" in note for note in payload["notes"])
    assert payload["epochs_seen"] == 4.1


def test_plateau_needs_two_points():
    curve = Curve(name="x")
    curve.steps, curve.values = [1], [1.0]
    analysis = analyse_plateau(curve)
    assert analysis.plateau_step is None
    assert analysis.notes


# ----------------------------------------------------------------------------------
# checkpoints
# ----------------------------------------------------------------------------------


def test_incomplete_checkpoint_is_not_a_resume_point(tmp_path):
    root = _write_run(
        tmp_path / "run", _healthy_records(400),
        checkpoints=((200, True), (400, False)), latest_pointer=400,
    )
    timeline = build_checkpoint_timeline(root / "checkpoints")
    assert timeline.resumable_step == 200
    assert any("not complete" in f for f in timeline.findings)
    assert any("latest.json points at step 400" in f for f in timeline.findings)


def test_missing_required_file_makes_a_checkpoint_incomplete(tmp_path):
    """The marker alone is not enough: a checkpoint that claims complete but has lost
    its optimizer state cannot resume training."""
    root = _write_run(tmp_path / "run", _healthy_records(200), checkpoints=((200, True),))
    (root / "checkpoints" / "step_000200" / "optimizer.pt").unlink()
    timeline = build_checkpoint_timeline(root / "checkpoints")
    assert timeline.resumable_step is None
    assert timeline.checkpoints[0].missing_files == ["optimizer.pt"]


def test_staging_directory_is_reported_as_a_crash(tmp_path):
    root = _write_run(tmp_path / "run", _healthy_records(200), checkpoints=((200, True),))
    (root / "checkpoints" / ".step_000400.incomplete").mkdir()
    timeline = build_checkpoint_timeline(root / "checkpoints")
    assert timeline.incomplete_staging == [".step_000400.incomplete"]
    assert timeline.resumable_step == 200


def test_no_checkpoints_directory_is_a_finding_not_a_crash():
    timeline = build_checkpoint_timeline(None)
    assert timeline.resumable_step is None
    assert timeline.findings


# ----------------------------------------------------------------------------------
# the whole run
# ----------------------------------------------------------------------------------


def test_reaching_max_steps_is_not_completion(tmp_path):
    """The loop finishing is a fact about the loop. Nothing else."""
    root = _write_run(
        tmp_path / "run", _resumed_records_with_the_bug(), max_steps=2000,
        checkpoints=((2000, True),),
    )
    analysis = analyse_run(root)
    assert analysis.reached_max_steps
    assert "loop reached max_steps" in analysis.loop_status
    assert "does not mean the experiment is complete" in analysis.to_dict()["completion_caveat"]
    assert any("POST_RUN_CHECKLIST" in f for f in analysis.findings)


def test_analysis_never_claims_model_quality(tmp_path):
    root = _write_run(tmp_path / "run", _healthy_records(400), checkpoints=((400, True),))
    payload = analyse_run(root).to_dict()
    assert "sanity_generate.py" in payload["quality_note"]
    assert "and and and" in analyse_run(root).render()


def test_analysis_reports_work_at_risk(tmp_path):
    """The gap between the newest log record and the newest complete checkpoint is
    exactly what a crash would cost."""
    root = _write_run(
        tmp_path / "run", _healthy_records(2000), checkpoints=((1400, True),)
    )
    analysis = analyse_run(root)
    assert any("600 steps behind" in f for f in analysis.findings)


def test_analysis_survives_a_run_that_is_still_training(tmp_path):
    """metrics.jsonl is appended to while this reads. A half-written final line costs one
    record and must not raise."""
    root = _write_run(tmp_path / "run", _healthy_records(400))
    with open(root / "metrics.jsonl", "a", encoding="utf-8") as stream:
        stream.write('{"step": 425, "loss": 0.9, "tokens_')
    analysis = analyse_run(root)
    assert analysis.records.last_step == 400
    assert not analysis.reached_max_steps


def test_analysis_survives_an_empty_run_directory(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    analysis = analyse_run(root)
    assert analysis.records.training == []
    assert set(analysis.files.missing()) == {
        "metrics", "latest_progress", "summary", "checkpoints_dir"
    }
    assert analysis.render()  # must render rather than raise


def test_validation_records_are_excluded_from_throughput(tmp_path):
    """A validation record carries no tokens_seen. Including it would invent a step of
    zero tokens over some seconds — a slowdown that never happened."""
    records = _healthy_records(400)
    records.append({"step": 400, "validation_loss": 1.1, "validation_bits_per_byte": 1.59})
    parsed = RunRecords.from_records(records)
    assert len(parsed.validation) == 1
    assert all("tokens_seen" in r for r in parsed.training)


def test_missing_validation_curve_is_a_finding(tmp_path):
    root = _write_run(tmp_path / "run", _healthy_records(400))
    analysis = analyse_run(root)
    assert any("no validation records" in f for f in analysis.findings)


def test_read_jsonl_skips_a_truncated_line(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n{"c":', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_read_jsonl_on_a_missing_file_is_empty(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []


# ----------------------------------------------------------------------------------
# plotting is optional
# ----------------------------------------------------------------------------------


def test_text_curve_renders_without_matplotlib():
    curve = Curve(name="validation bits/byte", baseline=8.0)
    curve.steps = list(range(0, 1000, 50))
    curve.values = [8.0 - 6.5 * (i / 20) for i in range(20)]
    rendered = render_text_curve(curve)
    assert "*" in rendered
    assert "baseline" in rendered


def test_text_curve_handles_a_flat_curve():
    curve = Curve(name="x")
    curve.steps, curve.values = [1, 2, 3], [1.0, 1.0, 1.0]
    assert "flat at" in render_text_curve(curve)


def test_plots_are_optional(tmp_path):
    """matplotlib absent must return an empty list, never raise: being unable to read
    your own logs without a plotting library would be the real failure."""
    root = _write_run(tmp_path / "run", _healthy_records(400))
    written = write_plots(analyse_run(root), tmp_path / "plots")
    assert isinstance(written, list)
    if written:
        assert all(p.endswith(".png") for p in written)
