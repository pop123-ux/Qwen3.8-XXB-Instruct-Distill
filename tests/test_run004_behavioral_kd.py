"""Preparation tests for Run 004's behavioural/delta-KD launcher."""

from __future__ import annotations

import json


def test_run004_command_and_manifest_are_explicit():
    from scripts.run004_behavioral_kd import (
        BEHAVIORAL_MODE,
        BEHAVIORAL_OBJECTIVE,
        build_kd_run_args,
    )

    class Args:
        teacher = "/teacher"
        revision = "dbdc473dea0d6a9763042881cc33d6058d1742d2"
        pretrained = "/student"
        text_path = "/corpus/train.txt"
        output = "/runs/run004_behavioral_kd"
        experiment_id = "run004_behavioral_kd"
        sequence_length = 1536
        max_tokens = 700000
        steps = 128
        batch_size = 1
        gradient_accumulation_steps = 1
        learning_rate = 2e-4
        kd_temperature = 2.0
        kd_top_k = 64
        chunk_pairs = 4
        lora_rank = 16
        lora_alpha = 32
        optimizer = "adamw"
        precision = "bf16"
        quantization = "4bit"
        seed = 0
        log_every = 1
        eval_every = 32
        save_every = 64

    args = build_kd_run_args(Args())
    assert BEHAVIORAL_OBJECTIVE == "behavioral_kd"
    assert BEHAVIORAL_MODE == "delta"
    assert args[args.index("--objective") + 1] == "layer_kd"
    assert args[args.index("--layer-kd-chunk-pairs") + 1] == "4"
    assert args[args.index("--max-tokens") + 1] == "700000"
    assert args[args.index("--sequence-length") + 1] == "1536"
    assert args[args.index("--steps") + 1] == "128"
    assert args[args.index("--name") + 1] == "run004_behavioral_kd"


def test_run004_manifest_declares_no_recurrent_state_projection(tmp_path):
    from scripts.run004_behavioral_kd import write_manifest

    path = write_manifest(
        tmp_path,
        command=["test-command"],
        dry_run=True,
        experiment_id="run004_behavioral_kd",
        max_tokens=700000,
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["objective"] == "behavioral_kd"
    assert manifest["behavioral_mode"] == "delta"
    assert manifest["experiment"] == "run004_behavioral_kd"
    assert manifest["max_tokens"] == 700000
    assert manifest["student"] == "qwen38_19b_h5120_l48_moe"
    assert manifest["teacher_revision"] == "dbdc473dea0d6a9763042881cc33d6058d1742d2"
    assert manifest["deltanet_state_matching"] is False
    assert "projection" in manifest["deltanet_state_note"]


def test_run004_recorder_payload_reads_the_real_summary_structure(tmp_path):
    """The trainer writes per-metric ``{first, final, mean}`` series under
    ``summary["distillation"]`` plus top-level loss/validation fields. The recorder must
    read *those*, not a non-existent ``initial``/``final``/``mean``/``trajectory``
    grouping that would leave the ledger metrics empty."""
    from scripts.record_run004_kd import payload

    summary = {
        "git_commit": "f4fc999",
        "first_loss": 2.53,
        "final_loss": 1.70,
        "first_validation_loss": 12.0,
        "final_validation_loss": 11.09,
        "best_validation_loss": 11.09,
        "tokens_per_second": 181.9,
        "runtime_s": 1080.7,
        "tokens_seen": 196608,
        "steps": 128,
        "parameter_counts": {"total_parameters": 13031508864, "trainable_parameters": 23003136},
        "memory": {"peak_allocated_gib": 38.95, "peak_reserved_gib": 40.77},
        "corpus": {"sha256": "abc123", "n_sequences": 455, "n_train": 433, "n_validation": 22},
        "config": {"data": {"max_sequence_length": 1536, "max_tokens": 700000},
                   "training": {"seed": 0}},
        "distillation": {
            "kd_temperature": 2.0,
            "kd_top_k": 64,
            "kd_loss": {"first": 6.67, "final": 4.74, "mean": 5.78},
            "ce_loss": {"first": 10.6, "final": 11.26, "mean": 11.44},
            "top1_agreement": {"first": 0.001, "final": 0.14, "mean": 0.10},
            "layer_kd_loss": {"first": 2.53, "final": 1.70, "mean": 1.88},
            "layer_magnitude": {"first": 1.69, "final": 1.14, "mean": 1.25},
            "layer_direction": {"first": 0.85, "final": 0.57, "mean": 0.63},
            "layer_norm_ratio": {"first": 0.57, "final": 0.67, "mean": 0.70},
            "layer_kd_definition": {"mode": "delta", "objective": "behavioral_kd",
                                    "n_supervised_pairs": 48, "topology_mismatch": "16 absorbed"},
        },
    }
    manifest = {"teacher": "Qwen/Qwen3.8-27B",
                "teacher_revision": "dbdc473dea0d6a9763042881cc33d6058d1742d2",
                "student": "qwen38_19b_h5120_l48_moe", "max_tokens": 700000}

    p = payload(summary, tmp_path, manifest)
    assert p["metrics"]["final_loss"] == 1.70
    assert p["metrics"]["final_validation_loss"] == 11.09
    assert p["metrics"]["series"]["layer_kd_loss"]["final"] == 1.70
    assert p["metrics"]["series"]["top1_agreement"]["final"] == 0.14
    assert set(p["metrics"]["series"]) >= {"kd_loss", "ce_loss", "top1_agreement",
                                           "layer_kd_loss", "layer_norm_ratio"}
    assert p["vram"]["peak_allocated_gib"] == 38.95
    assert p["throughput"]["tokens_per_second"] == 181.9
    assert p["git_commit"] == "f4fc999"
    assert p["seed"] == 0
    assert p["max_tokens"] == 700000
    assert p["sequence_length"] == 1536
    assert p["kd_temperature"] == 2.0 and p["kd_top_k"] == 64
    assert p["student_parameter_counts"]["total_parameters"] == 13031508864


def test_run004_recorder_rejects_pointwise_summary(tmp_path):
    from scripts.record_run004_kd import validate

    manifest = {
        "objective": "behavioral_kd",
        "behavioral_mode": "delta",
        "student": "qwen38_19b_h5120_l48_moe",
        "teacher": "Qwen/Qwen3.8-27B",
        "teacher_revision": "dbdc473dea0d6a9763042881cc33d6058d1742d2",
    }
    summary = {
        "objective": "layer_kd",
        "distillation": {
            "layer_kd_definition": {
                "mode": "pointwise",
                "objective": "layer_kd",
                "n_supervised_pairs": 48,
                "topology_mismatch": "16 removed",
            }
        },
    }
    problems = validate(summary, manifest)
    assert any("mode" in problem for problem in problems)
    assert any("objective" in problem for problem in problems)
