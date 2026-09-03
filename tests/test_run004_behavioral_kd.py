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
