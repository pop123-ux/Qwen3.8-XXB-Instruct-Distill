from __future__ import annotations

import json
from pathlib import Path

from scripts.guard_vram import parse_args
from scripts.run004_behavioral_kd import build_kd_run_args

ROOT = Path(__file__).resolve().parents[1]


def test_campaign_declares_hard_evidence_gates() -> None:
    campaign = json.loads((ROOT / "experiments" / "research_campaign.json").read_text())
    assert campaign["training_vram_ceiling_gib"] == 45.0
    gate_ids = {gate["id"] for gate in campaign["evidence_gates"]}
    assert gate_ids == {"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"}
    assert campaign["rules"]["no_fabricated_results"] is True


def test_matched_behavioral_protocol_reproduces_run003_data_cap() -> None:
    class Args:
        teacher = "/teacher"
        revision = "dbdc473dea0d6a9763042881cc33d6058d1742d2"
        pretrained = "/student"
        text_path = "/corpus/train.txt"
        output = "/runs/matched"
        experiment_id = "run004_matched_behavioral_kd"
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

    command = build_kd_run_args(Args())
    assert command[command.index("--max-tokens") + 1] == "700000"
    assert command[command.index("--sequence-length") + 1] == "1536"
    assert command[command.index("--steps") + 1] == "128"
    assert command[command.index("--name") + 1] == "run004_matched_behavioral_kd"


def test_vram_guard_defaults_to_45_gib() -> None:
    args = parse_args(["--", "python", "-c", "pass"])
    assert args.max_vram_gib == 45.0
    assert args.command == ["python", "-c", "pass"]
