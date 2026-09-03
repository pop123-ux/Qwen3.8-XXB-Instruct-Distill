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
    assert gate_ids == {"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12"}
    assert campaign["contribution_working_name"] == "computational_span_distillation"
    assert campaign["rules"]["no_fabricated_results"] is True
    assert campaign["rules"]["novelty_claim_requires_prior_art_comparator"] is True
    assert campaign["rules"]["central_result_requires_seed_replication"] is True


def test_literature_review_contains_closest_prior_art() -> None:
    text = (ROOT / "docs" / "LITERATURE_REVIEW.md").read_text()
    for needle in (
        "Beyond Logits: Aligning Feature Dynamics",
        "MTA: Multi-Granular Trajectory Alignment",
        "One-for-All: Bridge the Gap Between Heterogeneous Architectures",
        "Every Expert Matters",
        "Short Data, Long Context",
    ):
        assert needle in text


def test_novelty_hardening_requires_adjacent_vs_span_ablation() -> None:
    text = (ROOT / "docs" / "NOVELTY_HARDENING.md").read_text()
    assert "Adjacent dynamics vs span dynamics" in text
    assert "functional influence profile" in text
    assert "seed" in text.lower()


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
