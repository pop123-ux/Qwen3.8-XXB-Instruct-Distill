#!/usr/bin/env python3
"""Record a completed Run 004 behavioural/delta-KD result.

The recorder is deliberately separate from training. It accepts only the trainer summary
plus the Run 004 manifest, verifies that the summary actually reports ``mode=delta`` and
that the frozen protocol fields are present, then writes one measured ledger entry. It
never invents missing metrics and never upgrades a calibration into a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.research.ledger import Ledger

EXPECTED_STUDENT = "qwen38_19b_h5120_l48_moe"
EXPECTED_TEACHER = "Qwen/Qwen3.8-27B"
EXPECTED_REVISION = "dbdc473dea0d6a9763042881cc33d6058d1742d2"
EXPECTED_OBJECTIVE = "behavioral_kd"
EXPECTED_MODE = "delta"


def validate(summary: dict, manifest: dict) -> list[str]:
    problems: list[str] = []
    if manifest.get("objective") != EXPECTED_OBJECTIVE:
        problems.append(f"manifest objective is {manifest.get('objective')!r}, expected {EXPECTED_OBJECTIVE!r}")
    if manifest.get("behavioral_mode") != EXPECTED_MODE:
        problems.append(f"manifest mode is {manifest.get('behavioral_mode')!r}, expected {EXPECTED_MODE!r}")
    if manifest.get("student") != EXPECTED_STUDENT:
        problems.append("manifest student does not identify the frozen canonical student")
    if manifest.get("teacher") != EXPECTED_TEACHER:
        problems.append("manifest teacher does not identify Qwen3.8-27B")
    if manifest.get("teacher_revision") != EXPECTED_REVISION:
        problems.append("manifest teacher revision is not the canonical pinned revision")

    distillation = summary.get("distillation") or {}
    definition = distillation.get("layer_kd_definition") or {}
    if summary.get("objective") != "layer_kd":
        problems.append("trainer summary did not execute through the validated layer-KD trainer path")
    if definition.get("mode") != EXPECTED_MODE:
        problems.append(f"trainer definition mode is {definition.get('mode')!r}, expected {EXPECTED_MODE!r}")
    if definition.get("objective") != EXPECTED_OBJECTIVE:
        problems.append(f"trainer definition objective is {definition.get('objective')!r}, expected {EXPECTED_OBJECTIVE!r}")
    if definition.get("n_supervised_pairs") != 48:
        problems.append("Run 004 must supervise all 48 canonical student layers")
    if definition.get("topology_mismatch") is None:
        problems.append("trainer summary is missing topology-mismatch provenance")
    return problems


def payload(summary: dict, run_dir: Path, manifest: dict) -> dict:
    distillation = summary["distillation"]
    definition = distillation["layer_kd_definition"]
    return {
        "arm": "behavioral_kd",
        "objective": EXPECTED_OBJECTIVE,
        "mode": EXPECTED_MODE,
        "run_dir": str(run_dir),
        "student": EXPECTED_STUDENT,
        "teacher": EXPECTED_TEACHER,
        "teacher_revision": EXPECTED_REVISION,
        "steps": summary.get("steps"),
        "tokens": summary.get("tokens_seen"),
        "outcome": summary.get("outcome"),
        "memory_gib": summary.get("memory"),
        "throughput": summary.get("throughput"),
        "loss": {
            "training": summary.get("loss"),
            "kd": distillation.get("kd_loss"),
            "ce": distillation.get("ce_loss"),
            "layer_behavior": distillation.get("layer_kd_loss"),
        },
        "agreement": distillation.get("top1_agreement"),
        "validation": summary.get("validation"),
        "behavioral": {
            "definition": definition,
            "magnitude": distillation.get("layer_magnitude"),
            "direction": distillation.get("layer_direction"),
            "norm_ratio": distillation.get("layer_norm_ratio"),
        },
        "protocol": {
            "sequence_length": summary["config"]["data"]["max_sequence_length"],
            "batch_size": summary["config"]["training"]["batch_size"],
            "gradient_accumulation_steps": summary["config"]["training"]["gradient_accumulation_steps"],
            "steps": summary["config"]["training"]["max_steps"],
            "seed": summary["config"]["training"]["seed"],
            "optimizer": summary["config"]["training"]["optimizer"],
            "precision": summary["config"]["training"]["precision"],
            "strategy": summary["config"]["training"]["strategy"],
            "lora_rank": summary["config"]["training"]["lora_rank"],
            "lora_alpha": summary["config"]["training"]["lora_alpha"],
            "kd_temperature": summary["config"]["training"]["kd_temperature"],
            "kd_top_k": summary["config"]["training"]["kd_top_k"],
            "chunk_pairs": summary["config"]["training"]["layer_kd_chunk_pairs"],
        },
        "manifest": manifest,
        "caveat": (
            "This is the first behavioural/delta-KD control. A 128-step run tests whether "
            "the objective is trainable and whether teacher-alignment diagnostics move; it "
            "is not a downstream capability or SOTA result. The first comparison is against "
            "Run 003's matched pointwise layer-KD control."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=Path("experiments/ledger.jsonl"))
    args = parser.parse_args(argv)

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    problems = validate(summary, manifest)
    if problems:
        print("REFUSED: Run 004 provenance validation failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 2

    ledger = Ledger(args.ledger)
    entry = ledger.measured(
        "canonical_kd",
        "Run 004: pure behavioural/delta KD on the frozen canonical student",
        payload(summary, args.summary.parent, manifest),
        arm="run004_behavioral_kd",
        tags=["run004", "canonical", "qlora", "behavioral_kd", "delta"],
    )
    print(f"recorded {entry.id}  {entry.kind}  {entry.title}")
    print(f"ledger: {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
