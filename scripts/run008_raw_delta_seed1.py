#!/usr/bin/env python3
"""Run008: exact seed-1 replication of Run006 raw residual-delta KD.

Scientific invariant: this launcher changes only the random seed (0 -> 1) relative to
Run006. It reuses the validated Run004 delta patch and the existing layer-KD trainer,
while explicitly disabling RMS normalisation exactly as Run006 did.
"""

from __future__ import annotations

import json
from pathlib import Path

from kd_run import main as kd_main
from run004_behavioral_kd import patch_trainer_for_delta, restore_trainer

TEACHER = Path("/workspace/models/qwen3.8-27b-dbdc473")
REVISION = "dbdc473dea0d6a9763042881cc33d6058d1742d2"
STUDENT = Path("/workspace/runs/pilot001/transferred")
CORPUS = Path("/workspace/corpora/gutenberg/train.txt")
OUTPUT = Path("/workspace/runs/run008_raw_delta_seed1")
EXPERIMENT_ID = "run008_raw_delta_seed1"
SEED = 1


def command() -> list[str]:
    return [
        "--teacher", str(TEACHER),
        "--revision", REVISION,
        "--quantization", "4bit",
        "--student", "canonical",
        "--pretrained", str(STUDENT),
        "--text-path", str(CORPUS),
        "--max-tokens", "700000",
        "--sequence-length", "1536",
        "--steps", "128",
        "--batch-size", "1",
        "--gradient-accumulation-steps", "1",
        "--learning-rate", "0.0002",
        "--objective", "layer_kd",
        "--layer-kd-direction-weight", "1.0",
        "--layer-kd-chunk-pairs", "4",
        "--layer-kd-no-normalise",
        "--kd-temperature", "2.0",
        "--kd-top-k", "64",
        "--strategy", "qlora",
        "--optimizer", "adamw",
        "--lora-rank", "16",
        "--lora-alpha", "32",
        "--precision", "bf16",
        "--seed", str(SEED),
        "--log-every", "1",
        "--eval-every", "32",
        "--save-every", "64",
        "--output", str(OUTPUT),
        "--name", EXPERIMENT_ID,
    ]


def main() -> int:
    if (OUTPUT / "summary.json").exists():
        raise SystemExit(f"Refusing to overwrite completed evidence: {OUTPUT / 'summary.json'}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    cmd = command()
    manifest = {
        "experiment": EXPERIMENT_ID,
        "purpose": "second-seed replication of Run006 raw/unnormalised residual matching",
        "reference_run": "run006_raw_delta_behavioral_kd",
        "only_intended_scientific_change": "seed: 0 -> 1",
        "objective": "behavioral_kd",
        "behavioral_mode": "delta",
        "normalise": False,
        "loss_equation": "mean_pairs[MSE(d_s,d_t) + (1-cos(d_s,d_t))] on raw deltas",
        "seed": SEED,
        "teacher_revision": REVISION,
        "command": cmd,
    }
    (OUTPUT / "arm_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    patched = patch_trainer_for_delta()
    try:
        return kd_main(cmd)
    finally:
        restore_trainer(patched)


if __name__ == "__main__":
    raise SystemExit(main())
