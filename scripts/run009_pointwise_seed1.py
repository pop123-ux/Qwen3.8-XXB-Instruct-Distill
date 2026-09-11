#!/usr/bin/env python3
"""Matched seed-1 pointwise hidden-state KD control for Run008.

This is the Run003 protocol with the same fresh-session environment and seed=1, so Run008
can be judged against a same-seed control rather than only the historical seed-0 anchor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kd_run import main as kd_main

TEACHER = Path("/workspace/models/qwen3.8-27b-dbdc473")
REVISION = "dbdc473dea0d6a9763042881cc33d6058d1742d2"
STUDENT = Path("/workspace/runs/pilot001/transferred")
CORPUS = Path("/workspace/corpora/gutenberg/train.txt")
OUTPUT = Path("/workspace/runs/run009_pointwise_seed1")
EXPERIMENT_ID = "run009_pointwise_seed1"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate config without loading weights")
    args = parser.parse_args()

    if not args.dry_run and (OUTPUT / "summary.json").exists():
        raise SystemExit(f"Refusing to overwrite completed evidence: {OUTPUT / 'summary.json'}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cmd = command()
    manifest = {
        "experiment": EXPERIMENT_ID,
        "purpose": "same-seed matched pointwise control for Run008",
        "reference_protocol": "run003_layer_kd",
        "only_intended_scientific_change_vs_run003": "seed: 0 -> 1",
        "objective": "pointwise_hidden_state_matching",
        "normalise": True,
        "seed": SEED,
        "teacher_revision": REVISION,
        "dry_run": args.dry_run,
        "command": cmd,
    }
    (OUTPUT / "arm_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return kd_main(cmd + (["--dry-run"] if args.dry_run else []))


if __name__ == "__main__":
    raise SystemExit(main())
