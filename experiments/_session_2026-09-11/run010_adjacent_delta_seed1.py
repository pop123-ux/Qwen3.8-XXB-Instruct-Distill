#!/usr/bin/env python3
"""Run010: ADJACENT-vs-SPAN abstraction ablation, seed 1.

Preregistered RQ1 gate: "span-vs-adjacent abstraction ablation".

Run008 (raw residual delta) supervises the student's residual contribution against the
contribution of the COMPLETE teacher span [a,b) assigned to that student layer, so the 16
removed teacher layers are explicitly charged to a student computation.

Run010 changes exactly ONE thing: the teacher target becomes the SINGLE anchor layer's own
contribution, h_t[m(l)+1] - h_t[m(l)], instead of the whole span h_t[b] - h_t[a]. The
student term, the raw/unnormalised treatment, and every other frozen value are identical
to Run008.

This isolates whether the SPAN AGGREGATION (the abstraction the project claims) is doing
the work, or whether matching any local adjacent teacher delta would do just as well.

LOCKED before execution. No coefficient is tuned. Compare against Run008 (span) and
Run009 (pointwise), all at seed 1 in this same package environment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/workspace/Qwen3.8-XXB-Instruct-Distill")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import qwen_distill.distillation.behavioral as bh
from kd_run import main as kd_main
from run004_behavioral_kd import patch_trainer_for_delta, restore_trainer

TEACHER = Path("/workspace/models/qwen3.8-27b-dbdc473")
REVISION = "dbdc473dea0d6a9763042881cc33d6058d1742d2"
STUDENT = Path("/workspace/runs/pilot001/transferred")
CORPUS = Path("/workspace/corpora/gutenberg/train.txt")
OUTPUT = Path("/workspace/runs/run010_adjacent_delta_seed1")
EXPERIMENT_ID = "run010_adjacent_delta_seed1"
SEED = 1

_orig_pair_tensors = bh._pair_tensors
_fired = {"n": 0}


def adjacent_pair_tensors(student_hidden, teacher_hidden, mapping, spans, mode, s):
    """Identical to the delta objective except the teacher target is the ADJACENT layer."""
    if mode == "delta":
        anchor = mapping[s]
        _fired["n"] += 1
        return (student_hidden[s + 1] - student_hidden[s],
                teacher_hidden[anchor + 1] - teacher_hidden[anchor])
    return _orig_pair_tensors(student_hidden, teacher_hidden, mapping, spans, mode, s)


def command() -> list[str]:
    return [
        "--teacher", str(TEACHER), "--revision", REVISION, "--quantization", "4bit",
        "--student", "canonical", "--pretrained", str(STUDENT),
        "--text-path", str(CORPUS), "--max-tokens", "700000",
        "--sequence-length", "1536", "--steps", "128",
        "--batch-size", "1", "--gradient-accumulation-steps", "1",
        "--learning-rate", "0.0002",
        "--objective", "layer_kd",
        "--layer-kd-direction-weight", "1.0",
        "--layer-kd-chunk-pairs", "4",
        "--layer-kd-no-normalise",
        "--kd-temperature", "2.0", "--kd-top-k", "64",
        "--strategy", "qlora", "--optimizer", "adamw",
        "--lora-rank", "16", "--lora-alpha", "32",
        "--precision", "bf16", "--seed", str(SEED),
        "--log-every", "1", "--eval-every", "32", "--save-every", "64",
        "--output", str(OUTPUT), "--name", EXPERIMENT_ID,
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and (OUTPUT / "summary.json").exists():
        raise SystemExit(f"Refusing to overwrite completed evidence: {OUTPUT/'summary.json'}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cmd = command()
    (OUTPUT / "arm_manifest.json").write_text(json.dumps({
        "experiment": EXPERIMENT_ID,
        "purpose": "span-vs-adjacent abstraction ablation (preregistered RQ1 gate)",
        "reference_run": "run008_raw_delta_seed1",
        "only_intended_scientific_change": (
            "teacher target: full assigned span h_t[b]-h_t[a] -> single adjacent anchor "
            "layer h_t[m(l)+1]-h_t[m(l)]"),
        "objective": "behavioral_kd", "behavioral_mode": "delta_adjacent",
        "normalise": False,
        "loss_equation": "mean_pairs[MSE(d_s, d_adj) + (1-cos(d_s, d_adj))] on raw deltas, "
                         "d_s = h_s[l+1]-h_s[l], d_adj = h_t[m(l)+1]-h_t[m(l)]",
        "seed": SEED, "teacher_revision": REVISION,
        "locked_before_execution": True,
        "dry_run": args.dry_run, "command": cmd,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Run010 manifest: {OUTPUT/'arm_manifest.json'}")
    print("ABLATION: teacher target = adjacent anchor layer (NOT the assigned span)")

    bh._pair_tensors = adjacent_pair_tensors
    patched = patch_trainer_for_delta()
    try:
        rc = kd_main(cmd + (["--dry-run"] if args.dry_run else []))
    finally:
        restore_trainer(patched)
        bh._pair_tensors = _orig_pair_tensors
        print(f"adjacent_pair_tensors invocations: {_fired['n']}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
