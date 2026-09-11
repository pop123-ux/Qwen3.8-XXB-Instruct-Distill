#!/usr/bin/env python3
"""Run012: CE-only (no-teacher) baseline, seed 1 — the "is KD buying anything?" floor.

Construction guarantee
----------------------
This does NOT hand-write a config. It parses RUN008'S OWN ARGV through kd_run's parser
and builds the ExperimentConfig with kd_run.build_config(), then changes exactly two
things:
  1. training.objective : "layer_kd" -> "sft"        (the intended scientific change)
  2. runtime.output_dir / name                        (artifact paths only)
and calls train(config, None, teacher=None).

Everything else — corpus path, tokenizer path (the teacher's, so the packed-token
identity is preserved), max_tokens, sequence_length, max_steps, batch_size,
gradient_accumulation_steps, learning_rate, optimizer, strategy (QLoRA), lora_rank,
lora_alpha, precision, gradient_checkpointing, seed, eval_every/save_every/log_every —
comes from Run008's argv unchanged.

Genuine CE-only
---------------
trainer.train() takes `teacher` as a live provider object. With teacher=None the loop
branch at trainer.py:707 executes `model(input_ids=batch, labels=batch)` and uses
`outputs.loss` (HF causal-LM cross-entropy). No teacher forward, no logit KD, no layer
KD, no behavioural loss, no hidden states are requested. _require_supported() also
*enforces* this: it raises if objective=='sft' and a teacher provider is passed.

Unavoidable, recorded implementation differences vs Run008
---------------------------------------------------------
  * No 4-bit teacher is resident, so peak VRAM is necessarily much lower. This is
    inherent to a no-teacher control and is not a protocol deviation.
  * kd_weight / kd_temperature / kd_top_k / layer_kd_* are carried in the config for
    provenance but are inert under objective='sft'.
  * config.teacher metadata (model + revision) is retained so the artifact records which
    tokenizer produced the packed stream; no teacher weights are loaded.

LOCKED before execution. No coefficient is tuned.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/workspace/Qwen3.8-XXB-Instruct-Distill")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import kd_run
import run008_raw_delta_seed1 as run008
from qwen_distill.training.trainer import train

OUTPUT = Path("/workspace/runs/run012_ce_only_seed1")
EXPERIMENT_ID = "run012_ce_only_seed1"


def build() -> tuple[object, dict]:
    """Build Run008's exact config, then flip only the objective."""
    argv = run008.command()
    args = kd_run.build_parser().parse_args(argv) if hasattr(kd_run, "build_parser") \
        else kd_run.parse_args(argv)
    pretrained = Path(args.pretrained)
    config = kd_run.build_config(args, pretrained)

    before = {
        "objective": config.training.objective,
        "text_path": config.data.text_path,
        "tokenizer_path": config.data.tokenizer_path,
        "max_tokens": config.data.max_tokens,
        "max_sequence_length": config.data.max_sequence_length,
        "max_steps": config.training.max_steps,
        "batch_size": config.training.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "optimizer": config.training.optimizer,
        "strategy": config.training.strategy,
        "lora_rank": config.training.lora_rank,
        "lora_alpha": config.training.lora_alpha,
        "precision": config.training.precision,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "seed": config.training.seed,
        "eval_every": config.training.eval_every,
        "save_every": config.training.save_every,
        "log_every": config.training.log_every,
        "pretrained": config.model.pretrained,
    }

    config.training.objective = "sft"          # the ONLY scientific change
    config.name = EXPERIMENT_ID
    config.runtime.output_dir = str(OUTPUT)    # artifact path only
    config.validate()

    after = dict(before); after["objective"] = config.training.objective
    changed = {k: [before[k], after[k]] for k in before if before[k] != after[k]}
    return config, {"matched_from_run008_argv": before,
                    "changed": changed,
                    "is_single_scientific_change": list(changed) == ["objective"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show-diff", action="store_true")
    args = ap.parse_args()

    config, proof = build()
    if args.show_diff:
        print(json.dumps(proof, indent=2))
        return 0 if proof["is_single_scientific_change"] else 1
    if not proof["is_single_scientific_change"]:
        raise SystemExit(f"REFUSING: more than the objective changed:\n{json.dumps(proof, indent=2)}")
    if not args.dry_run and (OUTPUT / "summary.json").exists():
        raise SystemExit(f"Refusing to overwrite completed evidence: {OUTPUT/'summary.json'}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "arm_manifest.json").write_text(json.dumps({
        "experiment": EXPERIMENT_ID,
        "purpose": "no-teacher CE-only floor: does teacher supervision buy anything over ordinary training?",
        "reference_run": "run008_raw_delta_seed1",
        "only_intended_scientific_change": "objective: layer_kd (raw residual-transition KD) -> sft (CE only)",
        "config_construction": "kd_run.build_config() applied to run008.command(); objective overridden to 'sft'",
        "config_match_proof": proof,
        "teacher_used": False,
        "teacher_forward_passes": 0,
        "loss": "HF causal-LM cross-entropy via model(input_ids=batch, labels=batch).loss",
        "enforced_by": ("trainer._require_supported raises if objective=='sft' and a teacher "
                        "provider is passed; train() is called with teacher=None"),
        "recorded_implementation_differences": [
            "no 4-bit teacher resident -> peak VRAM necessarily much lower (inherent to the control)",
            "kd_weight/kd_temperature/kd_top_k/layer_kd_* carried for provenance but inert under sft",
            "config.teacher metadata retained only to record which tokenizer produced the packed stream",
        ],
        "seed": 1,
        "locked_before_execution": True,
        "preregistered_interpretation": {
            "if_ce_only_much_worse_than_all_kd_arms": "teacher supervision is buying real signal",
            "if_ce_only_comparable_to_kd_arms": ("teacher supervision is NOT buying much at this "
                                                 "budget; the whole KD comparison loses significance "
                                                 "and must be reported as such"),
            "if_ce_only_better": "report as-is; do not rescue the KD arms",
            "reference_points_seed1_same_environment": {
                "run009_pointwise": 8.73926, "run008_raw_delta": 7.41131,
                "run010_adjacent_raw_delta": 7.19814},
        },
        "dry_run": args.dry_run,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"Run012 manifest: {OUTPUT/'arm_manifest.json'}")
    print("CE-ONLY: objective=sft, teacher=None (no teacher forward pass)")
    print(f"single scientific change: {proof['is_single_scientific_change']}")
    if args.dry_run:
        print(json.dumps(proof, indent=2))
        print("\n  dry run: nothing was loaded and nothing was trained.")
        return 0
    return train(config, None, teacher=None)


if __name__ == "__main__":
    raise SystemExit(main())
