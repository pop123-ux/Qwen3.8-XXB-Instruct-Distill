#!/usr/bin/env python3
"""Run the preregistered composite matrix at seed 1, loading the teacher ONCE.

Arms run in the order A0, A2, A1, A3. Each builds its config via kd_run.build_config()
from run008's argv (envelope matched by construction, verified per arm), changes only the
objective / composite weights / hidden normalisation / artifact paths, and trains.

Hidden normalisation is explicit per arm, not inherited:
  A1 pointwise -> normalised (the repository's pointwise convention and
     CompositeLossConfig's default; an unnormalised pointwise MSE is dominated by
     whichever mapped pairs sit deepest, which would be a strawman baseline)
  A3 delta     -> raw (the variant the mechanism probes identified as the live one)
Both are recorded. The clean matched-normalisation contrasts are queued separately.
"""
from __future__ import annotations
import json, sys, time, traceback
from pathlib import Path

ROOT = Path("/workspace/Qwen3.8-XXB-Instruct-Distill")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import kd_run, run008_raw_delta_seed1 as run008
from qwen_distill.distillation.backends import TransformersTeacher
from qwen_distill.distillation.behavioral import HIDDEN_DELTA, HIDDEN_POINTWISE
from qwen_distill.research.ablations import ARMS
from qwen_distill.training.trainer import train

PLAN = [("A0", "raw"), ("A2", "raw"), ("A1", "normalised"), ("A3", "raw")]
INHERITED = ("max_steps","batch_size","gradient_accumulation_steps","learning_rate",
             "optimizer","strategy","lora_rank","lora_alpha","precision",
             "gradient_checkpointing","seed","eval_every","save_every","log_every",
             "layer_kd_chunk_pairs","layer_kd_direction_weight")
DATA = ("text_path","tokenizer_path","max_tokens","max_sequence_length")


def make(arm_name, norm_tag):
    arm = ARMS[arm_name]
    weights = dict(arm.loss_weights)
    args = kd_run.parse_args(run008.command())
    cfg = kd_run.build_config(args, Path(args.pretrained))
    before = {k: getattr(cfg.training, k) for k in INHERITED}
    before |= {k: getattr(cfg.data, k) for k in DATA}
    before["pretrained"] = cfg.model.pretrained

    eid = f"run013_{arm_name}_{arm.name}_{norm_tag}_seed1"
    out = Path("/workspace/runs") / eid
    cfg.training.objective = "composite"
    cfg.training.composite_weights = weights
    cfg.training.layer_kd_normalise = (norm_tag == "normalised")
    cfg.name = eid
    cfg.runtime.output_dir = str(out)
    cfg.validate()

    after = {k: getattr(cfg.training, k) for k in INHERITED}
    after |= {k: getattr(cfg.data, k) for k in DATA}
    after["pretrained"] = cfg.model.pretrained
    changed = {k: [before[k], after[k]] for k in before if before[k] != after[k]}
    hidden = (HIDDEN_DELTA if weights.get(HIDDEN_DELTA)
              else HIDDEN_POINTWISE if weights.get(HIDDEN_POINTWISE) else None)
    proof = {"arm": arm_name, "arm_name": arm.name,
             "resolved_coefficients": dict(sorted(weights.items())),
             "hidden_term": hidden, "hidden_normalise": cfg.training.layer_kd_normalise,
             "inherited_from_run008_argv": before,
             "non_objective_fields_changed": changed,
             "is_envelope_matched": changed == {}}
    return cfg, arm, out, eid, proof, hidden


def main() -> int:
    args = kd_run.parse_args(run008.command())
    print("loading teacher once for the whole matrix ...", flush=True)
    backend = TransformersTeacher(model=args.teacher_model, revision=args.revision,
                                  local_path=str(args.teacher),
                                  quantization=args.quantization,
                                  strict_architecture=True)
    backend.load()
    # capture hidden states unconditionally: one teacher serves arms with and without a
    # hidden term, and the extra tensors cost nothing for the arms that ignore them.
    teacher = backend.signal_provider(top_k=args.kd_top_k or None,
                                      temperature=args.kd_temperature,
                                      capture_hidden_states=True)
    print(f"teacher ready: {teacher.describe()}", flush=True)

    results = {}
    for arm_name, norm_tag in PLAN:
        cfg, arm, out, eid, proof, hidden = make(arm_name, norm_tag)
        if (out / "summary.json").exists():
            print(f"\n=== {arm_name} already complete, skipping ===", flush=True)
            continue
        if not proof["is_envelope_matched"]:
            print(f"REFUSING {arm_name}: {proof['non_objective_fields_changed']}", flush=True)
            results[arm_name] = "refused"; continue
        out.mkdir(parents=True, exist_ok=True)
        (out / "arm_manifest.json").write_text(json.dumps({
            "experiment": eid, "preregistered_arm": arm_name, "arm_name": arm.name,
            "question": arm.question, "prediction": arm.prediction,
            "falsified_if": arm.falsified_if,
            "resolved_coefficients": proof["resolved_coefficients"],
            "router_aux_loss_coef": 0.001,
            "hidden_term": hidden, "hidden_normalise": proof["hidden_normalise"],
            "objective": "composite", "envelope_match_proof": proof,
            "reference_run_for_envelope": "run008_raw_delta_seed1",
            "framing_note": ("run008-run011 are PURE hidden-matching mechanism probes and "
                             "are NOT these cells; run012 is a CE-only floor lacking A0's "
                             "router_balance term and is not reused as A0. Historical runs "
                             "are not relabelled."),
            "locked_before_execution": True,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\n{'='*70}\n=== {arm_name} {arm.name} [{norm_tag}] -> {eid}\n"
              f"=== weights {proof['resolved_coefficients']} hidden={hidden}\n{'='*70}",
              flush=True)
        t0 = time.time()
        try:
            rc = train(cfg, None, teacher=teacher)
            results[arm_name] = f"rc={rc} in {time.time()-t0:.1f}s"
        except Exception:
            traceback.print_exc()
            results[arm_name] = "EXCEPTION"
        print(f"=== {arm_name} done: {results[arm_name]}", flush=True)

    print("\n=== MATRIX COMPLETE ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
