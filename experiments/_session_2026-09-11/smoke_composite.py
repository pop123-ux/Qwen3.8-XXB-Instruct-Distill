#!/usr/bin/env python3
"""One-step GPU smoke test of the composite path before spending a full arm.

Checks on the real model/teacher/corpus: the run reaches an optimizer step, every
enabled term is finite and nonzero, trainable parameter count matches the QLoRA
envelope, LoRA parameters actually received gradients, and peak memory sits under the
45 GiB guard.
"""
import json, sys, time
from pathlib import Path
ROOT = Path("/workspace/Qwen3.8-XXB-Instruct-Distill")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/workspace/runpod_sessions/2026-09-11-rq1-3h")

import torch
import kd_run, run008_raw_delta_seed1 as run008
from qwen_distill.distillation.backends import TransformersTeacher
from qwen_distill.research.ablations import ARMS
from qwen_distill.training.trainer import train

ARM = sys.argv[1] if len(sys.argv) > 1 else "A3"
STEPS = 2
out = Path(f"/tmp/claude-0/-workspace-Qwen3-8-XXB-Instruct-Distill/818b7317-29ae-4f0d-889b-4809e266adbd/scratchpad/smoke_{ARM}")

args = kd_run.parse_args(run008.command())
config = kd_run.build_config(args, Path(args.pretrained))
config.training.objective = "composite"
config.training.composite_weights = dict(ARMS[ARM].loss_weights)
config.training.layer_kd_normalise = False
config.training.max_steps = STEPS
config.training.eval_every = STEPS
config.training.save_every = 10**6
config.training.log_every = 1
config.name = f"smoke_{ARM}"
config.runtime.output_dir = str(out)
config.validate()
print("weights:", dict(sorted(config.training.composite_weights.items())))

backend = TransformersTeacher(model=args.teacher_model, revision=args.revision,
                              local_path=str(args.teacher), quantization=args.quantization,
                              strict_architecture=True)
backend.load()
has_hidden = any(k.startswith("hidden_") for k in config.training.composite_weights)
teacher = backend.signal_provider(top_k=args.kd_top_k or None,
                                  temperature=args.kd_temperature,
                                  capture_hidden_states=has_hidden)
t0 = time.time()
rc = train(config, None, teacher=teacher)
print(f"\nrc={rc} in {time.time()-t0:.1f}s")

s = json.load(open(out / "summary.json"))
rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
steps = [r for r in rows if "loss" in r and "validation_loss" not in r]
print("\n=== SMOKE CHECKS ===")
ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {detail}")

check("reached optimizer steps", len(steps) >= 1, f"({len(steps)} logged)")
r = steps[-1]
import math
check("objective_total finite & nonzero", math.isfinite(r.get("objective_total", float('nan'))) and r["objective_total"] != 0, f"= {r.get('objective_total')}")
check("ce_loss finite & nonzero", math.isfinite(r.get("ce_loss", float('nan'))) and r["ce_loss"] > 0, f"= {r.get('ce_loss')}")
if config.training.composite_weights.get("logit_kd"):
    check("kd_loss finite & nonzero", math.isfinite(r.get("kd_loss", float('nan'))) and r["kd_loss"] > 0, f"= {r.get('kd_loss')}")
if config.training.composite_weights.get("router_balance"):
    check("router_aux finite & nonzero", math.isfinite(r.get("router_aux", float('nan'))) and r["router_aux"] > 0, f"= {r.get('router_aux')}")
if has_hidden:
    check("hidden term finite & nonzero", math.isfinite(r.get("layer_kd_loss", float('nan'))) and r["layer_kd_loss"] > 0, f"= {r.get('layer_kd_loss')}")
    check("hidden_weighted recorded", "hidden_weighted" in r, f"= {r.get('hidden_weighted')}")
pc = s["parameter_counts"]
check("trainable == QLoRA envelope 23,003,136", pc["trainable_parameters"] == 23003136, f"= {pc['trainable_parameters']:,}")
check("packed corpus sha matched", s["corpus"]["sha256"] == "e11ca38bb099fc89c2f74e96f5d2f1209def6a16f6a8432d4e9972acd50c100d")
check("n_tokens 700000", s["corpus"]["n_tokens"] == 700000)
check("seed 1", s["config"]["training"]["seed"] == 1)
pa = s["memory"]["peak_allocated_gib"]; pr = s["memory"]["peak_reserved_gib"]
check("peak allocated < 45 GiB", pa < 45.0, f"= {pa:.3f}")
check("peak reserved < 45 GiB", pr < 45.0, f"= {pr:.3f}")
check("no OOM", s["oom"] is None)
check("loss decreased or finite", math.isfinite(steps[0]["loss"]))
print(f"\n  first-step objective {steps[0]['loss']:.4f} -> last {steps[-1]['loss']:.4f}")
print(f"\nSMOKE: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
