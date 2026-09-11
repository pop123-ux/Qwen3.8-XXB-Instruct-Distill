#!/usr/bin/env python3
"""Compact seed-1 comparison: Run009 / Run008 / Run010 / Run011.
All same seed, same machine, same package environment, same packed corpus.
"""
import json
from pathlib import Path

ARMS = [
    ("run009_pointwise_seed1",        "pointwise (normalized)",      "control"),
    ("run011_normalized_delta_seed1", "residual-delta (normalized)", "treatment"),
    ("run008_raw_delta_seed1",        "residual-delta (raw)",        "treatment"),
    ("run010_adjacent_delta_seed1",   "adjacent-delta (raw)",        "treatment"),
]

rows = []
for run, label, kind in ARMS:
    p = Path(f"/workspace/runs/{run}")
    if not (p / "summary.json").exists():
        continue
    s = json.load(open(p / "summary.json")); d = s["distillation"]
    m = [json.loads(l) for l in open(p / "metrics.jsonl")]
    ev = {r["step"]: r["validation_loss"] for r in m if "validation_loss" in r}
    rows.append(dict(run=run, label=label, kind=kind, s=s, d=d, ev=ev,
                     norm=s["config"]["training"]["layer_kd_normalise"],
                     mode=d.get("layer_kd_definition", {}).get("mode")))

print("=" * 104)
print("SEED-1 ARMS — identical seed / machine / package env / packed corpus (e11ca38bb099fc89...)")
print("=" * 104)
print(f"{'arm':<30}{'norm':>6}{'mode':>11}{'final_val':>11}{'CE':>10}{'top1':>9}{'logitKD':>10}{'tok/s':>8}")
print("-" * 104)
for r in rows:
    s, d = r["s"], r["d"]
    print(f"{r['label']:<30}{str(r['norm']):>6}{str(r['mode']):>11}"
          f"{s['final_validation_loss']:>11.4f}{d['ce_loss']['final']:>10.4f}"
          f"{d['top1_agreement']['final']:>9.4f}{d['kd_loss']['final']:>10.4f}"
          f"{s['tokens_per_second']:>8.1f}")

ctrl = next(r for r in rows if r["kind"] == "control")
cv = ctrl["s"]["final_validation_loss"]
print()
print(f"{'vs pointwise control':<30}{'d final_val':>14}{'pct':>9}{'d CE':>11}{'d top1':>10}")
print("-" * 104)
for r in rows:
    if r["kind"] == "control":
        continue
    dv = r["s"]["final_validation_loss"] - cv
    dc = r["d"]["ce_loss"]["final"] - ctrl["d"]["ce_loss"]["final"]
    dt = r["d"]["top1_agreement"]["final"] - ctrl["d"]["top1_agreement"]["final"]
    print(f"{r['label']:<30}{dv:>+14.4f}{dv/cv*100:>+8.2f}%{dc:>+11.4f}{dt:>+10.4f}")

print()
print("DECISIVE NORMALIZATION ABLATION (identical argv except --layer-kd-no-normalise)")
print("-" * 104)
raw = next(r for r in rows if r["run"] == "run008_raw_delta_seed1")
nrm = next((r for r in rows if r["run"] == "run011_normalized_delta_seed1"), None)
if nrm:
    dv = nrm["s"]["final_validation_loss"] - raw["s"]["final_validation_loss"]
    print(f"  raw residual-delta (run008)        final_val {raw['s']['final_validation_loss']:.4f}")
    print(f"  normalized residual-delta (run011) final_val {nrm['s']['final_validation_loss']:.4f}")
    print(f"  effect of ENABLING normalization   {dv:+.4f}  ({dv/raw['s']['final_validation_loss']*100:+.2f}%)")
    print(f"  fraction of the raw-delta-vs-pointwise gap destroyed: "
          f"{dv / (cv - raw['s']['final_validation_loss']) * 100:.1f}%")

print()
print("VALIDATION TRAJECTORY")
print("-" * 104)
steps = sorted({st for r in rows for st in r["ev"]})
print(f"{'step':>6}" + "".join(f"{r['label'][:20]:>22}" for r in rows))
for st in steps:
    print(f"{st:>6}" + "".join(
        f"{r['ev'][st]:>22.4f}" if st in r["ev"] else f"{'-':>22}" for r in rows))
print("=" * 104)
