#!/usr/bin/env python3
"""Preregistered composite matrix, seed 1, 128 steps, identical envelope."""
import json
from pathlib import Path

ARMS = [
    ("A0", "run013_A0_ce_only_raw_seed1", "CE only"),
    ("A2", "run013_A2_logits_only_raw_seed1", "CE + logit KD"),
    ("A1", "run013_A1_pointwise_layer_matching_normalised_seed1", "+ pointwise hidden (norm)"),
    ("A3", "run013_A3_behavioural_delta_raw_seed1", "+ residual-transition (raw)"),
]
rows = []
for arm, run, label in ARMS:
    p = Path(f"/workspace/runs/{run}")
    if not (p / "summary.json").exists():
        print(f"MISSING {arm}"); continue
    s = json.load(open(p / "summary.json")); d = s.get("distillation") or {}
    m = [json.loads(l) for l in open(p / "metrics.jsonl")]
    steps = [r for r in m if "loss" in r and "validation_loss" not in r]
    ev = {r["step"]: r["validation_loss"] for r in m if "validation_loss" in r}
    mf = json.load(open(p / "arm_manifest.json"))
    last = steps[-1]
    rows.append(dict(arm=arm, run=run, label=label, s=s, d=d, ev=ev, last=last, mf=mf))

print("="*118)
print("PREREGISTERED COMPOSITE MATRIX - seed 1, 128 steps, packed corpus e11ca38bb099fc89...")
print("="*118)
print(f"{'arm':<4}{'terms':<28}{'final_val':>10}{'best_val':>10}{'CE':>9}{'logitKD':>9}"
      f"{'top1':>8}{'hidden':>9}{'aux':>8}{'objtot':>9}{'tok/s':>8}{'peakGiB':>9}")
print("-"*118)
for r in rows:
    s,d,l = r["s"], r["d"], r["last"]
    g=lambda k: l.get(k)
    def f(v,w,p=4): return f"{v:>{w}.{p}f}" if isinstance(v,(int,float)) else f"{'-':>{w}}"
    print(f"{r['arm']:<4}{r['label']:<28}"
          f"{s['final_validation_loss']:>10.4f}{s['best_validation_loss']:>10.4f}"
          f"{f(g('ce_loss'),9)}{f(g('kd_loss'),9)}{f(g('top1_agreement'),8)}"
          f"{f(g('layer_kd_loss'),9)}{f(g('router_aux'),8)}{f(g('objective_total'),9)}"
          f"{s['tokens_per_second']:>8.1f}{s['memory']['peak_allocated_gib']:>9.2f}")

print("\nDECISIVE CONTRASTS (final validation loss; lower is better)")
print("-"*118)
V = {r["arm"]: r["s"]["final_validation_loss"] for r in rows}
T = {r["arm"]: r["last"].get("top1_agreement") for r in rows}
def contrast(a,b,q):
    if a in V and b in V:
        dv=V[a]-V[b]; dt=(T[a]-T[b]) if (T.get(a) is not None and T.get(b) is not None) else None
        ts=f"  d_top1 {dt:+.4f}" if dt is not None else ""
        print(f"  {a} vs {b}: {dv:+.4f} ({dv/V[b]*100:+.2f}%){ts}   {q}")
contrast("A2","A0","value of ordinary output/logit KD")
contrast("A1","A2","marginal value of pointwise hidden supervision")
contrast("A3","A2","marginal value of raw residual-transition supervision")
contrast("A3","A1","residual-transition vs pointwise in a realistic recipe")

print("\nVALIDATION TRAJECTORY")
print("-"*118)
steps = sorted({s for r in rows for s in r["ev"]})
print(f"{'step':>6}"+"".join(f"{r['arm']:>12}" for r in rows))
for st in steps:
    print(f"{st:>6}"+"".join(f"{r['ev'][st]:>12.4f}" if st in r["ev"] else f"{'-':>12}" for r in rows))

print("\nINTEGRITY")
print("-"*118)
for r in rows:
    s,mf = r["s"], r["mf"]
    ok = (s["corpus"]["sha256"]=="e11ca38bb099fc89c2f74e96f5d2f1209def6a16f6a8432d4e9972acd50c100d"
          and s["steps"]==128 and s["config"]["training"]["seed"]==1
          and s["corpus"]["n_tokens"]==700000 and s["oom"] is None
          and s["outcome"]=="completed"
          and s["parameter_counts"]["trainable_parameters"]==23003136
          and mf["envelope_match_proof"]["is_envelope_matched"])
    print(f"  [{'OK ' if ok else 'CHK'}] {r['arm']} coeff={mf['resolved_coefficients']} "
          f"hidden={mf['hidden_term']} norm={mf['hidden_normalise']} "
          f"peak_res={s['memory']['peak_reserved_gib']:.2f}")
print("="*118)
print("NOTE: component columns (CE, logitKD, hidden, aux) are UNWEIGHTED magnitudes.")
print("      Only 'objtot' is the weighted objective actually optimised.")
