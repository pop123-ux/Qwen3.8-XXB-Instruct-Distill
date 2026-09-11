#!/usr/bin/env python3
"""Run008 (raw residual delta) vs Run009 (pointwise) - matched seed-1 pair.

Same seed, same machine, same package environment, same corpus/teacher/student.
Only the objective differs (layer_kd_normalise False vs True + delta patch).
"""
import json
from pathlib import Path

R8 = Path('/workspace/runs/run008_raw_delta_seed1')
R9 = Path('/workspace/runs/run009_pointwise_seed1')

def load(p):
    s = json.load(open(p / 'summary.json'))
    rows = [json.loads(l) for l in open(p / 'metrics.jsonl')]
    ev = [(r['step'], r['validation_loss']) for r in rows if 'validation_loss' in r]
    d = s['distillation']
    return dict(s=s, d=d, ev=ev)

a, b = load(R8), load(R9)          # a = treatment (raw delta), b = control (pointwise)
sa, sb, da, db = a['s'], b['s'], a['d'], b['d']

print("=" * 92)
print("MATCHED SEED-1 PAIR — same seed / machine / package env; objective is the only difference")
print("=" * 92)

print("\n--- INTEGRITY (must match across the pair) ---")
inv = [('packed_corpus_sha256', sa['corpus']['sha256'],           sb['corpus']['sha256']),
       ('teacher_revision',     da['teacher']['teacher_revision'], db['teacher']['teacher_revision']),
       ('seed',                 sa['config']['training']['seed'],  sb['config']['training']['seed']),
       ('steps',                sa['steps'],                       sb['steps']),
       ('sequence_length',      sa['corpus']['sequence_length'],   sb['corpus']['sequence_length']),
       ('n_tokens',             sa['corpus']['n_tokens'],          sb['corpus']['n_tokens']),
       ('learning_rate',        sa['config']['training']['learning_rate'], sb['config']['training']['learning_rate']),
       ('lora_rank/alpha',      f"{sa['config']['training']['lora_rank']}/{sa['config']['training']['lora_alpha']}",
                                f"{sb['config']['training']['lora_rank']}/{sb['config']['training']['lora_alpha']}"),
       ('pretrained',           sa['config']['model']['pretrained'], sb['config']['model']['pretrained']),
       ('outcome',              sa['outcome'],                     sb['outcome'])]
allok = True
for name, x, y in inv:
    ok = (x == y); allok &= ok
    print(f"  [{'OK ' if ok else 'DIFF'}] {name:22s} {x}")
print(f"\n  INTENDED DIFFERENCE:")
print(f"    layer_kd_normalise : run008={sa['config']['training']['layer_kd_normalise']}  run009={sb['config']['training']['layer_kd_normalise']}")
print(f"    behavioral mode    : run008={da.get('layer_kd_definition',{}).get('mode')}  run009={db.get('layer_kd_definition',{}).get('mode')}")
print(f"  INVARIANT GATE: {'PASS' if allok else 'FAIL'}")

print("\n" + "=" * 92)
print("PRIMARY COMPARISON  (validation loss is the protocol's primary metric)")
print("=" * 92)
print(f"{'metric':<34}{'Run009 pointwise':>18}{'Run008 raw-delta':>18}{'delta':>12}{'pct':>10}")
print("-" * 92)

def row(label, ctrl, trt, lower_better=True, pct=True):
    if ctrl is None or trt is None:
        print(f"{label:<34}{'-':>18}{'-':>18}{'-':>12}{'-':>10}"); return
    d = trt - ctrl
    p = (d / ctrl * 100) if (pct and ctrl) else None
    mark = ""
    if lower_better:   mark = "  raw-delta better" if d < 0 else "  pointwise better" if d > 0 else ""
    else:              mark = "  raw-delta better" if d > 0 else "  pointwise better" if d < 0 else ""
    ps = f"{p:+.2f}%" if p is not None else "-"
    print(f"{label:<34}{ctrl:>18.5f}{trt:>18.5f}{d:>+12.5f}{ps:>10}{mark}")

row("final validation loss",   sb['final_validation_loss'], sa['final_validation_loss'])
row("best validation loss",    sb['best_validation_loss'],  sa['best_validation_loss'])
row("final CE loss",           db['ce_loss']['final'],      da['ce_loss']['final'])
row("final logit KD loss",     db['kd_loss']['final'],      da['kd_loss']['final'])
row("final top-1 agreement",   db['top1_agreement']['final'], da['top1_agreement']['final'], lower_better=False)
row("mean top-1 agreement",    db['top1_agreement']['mean'],  da['top1_agreement']['mean'],  lower_better=False)

print("\n--- OPERATIONAL ---")
row("runtime (s)",             sb['runtime_s'],             sa['runtime_s'])
row("tokens/sec",              sb['tokens_per_second'],     sa['tokens_per_second'], lower_better=False)
row("peak allocated (GiB)",    sb['memory']['peak_allocated_gib'], sa['memory']['peak_allocated_gib'])
row("peak reserved (GiB)",     sb['memory']['peak_reserved_gib'],  sa['memory']['peak_reserved_gib'])
print(f"{'tokens seen':<34}{sb['tokens_seen']:>18}{sa['tokens_seen']:>18}")

print("\n--- OBJECTIVE-INTERNAL (NOT comparable across arms: different objectives) ---")
print(f"{'':<34}{'Run009 pointwise':>18}{'Run008 raw-delta':>18}")
for lbl, key in [('layer loss first','first'), ('layer loss final','final')]:
    print(f"{lbl:<34}{db['layer_kd_loss'][key]:>18.5f}{da['layer_kd_loss'][key]:>18.5f}")
print(f"{'final norm ratio':<34}{db['layer_norm_ratio']['final']:>18.5f}{da['layer_norm_ratio']['final']:>18.5f}")

print("\n--- VALIDATION TRAJECTORY ---")
print(f"{'step':>6}{'Run009 pointwise':>20}{'Run008 raw-delta':>20}{'delta':>12}")
m9 = dict(b['ev']); m8 = dict(a['ev'])
for st in sorted(set(m9) | set(m8)):
    x, y = m9.get(st), m8.get(st)
    ds = f"{y-x:+.4f}" if (x is not None and y is not None) else "-"
    print(f"{st:>6}{(f'{x:.4f}' if x is not None else '-'):>20}{(f'{y:.4f}' if y is not None else '-'):>20}{ds:>12}")

print("\n" + "=" * 92)
dv = sa['final_validation_loss'] - sb['final_validation_loss']
dt = da['top1_agreement']['final'] - db['top1_agreement']['final']
print(f"PRIMARY (final validation loss): {dv:+.5f}  ({dv/sb['final_validation_loss']*100:+.2f}%)  "
      f"-> {'raw-delta better' if dv < 0 else 'pointwise better'}")
print(f"DIAGNOSTIC (final top-1 agree) : {dt:+.5f}  -> {'raw-delta better' if dt > 0 else 'pointwise better'}")
print("=" * 92)
