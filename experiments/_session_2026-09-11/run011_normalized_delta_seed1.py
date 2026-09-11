#!/usr/bin/env python3
"""Run011: NORMALIZED residual-delta KD, seed 1 — direct normalization ablation of Run008.

Scientific question
-------------------
Run008 (raw residual-delta) beat the matched pointwise control Run009 by -1.328 validation
at seed 1, replicating seed 0. Run010 showed span aggregation is NOT the mechanism. The
remaining working hypothesis is that preserving representation-UPDATE MAGNITUDE is what
drives the advantage.

Run011 tests that directly: it is Run008 with residual-delta normalization ENABLED and
nothing else changed. Under RMS normalization the per-pair magnitude term collapses
(MSE(nds, ndt) == 2*(1-cos(nds, ndt)) identically), so the objective becomes
direction-only and carries no update-magnitude information.

Construction guarantee
----------------------
The argv is DERIVED from run008's own command() by removing exactly the single token
"--layer-kd-no-normalise". It is not hand-retyped, so drift in any other flag is
impossible by construction. The derivation is asserted at runtime.

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

from kd_run import main as kd_main
from run004_behavioral_kd import patch_trainer_for_delta, restore_trainer
import run008_raw_delta_seed1 as run008

OUTPUT = Path("/workspace/runs/run011_normalized_delta_seed1")
EXPERIMENT_ID = "run011_normalized_delta_seed1"
DROPPED_FLAG = "--layer-kd-no-normalise"


def command() -> list[str]:
    """Run008's argv minus exactly one flag, minus its output/name redirection."""
    base = run008.command()
    assert DROPPED_FLAG in base, "run008 no longer carries the flag this ablation removes"
    derived = [tok for tok in base if tok != DROPPED_FLAG]
    assert len(derived) == len(base) - 1, "expected to remove exactly one token"

    # redirect artifacts only; these are not scientific variables
    out_i = derived.index("--output")
    derived[out_i + 1] = str(OUTPUT)
    name_i = derived.index("--name")
    derived[name_i + 1] = EXPERIMENT_ID
    return derived


def diff_vs_run008() -> dict:
    """Mechanical proof that normalization is the only scientific difference."""
    a, b = run008.command(), command()

    def as_map(argv):
        m, i = {}, 0
        while i < len(argv):
            tok = argv[i]
            if tok.startswith("--"):
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    m[tok] = argv[i + 1]; i += 2
                else:
                    m[tok] = True; i += 1
            else:
                i += 1
        return m

    ma, mb = as_map(a), as_map(b)
    only_a = {k: ma[k] for k in ma if k not in mb}
    only_b = {k: mb[k] for k in mb if k not in ma}
    changed = {k: [ma[k], mb[k]] for k in ma if k in mb and ma[k] != mb[k]}
    artifact_only = {"--output", "--name"}
    scientific_changed = {k: v for k, v in changed.items() if k not in artifact_only}
    return {
        "flags_only_in_run008": only_a,
        "flags_only_in_run011": only_b,
        "values_changed": changed,
        "scientific_values_changed": scientific_changed,
        "artifact_only_changes": {k: changed[k] for k in changed if k in artifact_only},
        "is_clean_single_variable_ablation": (
            only_a == {DROPPED_FLAG: True} and only_b == {} and scientific_changed == {}
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show-diff", action="store_true", help="print the argv diff and exit")
    args = ap.parse_args()

    d = diff_vs_run008()
    if args.show_diff:
        print(json.dumps(d, indent=2))
        return 0 if d["is_clean_single_variable_ablation"] else 1
    if not d["is_clean_single_variable_ablation"]:
        raise SystemExit(f"REFUSING: argv diff vs run008 is not a clean single-variable "
                         f"ablation:\n{json.dumps(d, indent=2)}")

    if not args.dry_run and (OUTPUT / "summary.json").exists():
        raise SystemExit(f"Refusing to overwrite completed evidence: {OUTPUT/'summary.json'}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    cmd = command()
    (OUTPUT / "arm_manifest.json").write_text(json.dumps({
        "experiment": EXPERIMENT_ID,
        "purpose": "direct normalization ablation: does preserving update MAGNITUDE drive the raw-delta advantage?",
        "reference_run": "run008_raw_delta_seed1",
        "only_intended_scientific_change": "residual-delta normalization DISABLED -> ENABLED",
        "argv_construction": "derived programmatically from run008.command() by removing exactly '--layer-kd-no-normalise'",
        "argv_diff_vs_run008": d,
        "objective": "behavioral_kd",
        "behavioral_mode": "delta",
        "normalise": True,
        "loss_equation": ("mean_pairs[ MSE(nd_s, nd_t) + 1.0*(1 - cos(nd_s, nd_t)) ], "
                          "nd = d / (RMS(d)+eps), d_s = h_s[l+1]-h_s[l], d_t = h_t[b]-h_t[a]"),
        "expected_implementation_sanity_check": ("with normalization ON the logged layer_magnitude "
                                                 "should equal 2 x layer_direction identically; this is "
                                                 "an implementation check, NOT a scientific result"),
        "seed": 1,
        "teacher_revision": run008.REVISION,
        "locked_before_execution": True,
        "preregistered_interpretation": {
            "if_run011_much_worse_than_run008": (
                "MAGNITUDE HYPOTHESIS SUPPORTED: removing update-magnitude information destroys the "
                "advantage, so magnitude preservation is the operative factor."),
            "if_run011_approx_equal_to_run008": (
                "MAGNITUDE HYPOTHESIS FALSIFIED: normalization is irrelevant and the raw-delta "
                "advantage must come from something else (e.g. the delta/transition form itself, "
                "or loss scale/effective-LR effects)."),
            "if_run011_between_run008_and_run009": (
                "MAGNITUDE HYPOTHESIS PARTIALLY SUPPORTED / WEAKENED: magnitude contributes but does "
                "not account for the whole gap."),
            "if_run011_better_than_run008": (
                "HYPOTHESIS FALSIFIED IN REVERSE: report as-is, do not rescue."),
            "reference_points_seed1_same_environment": {
                "run009_pointwise_normalized": 8.73926,
                "run008_raw_delta": 7.41131,
                "run010_adjacent_raw_delta": 7.19814},
            "note": ("n is small and no variance estimate exists. A difference smaller than the "
                     "run010-vs-run008 spread (0.213) must NOT be treated as meaningful.")},
        "dry_run": args.dry_run,
        "command": cmd,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"Run011 manifest: {OUTPUT/'arm_manifest.json'}")
    print("ABLATION: residual-delta normalization ENABLED (Run008 argv minus --layer-kd-no-normalise)")
    print(f"clean single-variable ablation: {d['is_clean_single_variable_ablation']}")

    patched = patch_trainer_for_delta()
    try:
        return kd_main(cmd + (["--dry-run"] if args.dry_run else []))
    finally:
        restore_trainer(patched)


if __name__ == "__main__":
    raise SystemExit(main())
