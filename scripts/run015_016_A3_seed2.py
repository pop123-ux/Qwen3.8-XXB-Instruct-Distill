#!/usr/bin/env python3
"""Run015 / Run016: seed-2 replication of the seed-1 composite A3 raw-vs-normalised pair.

Seed 1 (run013) under the composite objective CE 0.1 + logit KD 1.0 + hidden_delta 0.5 +
router_balance 1.0:
    A3 raw         4.936469793319702
    A3 normalised  4.838203148408369     contrast (normalised - raw) = -0.0982666

Configs are built by the CANONICAL construction path that produced the seed-1 arms --
``make()`` in experiments/_session_2026-09-11/run_matrix_composite.py -- and then exactly
three fields are changed: ``training.seed`` 1 -> 2, ``name``, ``runtime.output_dir``.
Every other resolved field is checked against both the freshly built seed-1 config AND the
archived executed seed-1 config in ``summary.json``; any other difference refuses the run.

Order is preregistered: Run015 (raw) then Run016 (normalised). Both configs are built and
verified before the first arm trains, so the first result cannot influence the second. The
teacher is loaded once and shared, as in the seed-1 matrix.
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

SEED = 2
PERMITTED = {"training.seed", "name", "runtime.output_dir"}
MATRIX = REPO / "experiments" / "_session_2026-09-11" / "run_matrix_composite.py"

#: Preregistered execution order. (arm, normalisation tag, run id, seed-1 reference run id)
PLAN: tuple[tuple[str, str, str, str], ...] = (
    ("A3", "raw", "run015_A3_behavioural_delta_raw_seed2",
     "run013_A3_behavioural_delta_raw_seed1"),
    ("A3", "normalised", "run016_A3_behavioural_delta_normalised_seed2",
     "run013_A3_behavioural_delta_normalised_seed1"),
)


#: Technical history recorded into the arm manifest. Not a scientific change.
EXECUTION_NOTES: dict[str, str] = {
    "run016_A3_behavioural_delta_normalised_seed2": (
        "Attempt 1 ran second in the shared process and OOMed on its first backward pass (step 0, "
        "allocated 39.03 / reserved 42.90 GiB) because run015's CUDA cache was not released: its "
        "baseline reserved memory was 41.9 GiB versus 16.71 GiB in a fresh process. Zero optimizer "
        "steps were taken. Evidence preserved at /workspace/runs/"
        "run016_A3_behavioural_delta_normalised_seed2_attempt1_oom. This attempt runs alone in a fresh "
        "process with the identical frozen config, matching the seed-1 normalised reference, which "
        "also ran in its own process."),
}


def load_matrix():
    spec = importlib.util.spec_from_file_location("run_matrix_composite", MATRIX)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        obj = {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict) or hasattr(v, "__dataclass_fields__"):
                out.update(flatten(v, key + "."))
            else:
                out[key] = v
        return out
    return {prefix.rstrip("."): obj}


def _normal(v: Any) -> Any:
    """JSON round-trip, so a live config compares equal to its archived serialisation."""
    return json.loads(json.dumps(v, default=str))


def diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, list[Any]]:
    return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b))
            if _normal(a.get(k)) != _normal(b.get(k))}


def build(matrix, arm: str, tag: str, run_id: str, ref_id: str) -> dict[str, Any]:
    ref_cfg, arm_obj, _out, ref_eid, ref_proof, hidden = matrix.make(arm, tag)
    if ref_eid != ref_id:
        raise SystemExit(f"REFUSED: canonical make() produced {ref_eid}, expected {ref_id}")

    cfg = copy.deepcopy(ref_cfg)
    cfg.training.seed = SEED
    cfg.name = run_id
    cfg.runtime.output_dir = str(Path("/workspace/runs") / run_id)
    cfg.validate()

    vs_built = diff(flatten(ref_cfg), flatten(cfg))
    archived = json.loads((REPO / "experiments" / ref_id / "summary.json").read_text())["config"]
    vs_archived = diff(flatten(archived), flatten(cfg))
    return {
        "cfg": cfg, "arm": arm_obj, "hidden": hidden, "ref_proof": ref_proof,
        "diff_vs_built_seed1": vs_built,
        "diff_vs_archived_seed1": vs_archived,
        "clean": set(vs_built) == PERMITTED and set(vs_archived) == PERMITTED,
    }


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=[row[2] for row in PLAN],
                    help="train a single arm in a fresh process (both arms are still verified)")
    args = ap.parse_args(argv)
    selected = [row for row in PLAN if args.only in (None, row[2])]

    matrix = load_matrix()
    built = {run_id: build(matrix, *row[:2], run_id, row[3]) for row in PLAN for run_id in [row[2]]}

    # Both arms frozen and verified before either trains.
    for run_id, b in built.items():
        print(f"  {run_id}: permitted-only diff = {b['clean']}")
        if not b["clean"]:
            print(f"REFUSED {run_id}:\n  vs built  : {b['diff_vs_built_seed1']}\n"
                  f"  vs archive: {b['diff_vs_archived_seed1']}", file=sys.stderr)
            return 2
    pair = diff(flatten(built[PLAN[0][2]]["cfg"]), flatten(built[PLAN[1][2]]["cfg"]))
    if set(pair) != {"training.layer_kd_normalise", "name", "runtime.output_dir"}:
        print(f"REFUSED: seed-2 raw vs normalised differ beyond normalisation: {pair}",
              file=sys.stderr)
        return 2

    status = git("status", "--porcelain")
    if args.dry_run:
        print("  dry run: both arms verified; nothing trained.")
        return 0
    if status:
        print(f"REFUSED: worktree dirty:\n{status}", file=sys.stderr)
        return 2
    head = git("rev-parse", "HEAD")

    import torch

    import kd_run
    import run008_raw_delta_seed1 as run008
    from qwen_distill.distillation.backends import TransformersTeacher
    from qwen_distill.training.trainer import train

    for _arm, _tag, run_id, _ref in selected:
        if (Path("/workspace/runs") / run_id / "summary.json").exists():
            print(f"REFUSED: completed evidence exists for {run_id}", file=sys.stderr)
            return 2

    targs = kd_run.parse_args(run008.command())
    backend = TransformersTeacher(model=targs.teacher_model, revision=targs.revision,
                                  local_path=str(targs.teacher), quantization=targs.quantization,
                                  strict_architecture=True)
    backend.load()
    teacher = backend.signal_provider(top_k=targs.kd_top_k or None,
                                      temperature=targs.kd_temperature,
                                      capture_hidden_states=True)
    print(f"teacher ready: {teacher.describe()}", flush=True)

    results = {}
    for position, (arm, tag, run_id, ref_id) in enumerate(selected):
        if position:
            # Release the previous arm before building the next. Without this, attempt 1 of
            # run016 started with 41.9 GiB reserved (vs 16.71 GiB in a fresh process) because
            # run015's CUDA cache was never returned, and OOMed on its first backward pass.
            gc.collect()
            torch.cuda.empty_cache()
        b = built[run_id]
        cfg = b["cfg"]
        out = Path(cfg.runtime.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "arm_manifest.json").write_text(json.dumps({
            "experiment": run_id,
            "replication_of": ref_id,
            "replication_seed": SEED,
            "preregistered_arm": arm, "arm_name": b["arm"].name,
            "hidden_term": b["hidden"], "hidden_normalise": cfg.training.layer_kd_normalise,
            "resolved_coefficients": b["ref_proof"]["resolved_coefficients"],
            "router_aux_loss_coef": 0.001,
            "objective": "composite",
            "permitted_changes_vs_seed1": sorted(PERMITTED),
            "diff_vs_built_seed1": b["diff_vs_built_seed1"],
            "diff_vs_archived_seed1": b["diff_vs_archived_seed1"],
            "preregistered_order": [row[2] for row in PLAN],
            "executed_in_fresh_process": args.only is not None,
            "execution_note": EXECUTION_NOTES.get(run_id),
            "preparation_commit": head,
            "locked_before_execution": True,
        }, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\n{'=' * 70}\n=== {run_id}  [{tag}]  seed={cfg.training.seed}\n{'=' * 70}",
              flush=True)
        t0 = time.time()
        rc = train(cfg, None, teacher=teacher)
        results[run_id] = f"rc={rc} in {time.time() - t0:.1f}s"
        print(f"=== {run_id} done: {results[run_id]}", flush=True)
        if rc != 0:
            print(f"STOPPING: {run_id} returned {rc}", file=sys.stderr)
            return rc
    print("\n=== PAIR COMPLETE ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
