#!/usr/bin/env python3
"""CPU-only static gate for the research laboratory.

Run this before renting a GPU. It deliberately imports no torch/transformers and touches no
model files. Its job is to catch repository drift, incomplete objective registration,
mislabelled FDD, hidden protocol changes and missing reproducibility infrastructure while
those checks are still free.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "research/plans/RQ1_OBJECTIVE_LAB_V1.json"
PROTOCOL = ROOT / "research/protocols/RQ1_OBJECTIVES_V2.json"

REQUIRED = (
    ".claude/skills/qwen38-distillation-research/SKILL.md",
    "docs/REPRODUCIBILITY.md",
    "environment/Dockerfile.research",
    "environment/research-baseline.json",
    "experiments/research_campaign.json",
    "requirements/research-rq1-direct.txt",
    "scripts/capture_research_environment.py",
    "scripts/research_guard.py",
    "tests/test_research_reproducibility_controls.py",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def inspect() -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            errors.append(f"missing required lab file: {rel}")
    for path in (PLAN, PROTOCOL):
        if not path.exists():
            errors.append(f"missing canonical research control file: {path.relative_to(ROOT)}")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}

    plan, protocol = _load(PLAN), _load(PROTOCOL)

    if protocol.get("plan") != str(PLAN.relative_to(ROOT)).replace("\\", "/"):
        errors.append("protocol does not point to the canonical RQ1 lab plan")
    if protocol.get("independent_variable") != "arm":
        errors.append("RQ1_OBJECTIVES_V2 must vary only the registered arm")
    if plan.get("canonical_teacher") != {
        "model": protocol["teacher"]["model"], "revision": protocol["teacher"]["revision"]
    }:
        errors.append("teacher identity differs between plan and protocol")
    if plan["canonical_student"]["id"] != protocol["student"]["id"]:
        errors.append("student identity differs between plan and protocol")
    if plan["matched_recipe"] != protocol["training"]:
        errors.append("matched scientific recipe differs between plan and protocol")

    plan_arms = {a["arm"]: a for a in plan["arms"]}
    protocol_arms = protocol["arm_registry"]
    if set(plan_arms) != set(protocol["allowed_arms"]) or set(plan_arms) != set(protocol_arms):
        errors.append("A-F arm membership differs across plan/protocol registry")

    # FDD is prediction-space trajectory + derivative + output KD. A raw residual-delta
    # experiment is useful as an ablation, but calling it FDD would invalidate the prior-art
    # comparator without producing any obvious runtime failure.
    fdd = plan_arms.get("B", {})
    if fdd.get("space") != "lm_head_prediction":
        errors.append("arm B cannot be called FDD unless it operates in LM-head prediction space")
    fdd_components = set(fdd.get("components", []))
    if not {"output_kd", "trajectory_kl", "derivative_cosine"} <= fdd_components:
        errors.append("arm B is missing a required FDD component")
    adjacent = plan_arms.get("D", {})
    if "fdd" in (adjacent.get("id") or "").lower():
        errors.append("arm D is an adjacent residual ablation and must not be labelled FDD")

    # An implementation status is an execution gate, not prose. Pending arms must remain
    # blocked until CPU tests are committed.
    ready = {"existing_cpu_tested", "cpu_tested", "ready"}
    for arm, spec in plan_arms.items():
        status = spec.get("implementation_status")
        gpu_status = spec.get("gpu_status")
        if status in ready and gpu_status == "blocked":
            errors.append(f"arm {arm} is CPU-ready but still marked GPU-blocked")
        if status not in ready and gpu_status != "blocked":
            errors.append(f"arm {arm} is not CPU-ready but is GPU-eligible")

    f = protocol_arms.get("F", {})
    if f.get("composite_weights") is None and not str(f.get("status", "")).startswith("blocked"):
        errors.append("arm F has no preregistered composite weights but is not blocked")

    env = protocol["runtime_environment"]
    if env.get("allocator_env_canonical") != "PYTORCH_ALLOC_CONF":
        errors.append("new protocols must use PYTORCH_ALLOC_CONF as the canonical allocator variable")
    docker = (ROOT / "environment/Dockerfile.research").read_text(encoding="utf-8")
    if "PYTORCH_ALLOC_CONF=expandable_segments:True" not in docker:
        errors.append("research image does not set the registered canonical allocator")
    if "requirements/research-rq1-direct.txt" not in docker:
        errors.append("research image is not built from the pinned direct RQ1 requirements")
    if "pip install -e . --no-deps" not in docker:
        errors.append("research image does not install this repository as its code payload")

    if (ROOT / "--bg").exists() or (ROOT / "Record").exists():
        warnings.append("zero-byte shell-artifact files --bg/Record are still present at repo root")

    dirty = _git("status", "--porcelain")
    state = {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty": bool(dirty) if dirty is not None else None,
    }
    if state["dirty"]:
        warnings.append("working tree is dirty; controlled GPU launch must record a clean commit")

    return {
        "ok": not errors,
        "plan_id": plan.get("plan_id"),
        "protocol_id": protocol.get("protocol_id"),
        "git": state,
        "ready_arms": [a for a in protocol["allowed_arms"] if protocol_arms[a]["status"] in ready],
        "blocked_arms": [a for a in protocol["allowed_arms"] if protocol_arms[a]["status"] not in ready],
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict-warnings", action="store_true")
    args = p.parse_args()
    report = inspect()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"LAB PREFLIGHT: {'PASS' if report['ok'] else 'FAIL'}")
        print(f"  plan: {report.get('plan_id')}  protocol: {report.get('protocol_id')}")
        print(f"  ready arms: {', '.join(report.get('ready_arms', [])) or 'none'}")
        print(f"  blocked arms: {', '.join(report.get('blocked_arms', [])) or 'none'}")
        for item in report.get("errors", []):
            print(f"  ERROR: {item}")
        for item in report.get("warnings", []):
            print(f"  WARNING: {item}")
    if not report["ok"]:
        return 2
    if args.strict_warnings and report.get("warnings"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
