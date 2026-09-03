#!/usr/bin/env python3
"""Record a completed Run 004 behavioural/delta-KD result.

The recorder is deliberately separate from training. It accepts only the trainer summary
plus the Run 004 manifest, verifies that the summary actually reports ``mode=delta`` and
that the frozen protocol fields are present, then writes one measured ledger entry. It
never invents missing metrics and never upgrades a calibration into a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_distill.research.ledger import Ledger

EXPECTED_STUDENT = "qwen38_19b_h5120_l48_moe"
EXPECTED_TEACHER = "Qwen/Qwen3.8-27B"
EXPECTED_REVISION = "dbdc473dea0d6a9763042881cc33d6058d1742d2"
EXPECTED_OBJECTIVE = "behavioral_kd"
EXPECTED_MODE = "delta"


def validate(summary: dict, manifest: dict) -> list[str]:
    problems: list[str] = []
    if manifest.get("objective") != EXPECTED_OBJECTIVE:
        problems.append(f"manifest objective is {manifest.get('objective')!r}, expected {EXPECTED_OBJECTIVE!r}")
    if manifest.get("behavioral_mode") != EXPECTED_MODE:
        problems.append(f"manifest mode is {manifest.get('behavioral_mode')!r}, expected {EXPECTED_MODE!r}")
    if manifest.get("student") != EXPECTED_STUDENT:
        problems.append("manifest student does not identify the frozen canonical student")
    if manifest.get("teacher") != EXPECTED_TEACHER:
        problems.append("manifest teacher does not identify Qwen3.8-27B")
    if manifest.get("teacher_revision") != EXPECTED_REVISION:
        problems.append("manifest teacher revision is not the canonical pinned revision")

    distillation = summary.get("distillation") or {}
    definition = distillation.get("layer_kd_definition") or {}
    if summary.get("objective") != "layer_kd":
        problems.append("trainer summary did not execute through the validated layer-KD trainer path")
    if definition.get("mode") != EXPECTED_MODE:
        problems.append(f"trainer definition mode is {definition.get('mode')!r}, expected {EXPECTED_MODE!r}")
    if definition.get("objective") != EXPECTED_OBJECTIVE:
        problems.append(f"trainer definition objective is {definition.get('objective')!r}, expected {EXPECTED_OBJECTIVE!r}")
    if definition.get("n_supervised_pairs") != 48:
        problems.append("Run 004 must supervise all 48 canonical student layers")
    if definition.get("topology_mismatch") is None:
        problems.append("trainer summary is missing topology-mismatch provenance")
    return problems


def payload(summary: dict, run_dir: Path, manifest: dict) -> dict:
    distillation = summary["distillation"]
    definition = distillation["layer_kd_definition"]
    return {
        "arm": "behavioral_kd",
        "objective": EXPECTED_OBJECTIVE,
        "mode": EXPECTED_MODE,
        "run_dir": str(run_dir),
        "teacher": manifest["teacher"],
        "teacher_revision": manifest["teacher_revision"],
        "student": manifest["student"],
        "summary": str(run_dir / "summary.json"),
        "definition": definition,
        "metrics": summary.get("metrics", {}),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing Run 004 summary: {summary_path}")
    if not args.manifest.exists():
        raise FileNotFoundError(f"missing Run 004 manifest: {args.manifest}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    problems = validate(summary, manifest)
    if problems:
        raise ValueError("Run 004 validation failed:\n- " + "\n- ".join(problems))

    ledger = Ledger(args.ledger)
    ledger.append(payload(summary, args.run_dir, manifest))
    print(f"Recorded Run 004 in {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
