#!/usr/bin/env python3
"""Record a completed Run 004 behavioural/delta-KD result.

The recorder is deliberately separate from training. It accepts only the trainer summary
plus the Run 004 manifest, verifies that the summary actually reports ``mode=delta`` and
that the frozen protocol fields are present, then writes one measured ledger entry.
It never invents missing metrics and never upgrades a calibration into a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

Ledger = import_module("qwen_distill.research.ledger").Ledger

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


#: The per-metric series the trainer actually writes under ``summary["distillation"]``.
#: Each is a ``{"first": ..., "final": ..., "mean": ...}`` dict. There is no
#: ``initial``/``final``/``mean``/``trajectory`` grouping — reading those keys yields
#: nothing, which is the bug this list fixes.
_DISTILLATION_SERIES = (
    "kd_loss", "ce_loss", "top1_agreement", "teacher_entropy", "teacher_tail_mass",
    "layer_kd_loss", "layer_magnitude", "layer_direction", "layer_norm_ratio",
)


def payload(summary: dict, run_dir: Path, manifest: dict) -> dict:
    distillation = summary["distillation"]
    definition = distillation["layer_kd_definition"]
    memory = summary.get("memory") or {}
    config = summary.get("config") or {}
    data_cfg = config.get("data") or {}
    training_cfg = config.get("training") or {}
    return {
        "arm": "behavioral_kd",
        "objective": EXPECTED_OBJECTIVE,
        "mode": EXPECTED_MODE,
        "run_dir": str(run_dir),
        "summary": str(run_dir / "summary.json"),
        # -- provenance so the run is reproducible from the ledger entry alone --------
        "git_commit": summary.get("git_commit"),
        "teacher": manifest["teacher"],
        "teacher_revision": manifest["teacher_revision"],
        "student": manifest["student"],
        "student_parameter_counts": summary.get("parameter_counts") or {},
        "sequence_length": data_cfg.get("max_sequence_length"),
        "max_tokens": manifest.get("max_tokens", data_cfg.get("max_tokens")),
        "seed": training_cfg.get("seed"),
        "kd_temperature": distillation.get("kd_temperature"),
        "kd_top_k": distillation.get("kd_top_k"),
        "corpus": summary.get("corpus") or {},
        "definition": definition,
        # -- measured metrics, read from the structure the trainer actually writes ----
        "metrics": {
            "first_loss": summary.get("first_loss"),
            "final_loss": summary.get("final_loss"),
            "first_validation_loss": summary.get("first_validation_loss"),
            "final_validation_loss": summary.get("final_validation_loss"),
            "best_validation_loss": summary.get("best_validation_loss"),
            "series": {
                name: distillation[name]
                for name in _DISTILLATION_SERIES
                if name in distillation
            },
        },
        "throughput": {
            "tokens_per_second": summary.get("tokens_per_second"),
            "runtime_s": summary.get("runtime_s"),
            "tokens_seen": summary.get("tokens_seen"),
            "steps": summary.get("steps"),
        },
        "vram": {
            "peak_allocated_gib": memory.get("peak_allocated_gib"),
            "peak_reserved_gib": memory.get("peak_reserved_gib"),
        },
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
    ledger.measured(
        kind="training_run",
        title="Run 004 behavioural/delta KD",
        payload=payload(summary, args.run_dir, manifest),
        arm="behavioral_kd",
        tags=["run004", "behavioral_kd", "delta", "rq1"],
    )
    print(f"Recorded Run 004 in {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
