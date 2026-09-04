#!/usr/bin/env python3
"""Canonical launcher for the post-Run004-M RQ1 objective campaign.

This launcher continues the existing research lineage. It never rewrites or relabels
Runs 001-004-M. Historical Run003 (Arm A) and Run004-M (Arm C) remain immutable anchors;
new A/C executions are permitted only when explicitly marked as bridge replications.

The launcher is deliberately orchestration-only and imports no torch. Environment capture
and the research guard run as short-lived child processes so no preflight CUDA context is
kept alive while the training child executes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "research/protocols/RQ1_OBJECTIVES_V2.json"
PLAN_PATH = ROOT / "research/plans/RQ1_OBJECTIVE_LAB_V1.json"
READY = {"existing_cpu_tested", "cpu_tested", "ready"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def _assert_fresh_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"output already exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise RuntimeError(
                f"refusing to overwrite a non-empty run directory: {path}. "
                "Historical and completed run artifacts are immutable."
            )
    else:
        path.mkdir(parents=True, exist_ok=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=list("ABCDEF"), required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--text-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bridge-replication",
        action="store_true",
        help="required for a new A/C run; historical A/C anchors are never replaced",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = _load(PROTOCOL_PATH)
    plan = _load(PLAN_PATH)
    spec = protocol["arm_registry"][args.arm]

    if spec.get("status") not in READY:
        print(
            f"REFUSED: arm {args.arm} is not CPU-ready: {spec.get('status')}",
            file=sys.stderr,
        )
        return 2
    if args.arm == "F":
        print("REFUSED: Arm F remains preregistration-blocked", file=sys.stderr)
        return 2
    if args.arm in {"A", "C"} and not args.bridge_replication:
        print(
            "REFUSED: A and C already exist as historical anchors. A new A/C execution "
            "must pass --bridge-replication and receive a new output/run identity.",
            file=sys.stderr,
        )
        return 2
    if args.bridge_replication and args.arm not in {"A", "C"}:
        print("REFUSED: --bridge-replication is only valid for A or C", file=sys.stderr)
        return 2

    image_digest = os.environ.get("RESEARCH_CONTAINER_DIGEST")
    if not image_digest:
        print(
            "REFUSED: set RESEARCH_CONTAINER_DIGEST to the immutable image digest used "
            "to start this container",
            file=sys.stderr,
        )
        return 2

    for label, path in {
        "teacher": args.teacher,
        "pretrained student": args.pretrained,
    }.items():
        if not path.is_dir():
            print(f"REFUSED: missing {label} directory: {path}", file=sys.stderr)
            return 2
    if not args.text_path.is_file():
        print(f"REFUSED: missing corpus: {args.text_path}", file=sys.stderr)
        return 2

    student_weights = args.pretrained / "model.safetensors"
    if not student_weights.is_file():
        print(
            f"REFUSED: canonical student materialization is missing: {student_weights}",
            file=sys.stderr,
        )
        return 2

    try:
        _assert_fresh_output(args.output)
        _run([sys.executable, "scripts/lab_preflight.py", "--json"])

        # Record the exact materialized student used by the continuation campaign. The
        # historical anchors predate this hash gate, so this is not retroactively claimed
        # as a historical digest; it is the identity lock for all new V2 executions.
        student_identity = {
            "path": str(student_weights.resolve()),
            "sha256": _sha256(student_weights),
            "size_bytes": student_weights.stat().st_size,
            "protocol_student": protocol["student"],
        }
        (args.output / "student_identity.json").write_text(
            json.dumps(student_identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        resolved = {"arm": args.arm, "training": protocol["training"]}
        resolved_path = args.output / "rq1_guard_resolved_config.json"
        resolved_path.write_text(
            json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        anchors = plan["historical_anchors"]
        launch = {
            "schema": 1,
            "campaign": "RQ1 continuation after Run004-M",
            "protocol_id": protocol["protocol_id"],
            "arm": args.arm,
            "objective_id": spec["id"],
            "continuation_of": [anchor["id"] for anchor in anchors],
            "historical_anchors": anchors,
            "historical_artifacts_immutable": True,
            "bridge_replication": bool(args.bridge_replication),
            "new_run_not_reset": True,
            "container_digest": image_digest,
            "image_git_sha": os.environ.get("RESEARCH_GIT_SHA"),
            "student_identity_sha256": student_identity["sha256"],
            "teacher": protocol["teacher"],
            "training": protocol["training"],
            "output": str(args.output.resolve()),
        }
        (args.output / "continuation_launch_manifest.json").write_text(
            json.dumps(launch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # This subprocess may initialize CUDA, but exits before the long-lived guard starts.
        _run([
            sys.executable,
            "scripts/capture_research_environment.py",
            "--output",
            str(args.output / "research_environment.json"),
        ])

        child = [
            sys.executable,
            "scripts/rq1_run.py",
            "--arm",
            args.arm,
            "--teacher",
            str(args.teacher),
            "--pretrained",
            str(args.pretrained),
            "--text-path",
            str(args.text_path),
            "--output",
            str(args.output),
        ]
        if args.dry_run:
            child.append("--dry-run")

        guard = [
            sys.executable,
            "scripts/research_guard.py",
            "--protocol",
            str(PROTOCOL_PATH),
            "--resolved-config",
            str(resolved_path),
            "--arm",
            args.arm,
            "--manifest",
            str(args.output / "research_guard_manifest.json"),
            "--",
            *child,
        ]
        completed = subprocess.run(guard, cwd=ROOT, check=False)
        return completed.returncode
    except RuntimeError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
