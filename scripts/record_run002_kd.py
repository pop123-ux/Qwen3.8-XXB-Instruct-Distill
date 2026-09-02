#!/usr/bin/env python3
"""Record Run 002 -- the pure logit-KD control -- in the experiment ledger.

Run 002 is a *control arm*, not a follow-on improvement to Run 001. Run 001 established
that the canonical KD mechanism executes at all (mixed_kd, alpha 0.5, sequence 1024);
Run 002 establishes a conventional pure-logit-KD reference point (alpha 1.0, at the
sequence length its own calibration cleared the 42 GiB gate at) that later layer-KD and
computational-behaviour/state-KD arms can be matched against. The two are not a before/after pair and this recorder does not present them as
one: no delta between the runs is computed here, because they differ in objective,
sequence length and step count simultaneously, and a difference across three axes at once
attributes to none of them.

Every number is copied from the summary the trainer wrote on the GPU. Nothing is
recomputed and nothing is entered by hand, so the ledger cannot drift from the run.

Usage::

    python scripts/record_run002_kd.py \\
        --calibration /workspace/runs/run002_calibration/summary.json \\
        --control /workspace/runs/run002_logit_kd/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from record_run001_kd import entry_payload

from qwen_distill.research.ledger import Ledger

#: The gate Run 002 was launched under. Recorded with the result so a reader can see the
#: threshold the measurement was judged against, not just the measurement.
PEAK_ALLOCATED_GATE_GIB = 42.0

CONTROL_CAVEAT = (
    "A control, not a capability result. Pure logit KD (alpha=1.0, CE reported but "
    "unweighted) on the frozen canonical student, recorded so that later layer-KD and "
    "computational-behaviour/state-KD arms have a conventional KD reference to be "
    "matched against. 128 steps cannot move a 13B model and no capability claim is made. "
    "It is also not a measured improvement over Run 001: the two runs differ in "
    "objective, sequence length and step count at once, so no margin between them "
    "attributes to any single cause."
)


def calibration_payload(summary: dict, run_dir: str, *, gate_gib: float) -> dict:
    """The memory calibration and the gate decision it produced."""
    memory = summary["memory"]
    peak_allocated = memory["peak_allocated_gib"]
    sequence_length = summary["config"]["data"]["max_sequence_length"]
    return {
        "run_directory": run_dir,
        "purpose": (
            f"one {sequence_length}-token optimizer step, run to measure peak VRAM before "
            "committing to 128 steps at a sequence length never previously exercised"
        ),
        "outcome": summary["outcome"],
        "sequence_length": sequence_length,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": memory["peak_reserved_gib"],
        "total_vram_gib": memory["total_vram_gib"],
        "stages_allocated_gib": {s["stage"]: s["allocated_gib"] for s in memory["snapshots"]},
        # Teacher is resident before the student is built, so the baseline snapshot is
        # the teacher alone and the model-creation delta is the quantised student.
        "teacher_resident_gib": next(
            (s["allocated_gib"] for s in memory["snapshots"] if s["stage"] == "baseline"), None
        ),
        "gate_gib": gate_gib,
        "gate_passed": peak_allocated <= gate_gib,
        "gate_decision": (
            f"peak allocated {peak_allocated:.2f} GiB "
            f"{'<=' if peak_allocated <= gate_gib else '>'} {gate_gib} GiB gate -> "
            f"{'proceeded to the 128-step control' if peak_allocated <= gate_gib else 'STOPPED'}"
        ),
    }


def control_payload(summary: dict, run_dir: str, checkpoints: list[str]) -> dict:
    """Run 002 proper, built from the trainer's own summary."""
    payload = entry_payload(summary, run_dir)
    payload["caveat"] = CONTROL_CAVEAT
    payload["arm"] = "logit_kd_control"
    payload["control_for"] = [
        "layer_kd (not yet run)",
        "computational_behaviour_state_kd (not yet run)",
    ]
    payload["relationship_to_run_001"] = (
        "Run 001 is the mechanism-validation pilot (mixed_kd alpha 0.5, sequence 1024, "
        "50 steps) and remains documented as such. Run 002 is a separate control arm. "
        "They are not a baseline/treatment pair."
    )
    payload["checkpoints_validated"] = checkpoints
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--control", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[],
                        help="repeatable; a checkpoint directory validated for this run")
    parser.add_argument("--ledger", type=Path, default=Path("experiments/ledger.jsonl"))
    args = parser.parse_args(argv)

    ledger = Ledger(args.ledger)
    written = []

    if args.calibration and args.calibration.is_file():
        summary = json.loads(args.calibration.read_text(encoding="utf-8"))
        calibration_sequence = summary["config"]["data"]["max_sequence_length"]
        written.append(ledger.measured(
            "memory_measurement",
            f"Run 002 calibration: peak VRAM for one {calibration_sequence}-token KD step "
            "on the canonical student",
            calibration_payload(summary, str(args.calibration.parent),
                                gate_gib=PEAK_ALLOCATED_GATE_GIB),
            arm="run002_memory_calibration",
            tags=["run002", "canonical", "qlora", "memory", "calibration"],
        ))

    if args.control and args.control.is_file():
        summary = json.loads(args.control.read_text(encoding="utf-8"))
        written.append(ledger.measured(
            "canonical_kd",
            f"Run 002: {summary['steps']} pure logit-KD steps on the frozen 13.01B "
            f"canonical student at sequence "
            f"{summary['config']['data']['max_sequence_length']}",
            control_payload(summary, str(args.control.parent),
                            [str(c) for c in args.checkpoint]),
            arm="run002_logit_kd_control",
            tags=["run002", "canonical", "qlora", "control", "logit_kd"],
        ))

    for entry in written:
        print(f"  recorded {entry.id}  {entry.kind}  {entry.title}")
    if not written:
        print("  nothing recorded: no summary files were found at the given paths")
        return 1
    print(f"\n  ledger: {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
