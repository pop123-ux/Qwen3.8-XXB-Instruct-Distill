#!/usr/bin/env python3
"""Record Run 003 -- the layer/intermediate-KD control -- in the experiment ledger.

Run 003 is the matched partner of Run 002, not a follow-on improvement. Everything is held
equal -- teacher, revision, quantisation, student checkpoint, corpus, tokenizer, sequence
length, batch, accumulation, steps, token budget, seed, optimizer, precision, gradient
checkpointing, LoRA geometry, learning rate, weight decay, warmup and schedule -- and the
single difference is the distillation objective:

    Run 002   pure logit KD          KL(teacher || student) over the output distribution
    Run 003   pure layer/intermediate KD   pointwise hidden-state matching at mapped layers

Because exactly one thing differs, a difference between the two arms attributes to the
objective. That is the whole reason the protocol is copied rather than re-chosen, and this
recorder checks it: :func:`protocol_diff` compares the two summaries field by field and the
entry carries the result, so a reader can see the comparison was controlled rather than
being asked to trust it.

Every number is copied from the summary the trainer wrote on the GPU. Nothing is recomputed
and nothing is entered by hand, so the ledger cannot drift from the run.

Usage::

    python scripts/record_run003_kd.py \\
        --calibration /workspace/runs/run003_calibration/summary.json \\
        --control /workspace/runs/run003_layer_kd/summary.json \\
        --reference experiments/run002_logit_kd/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from record_run001_kd import entry_payload
from record_run002_kd import PEAK_ALLOCATED_GATE_GIB, calibration_payload

from qwen_distill.research.ledger import Ledger

#: The fields that must be equal for Run 002 and Run 003 to be a controlled pair. The
#: objective is deliberately absent: it is the variable under test.
PROTOCOL_FIELDS = (
    ("data", "max_sequence_length"),
    ("training", "max_steps"),
    ("training", "batch_size"),
    ("training", "gradient_accumulation_steps"),
    ("training", "seed"),
    ("training", "optimizer"),
    ("training", "precision"),
    ("training", "strategy"),
    ("training", "lora_rank"),
    ("training", "lora_alpha"),
    ("training", "lora_dropout"),
    ("training", "learning_rate"),
    ("training", "weight_decay"),
    ("training", "warmup_steps"),
    ("training", "scheduler"),
    ("training", "gradient_checkpointing"),
)

CONTROL_CAVEAT = (
    "A control, not a capability result. Pure layer/intermediate KD -- each student "
    "layer's output hidden state matched to the output of its mapped teacher layer, with "
    "the logit KD divergence and CE reported but unweighted -- on the frozen canonical "
    "student, run at Run 002's exact protocol so the two differ only in objective. 128 "
    "steps cannot move a 13B model and no capability claim is made. The comparison this "
    "enables is between two distillation objectives at a matched token budget, on "
    "held-out loss and teacher agreement; it is not evidence about downstream capability."
)


def protocol_diff(control: dict, reference: dict) -> dict:
    """Which protocol fields differ between this run and its reference arm.

    An empty ``differences`` is the claim that the comparison is controlled. A non-empty
    one is recorded rather than suppressed: a reader must be able to see that the arms
    drifted, because a margin across two axes at once attributes to neither.
    """
    differences = {}
    for section, field in PROTOCOL_FIELDS:
        mine = (control["config"].get(section) or {}).get(field)
        theirs = (reference["config"].get(section) or {}).get(field)
        if mine != theirs:
            differences[f"{section}.{field}"] = {"run003": mine, "reference": theirs}
    for name, path in (("corpus.sha256", ("corpus", "sha256")),
                       ("teacher.revision", ("config", "teacher", "revision"))):
        mine, theirs = control, reference
        for key in path:
            mine = (mine or {}).get(key) if isinstance(mine, dict) else None
            theirs = (theirs or {}).get(key) if isinstance(theirs, dict) else None
        if mine != theirs:
            differences[name] = {"run003": mine, "reference": theirs}
    return {
        "reference_experiment": reference.get("experiment"),
        "reference_objective": reference.get("objective"),
        "fields_compared": [f"{section}.{field}" for section, field in PROTOCOL_FIELDS]
                           + ["corpus.sha256", "teacher.revision"],
        "differences": differences,
        "controlled": not differences,
        "verdict": (
            "matched: the arms differ only in the distillation objective"
            if not differences else
            "NOT matched: " + ", ".join(sorted(differences))
            + " differ as well as the objective, so a margin between the arms does not "
              "attribute to the objective alone"
        ),
    }


def control_payload(summary: dict, run_dir: str, checkpoints: list[str],
                    reference: dict | None) -> dict:
    """Run 003 proper, built from the trainer's own summary."""
    payload = entry_payload(summary, run_dir)
    distillation = summary["distillation"]
    payload["caveat"] = CONTROL_CAVEAT
    payload["arm"] = "layer_kd_control"
    payload["layer_kd"] = {
        "definition": distillation.get("layer_kd_definition"),
        "layer_kd_loss": distillation.get("layer_kd_loss"),
        "layer_magnitude": distillation.get("layer_magnitude"),
        "layer_direction": distillation.get("layer_direction"),
        "layer_norm_ratio": distillation.get("layer_norm_ratio"),
    }
    payload["loss"]["layer_kd"] = distillation.get("layer_kd_loss")
    payload["matched_against_run_002"] = (
        protocol_diff(summary, reference) if reference else
        {"controlled": None, "verdict": "no reference summary was supplied to compare "
                                        "against; the match is unverified"}
    )
    payload["relationship_to_run_002"] = (
        "Run 002 (pure logit KD) and Run 003 (pure layer/intermediate KD) are matched "
        "arms of one controlled comparison. Neither is a baseline for the other in a "
        "before/after sense: they were run at the same protocol and are read side by "
        "side."
    )
    payload["checkpoints_validated"] = checkpoints
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--control", type=Path, default=None)
    parser.add_argument("--reference", type=Path,
                        default=Path("experiments/run002_logit_kd/summary.json"),
                        help="the arm this run is matched against")
    parser.add_argument("--checkpoint", type=Path, action="append", default=[],
                        help="repeatable; a checkpoint directory validated for this run")
    parser.add_argument("--ledger", type=Path, default=Path("experiments/ledger.jsonl"))
    args = parser.parse_args(argv)

    ledger = Ledger(args.ledger)
    reference = (json.loads(args.reference.read_text(encoding="utf-8"))
                 if args.reference and args.reference.is_file() else None)
    written = []

    if args.calibration and args.calibration.is_file():
        summary = json.loads(args.calibration.read_text(encoding="utf-8"))
        sequence_length = summary["config"]["data"]["max_sequence_length"]
        written.append(ledger.measured(
            "memory_measurement",
            f"Run 003 calibration: peak VRAM for one {sequence_length}-token layer-KD "
            "step on the canonical student",
            calibration_payload(summary, str(args.calibration.parent),
                                gate_gib=PEAK_ALLOCATED_GATE_GIB)
            | {"objective": summary.get("objective"),
               "why_this_is_not_run_002s_calibration": (
                   "layer_kd makes the teacher return its hidden states and the student "
                   "retain its own, which Run 002's logit-KD calibration never measured. "
                   "The gate has to be cleared again for this objective."
               )},
            arm="run003_memory_calibration",
            tags=["run003", "canonical", "qlora", "memory", "calibration", "layer_kd"],
        ))

    if args.control and args.control.is_file():
        summary = json.loads(args.control.read_text(encoding="utf-8"))
        written.append(ledger.measured(
            "canonical_kd",
            f"Run 003: {summary['steps']} pure layer/intermediate-KD steps on the frozen "
            f"13.01B canonical student at sequence "
            f"{summary['config']['data']['max_sequence_length']}",
            control_payload(summary, str(args.control.parent),
                            [str(c) for c in args.checkpoint], reference),
            arm="run003_layer_kd_control",
            tags=["run003", "canonical", "qlora", "control", "layer_kd"],
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
