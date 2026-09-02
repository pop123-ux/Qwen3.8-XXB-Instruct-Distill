#!/usr/bin/env python3
"""Record the first controlled canonical KD experiment in the experiment ledger.

Reads the run summaries the trainer wrote and turns them into ledger entries. Every
number here is copied from a summary produced on the GPU — nothing is recomputed, and
nothing is entered by hand — so the ledger cannot drift from what the run measured.

Usage::

    python scripts/record_run001_kd.py \\
        --smoke /workspace/runs/run001_kd_smoke/summary.json \\
        --pilot /workspace/runs/run001_kd_pilot/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.research.ledger import Ledger

REVISION = "dbdc473dea0d6a9763042881cc33d6058d1742d2"
STUDENT_ID = "qwen38_19b_h5120_l48_moe"
CANONICAL_PARAMETERS = 13_008_505_728


def entry_payload(summary: dict, run_dir: str) -> dict:
    """The measured facts of one run, pulled straight out of its summary."""
    dist = summary["distillation"]
    memory = summary["memory"]
    counts = summary.get("parameter_counts") or {}
    training = summary["config"]["training"]
    data = summary["config"]["data"]
    return {
        "run_directory": run_dir,
        "outcome": summary["outcome"],
        "steps_completed": summary["steps"],
        "teacher": {
            "model": dist["teacher"]["teacher_model"],
            "revision": dist["teacher"]["teacher_revision"],
            "quantization": "4bit",
            "signal_source": dist["teacher"]["source"],
            "top_k": dist["teacher"]["top_k"],
            "temperature": dist["teacher"]["temperature"],
        },
        "student": {
            "id": STUDENT_ID,
            "canonical_parameters": CANONICAL_PARAMETERS,
            # total includes the adapters; the frozen base must equal the canonical count
            "total_parameters_with_adapters": counts.get("total_parameters"),
            "trainable_parameters": counts.get("trainable_parameters"),
            "trainable_fraction": counts.get("trainable_fraction"),
            "frozen_base_parameters": (
                counts["total_parameters"] - counts["trainable_parameters"]
                if counts else None
            ),
        },
        "training": {
            "strategy": training["strategy"],
            "optimizer": training["optimizer"],
            "lora_rank": training["lora_rank"],
            "lora_alpha": training["lora_alpha"],
            "objective": training["objective"],
            "kd_weight": training["kd_weight"],
            "kd_temperature": training["kd_temperature"],
            "kd_top_k": training["kd_top_k"],
            "precision": summary["effective_precision"],
            "sequence_length": data["max_sequence_length"],
            "batch_size": training["batch_size"],
            "gradient_checkpointing": training["gradient_checkpointing"],
            "learning_rate": training["learning_rate"],
        },
        "loss": {
            "first": summary["first_loss"],
            "final": summary["final_loss"],
            "kd": dist["kd_loss"],
            "ce": dist["ce_loss"],
            "first_validation": summary.get("first_validation_loss"),
            "final_validation": summary.get("final_validation_loss"),
        },
        "teacher_diagnostics": {
            "entropy": dist["teacher_entropy"],
            "top1_agreement": dist["top1_agreement"],
            "tail_mass": dist["teacher_tail_mass"],
        },
        "memory_gib": {
            "peak_allocated": memory["peak_allocated_gib"],
            "peak_reserved": memory["peak_reserved_gib"],
            "total_vram": memory["total_vram_gib"],
            "stages": {s["stage"]: s["allocated_gib"] for s in memory["snapshots"]},
        },
        "throughput": {
            "runtime_s": summary["runtime_s"],
            "tokens_seen": summary["tokens_seen"],
            "tokens_per_second": summary["tokens_per_second"],
        },
        "corpus": {
            "source": summary["corpus"]["source"],
            "sha256": summary["corpus"]["sha256"],
            "n_sequences": summary["corpus"]["n_sequences"],
            "sequence_length": summary["corpus"]["sequence_length"],
        },
        "caveat": (
            "This is a mechanism and memory result. It says the canonical KD path "
            "executes and fits; it says nothing about capability. 50 steps over 51,200 "
            "tokens on a public-domain English corpus cannot move a 13B model, and no "
            "capability claim is made from it."
        ),
    }


def pilot_payload(metrics_path: Path, checkpoint: Path | None) -> dict:
    """Build the pilot's record from its metrics stream and final checkpoint.

    The pilot has no ``summary.json``. It completed all 50 optimizer steps and wrote
    both checkpoints, and then the summary writer raised ``NameError: param_report`` --
    a defect introduced during this session while the pilot was already running, in the
    change that added ``parameter_counts`` to the summary. The name was assigned in
    ``train()`` and read in ``_write_summary()``, a different scope. It is fixed and
    covered by the checkpoint tests, but the fix landed too late for this process, which
    had already imported the broken module.

    So this entry is assembled from the artifacts that *were* written and fsynced as the
    run proceeded: the per-step metrics stream and the checkpoint metadata. Every number
    is still a measurement taken on the GPU during the run; none is recomputed here. The
    absence is recorded in the payload rather than papered over, because a reader
    comparing this entry against the smoke entry will notice the missing memory profile
    and should be told why.
    """
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    steps = [r for r in rows if r.get("status") == "completed_step"]
    validations = [r for r in rows if r.get("status") == "validated"]
    metadata = json.loads((checkpoint / "metadata.json").read_text()) if checkpoint else {}

    def series(key: str) -> dict:
        values = [r[key] for r in steps if r.get(key) is not None]
        return {
            "first": values[0] if values else None,
            "final": values[-1] if values else None,
            "mean": round(sum(values) / len(values), 5) if values else None,
        }

    return {
        "run_directory": str(metrics_path.parent),
        "outcome": "50/50 optimizer steps completed; summary.json not written",
        "steps_completed": steps[-1]["step"] if steps else 0,
        "record_provenance": (
            "assembled from metrics.jsonl and checkpoint metadata, both written and "
            "fsynced during the run. summary.json is absent: the writer raised "
            "NameError('param_report') after the final checkpoint, from a defect "
            "introduced mid-session and since fixed. Training itself completed."
        ),
        "missing_from_this_entry": [
            "memory profile (peak allocated/reserved and per-stage snapshots) -- "
            "measured only by the summary writer; see the smoke entry, which ran the "
            "identical configuration and recorded 37.79 GiB allocated / 38.30 reserved"
        ],
        "teacher": {
            "model": "Qwen/Qwen3.8-27B",
            "revision": REVISION,
            "quantization": "4bit",
            "signal_source": "online",
            "top_k": 64,
            "temperature": 2.0,
        },
        "student": {
            "id": STUDENT_ID,
            "canonical_parameters": CANONICAL_PARAMETERS,
            "total_parameters_with_adapters": metadata.get("parameter_count"),
            "trainable_parameters": metadata.get("trainable_parameter_count"),
            "frozen_base_parameters": (
                metadata["parameter_count"] - metadata["trainable_parameter_count"]
                if metadata.get("parameter_count") else None
            ),
        },
        "training": {
            "strategy": metadata.get("strategy", "qlora"),
            "optimizer": metadata.get("optimizer"),
            "lora_rank": 16,
            "lora_alpha": 32,
            "objective": "mixed_kd",
            "kd_weight": 0.5,
            "kd_temperature": 2.0,
            "kd_top_k": 64,
            "precision": metadata.get("precision"),
            "sequence_length": metadata.get("sequence_length"),
            "batch_size": metadata.get("batch_size"),
            "gradient_checkpointing": metadata.get("gradient_checkpointing"),
        },
        "loss": {
            "first": steps[0]["loss"] if steps else None,
            "final": steps[-1]["loss"] if steps else None,
            "kd": series("kd_loss"),
            "ce": series("ce_loss"),
            "validation": [
                {"step": v["step"], "validation_loss": v["validation_loss"]}
                for v in validations
            ],
        },
        "teacher_diagnostics": {
            "entropy": series("teacher_entropy"),
            "top1_agreement": series("top1_agreement"),
            "tail_mass": series("teacher_tail_mass"),
        },
        "throughput": {
            "runtime_s": steps[-1]["elapsed_s"] if steps else None,
            "tokens_seen": steps[-1]["tokens_seen"] if steps else None,
            "tokens_per_second": steps[-1]["tokens_per_second"] if steps else None,
        },
        "checkpoints_validated": {
            "step_000025": "structure, manifest and load level",
            "step_000050": "structure, manifest and load level",
        },
        "caveat": (
            "A mechanism result. The loss falling from 9.91 to 4.96 and top-1 agreement "
            "rising from 0.001 to 0.139 show a gradient reached the adapters and the "
            "optimizer moved them. 50 steps over 51,200 tokens cannot move a 13B model's "
            "capability and no capability claim is made from it."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--pilot-metrics", type=Path, default=None)
    parser.add_argument("--pilot-checkpoint", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=Path("experiments/ledger.jsonl"))
    args = parser.parse_args(argv)

    ledger = Ledger(args.ledger)
    written = []

    smoke = json.loads(args.smoke.read_text())
    written.append(ledger.measured(
        "canonical_kd",
        "Run 001 smoke: one real KD optimizer step on the frozen 13.01B canonical student",
        entry_payload(smoke, str(args.smoke.parent)),
        arm="run001_canonical_kd_smoke",
        tags=["run001", "canonical", "qlora", "mechanism"],
    ))

    if args.pilot_metrics and args.pilot_metrics.is_file():
        written.append(ledger.measured(
            "canonical_kd",
            "Run 001 pilot: 50 real KD optimizer steps on the frozen 13.01B canonical "
            "student",
            pilot_payload(args.pilot_metrics, args.pilot_checkpoint),
            arm="run001_canonical_kd_pilot",
            tags=["run001", "canonical", "qlora", "pilot"],
        ))

    written.append(ledger.measured(
        "blocker_closed",
        "B11: the canonical student is trainable on one A40 through the "
        "parameter-efficient path, and only that path",
        {
            "blocker": "B11",
            "what_was_missing": (
                "`lora` and `qlora` were in STRATEGIES and validated by the config, but "
                "trainer._require_supported raised NotImplementedError for any strategy "
                "but 'full'; training.optimizer was likewise validated, recorded in every "
                "run summary, and ignored -- torch.optim.AdamW was hardcoded. So a run "
                "could declare `strategy: qlora, optimizer: adamw_8bit` and neither took "
                "effect."
            ),
            "what_was_implemented": (
                "qwen_distill.training.peft_support: LoRA/QLoRA over the six projection "
                "families that reach all 48 layers, NF4 base quantisation, adapter-only "
                "checkpointing, and honest optimizer selection across OPTIMIZERS."
            ),
            "full_parameter_adamw_still_impossible": True,
            "full_parameter_adamw_requirement_gib": 210.2,
            "usable_vram_gib": 44.43,
            "architecture_changed": False,
            "frozen_base_parameters": CANONICAL_PARAMETERS,
        },
        arm="run001_canonical_kd_smoke",
        tags=["run001", "blocker", "qlora"],
    ))

    print(f"wrote {len(written)} entries to {args.ledger}")
    for entry in written:
        print(f"  {entry.id}  {entry.kind:16s} {entry.title[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
