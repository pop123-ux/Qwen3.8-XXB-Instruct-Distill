#!/usr/bin/env python3
"""Launch Run 004: residual-contribution (delta) behavioural KD.

Run 004 is the first direct test of the project's "beyond layer matching" hypothesis.
The repository already contains the validated training/checkpoint machinery and the
behavioural loss implementation, including the mathematically equivalent chunked path.
This launcher reuses that machinery while forcing the behavioural loss into ``delta``
mode:

    student contribution     h_s[l+1] - h_s[l]
    teacher target contribution h_t[b]   - h_t[a]

where ``(a, b)`` is the complete teacher span assigned to student layer ``l``. The
teacher spans tile the full teacher depth, so removed teacher layers are charged to a
student computation instead of being dropped as they are in conventional pointwise layer
matching.

The underlying validated trainer currently names this execution path ``layer_kd``. This
launcher deliberately does not fork or duplicate that large training loop. Instead it
patches only the two imported behavioural-loss functions and the definition writer in the
live trainer process, then writes a sidecar manifest declaring the actual Run 004 mode.
The trainer's internal config remains ``layer_kd`` for compatibility; the sidecar is the
canonical Run 004 protocol record and the recorder validates it before writing the ledger.

This is preparation infrastructure only. It never changes the frozen student architecture
and it does not execute training unless the command is invoked without ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

trainer_module = import_module("qwen_distill.training.trainer")

TEACHER_REVISION = "dbdc473dea0d6a9763042881cc33d6058d1742d2"
STUDENT_ID = "qwen38_19b_h5120_l48_moe"
UNDERLYING_OBJECTIVE = "layer_kd"
BEHAVIORAL_OBJECTIVE = "behavioral_kd"
BEHAVIORAL_MODE = "delta"
DEFAULT_CHUNK_PAIRS = 4


def patch_trainer_for_delta() -> tuple[Any, Any, Any]:
    """Force the existing trainer's layer-KD branch to use the delta objective.

    Only function arguments are changed. The validated loss implementation, chunking
    semantics, optimizer, checkpointing and validation remain those already exercised by
    Run 003.
    """

    original_loss = trainer_module.behavioral_loss
    original_chunked = trainer_module.behavioral_loss_chunked
    original_definition = trainer_module._layer_kd_definition

    def delta_loss(*args, **kwargs):
        kwargs["mode"] = BEHAVIORAL_MODE
        return original_loss(*args, **kwargs)

    def delta_chunked(*args, **kwargs):
        kwargs["mode"] = BEHAVIORAL_MODE
        return original_chunked(*args, **kwargs)

    def delta_definition(config, mapping):
        definition = original_definition(config, mapping)
        definition.update(
            {
                "objective": BEHAVIORAL_OBJECTIVE,
                "mode": BEHAVIORAL_MODE,
                "implementation": "qwen_distill.distillation.behavioral.behavioral_loss",
                "teacher_representation": (
                    "hidden_states[b] - hidden_states[a], the telescoped contribution of "
                    "the complete teacher span assigned to the student layer"
                ),
                "student_representation": (
                    "hidden_states[l + 1] - hidden_states[l], the residual contribution "
                    "of the student layer"
                ),
                "mapping_strategy": mapping.strategy,
                "n_supervised_pairs": len(mapping.mapping),
                "span_semantics": (
                    "Every teacher layer belongs to exactly one half-open span [a,b); the "
                    "teacher target is h_t[b]-h_t[a], so removed teacher layers are "
                    "explicitly charged to a student computation."
                ),
                "topology_mismatch": (
                    f"{len(mapping.removed_teacher_layers)} teacher layers are absorbed "
                    "into neighbouring student spans rather than left unsupervised."
                ),
            }
        )
        evaluation = definition.get("evaluation")
        if evaluation is not None:
            evaluation["mode"] = BEHAVIORAL_MODE
        return definition

    trainer_module.behavioral_loss = delta_loss
    trainer_module.behavioral_loss_chunked = delta_chunked
    trainer_module._layer_kd_definition = delta_definition
    return original_loss, original_chunked, original_definition


def restore_trainer(originals: tuple[Any, Any, Any]) -> None:
    """Restore trainer globals after the delegated run."""
    original_loss, original_chunked, original_definition = originals
    trainer_module.behavioral_loss = original_loss
    trainer_module.behavioral_loss_chunked = original_chunked
    trainer_module._layer_kd_definition = original_definition


def write_manifest(output: Path, *, command: list[str], dry_run: bool) -> Path:
    """Write the authoritative Run 004 execution manifest."""
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "run004_behavioral_kd",
        "objective": BEHAVIORAL_OBJECTIVE,
        "behavioral_mode": BEHAVIORAL_MODE,
        "student": STUDENT_ID,
        "teacher": "Qwen/Qwen3.8-27B",
        "teacher_revision": TEACHER_REVISION,
        "underlying_trainer_objective": UNDERLYING_OBJECTIVE,
        "implementation": "qwen_distill.distillation.behavioral.behavioral_loss",
        "student_term": "h_s[l+1] - h_s[l]",
        "teacher_term": "h_t[b] - h_t[a] where [a,b) is the assigned teacher span",
        "span_property": "teacher layers are tiled exactly once across student spans",
        "projection": "none; student/teacher hidden size is 5120",
        "normalisation": "per-token RMS scaling, inherited from the validated behavioral loss",
        "direction_weight": 1.0,
        "chunk_pairs": DEFAULT_CHUNK_PAIRS,
        "deltanet_state_matching": False,
        "deltanet_state_note": (
            "No recurrent-state projection is performed: teacher/student recurrent-state "
            "shapes differ, so the run uses residual-interface computational behavior only."
        ),
        "dry_run": dry_run,
        "command": command,
    }
    path = output / "run004_behavioral_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--revision", default=TEACHER_REVISION)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--text-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/run004_behavioral_kd"))
    parser.add_argument("--sequence-length", type=int, default=1536)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--kd-top-k", type=int, default=64)
    parser.add_argument("--chunk-pairs", type=int, default=DEFAULT_CHUNK_PAIRS)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--quantization", choices=("4bit", "8bit"), default="4bit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_kd_run_args(args: argparse.Namespace) -> list[str]:
    """Translate Run 004's public protocol into the existing kd_run CLI."""
    return [
        "--teacher", str(args.teacher),
        "--revision", args.revision,
        "--quantization", args.quantization,
        "--student", "canonical",
        "--pretrained", str(args.pretrained),
        "--text-path", str(args.text_path),
        "--sequence-length", str(args.sequence_length),
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--learning-rate", str(args.learning_rate),
        "--objective", UNDERLYING_OBJECTIVE,
        "--layer-kd-direction-weight", "1.0",
        "--layer-kd-chunk-pairs", str(args.chunk_pairs),
        "--kd-temperature", str(args.kd_temperature),
        "--kd-top-k", str(args.kd_top_k),
        "--strategy", "qlora",
        "--optimizer", args.optimizer,
        "--lora-rank", str(args.lora_rank),
        "--lora-alpha", str(args.lora_alpha),
        "--precision", args.precision,
        "--seed", str(args.seed),
        "--log-every", str(args.log_every),
        "--eval-every", str(args.eval_every),
        "--save-every", str(args.save_every),
        "--output", str(args.output),
        "--name", "run004_behavioral_kd",
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_kd_run_args(args)
    manifest = write_manifest(args.output, command=command, dry_run=args.dry_run)
    print(f"Run 004 manifest: {manifest}")
    print("Objective: behavioral_kd / delta")
    print("Teacher contribution: h_t[b] - h_t[a]")
    print("Student contribution: h_s[l + 1] - h_s[l]")
    print(f"Chunk pairs: {args.chunk_pairs}")

    from kd_run import main as kd_main

    patched = patch_trainer_for_delta()
    try:
        if args.dry_run:
            return kd_main(command + ["--dry-run"])
        return kd_main(command)
    finally:
        restore_trainer(patched)


if __name__ == "__main__":
    raise SystemExit(main())
