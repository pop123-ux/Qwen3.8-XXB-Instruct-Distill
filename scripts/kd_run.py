#!/usr/bin/env python3
"""Drive a real teacher-in-the-loop KD run: teacher forward, student forward, KD loss,
backward, optimizer step.

This is the glue the roadmap was missing. Every piece it uses already existed and is
unchanged; nothing here reimplements a loader, a loss or a training loop:

* the teacher is loaded through :class:`TransformersTeacher`, whose missing-weight gate is
  fatal — never a bare ``from_pretrained``;
* the signal is an :class:`OnlineTeacher` (``objective.signal_source: online``), the only
  provider the trainer accepts for KD over a text corpus;
* the corpus is the tokenizer-backed path added in ``a0c2c39``, encoded with the teacher's
  own tokenizer at the student's vocabulary;
* the loop is :func:`qwen_distill.training.trainer.train`, unmodified.

Two student modes, deliberately separated so a mechanism check can never be mistaken for a
research result:

``--student canonical``
    The frozen target ``qwen38_19b_h5120_l48_moe`` (13,008,505,728 parameters), loaded from
    a checkpoint materialised by ``scripts/distill_pilot.py``. **There are no geometry
    flags**, for the same reason the pilot has none: a run that could quietly train a
    different architecture produces a number nobody can attribute.

``--student preflight``
    ``moe_student.tiny_fixture`` at the teacher's vocabulary — the project's own
    scaled-down member of the *same* architecture family (same block pattern, same GQA and
    DeltaNet head ratios, same top-k routing with a shared expert, same
    ``Qwen3_5MoeForCausalLM`` class). It exists to prove the chain runs end to end against
    the real teacher. **Its loss says nothing about capability** and it is never a
    substitute for the canonical run.

Exit codes: ``0`` the run completed, ``1`` it ran and failed (OOM included), ``2`` refused
before any weights were loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.architecture.moe_student import FROZEN_STUDENT
from qwen_distill.distillation.backends import TransformersTeacher
from qwen_distill.distillation.real_teacher import DEFAULT_TEACHER_MODEL
from qwen_distill.training.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
)
from qwen_distill.training.trainer import train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    teacher = parser.add_argument_group("teacher")
    teacher.add_argument("--teacher", type=Path, required=True,
                         help="local path to the downloaded checkpoint")
    teacher.add_argument("--revision", required=True,
                         help="exact commit SHA; recorded on every artifact")
    teacher.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL,
                         help=argparse.SUPPRESS)
    teacher.add_argument("--quantization", choices=("4bit", "8bit"), default=None,
                         help="quantise the resident teacher; it is frozen, so this "
                              "costs signal fidelity only, not trainability")

    student = parser.add_argument_group("student")
    student.add_argument("--student", choices=("canonical", "preflight"), required=True,
                         help="'canonical' is the frozen target and takes no geometry "
                              "flags; 'preflight' is the mechanism check")
    student.add_argument("--pretrained", type=Path, default=None,
                         help="materialised student checkpoint; required for 'canonical'")

    data = parser.add_argument_group("data")
    data.add_argument("--text-path", type=Path, required=True,
                      help="plain UTF-8 corpus")
    data.add_argument("--sequence-length", type=int, default=1024)
    data.add_argument("--max-documents", type=int, default=None)
    data.add_argument("--max-tokens", type=int, default=None)

    run = parser.add_argument_group("run")
    run.add_argument("--steps", type=int, default=100, help="optimizer steps")
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--gradient-accumulation-steps", type=int, default=1)
    run.add_argument("--learning-rate", type=float, default=2e-4)
    run.add_argument("--objective", choices=("logit_kd", "mixed_kd"), default="logit_kd",
                     help="'logit_kd' is pure KD and ignores --kd-weight; 'mixed_kd' is "
                          "--kd-weight of KD and the remainder cross-entropy")
    run.add_argument("--kd-weight", type=float, default=0.5,
                     help="KD share of the loss under --objective mixed_kd; the rest is CE")
    run.add_argument("--kd-temperature", type=float, default=2.0)
    run.add_argument("--kd-top-k", type=int, default=64,
                     help="teacher shortlist; the tail is kept exact via logsumexp")
    run.add_argument("--strategy", choices=("full", "lora", "qlora"), default="qlora",
                     help="'qlora' (default) quantises the frozen base to NF4 and trains "
                          "LoRA adapters; 'lora' keeps the base in --precision; 'full' "
                          "trains every parameter and does not fit the canonical student "
                          "on one 48 GB card")
    run.add_argument("--optimizer", choices=("adamw", "adamw_8bit", "adafactor", "sgd"),
                     default="adamw")
    run.add_argument("--lora-rank", type=int, default=16)
    run.add_argument("--lora-alpha", type=int, default=32)
    run.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output", type=Path, default=Path("runs/kd_run"))
    run.add_argument("--name", default=None, help="experiment name for the record")
    run.add_argument("--dry-run", action="store_true",
                     help="build the config and print it; load nothing, train nothing")
    return parser.parse_args(argv)


def build_preflight_student(destination: Path, *, sequence_length: int) -> Path:
    """Materialise the tiny same-family student so it loads through ``pretrained``.

    Saving it first, rather than handing the trainer a live module, is deliberate: it makes
    the preflight traverse exactly the code path the canonical run traverses
    (``ModelConfig(pretrained=...)`` -> ``AutoModelForCausalLM.from_pretrained``), so a
    defect in that path cannot hide behind a shortcut only the preflight takes.
    """
    from qwen_distill.architecture.moe_student import build_model, tiny_fixture

    spec = tiny_fixture(
        vocab_size=FROZEN_STUDENT.vocab_size,
        max_position_embeddings=max(sequence_length, 512),
    )
    model = build_model(spec, meta=False)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  preflight student : {type(model).__name__}, {n_params:,} parameters")
    print(f"                      vocab {spec.vocab_size:,}, hidden {spec.hidden_size}, "
          f"{spec.num_hidden_layers} layers, {spec.num_experts} experts top-"
          f"{spec.num_experts_per_tok}")
    print(f"                      saved to {destination}")
    del model
    return destination


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()

    if args.student == "canonical" and args.pretrained is None:
        print("  REFUSED: --student canonical needs --pretrained, the checkpoint written "
              "by scripts/distill_pilot.py. The frozen student is never built from "
              "scratch here: an untransferred 13B student would train, fall, and mean "
              "nothing.", file=sys.stderr)
        return 2
    if not args.text_path.is_file():
        print(f"  REFUSED: no corpus at {args.text_path}", file=sys.stderr)
        return 2
    if not (args.teacher / "config.json").is_file():
        print(f"  REFUSED: no checkpoint at {args.teacher}", file=sys.stderr)
        return 2

    name = args.name or f"kd_{args.student}"
    args.output.mkdir(parents=True, exist_ok=True)

    # -- the student -----------------------------------------------------------------
    if args.student == "canonical":
        pretrained = args.pretrained
        print(f"  canonical student : {FROZEN_STUDENT.name}")
        print(f"                      loaded from {pretrained}")
    else:
        pretrained = build_preflight_student(
            args.output / "preflight_student", sequence_length=args.sequence_length
        )

    # -- the configuration -----------------------------------------------------------
    config = ExperimentConfig(
        name=name,
        description=(
            "real teacher-in-the-loop logit KD against "
            + ("the frozen canonical student" if args.student == "canonical"
               else "a same-family mechanism-check student")
        ),
        objective={"signal_source": "online"},
        teacher={"model": args.teacher_model, "revision": args.revision},
        model=ModelConfig(pretrained=str(pretrained)),
        data=DataConfig(
            tokenized_text=True,
            text_path=str(args.text_path),
            tokenizer_path=str(args.teacher),
            expected_vocab_size=FROZEN_STUDENT.vocab_size,
            document_separator="blank_line",
            max_sequence_length=args.sequence_length,
            max_documents=args.max_documents,
            max_tokens=args.max_tokens,
        ),
        training=TrainingConfig(
            strategy=args.strategy,
            optimizer=args.optimizer,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            objective=args.objective,
            precision=args.precision,
            learning_rate=args.learning_rate,
            max_steps=args.steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=True,
            kd_weight=args.kd_weight,
            kd_temperature=args.kd_temperature,
            kd_top_k=args.kd_top_k,
            seed=args.seed,
            eval_every=max(1, args.steps // 4),
            save_every=max(1, args.steps // 2),
            log_every=max(1, args.steps // 20),
        ),
        runtime=RuntimeConfig(output_dir=str(args.output)),
    )
    config.validate()

    if args.dry_run:
        print(json.dumps(
            {"student": args.student, "pretrained": str(pretrained),
             "teacher": str(args.teacher), "revision": args.revision,
             "steps": args.steps, "sequence_length": args.sequence_length,
             "kd_top_k": args.kd_top_k, "kd_temperature": args.kd_temperature,
             "objective": args.objective, "kd_weight": args.kd_weight,
             "strategy": args.strategy, "optimizer": args.optimizer,
             "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
             "expected_vocab_size": FROZEN_STUDENT.vocab_size},
            indent=2))
        print("\n  dry run: nothing was loaded and nothing was trained.")
        return 0

    # -- the teacher -----------------------------------------------------------------
    print(f"\n  loading teacher {args.teacher_model} @ {args.revision}")
    print(f"    quantization: {args.quantization or 'none (native dtype)'}")
    backend = TransformersTeacher(
        model=args.teacher_model,
        revision=args.revision,
        local_path=str(args.teacher),
        quantization=args.quantization,
        strict_architecture=True,
    )
    backend.load()   # missing weights are fatal here, never a warning
    # The capture temperature must equal training.kd_temperature: the stored logsumexp is
    # taken at the capture temperature, and the trainer refuses a mismatch rather than
    # letting the tail term be computed against a distribution it does not describe.
    teacher = backend.signal_provider(
        top_k=args.kd_top_k or None, temperature=args.kd_temperature
    )
    print(f"    teacher ready: {teacher.describe()}")

    print(f"\n  DISTILLING {args.steps} steps "
          f"(batch {args.batch_size} x {args.gradient_accumulation_steps} accum "
          f"@ {args.sequence_length} tokens) ...")
    exit_code = train(config, None, teacher=teacher)

    elapsed = time.time() - started
    print(f"\n  exit code {exit_code} in {elapsed / 60:.1f} min")
    print(f"  record: {args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
