#!/usr/bin/env python3
"""The first real teacher -> student distillation, end to end and in one command.

This script exists to answer one question before any money is spent on a rented GPU:
**does the chain work at all?** Plan a transfer, materialise the weights, measure the
student cold, distil against the teacher's distribution, measure it again, checkpoint.
Each of those pieces is tested on its own; nothing until now has run them in sequence.

Two stages, and they prove different things.

``--stand-in`` (**Stage 0**)
    A small randomly-initialised teacher, on a T4 or a laptop, in minutes. It proves the
    *mechanism*: the plan applies, the shapes line up, the KD gradient flows, the
    checkpoint is resumable. It proves **nothing whatsoever about capability** — the
    teacher knows nothing, so a student that matches it has learned nothing. Any claim
    about model quality drawn from a Stage-0 run is a claim about noise.

``--teacher DIR`` (**Stage 1**)
    The real Qwen3.8-27B. The same code path, on hardware that can hold it.

The cheapest informative measurement is taken first and for free: the transferred student
is evaluated **before any training**. If a 6x parameter reduction leaves it at chance, the
transfer strategy is the problem and no amount of distillation budget will find that out
faster.

Exit codes: ``0`` the chain ran, ``1`` it did not, ``2`` the request could not be set up.

Examples::

    # Stage 0: prove the chain, free hardware, minutes
    python scripts/distill_pilot.py --stand-in --output runs/stage0

    # what would be transferred, without loading or training anything
    python scripts/distill_pilot.py --teacher ./qwen3.8-27b --hidden 3072 --layers 28 \\
        --kv-heads 2 --dn-key-heads 8 --dry-run

    # Stage 1
    python scripts/distill_pilot.py --teacher ./qwen3.8-27b --hidden 3072 --layers 28 \\
        --kv-heads 2 --dn-key-heads 8 --steps 2000 --output runs/pilot1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.architecture.materialize import (
    SafetensorsSource,
    StateDictSource,
    UnsupportedReduction,
    initialise_student,
)
from qwen_distill.architecture.params import count_parameters, format_params
from qwen_distill.architecture.spec import HybridArchSpec
from qwen_distill.architecture.transfer import build_transfer_plan, student_from_teacher

#: The stand-in teacher. Deliberately tiny and deliberately *structured like the real
#: one*: a period-4 hybrid, the teacher's GQA ratio (6 query heads per KV head) and its
#: DeltaNet ratio (3 value heads per key head). Every constraint the transfer enforces is
#: therefore exercised; only the scale is fake.
STAND_IN = dict(
    hidden_size=256, num_hidden_layers=16, intermediate_size=768, vocab_size=256,
    num_attention_heads=12, num_key_value_heads=2, head_dim=32,
    linear_num_key_heads=4, linear_num_value_heads=12,
    linear_key_head_dim=32, linear_value_head_dim=32,
    full_attention_interval=4, tie_word_embeddings=True, max_position_embeddings=1024,
)


def build_model(spec: HybridArchSpec, *, seed: int | None = None):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if seed is not None:
        torch.manual_seed(seed)
    fields = {k: v for k, v in spec.to_hf_text_config().items() if k != "model_type"}
    return AutoModelForCausalLM.from_config(AutoConfig.for_model("qwen3_5_text", **fields))


def load_teacher_spec(directory: Path) -> HybridArchSpec:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    return HybridArchSpec.from_hf_config(config, name=directory.name)


def evaluate(model, sequences, *, batch_size: int, device: str, limit: int = 16) -> float:
    """Mean cross-entropy over a held-out slice, in nats per token."""
    import torch

    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for start in range(0, min(len(sequences), limit * batch_size) - batch_size + 1, batch_size):
            batch = torch.tensor(
                sequences[start : start + batch_size], dtype=torch.long, device=device
            )
            total += float(model(input_ids=batch, labels=batch).loss)
            batches += 1
    model.train()
    return total / max(batches, 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--teacher", type=Path, help="directory holding the teacher's config.json and weights")
    source.add_argument("--stand-in", action="store_true", help="Stage 0: a small random teacher, proving the chain only")

    student = parser.add_argument_group("student geometry (defaults inherit the teacher)")
    student.add_argument("--hidden", type=int)
    student.add_argument("--layers", type=int, help="must be a whole number of hybrid groups")
    student.add_argument("--ffn", type=int, dest="intermediate")
    student.add_argument("--kv-heads", type=int, help="query heads follow at the teacher's GQA ratio")
    student.add_argument("--dn-key-heads", type=int, help="DeltaNet value heads follow at the teacher's ratio")
    student.add_argument("--untie-embeddings", action="store_true")

    transfer = parser.add_argument_group("transfer")
    transfer.add_argument("--layer-selection", default="group",
                          choices=("group", "first", "last", "uniform", "interleave"))
    transfer.add_argument("--width-reduction", default="slice",
                          choices=("slice", "mean_pool", "importance"))

    distil = parser.add_argument_group("distillation")
    distil.add_argument("--objective", default="logit_kd", choices=("logit_kd", "mixed_kd", "sft"))
    distil.add_argument("--top-k", type=int, default=64, help="0 keeps the full distribution")
    distil.add_argument("--temperature", type=float, default=1.0)
    distil.add_argument("--kd-weight", type=float, default=0.5, help="mixed_kd only")
    distil.add_argument("--kd-tail", default="bucket", choices=("bucket", "renormalize"))

    run = parser.add_argument_group("run")
    run.add_argument("--corpus", type=Path, help="a UTF-8 text file; procedural text if omitted")
    run.add_argument("--corpus-bytes", type=int, default=0,
                     help="procedural corpus size; 0 sizes it to the run (default)")
    run.add_argument("--steps", type=int, default=200)
    run.add_argument("--batch-size", type=int, default=2)
    run.add_argument("--seq-len", type=int, default=256)
    run.add_argument("--learning-rate", type=float, default=3e-4)
    run.add_argument("--device", default="auto")
    run.add_argument("--precision", default="fp32", choices=("fp32", "fp16", "bf16"))
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output", type=Path, default=Path("runs/distill_pilot"))
    run.add_argument("--dry-run", action="store_true", help="plan and report, load and train nothing")
    run.add_argument("--transfer-only", action="store_true", help="transfer and checkpoint, do not train")
    run.add_argument("--json", type=Path, help="write the record here as well as to the output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record: dict[str, object] = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # --- the two architectures ------------------------------------------
    if args.stand_in:
        teacher_spec = HybridArchSpec(name="stand-in", **STAND_IN)
    else:
        if not (args.teacher / "config.json").exists():
            print(f"  no config.json in {args.teacher}", file=sys.stderr)
            return 2
        teacher_spec = load_teacher_spec(args.teacher)

    try:
        student_spec = student_from_teacher(
            teacher_spec,
            name=f"{teacher_spec.name}-student",
            hidden_size=args.hidden,
            num_hidden_layers=args.layers,
            intermediate_size=args.intermediate,
            num_key_value_heads=args.kv_heads,
            linear_num_key_heads=args.dn_key_heads,
            tie_word_embeddings=False if args.untie_embeddings else None,
            max_position_embeddings=min(teacher_spec.max_position_embeddings, 8192),
        )
    except ValueError as exc:
        print(f"  cannot derive a transferable student: {exc}", file=sys.stderr)
        return 2

    teacher_params = count_parameters(teacher_spec)
    student_params = count_parameters(student_spec)
    print("\n  ARCHITECTURES")
    print(f"    teacher : {teacher_spec.name}  {format_params(teacher_params.total)}  "
          f"h{teacher_spec.hidden_size} L{teacher_spec.num_hidden_layers} "
          f"ff{teacher_spec.intermediate_size} V{teacher_spec.vocab_size}")
    print(f"    student : {student_spec.name}  {format_params(student_params.total)}  "
          f"h{student_spec.hidden_size} L{student_spec.num_hidden_layers} "
          f"ff{student_spec.intermediate_size} V{student_spec.vocab_size}")
    print(f"    reduction: {teacher_params.total / student_params.total:.2f}x  "
          f"(embedding is {student_params.embedding / student_params.total:.1%} of the student)")
    record["teacher"] = {**teacher_spec.to_dict(), "parameters": teacher_params.total}
    record["student"] = {**student_spec.to_dict(), "parameters": student_params.total}

    # --- the plan --------------------------------------------------------
    plan = build_transfer_plan(
        teacher_spec, student_spec,
        layer_selection=args.layer_selection, width_reduction=args.width_reduction,
    )
    print("\n  TRANSFER PLAN")
    print(f"    strategy   : {plan.strategy}")
    print(f"    layer map  : {plan.layer_map[0]} .. {plan.layer_map[max(plan.layer_map)]} "
          f"({len(plan.layer_map)} student layers over {teacher_spec.num_hidden_layers} teacher layers)")
    print(f"    coverage   : {plan.coverage:.1%} of student tensors")
    for warning in plan.warnings:
        print(f"    ! {warning}")
    record["plan"] = plan.to_dict()

    if args.dry_run:
        _write_record(args, record)
        print("\n  dry run: nothing was loaded, transferred or trained.")
        return 0

    # --- the corpus ------------------------------------------------------
    from qwen_distill.training.text_data import BYTE_VOCAB_SIZE, prepare_corpus

    if student_spec.vocab_size != BYTE_VOCAB_SIZE:
        # The one place a real teacher and this script part company today. The corpus
        # pipeline is byte-level; a Qwen-vocabulary student needs the teacher's tokenizer,
        # and vendor/qwen38-metadata carries the config and chat template but no
        # tokenizer.json. Saying so beats training against mis-encoded text.
        print(
            f"\n  the student's vocabulary is {student_spec.vocab_size}, but the corpus "
            f"pipeline emits byte-level tokens (vocab {BYTE_VOCAB_SIZE}).\n"
            "  Tokenise the corpus with the teacher's tokenizer first — it is not in "
            "vendor/qwen38-metadata, so it has to come from the teacher checkout.",
            file=sys.stderr,
        )
        return 2

    # Sized to the run rather than fixed: a 2-step smoke test does not need 400 KB of
    # procedural text, and generating it twice (here and again inside the trainer) is most
    # of the wall clock on a short pilot.
    needed = args.steps * args.batch_size * args.seq_len
    corpus_bytes = args.corpus_bytes or max(20_000, min(400_000, needed * 8))
    train_sequences, validation_sequences, corpus_stats = prepare_corpus(
        text_path=str(args.corpus) if args.corpus else None,
        sequence_length=args.seq_len,
        procedural_bytes=corpus_bytes,
        validation_fraction=0.1,
        seed=args.seed,
    )
    print(f"\n  CORPUS  {corpus_stats.source}: {corpus_stats.n_bytes:,} bytes, "
          f"{corpus_stats.n_train} train / {corpus_stats.n_validation} validation sequences")
    epochs = needed / max(corpus_stats.n_bytes, 1)
    if not args.corpus and epochs > 2:
        # Procedural text is a smoke-test corpus, not a training corpus. Reading it many
        # times over would still produce a falling loss, and that fall would be
        # memorisation of a generated pattern rather than anything about the teacher.
        print(f"    ! this run reads the procedural corpus {epochs:.1f} times over. "
              "A falling loss\n      would then be memorisation of generated text, not "
              "distillation. Pass --corpus\n      (or raise --corpus-bytes) for any run "
              "whose result is meant to mean something.", file=sys.stderr)
    record["corpus"] = {"source": corpus_stats.source, "bytes": corpus_stats.n_bytes,
                        "sha256": corpus_stats.sha256}

    # --- materialise -----------------------------------------------------
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher_model = None
    if args.stand_in:
        teacher_model = build_model(teacher_spec, seed=args.seed + 1).to(device)
        source = StateDictSource(teacher_model.state_dict())
    else:
        source = SafetensorsSource(args.teacher)

    student_model = build_model(student_spec, seed=args.seed)
    try:
        report = initialise_student(
            student_model, plan, teacher_spec, student_spec, source,
            width_reduction=args.width_reduction,
        )
    except UnsupportedReduction as exc:
        print(f"\n  transfer refused: {exc}", file=sys.stderr)
        return 2
    print()
    print(report.render())
    record["transfer"] = report.to_dict()
    if hasattr(source, "close"):
        source.close()
    student_model = student_model.to(device)

    # --- the free measurement -------------------------------------------
    baseline = build_model(student_spec, seed=args.seed + 99).to(device)
    cold = evaluate(student_model, validation_sequences, batch_size=args.batch_size, device=device)
    untrained = evaluate(baseline, validation_sequences, batch_size=args.batch_size, device=device)
    del baseline
    print("\n  BEFORE ANY TRAINING")
    print(f"    transferred student : {cold:.4f} nats/token")
    print(f"    random init         : {untrained:.4f} nats/token")
    print(f"    the transfer is worth {untrained - cold:+.4f} nats before a single step")
    if args.stand_in:
        print("    (the teacher is random, so ~0 is the correct answer here. A gain would "
              "mean\n     the measurement is wrong, not that the transfer is good.)")
    record["cold_evaluation"] = {
        "transferred_nats": cold, "random_init_nats": untrained, "delta": untrained - cold,
    }

    # Written out unconditionally, and *not* only for --transfer-only. The trainer builds
    # its student from the config; handing it a spec would rebuild a random model and
    # silently discard everything above, so the transferred weights have to reach it as a
    # checkpoint. Keeping the artifact is worth it on its own at Stage 1 — a 54 GB
    # transfer is not something to redo per run.
    args.output.mkdir(parents=True, exist_ok=True)
    transferred = args.output / "transferred"
    student_model.save_pretrained(transferred)
    print(f"\n  transferred student written to {transferred}")
    record["transferred_path"] = str(transferred)

    if args.transfer_only:
        _write_record(args, record)
        print("  transfer only: nothing was trained.")
        return 0

    # --- distil ----------------------------------------------------------
    if teacher_model is None:
        from transformers import AutoModelForCausalLM

        print("\n  loading the teacher for online distillation ...")
        teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher, dtype="auto").to(device)

    from qwen_distill.distillation.teacher_signal import OnlineTeacher
    from qwen_distill.training.config import ExperimentConfig, ModelConfig
    from qwen_distill.training.trainer import train

    config = ExperimentConfig(name="distill_pilot")
    # `pretrained`, not `architecture`: this is what carries the transfer into training.
    config.model = ModelConfig(pretrained=str(transferred))
    config.data.text_corpus = True
    config.data.text_path = str(args.corpus) if args.corpus else None
    config.data.procedural_bytes = corpus_bytes
    config.data.max_sequence_length = args.seq_len
    config.data.validation_fraction = 0.1
    config.data.shuffle_seed = args.seed
    config.training.objective = args.objective
    config.training.strategy = "full"
    config.training.precision = args.precision
    config.training.max_steps = args.steps
    config.training.batch_size = args.batch_size
    config.training.gradient_accumulation_steps = 1
    config.training.learning_rate = args.learning_rate
    config.training.kd_temperature = args.temperature
    config.training.kd_weight = args.kd_weight
    config.training.kd_tail = args.kd_tail
    config.training.kd_top_k = args.top_k or None
    config.training.seed = args.seed
    config.training.log_every = max(1, args.steps // 20)
    # Once, at the end. The trainer validates over the *whole* held-out split, which on a
    # short pilot costs far more wall clock than the training it is measuring — and the
    # before/after numbers this script prints are the measurement that matters here.
    config.training.eval_every = args.steps
    config.training.save_every = args.steps
    config.training.gradient_checkpointing = False
    config.objective = {"signal_source": "online"}
    config.runtime.output_dir = str(args.output)
    config.runtime.device = device
    config.validate()

    teacher = (
        None if args.objective == "sft"
        else OnlineTeacher(
            model=teacher_model, top_k=args.top_k or None, temperature=args.temperature,
            teacher_model=teacher_spec.name,
        )
    )
    print(f"\n  DISTILLING {args.steps} steps ...")
    exit_code = train(config, student_spec, teacher=teacher)
    record["training_exit_code"] = exit_code

    from qwen_distill.training.checkpoints import load_checkpoint, resolve_checkpoint

    del student_model   # the trainer holds its own copy, loaded from the checkpoint above
    trained_path = resolve_checkpoint(args.output / "checkpoints", "latest")
    if trained_path is None:
        print("\n  no checkpoint to re-evaluate; skipping the after measurement",
              file=sys.stderr)
        _write_record(args, record)
        return exit_code
    # NOT `from_pretrained`: a training checkpoint's config.json is the *experiment*
    # config, not a model config, so the auto-loader cannot read it. The checkpoint module
    # restores weights into a model built from the spec, and validates the directory on the
    # way in rather than loading a partial one.
    trained = build_model(student_spec).to(device)
    load_checkpoint(trained_path, model=trained, map_location=device)
    warm = evaluate(trained, validation_sequences, batch_size=args.batch_size, device=device)
    print("\n  AFTER DISTILLATION")
    print(f"    student             : {warm:.4f} nats/token  ({warm - cold:+.4f} vs cold)")
    record["warm_evaluation"] = {"nats": warm, "delta_vs_cold": warm - cold}

    summary_path = args.output / "summary.json"
    if summary_path.exists():
        record["run_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    _write_record(args, record)

    print("\n  WHAT THIS RUN ESTABLISHED")
    print("    the chain ran: plan -> materialise -> KD -> checkpoint" if exit_code == 0
          else "    the chain did NOT complete; see the error above")
    if args.stand_in:
        print("    the teacher was random, so nothing here says anything about capability.")
        print("    expect validation loss to RISE: the student is being pulled toward a "
              "random\n    distribution and away from the corpus. That rise is the "
              "clearest evidence the\n    KD term is real rather than falling through to "
              "cross-entropy.")
    return exit_code


def _write_record(args: argparse.Namespace, record: dict[str, object]) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "pilot_record.json"
    payload = json.dumps(record, indent=2, default=str)
    destination.write_text(payload, encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload, encoding="utf-8")
    print(f"\n  record: {destination}")


if __name__ == "__main__":
    raise SystemExit(main())
