#!/usr/bin/env python3
"""Train a student model, from toy prototypes to the final run, from one config.

**Run `--dry-run` first, always.** It loads the config, resolves the architecture,
estimates parameters and peak VRAM against the detected hardware, and tells you whether
the run is plausible — in seconds, before any weights load. That is the difference
between discovering an OOM now and discovering it forty minutes into rented GPU time.

One script spans every scale deliberately: a recipe validated on a T4 is the *same*
recipe scaled up, not a different code path. See docs/TRAINING_ON_LIMITED_HARDWARE.md
for the development ladder these configs sit on.

Examples::

    python scripts/train_student.py --config configs/experiments/t4_prototype.yaml --dry-run
    python scripts/train_student.py --config configs/experiments/t4_prototype.yaml
    python scripts/train_student.py --config ... --dry-run --simulate-vram 16
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.architecture.params import count_parameters, format_params
from qwen_distill.training.config import ExperimentConfig
from qwen_distill.training.feasibility import check_feasibility

RULE = "=" * 60


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="resolve everything and report feasibility, but do not train",
    )
    parser.add_argument(
        "--simulate-vram", type=float, metavar="GIB",
        help="check feasibility against a hypothetical GPU of this size",
    )
    parser.add_argument("--output-dir", type=Path, help="override runtime.output_dir")
    parser.add_argument("--max-steps", type=int, help="override training.max_steps")
    parser.add_argument("--seed", type=int, help="override training.seed")
    parser.add_argument(
        "--resume", metavar="REF",
        help='resume training: "latest" for the newest verified checkpoint, a step '
             'number, or a checkpoint directory. Refuses to start if nothing valid '
             'matches, rather than silently restarting from step 0.',
    )
    parser.add_argument(
        "--resume-from", type=Path,
        help="deprecated alias for --resume, kept so existing commands keep working",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="report where this experiment got to and how to resume it, then exit. "
             "Reads local disk and, when configured, persistent storage — so it works "
             "in a fresh Colab that has never seen the run.",
    )
    parser.add_argument(
        "--restore", action="store_true",
        help="copy the run back from persistent storage before training. Use with "
             "--resume latest in a fresh session.",
    )
    parser.add_argument(
        "--persistent-dir", metavar="PATH",
        help="override training.persistent_backup for this invocation",
    )
    parser.add_argument("--json", type=Path, help="write the dry-run report here")
    parser.add_argument(
        "--force", action="store_true",
        help="train even when the feasibility check says NOT FEASIBLE (expect an OOM)",
    )
    return parser


def _report_persistent_status(config: ExperimentConfig) -> None:
    """What is on persistent storage — the only durable answer after a runtime dies."""
    destination = config.training.persistent_backup
    if not destination:
        print("\n  persistent storage: not configured (local only)")
        return

    from qwen_distill.training.persist import persistent_status

    status = persistent_status(destination)
    print(f"\n  persistent storage: {status['destination']}")
    if not status["exists"]:
        print("    NOT PRESENT — is Drive mounted, and is the path right?")
        return
    print(f"    checkpoints     : {len(status['checkpoints'])}")
    for name in status["checkpoints"][-3:]:
        print(f"      {name}")
    progress = status["latest_progress"]
    if progress:
        print(f"    latest step     : {progress.get('step')}")
        for key in ("loss", "validation_loss", "validation_bits_per_byte"):
            if progress.get(key) is not None:
                print(f"    {key:<16}: {progress[key]}")
    if status["resumable_checkpoint"]:
        print(f"    resumable at    : step {status['resumable_step']}")
        print("    Restore and continue with:")
        print("      python scripts/train_student.py --config <config> "
              "--restore --resume latest")
    else:
        print("    no complete checkpoint there yet")


def restore_experiment(config: ExperimentConfig) -> int:
    """Pull a persisted run back onto local disk, for a fresh Colab session."""
    from qwen_distill.training.persist import restore_run

    destination = config.training.persistent_backup
    if not destination:
        print("--restore needs a persistent location: set training.persistent_backup, "
              "or pass --persistent-dir.", file=sys.stderr)
        return 2

    print(f"{RULE}\nRESTORE FROM PERSISTENT STORAGE\n{RULE}\n")
    try:
        result = restore_run(destination, config.runtime.output_dir)
    except (OSError, FileNotFoundError) as exc:
        print(f"  restore failed: {exc}", file=sys.stderr)
        return 2

    print(f"  from : {result['destination']}")
    print(f"  to   : {result['local']}")
    if result["restored"]:
        print(f"  restored {len(result['restored'])} checkpoint(s): "
              f"{', '.join(result['restored'])}")
    else:
        print("  nothing to copy — local disk already has every complete checkpoint")
    if result["skipped_incomplete"]:
        # A copy interrupted by a dying runtime. Never restored, never resumed from.
        print(f"  skipped {len(result['skipped_incomplete'])} incomplete checkpoint(s): "
              f"{', '.join(result['skipped_incomplete'])}")
    if result["pointer"]:
        print(f"  resumable at step {result['pointer']['step']} "
              f"({result['pointer']['path']})")
    else:
        print("  no complete checkpoint available — nothing to resume from",
              file=sys.stderr)
        return 2
    print()
    return 0


def report_status(config: ExperimentConfig) -> int:
    """Where did this experiment get to, and how do I continue it?

    Answers from files on disk alone, so it works in a fresh Colab session that has
    never seen the run — which is precisely when the question gets asked.
    """
    from qwen_distill.training.checkpoints import (
        list_checkpoints,
        read_latest_pointer,
        resolve_checkpoint,
    )
    from qwen_distill.training.progress import ProgressWriter

    output = Path(config.runtime.output_dir)
    checkpoint_root = output / "checkpoints"
    print(f"{RULE}\nEXPERIMENT STATUS: {config.name}\n{RULE}\n")
    print(f"  run directory : {output}")

    if not output.is_dir():
        print("\n  Nothing on local disk — this experiment has not been run here.")
        # A fresh Colab has no local run, which is exactly when persistent storage is
        # the only thing that can answer the question.
        _report_persistent_status(config)
        print("\n  Start it with:\n    python scripts/train_student.py --config <config>")
        return 0

    latest = ProgressWriter(output).read_latest()
    if latest:
        print("\n  latest progress record")
        print(f"    step            : {latest.get('step')} of {config.training.max_steps}")
        for key, label in (("loss", "training loss"), ("validation_loss", "validation loss"),
                           ("bits_per_byte", "bits per byte"),
                           ("validation_bits_per_byte", "validation bpb"),
                           ("tokens_seen", "tokens seen")):
            if latest.get(key) is not None:
                print(f"    {label:<16}: {latest[key]}")
        print(f"    recorded at     : {latest.get('timestamp')}")
    else:
        print("\n  no progress records — the run has not logged a step yet")

    checkpoints = list_checkpoints(checkpoint_root)
    pointer = read_latest_pointer(checkpoint_root)
    print(f"\n  complete checkpoints: {len(checkpoints)}")
    for path in checkpoints[-5:]:
        age = ""
        try:
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            age = f"  ({metadata.get('created_at', '')})"
        except (OSError, json.JSONDecodeError):
            pass
        print(f"    {path.name}{age}")
    if len(checkpoints) > 5:
        print(f"    ... and {len(checkpoints) - 5} older")

    _report_persistent_status(config)

    resolved = resolve_checkpoint(checkpoint_root, "latest")
    if resolved is None:
        print("\n  RESUMABLE: no. No complete checkpoint exists.")
        if pointer:
            print(f"  (latest.json names {pointer.get('path')!r}, which did not verify)")
        return 0

    print(f"\n  RESUMABLE: yes, from {resolved.name}")
    print("  Resume with:")
    print("    python scripts/train_student.py --config <config> --resume latest")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = ExperimentConfig.load(args.config)
    except (ValueError, OSError) as exc:
        print(f"Could not load {args.config}:\n{exc}", file=sys.stderr)
        return 2

    if args.output_dir:
        config.runtime.output_dir = str(args.output_dir)
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.seed is not None:
        config.training.seed = args.seed
    # --resume supersedes --resume-from; both set the same field.
    if args.resume:
        config.runtime.resume_from = str(args.resume)
    elif args.resume_from:
        config.runtime.resume_from = str(args.resume_from)

    if args.persistent_dir:
        config.training.persistent_backup = args.persistent_dir

    if args.status:
        return report_status(config)

    if args.restore:
        code = restore_experiment(config)
        if code != 0:
            return code

    print(RULE)
    print(f"EXPERIMENT: {config.name}")
    print(RULE)
    if config.level:
        print(f"  ladder level : {config.level}")
    if config.description:
        print(f"  purpose      : {' '.join(config.description.split())}")
    print(f"  objective    : {config.training.objective}")
    print(f"  strategy     : {config.training.strategy} / {config.training.optimizer} "
          f"/ {config.training.precision}")
    print(f"  steps        : {config.training.max_steps} "
          f"(effective batch {config.training.effective_batch_size})")
    print(f"  sequence len : {config.data.max_sequence_length}")
    print(f"  output       : {config.runtime.output_dir}")

    spec = config.model.resolve_spec(base_dir=Path.cwd())
    if spec is not None:
        params = count_parameters(spec)
        print(f"\n  student      : {format_params(params.total)} parameters "
              f"({format_params(params.non_embedding)} non-embedding)")
        print(f"  layout       : {spec.num_hidden_layers} layers = "
              f"{spec.num_linear_attention_layers} linear + "
              f"{spec.num_full_attention_layers} full attention")
    elif config.model.pretrained:
        print(f"\n  student      : initialised from {config.model.pretrained}")
        print("                 (parameter count unknown until the checkpoint loads)")

    report = check_feasibility(config, spec, available_gib=args.simulate_vram)
    if args.simulate_vram is not None:
        print(f"\n  (feasibility checked against a simulated {args.simulate_vram:.2f} GiB GPU)")
    print()
    print(report.render())

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "config": config.to_dict(),
        "student_parameters": count_parameters(spec).as_dict() if spec else None,
        "feasibility": report.to_dict(),
        "dry_run": bool(args.dry_run),
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    if args.dry_run:
        print("\n  DRY RUN: nothing was trained.")
        if report.status.startswith("PLAUSIBLE"):
            print("  This configuration looks runnable. Re-run without --dry-run to train.")
            return 0
        if report.status == "UNKNOWN":
            print("  Feasibility could not be determined; see the reason above.")
            return 0
        print("  This configuration is NOT expected to fit. Apply a suggestion above.")
        return 1

    if report.status == "NOT FEASIBLE" and not args.force:
        print("\n  Refusing to start: the feasibility check says this will not fit.")
        print("  Apply one of the suggestions above, or pass --force to try anyway.")
        return 1

    from qwen_distill.training.trainer import train

    return train(config, spec)


if __name__ == "__main__":
    sys.exit(main())
