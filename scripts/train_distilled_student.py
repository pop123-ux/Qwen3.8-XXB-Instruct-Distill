#!/usr/bin/env python3
"""Train a student from a saved teacher dataset.

Consumes what `generate_teacher_data.py` produced. The teacher does not need to be
present, or to exist any more — that separation is what lets a 94M student train on a
free T4 from a dataset a rented A100 wrote weeks earlier.

Reuses the existing training stack (checkpointing, resume, Drive persistence, the memory
estimator) rather than adding a second trainer. What is new here is the *objective* and
the *data provenance*.

**Objectives**: ``sft`` is implemented. ``logit_kd`` and ``mixed_kd`` are declared and
raise — no teacher logits exist yet, and a KD run that silently ran SFT would invalidate
the comparison this project exists to make.

Examples::

    python scripts/train_distilled_student.py --config configs/distillation/sft_smoke.yaml --dry-run
    python scripts/train_distilled_student.py --config configs/distillation/sft_smoke.yaml --inspect-data
    python scripts/train_distilled_student.py --list-objectives
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.distillation.dataset import DatasetFilter, load_teacher_dataset
from qwen_distill.distillation.objectives import (
    ObjectiveConfig,
    ObjectiveUnavailable,
    check_dataset_supports,
    describe_objectives,
)
from qwen_distill.training.config import ExperimentConfig

RULE = "=" * 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="distillation experiment YAML")
    parser.add_argument("--dataset", type=Path, help="override the teacher dataset path")
    parser.add_argument("--objective", help="override objective.type")
    parser.add_argument("--limit", type=int, help="load at most this many records")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate everything and report; train nothing")
    parser.add_argument("--inspect-data", action="store_true",
                        help="load and describe the teacher dataset, then exit")
    parser.add_argument("--list-objectives", action="store_true",
                        help="show which objectives are implemented, and exit")
    parser.add_argument("--json", type=Path, help="write the validation report here")
    return parser


def load_objective(config: ExperimentConfig, override: str | None) -> ObjectiveConfig:
    """Read the objective from the config's extras, honouring a CLI override."""
    raw = dict(getattr(config, "objective", None) or {})
    if override:
        raw["type"] = override
    known = set(ObjectiveConfig.__dataclass_fields__)
    return ObjectiveConfig(**{k: v for k, v in raw.items() if k in known})


def describe_dataset(dataset, *, verbose: bool = True) -> None:
    print(f"  source     : {dataset.source}")
    print(f"  records    : {len(dataset):,}")
    print(dataset.stats.render())
    manifest = dataset.manifest or {}
    if manifest:
        run = manifest.get("run_manifest") or {}
        teacher = run.get("teacher") or {}
        if teacher:
            print(f"  teacher    : {teacher.get('model')} "
                  f"@ {teacher.get('revision') or 'UNPINNED'}")
            if teacher.get("is_synthetic"):
                print("  *** SYNTHETIC DATASET — mock teacher. Any model trained on this")
                print("      learns nothing about the real teacher. ***")
        if run.get("reasoning_mode"):
            print(f"  reasoning  : {run['reasoning_mode']}")
        if manifest.get("n_incomplete_shards"):
            print(f"  ! {manifest['n_incomplete_shards']} incomplete shard(s) were skipped")
    if verbose and dataset.examples:
        example = dataset.examples[0]
        print(f"\n  first record: {example.example_id}")
        print(f"    prompt  : {example.prompt[:70]}")
        print(f"    answer  : {example.teacher_answer[:70]}")
        print(f"    tokens  : reasoning {example.teacher_thinking_tokens}, "
              f"answer {example.teacher_answer_tokens}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_objectives:
        print(f"{RULE}\nDISTILLATION OBJECTIVES\n{RULE}\n")
        print(describe_objectives())
        return 0

    if not args.config and not args.dataset:
        print("need --config or --dataset (or --list-objectives)", file=sys.stderr)
        return 2

    config = None
    if args.config:
        try:
            config = ExperimentConfig.load(args.config)
        except (ValueError, OSError) as exc:
            print(f"Could not load {args.config}:\n{exc}", file=sys.stderr)
            return 2

    dataset_path = args.dataset or (
        Path(config.data.train_path) if config and config.data.train_path else None
    )
    if dataset_path is None:
        print("no teacher dataset: set data.train_path in the config, or pass --dataset",
              file=sys.stderr)
        return 2

    objective = load_objective(config, args.objective) if config else ObjectiveConfig(
        type=args.objective or "sft"
    )

    print(f"{RULE}\nDISTILLATION\n{RULE}\n")
    print(f"  objective  : {objective.type} ({objective.spec().status})")
    print(f"  dataset    : {dataset_path}")
    print()

    try:
        dataset = load_teacher_dataset(
            dataset_path, filter=DatasetFilter(), limit=args.limit
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"  could not load the teacher dataset: {exc}", file=sys.stderr)
        return 2

    describe_dataset(dataset, verbose=args.inspect_data)

    if args.inspect_data:
        train, validation = dataset.split()
        print(f"\n  deterministic split: {len(train):,} train / {len(validation):,} validation")
        print(f"  KD targets present : {dataset.stats.n_with_kd_targets:,} record(s)")
        return 0

    problems = check_dataset_supports(objective, dataset)
    report = {
        "objective": objective.to_dict(),
        "dataset": {"path": str(dataset_path), **dataset.stats.to_dict()},
        "problems": problems,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\n  wrote {args.json}")

    if problems:
        print("\n  CANNOT TRAIN:", file=sys.stderr)
        for problem in problems:
            print(f"    - {problem}", file=sys.stderr)
        return 2

    if args.dry_run:
        print("\n  DRY RUN: configuration and dataset validate. Nothing was trained.")
        return 0

    try:
        objective.require_available()
    except ObjectiveUnavailable as exc:
        print(f"\n  {exc}", file=sys.stderr)
        return 2

    # The SFT path over teacher text is not wired into the trainer yet: the trainer's
    # text mode currently consumes byte-level corpora, and connecting it needs a
    # tokenizer decision that Level 2's result should inform. Stopping here is the
    # honest position — the alternative is a training run that reports success for an
    # objective it did not implement.
    print("\n  Teacher-text SFT is not yet connected to the trainer.", file=sys.stderr)
    print("  The dataset, objective and provenance all validate; what remains is the",
          file=sys.stderr)
    print("  tokenizer decision, which Level 2's result should inform.", file=sys.stderr)
    print("  Use --dry-run or --inspect-data until then.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
