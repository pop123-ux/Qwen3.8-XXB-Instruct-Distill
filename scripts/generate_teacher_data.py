#!/usr/bin/env python3
"""Generate teacher responses into sharded, resumable JSONL.

This is the expensive stage: a GPU big enough for a 27B model, producing thousands of
long generations. The output is a durable artifact that a T4 can train from weeks later
— the teacher and the student never need to coexist.

**The real teacher backend is not implemented yet.** Selecting it raises with an
explanation. The mock backend produces deterministic synthetic data for testing the
pipeline, and must be asked for by name: there is no fallback, because a synthetic
dataset that looks real is the most expensive failure available here.

Examples::

    # validate a configuration without generating anything
    python scripts/generate_teacher_data.py --input prompts.jsonl \\
        --output data/teacher --reasoning-mode xhigh --dry-run

    # exercise the pipeline end to end with synthetic data
    python scripts/generate_teacher_data.py --input prompts.jsonl \\
        --output data/teacher --backend mock --reasoning-mode xhigh

    # resume after a rented instance was terminated
    python scripts/generate_teacher_data.py --input prompts.jsonl \\
        --output data/teacher --backend mock --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.distillation import (
    DatasetManifest,
    make_backend,
    read_prompts,
    resolve_mode,
    scan_completed_ids,
)
from qwen_distill.distillation.backends import MOCK_BACKEND, TRANSFORMERS_BACKEND
from qwen_distill.distillation.generation import generate_dataset
from qwen_distill.distillation.provenance import TeacherIdentity
from qwen_distill.distillation.reasoning_modes import SUPPORTED_MODES, UnsupportedReasoningMode

RULE = "=" * 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="prompts JSONL")
    parser.add_argument("--output", type=Path, required=True, help="dataset directory")
    parser.add_argument(
        "--backend", default=TRANSFORMERS_BACKEND,
        help=f"{TRANSFORMERS_BACKEND!r} (real, not yet implemented) or {MOCK_BACKEND!r} "
             "(synthetic, tests only). Never selected implicitly.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B", help="teacher model id or path")
    parser.add_argument(
        "--revision",
        help="teacher revision. Without it the dataset is not reproducible from the "
             "model id alone, and the manifest records that gap.",
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=Path("vendor/qwen38-metadata"),
        help="teacher metadata, hashed into the manifest so prompt rendering is verifiable",
    )
    parser.add_argument("--reasoning-mode", default="xhigh", choices=sorted(SUPPORTED_MODES))
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--limit", type=int, help="generate at most this many new records")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="skip prompts already present on disk (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="regenerate everything, ignoring existing records")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate configuration and report the plan; generate nothing")
    parser.add_argument("--status", action="store_true",
                        help="report an existing dataset's manifest and exit")
    parser.add_argument("--mock-failure-rate", type=float, default=0.0,
                        help="mock backend only: fraction of prompts that fail")
    return parser


def report_status(directory: Path) -> int:
    print(f"{RULE}\nTEACHER DATASET STATUS\n{RULE}\n")
    print(f"  directory : {directory}")
    if not directory.is_dir():
        print("\n  Nothing here — no generation has run into this directory.")
        return 0
    manifest = DatasetManifest.read(directory)
    if manifest is None:
        print("\n  No manifest. Records on disk cannot be trusted as a complete dataset.")
        done, counts = scan_completed_ids(directory)
        print(f"  {len(done)} record(s) found across {len(counts)} file(s)")
        return 0
    print()
    print(manifest.render())
    verification = manifest.verify(directory)
    print(f"\n  integrity : {'OK' if verification['ok'] else 'PROBLEMS'}")
    for problem in verification["problems"]:
        print(f"    ! {problem}")
    for stray in verification["unlisted_shards"]:
        print(f"    ! {stray}: on disk but not in the manifest (interrupted write?)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.status:
        return report_status(args.output)

    try:
        mode = resolve_mode(args.reasoning_mode)
    except UnsupportedReasoningMode as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    backend_kwargs = {}
    if args.backend == MOCK_BACKEND:
        backend_kwargs = {"seed": args.seed, "failure_rate": args.mock_failure_rate}
    else:
        backend_kwargs = {
            "model": args.model, "revision": args.revision,
            "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        }
    try:
        backend = make_backend(args.backend, **backend_kwargs)
    except (ValueError, TypeError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    try:
        prompts = read_prompts(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    identity = TeacherIdentity.from_metadata_dir(
        args.model, args.metadata_dir, revision=args.revision
    )
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "reasoning_mode": mode.name,
        "template_kwargs": mode.template_kwargs(),
    }

    already, _ = scan_completed_ids(args.output) if args.resume else (set(), {})
    pending = [p for p in prompts if p.id not in already]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"{RULE}\nTEACHER DATA GENERATION\n{RULE}\n")
    print(f"  backend        : {args.backend}"
          + ("   *** SYNTHETIC — not real teacher output ***"
             if backend.describe().get("is_synthetic") else ""))
    print(f"  teacher        : {args.model} @ {args.revision or 'UNPINNED'}")
    print(f"  reasoning mode : {mode.name}  -> {mode.template_kwargs()}")
    print(f"  template sha   : {(identity.chat_template_sha256 or 'unknown')[:16]}")
    print(f"  prompts        : {len(prompts):,}")
    print(f"  already done   : {len(already):,}")
    print(f"  to generate    : {len(pending):,}")
    print(f"  output         : {args.output}  (shards of {args.shard_size:,})")
    if not identity.is_pinned:
        print("\n  ! teacher revision is not pinned: this dataset will not be fully "
              "reproducible\n    from the model id alone. Pass --revision to fix.")

    if args.dry_run:
        print("\n  DRY RUN: nothing was generated.")
        return 0

    if not pending:
        print("\n  Nothing to do — every prompt already has a record.")
        return 0

    def progress(position: int, total: int, prompt_id: str) -> None:
        if position % 50 == 0 or position == total:
            print(f"  {position:>6}/{total}  {prompt_id}")

    print()
    try:
        manifest, stats = generate_dataset(
            prompts, backend, mode, args.output,
            teacher_model=args.model, teacher_revision=args.revision,
            chat_template_sha256=identity.chat_template_sha256,
            generation_config=generation_config,
            shard_size=args.shard_size, resume=args.resume, limit=args.limit,
            on_progress=progress,
        )
    except NotImplementedError as exc:
        # An unimplemented backend is a configuration answer, not a crash. It must never
        # be followed by a fallback to synthetic data.
        print(f"\n  cannot generate: {exc}", file=sys.stderr)
        print("\n  To exercise the pipeline without a teacher, pass --backend mock. "
              "That produces\n  synthetic data, is marked as such in every record and "
              "in the manifest, and is\n  never selected implicitly.", file=sys.stderr)
        return 2

    print()
    print(manifest.render())
    print(f"\n  generated {stats.generated:,}, failed {stats.failed:,}, "
          f"skipped {stats.skipped_existing:,} already present")
    if stats.failed:
        print(f"  ! {stats.failed} prompt(s) failed; ids recorded in the manifest",
              file=sys.stderr)
    print(f"  manifest: {args.output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
