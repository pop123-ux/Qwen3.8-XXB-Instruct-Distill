#!/usr/bin/env python3
"""Verify a training checkpoint reloads and behaves identically.

Checks parameter identity, logit identity through a fresh model build, generation
determinism, and that the saved step/history restore correctly for resume. A checkpoint
that writes cleanly but reloads into a different model is a silent failure that
invalidates every result after it.

Example::

    python scripts/validate_checkpoint.py --checkpoint experiments/runs/t4_level2_100m/final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.training.validate_checkpoint import (
    DEFAULT_PROMPTS,
    validate_checkpoint,
    validate_resume,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # A single required path reads better positionally; --checkpoint stays accepted so
    # existing invocations keep working.
    parser.add_argument("checkpoint", type=Path, nargs="?", help="checkpoint directory")
    parser.add_argument("--checkpoint", dest="checkpoint_flag", type=Path,
                        help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cpu", help="device to reload onto (default cpu)")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--json", type=Path, help="write the report here")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint or args.checkpoint_flag
    if args.checkpoint is None:
        parser.error("a checkpoint directory is required")
    if not args.checkpoint.is_dir():
        print(f"Checkpoint directory not found: {args.checkpoint}", file=sys.stderr)
        return 2

    report = validate_checkpoint(
        args.checkpoint, device=args.device,
        prompts=tuple(args.prompts), max_new_tokens=args.max_new_tokens,
    )
    print(report.render())

    resume = validate_resume(args.checkpoint)
    print(f"\n  resume: step {resume.resumed_step} (expected {resume.expected_step}), "
          f"history preserved: {resume.history_preserved} -> "
          f"{'PASS' if resume.passed else 'FAIL'}")
    if resume.error:
        print(f"  ERROR: {resume.error}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"checkpoint": report.to_dict(), "resume": resume.to_dict()}, indent=2)
            + "\n", encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    return 0 if (report.passed and resume.passed) else 1


if __name__ == "__main__":
    sys.exit(main())
