#!/usr/bin/env python3
"""Compare two experiments without pretending they measured the same thing.

Level 2 reached validation bits-per-byte 1.270 on procedurally generated text. Level 2R
will reach some number on real English. Subtracting them would produce a fabricated
result: the two validation sets have different intrinsic entropy, and the difference
between the numbers is dominated by the corpora rather than by the models. Level 2's
1.270 coexisted with ``"and and and"`` precisely because its corpus had a low floor.

So this tool **refuses to compute that delta**. It reports both values, says why no delta
is available, and prints the protocol that would produce a comparable number: evaluate
both checkpoints on one shared held-out corpus.

What it does compare, because a corpus change leaves them valid: throughput, memory,
parameter count, training stability — everything that measures the implementation rather
than the data. It also checks the claim that only the corpus changed, rather than taking
it on trust.

Accepts a published ``RESULT.json``, a run directory containing one, or a live run
directory with ``metrics.jsonl``. A run that is still training compares fine; its
unmeasured quantities report UNKNOWN.

Examples::

    python scripts/compare_runs.py \\
        experiments/runs/t4_level2_100m_ckpt_complete \\
        experiments/runs/t4_level2r_100m_real_english

    python scripts/compare_runs.py <left> <right> --json comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.analysis.compare import compare_runs, load_run_facts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("left", type=Path, help="baseline run (RESULT.json or run directory)")
    parser.add_argument("right", type=Path, help="run being compared against it")
    parser.add_argument("--left-name", help="override the baseline's name")
    parser.add_argument("--right-name", help="override the other run's name")
    parser.add_argument("--json", type=Path, help="write the full comparison here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    for path in (args.left, args.right):
        if not path.exists():
            print(f"no such path: {path}", file=sys.stderr)
            return 2

    try:
        left = load_run_facts(args.left, name=args.left_name)
        right = load_run_facts(args.right, name=args.right_name)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"could not read a run record: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    comparison = compare_runs(left, right)
    print(comparison.render())

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(comparison.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json}")

    # Exit 1 when the two runs are not a controlled comparison: something besides the
    # corpus moved, so even the process metrics are confounded and the caller should
    # know without reading the output.
    return 0 if comparison.controlled_experiment else 1


if __name__ == "__main__":
    sys.exit(main())
