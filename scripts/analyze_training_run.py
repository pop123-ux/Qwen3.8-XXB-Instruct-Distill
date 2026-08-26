#!/usr/bin/env python3
"""Read back what a training run left on disk, and say what it does and does not show.

Level 2 reached validation BPB 1.270, logged 139,256 tok/s, and generated
``"and and and"``. Every one of those was recoverable from files the run wrote. Nobody
looked until the run was over.

This looks. It reads ``metrics.jsonl``, ``progress/latest.json``, ``summary.json`` and
``checkpoints/`` — no model, no GPU, no writes into the run directory — so it is safe to
point at an experiment that is still training. Run it on Level-2R mid-flight without
touching it.

What it reports:

* **loss and bits-per-byte curves**, against the 8.0 uniform-byte baseline, because
  "1.27" means nothing without a scale
* **throughput at three scopes**, never collapsed into one number: per log record,
  over a rolling window, and run-wide across every session
* **an audit of the logged rate** against the cumulative counters in the same record,
  which is how the Level-2 bug is caught in a log written by code that still has it
* **when improvement stopped**, and how much of the run came after
* **the checkpoint timeline**: which are complete, which are resumable, and how much
  work a crash right now would cost

Plots are written if matplotlib is importable. It is not a dependency; text curves are
rendered either way.

Examples::

    python scripts/analyze_training_run.py experiments/runs/t4_level2r_100m_real_english
    python scripts/analyze_training_run.py <run-dir> --json analysis.json --plot-dir plots/
    python scripts/analyze_training_run.py <run-dir> --window 8 --quiet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.analysis import analyse_run, write_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", type=Path, help="run directory (the one containing metrics.jsonl)")
    parser.add_argument("--window", type=int, default=4,
                        help="records per rolling interval window (default: 4)")
    parser.add_argument("--name", help="override the run name used in the report")
    parser.add_argument("--json", type=Path, help="write the full analysis here")
    parser.add_argument("--plot-dir", type=Path,
                        help="write PNG plots here if matplotlib is available")
    parser.add_argument("--no-text-plot", action="store_true",
                        help="skip the ASCII curve in the report")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the findings, not the full report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.run.is_dir():
        print(f"not a directory: {args.run}", file=sys.stderr)
        return 2

    analysis = analyse_run(args.run, window=args.window, name=args.name)

    if args.quiet:
        print(f"{analysis.name}: {analysis.loop_status}")
        run_wide = analysis.throughput.run_wide_tokens_per_second
        if run_wide:
            print(f"  run-wide throughput: {run_wide:,.1f} tok/s")
        for finding in analysis.findings:
            print(f"  ! {finding}")
    else:
        print(analysis.render(plots=not args.no_text_plot))

    if args.plot_dir:
        written = write_plots(analysis, args.plot_dir)
        if written:
            print("\nplots:")
            for path in written:
                print(f"  {path}")
        else:
            # Not an error. Being unable to read your own logs without a plotting library
            # would be the error.
            print(
                "\nno plots written — matplotlib is not installed. The text curves and "
                "the numbers above are complete without it."
            )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(analysis.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    # Exit code reports whether there is anything to look at, not whether the run was
    # good: a clean analysis of a run that produced a useless model still exits 0.
    return 0 if analysis.records.training else 1


if __name__ == "__main__":
    sys.exit(main())
