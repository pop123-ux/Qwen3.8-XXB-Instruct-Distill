#!/usr/bin/env python3
"""Recover correct throughput from logs written before the resume bug was fixed.

Runs that resumed reported an instantaneous rate computed as **cumulative tokens across
all sessions / this session's elapsed time**. The Level-2 run's true rate was ~2,090
tok/s; its logs claimed 139,256 immediately after resuming at step 1600, decaying as 1/n
to 10,605 by step 2000.

The logs are repairable because every record still carries the two cumulative fields that
matter — ``tokens_seen`` and ``elapsed_s``. From those:

* run-wide rate = tokens_seen / elapsed_s                     (exact)
* interval rate = delta tokens / delta seconds between records (exact)

The originally logged value is kept alongside as ``logged_tokens_per_second`` rather than
overwritten, so a corrected log can be diffed against what was printed at the time.

Usage::

    python scripts/recompute_throughput.py experiments/runs/t4_level2_100m_ckpt/metrics.jsonl
    python scripts/recompute_throughput.py <metrics.jsonl> --json corrected.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.training.throughput import recompute_from_history

RULE = "=" * 78


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("metrics", type=Path, help="metrics.jsonl from a training run")
    parser.add_argument("--json", type=Path, help="write the corrected records here")
    parser.add_argument("--limit", type=int, default=20, help="rows to print")
    return parser


def read_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a kill mid-append; skip the partial record, keep the rest
    return records


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.metrics.is_file():
        print(f"metrics file not found: {args.metrics}", file=sys.stderr)
        return 2

    records = [r for r in read_records(args.metrics) if "tokens_seen" in r]
    if not records:
        print("no records carrying tokens_seen — nothing to recompute", file=sys.stderr)
        return 2

    corrected = recompute_from_history(records)
    print(f"{RULE}\nTHROUGHPUT RECOMPUTED FROM {args.metrics}\n{RULE}\n")
    print(f"  {'step':>7}{'tokens':>14}{'elapsed s':>12}"
          f"{'logged':>12}{'run-wide':>12}{'interval':>12}")
    shown = corrected[-args.limit:] if args.limit else corrected
    def cell(row: dict, key: str) -> str:
        value = row.get(key)
        return "-" if value is None else f"{value:,.0f}"

    for row in shown:
        print(f"  {row['step']:>7}{row['tokens_seen']:>14,}{row['elapsed_s']:>12,.0f}"
              f"{cell(row, 'logged_tokens_per_second'):>12}"
              f"{cell(row, 'tokens_per_second'):>12}"
              f"{cell(row, 'interval_tokens_per_second'):>12}")

    final = corrected[-1]
    print(f"\n  run-wide throughput: {final['tokens_per_second']:,.0f} tok/s "
          f"({final['tokens_seen']:,} tokens / {final['elapsed_s']:,.0f} s)")

    overstated = [
        r for r in corrected
        if r.get("logged_tokens_per_second") and r.get("tokens_per_second")
        and r["logged_tokens_per_second"] > 2 * r["tokens_per_second"]
    ]
    if overstated:
        worst = max(overstated, key=lambda r: r["logged_tokens_per_second"])
        print(f"\n  ! {len(overstated)} record(s) were logged at more than 2x the true "
              f"run-wide rate.")
        print(f"    Worst: step {worst['step']} logged "
              f"{worst['logged_tokens_per_second']:,.0f} tok/s against a true "
              f"{worst['tokens_per_second']:,.0f}.")
        print("    That is the resume bug: cumulative tokens over session-only elapsed.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(corrected, indent=2) + "\n", encoding="utf-8")
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
