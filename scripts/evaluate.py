#!/usr/bin/env python3
"""Run an evaluation suite against a model and write machine-readable results.

Implements the three tiers from ``docs/EVALUATION_PLAN.md``. Sample selection is
deterministic (``--limit`` takes a stable prefix, never a shuffle), so two runs of the
same command are comparable.

Examples::

    # verify the backend actually works before trusting a run
    python scripts/evaluate.py --model /models/Qwen3.8-27B --probe-only

    # tier-1 development sweep
    python scripts/evaluate.py --model /models/Qwen3.8-27B --suite tier1 \\
        --reasoning-effort low --output evaluations/baselines/teacher/tier1_low
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.evaluation.metrics import format_summary, summarise
from qwen_distill.evaluation.runner import TransformersBackend, run_tasks
from qwen_distill.evaluation.tasks import long_context_suite, reasoning_dev_set
from qwen_distill.utils.hardware import collect_hardware

SUITES = {
    "tier1": lambda: reasoning_dev_set() + long_context_suite((1024, 4096), (0.1, 0.5, 0.9)),
    "reasoning": reasoning_dev_set,
    "long_context": lambda: long_context_suite((1024, 4096, 16384), (0.1, 0.5, 0.9)),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="repo id or local checkpoint path")
    parser.add_argument("--suite", default="tier1", choices=sorted(SUITES))
    parser.add_argument("--limit", type=int, help="evaluate only the first N tasks (deterministic)")
    parser.add_argument("--reasoning-effort", help="value passed to the chat template")
    parser.add_argument(
        "--no-thinking", action="store_true", help="pass enable_thinking=False to the template"
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path, help="directory for results and generations")
    parser.add_argument(
        "--probe-only", action="store_true",
        help="run a single real generation to prove the backend works, then exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    backend = TransformersBackend(
        args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=False if args.no_thinking else None,
    )

    if args.probe_only:
        probe = backend.probe()
        print(json.dumps(probe.to_dict(), indent=2))
        if not probe.ok:
            print("\nBackend probe FAILED. Do not trust evaluation results from this stack.")
        return 0 if probe.ok else 1

    tasks = SUITES[args.suite]()
    if args.limit is not None:
        tasks = tasks[: args.limit]

    output_dir = args.output
    generations_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        generations_path = output_dir / "generations.jsonl"

    print(f"model  : {args.model}")
    print(f"suite  : {args.suite} ({len(tasks)} tasks)")
    print(f"effort : {args.reasoning_effort or '(default)'}"
          f"{' thinking=off' if args.no_thinking else ''}\n")

    results = run_tasks(backend, tasks, output_path=generations_path, progress=True)
    summary = summarise(results)
    print("\n" + format_summary(summary))

    if output_dir is not None:
        metadata = {
            "model": args.model,
            "suite": args.suite,
            "n_tasks": len(tasks),
            "limit": args.limit,
            "reasoning_effort": args.reasoning_effort,
            "enable_thinking": False if args.no_thinking else None,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "backend": backend.name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task_ids": [t.task_id for t in tasks],
            "hardware": collect_hardware().to_dict(),
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (output_dir / "benchmark_results.json").write_text(
            json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {output_dir}/{{metadata,benchmark_results}}.json and generations.jsonl")

    return 0


if __name__ == "__main__":
    sys.exit(main())
