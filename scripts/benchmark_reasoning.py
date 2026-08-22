#!/usr/bin/env python3
"""Measure what the reasoning controls actually do.

Two checks, cheapest first:

1. **Template check** (``--template-only``) renders the chat template at every
   reasoning setting and diffs the results. Needs only the tokenizer. If two settings
   render byte-identical prompts, one is a no-op *by construction* — which is the
   behaviour reported for Qwen3.8's ``medium``, and this check either confirms or
   refutes it directly.

2. **Generation sweep** runs the difficulty-stratified dev set at each setting and
   compares measured thinking-token counts.

Together these produce the teacher's accuracy-vs-reasoning-cost curve, which is the
reference the student must beat on efficiency and match on capability.

Examples::

    python scripts/benchmark_reasoning.py --model Qwen/Qwen3.8-27B --template-only
    python scripts/benchmark_reasoning.py --model /models/Qwen3.8-27B \\
        --output evaluations/baselines/teacher/reasoning
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.evaluation.metrics import format_summary
from qwen_distill.evaluation.reasoning import (
    DEFAULT_SETTINGS,
    compare_rendered_prompts,
    sweep_reasoning_settings,
)
from qwen_distill.evaluation.runner import TransformersBackend
from qwen_distill.evaluation.tasks import reasoning_dev_set
from qwen_distill.utils.hardware import collect_hardware


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--settings", nargs="+", default=None,
        help="reasoning_effort values to sweep (default: default/low/medium/xhigh)",
    )
    parser.add_argument(
        "--template-only", action="store_true",
        help="only diff rendered prompts; no weights needed",
    )
    parser.add_argument("--limit", type=int, help="use only the first N dev-set tasks")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = tuple(args.settings) if args.settings else DEFAULT_SETTINGS
    payload: dict[str, object] = {
        "model": args.model,
        "settings": [s if s is not None else "(default)" for s in settings],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": collect_hardware().to_dict(),
    }

    # --- 1. template-level check --------------------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    comparison = compare_rendered_prompts(tokenizer, settings=settings)
    payload["template_comparison"] = comparison.to_dict()

    print("--- rendered prompt per reasoning setting ---")
    print(f"{'setting':<22}{'sha256[:12]':<16}{'chars':>7}")
    for rendering in comparison.renderings:
        line = f"{str(rendering.setting):<22}{rendering.sha256[:12]:<16}{rendering.n_chars:>7}"
        if rendering.error:
            line += f"   ! {rendering.error}"
        print(line)
    print(f"\ndistinct prompts    : {comparison.n_distinct} of {len(comparison.renderings)}")
    if comparison.has_noop_settings:
        for group in comparison.identical_groups:
            print(f"IDENTICAL PROMPTS   : {' == '.join(group)}")
        print("=> these settings are indistinguishable at the template level:")
        print("   selecting one over another cannot change behaviour via the prompt.")
    else:
        print("every setting renders a distinct prompt")

    if args.template_only:
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "reasoning_template.json").write_text(
                json.dumps(payload, indent=2) + "\n"
            )
            print(f"\nwrote {args.output}/reasoning_template.json")
        return 0

    # --- 2. generation sweep ------------------------------------------
    tasks = reasoning_dev_set()
    if args.limit is not None:
        tasks = tasks[: args.limit]

    def make_backend(setting: str | None) -> TransformersBackend:
        return TransformersBackend(
            args.model, device=args.device, dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            seed=args.seed, reasoning_effort=setting,
        )

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    sweep = sweep_reasoning_settings(
        make_backend, tasks, settings, output_dir=args.output, progress=True
    )
    payload["sweep"] = sweep.to_dict()

    print("\n\n=== reasoning cost by setting and difficulty ===")
    for label, summary in sweep.per_setting.items():
        print(f"\n--- reasoning_effort = {label} ---")
        print(format_summary(summary))

    print("\n=== mean thinking tokens by setting ===")
    for label, summary in sweep.per_setting.items():
        accuracy = summary.overall.accuracy
        acc = "-" if accuracy is None else f"{accuracy * 100:.1f}%"
        print(f"  {label:<16}{summary.overall.mean_thinking_tokens:>10.0f} tokens   acc {acc}")

    indistinguishable = sweep.indistinguishable_settings()
    if indistinguishable:
        print("\nSettings whose measured thinking cost differs by <5%:")
        for a, b in indistinguishable:
            print(f"  {a} ~= {b}")
        print("  => the control had little or no measurable effect on reasoning length.")

    if args.output:
        (args.output / "reasoning_baseline.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}/reasoning_baseline.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
