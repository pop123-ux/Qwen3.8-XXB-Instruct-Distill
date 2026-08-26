#!/usr/bin/env python3
"""Where the next size step fits: 250M / 350M / 500M against 12, 16, 24 and 48 GB.

Level 2 established one point — a 94.48M hybrid trains on a T4 at ~2,090 tok/s. This
answers what the next rung costs, and on which cards.

**No new estimator.** Every memory figure comes from
``qwen_distill.diagnostics.fit.estimate_training_memory`` — the same model that diagnosed
the Level-2 OOM, including the Gated DeltaNet activation term whose absence caused it,
calibrated against six measured configurations. A second estimator would produce a second
answer and no way to choose.

**Parameter counts are measured, not chosen.** Each candidate is built from the Level-2
shape rule and counted with ``architecture.params.count_parameters``. 250M/350M/500M are
scale *classes*; the candidates land where the shape rule puts them and the measured count
is what gets reported.

Two candidates per class bracket it — one wider and shallower, one narrower and deeper —
because two models of the same size with different aspect ratios have visibly different
activation costs.

The report is anchored on the configuration Level 2 actually ran, so the estimator can be
checked against a real run rather than trusted.

Examples::

    python scripts/evaluate_scaling_candidates.py
    python scripts/evaluate_scaling_candidates.py --json scaling.json
    python scripts/evaluate_scaling_candidates.py --precision bf16 --throughput
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.analysis.scaling import (
    DEVICE_BUDGETS,
    SCALE_CLASSES,
    build_candidates,
    build_matrix,
    extrapolated_tokens_per_second,
)
from qwen_distill.architecture.params import count_parameters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--precision", default="fp16",
                        choices=("fp16", "bf16", "pure_bf16", "fp32"),
                        help="fp16 for Turing (T4 has no bf16); bf16 on Ampere and later")
    parser.add_argument("--devices", nargs="+", type=int, metavar="GB",
                        help="restrict to these nominal VRAM sizes (default: all four)")
    parser.add_argument("--throughput", action="store_true",
                        help="also print the UNVALIDATED FLOP-ratio throughput extrapolation")
    parser.add_argument("--json", type=Path, help="write the full matrix here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    devices = DEVICE_BUDGETS
    if args.devices:
        wanted = set(args.devices)
        devices = tuple(d for d in DEVICE_BUDGETS if d.nominal_gb in wanted)
        if not devices:
            available = ", ".join(str(d.nominal_gb) for d in DEVICE_BUDGETS)
            print(f"no such device budget; known: {available}", file=sys.stderr)
            return 2

    matrix = build_matrix(devices=devices, precision=args.precision)
    print(matrix.render())

    print("\n" + "-" * 92)
    print("CANDIDATE ARCHITECTURES — measured parameter counts")
    print("-" * 92)
    targets = {scale.label: scale.target_parameters for scale in SCALE_CLASSES}
    for spec in build_candidates():
        measured = count_parameters(spec).total
        target = targets[spec.name.split("_", 1)[0]]
        print(
            f"  {spec.name:<20} {measured:>13,}  ({measured / target - 1:+6.1%} vs the "
            f"{target / 1e6:.0f}M class label)"
        )
        print(
            f"  {'':<20} hidden {spec.hidden_size}, {spec.num_hidden_layers} layers "
            f"({spec.num_linear_attention_layers} DeltaNet + "
            f"{spec.num_full_attention_layers} full attention), ff {spec.intermediate_size}, "
            f"{spec.num_attention_heads} heads / {spec.num_key_value_heads} kv"
        )
    print("\n  The class labels are targets. The counts are results, and they are what")
    print("  gets reported anywhere else.")

    if args.throughput:
        print("\n" + "-" * 92)
        print("THROUGHPUT — UNVALIDATED EXTRAPOLATION, NOT A MEASUREMENT")
        print("-" * 92)
        for spec in build_candidates():
            estimate = extrapolated_tokens_per_second(spec)
            hours = None
            rate = estimate["extrapolated_tokens_per_second"]
            if rate:
                hours = 32_768_000 / rate / 3600      # Level 2's token budget
            print(
                f"  {spec.name:<20} {estimate['flops_ratio_vs_level2']:>6.2f}x FLOPs  "
                f"~{rate:>8,.0f} tok/s   "
                + (f"~{hours:5.1f} h for Level 2's 32.8M tokens" if hours else "")
            )
        print("\n  Derived from ONE measured point (Level 2: 2,089.2 tok/s on a T4) by FLOP")
        print("  ratio. It ignores memory bandwidth, kernel efficiency and occupancy, all of")
        print("  which change with shape. An order of magnitude, not a prediction.")

    if args.json:
        payload = matrix.to_dict()
        payload["candidates_detail"] = [
            {
                "name": spec.name,
                "measured_parameters": count_parameters(spec).total,
                "spec": spec.to_dict() if hasattr(spec, "to_dict") else None,
                "throughput_extrapolation": extrapolated_tokens_per_second(spec),
            }
            for spec in build_candidates()
        ]
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
