#!/usr/bin/env python3
"""Estimate the full VRAM envelope of an architecture across contexts and quantisations.

Reports ``weights + KV cache + recurrent state + activations + runtime overhead``,
never just weight size. All numbers are analytical estimates; measure real peak VRAM
with ``scripts/benchmark_memory.py`` before publishing any deployment claim.

Examples::

    python scripts/estimate_vram.py --preset teacher --matrix
    python scripts/estimate_vram.py --spec configs/student/candidate.json --vram 16 --max-context
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.architecture.flops import (
    bandwidth_bound_tokens_per_second,
    decode_flops_per_token,
    format_flops,
)
from qwen_distill.architecture.memory import (
    QUANT_BYTES_PER_PARAM,
    DeploymentConfig,
    estimate_memory,
    max_context_within,
)
from qwen_distill.architecture.params import count_parameters, format_params
from qwen_distill.architecture.spec import HybridArchSpec

PRESETS = {
    "teacher": HybridArchSpec(
        name="Qwen3.8-27B (published spec, pending checkpoint verification)"
    ),
}

DEFAULT_CONTEXTS = (8192, 32768, 65536, 131072, 262144)
DEFAULT_QUANTS = ("bf16", "int8", "q6_k", "q5_k_m", "q4_k_m")


def load_spec(args: argparse.Namespace) -> HybridArchSpec:
    if args.preset:
        return PRESETS[args.preset]
    return HybridArchSpec.load(args.spec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=sorted(PRESETS), help="built-in architecture")
    source.add_argument("--spec", type=Path, help="path to a saved HybridArchSpec JSON")
    parser.add_argument("--vram", type=float, default=16.0, help="target VRAM in GiB (default 16)")
    parser.add_argument(
        "--reserved", type=float, default=1.0,
        help="GiB withheld for driver/desktop (default 1.0)",
    )
    parser.add_argument("--context", type=int, default=32768, help="context length for the single-point report")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--quant", default="q4_k_m", choices=sorted(QUANT_BYTES_PER_PARAM))
    parser.add_argument(
        "--embedding-quant", default="q6_k",
        choices=sorted(QUANT_BYTES_PER_PARAM) + ["none"],
        help="precision for embedding/lm_head; 'none' means same as --quant",
    )
    parser.add_argument("--kv-dtype", default="fp16", choices=["fp32", "bf16", "fp16", "fp8"])
    parser.add_argument("--matrix", action="store_true", help="print a context x quantisation matrix")
    parser.add_argument("--max-context", action="store_true", help="report the largest fitting context")
    parser.add_argument("--json", type=Path, help="write results as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_spec(args)
    budget = args.vram - args.reserved

    base = DeploymentConfig(
        context_length=args.context,
        batch_size=args.batch_size,
        weight_quant=args.quant,
        embedding_quant=None if args.embedding_quant == "none" else args.embedding_quant,
        kv_cache_dtype=args.kv_dtype,
    )

    params = count_parameters(spec)
    print(f"architecture : {spec.name}")
    print(f"parameters   : {format_params(params.total)} total, "
          f"{format_params(params.non_embedding)} non-embedding")
    print(f"layout       : {spec.num_hidden_layers} layers = "
          f"{spec.num_linear_attention_layers} linear + {spec.num_full_attention_layers} full attention")
    print(f"budget       : {args.vram:.1f} GiB card - {args.reserved:.1f} GiB reserved "
          f"= {budget:.1f} GiB usable\n")

    est = estimate_memory(spec, base)
    print(f"--- envelope @ {args.context:,} ctx, {args.quant} weights, batch {args.batch_size} ---")
    for key, value in est.as_dict().items():
        print(f"  {key:<24}{value:8.3f} GiB")
    verdict = "FITS" if est.total_gib <= budget else "DOES NOT FIT"
    print(f"  {'verdict':<24}{verdict} (headroom {budget - est.total_gib:+.2f} GiB)")

    decode = decode_flops_per_token(spec, args.context)
    print(f"\n  decode FLOPs/token      {format_flops(decode.total)}")
    for bandwidth, label in ((320.0, "T4 ~320 GB/s"), (448.0, "3060 Ti ~448 GB/s"), (672.0, "5070 ~672 GB/s")):
        ceiling = bandwidth_bound_tokens_per_second(spec, bandwidth, args.quant)
        print(f"  bandwidth ceiling       {ceiling:6.1f} tok/s  ({label})")

    results: dict[str, object] = {
        "spec": spec.to_dict(),
        "params": params.as_dict(),
        "budget_gib": budget,
        "point_estimate": est.as_dict(),
    }

    if args.max_context:
        ctx = max_context_within(spec, budget, base)
        print(f"\n  max context within {budget:.1f} GiB: {ctx:,} tokens")
        results["max_context"] = ctx

    if args.matrix:
        print(f"\n--- peak VRAM (GiB) by context x quantisation, budget {budget:.1f} GiB ---")
        header = "  " + f"{'quant':<10}" + "".join(f"{c // 1024:>8}k" for c in DEFAULT_CONTEXTS)
        print(header)
        matrix: dict[str, dict[str, float]] = {}
        for quant in DEFAULT_QUANTS:
            row, cells = {}, []
            for ctx in DEFAULT_CONTEXTS:
                total = estimate_memory(
                    spec, replace(base, context_length=ctx, weight_quant=quant)
                ).total_gib
                row[str(ctx)] = round(total, 2)
                mark = " " if total <= budget else "*"
                cells.append(f"{total:>8.1f}{mark}")
            matrix[quant] = row
            print(f"  {quant:<10}" + "".join(cells))
        print("  (* exceeds budget)")
        results["matrix"] = matrix

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
