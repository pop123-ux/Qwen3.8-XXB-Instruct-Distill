#!/usr/bin/env python3
"""Search the hybrid architecture space for candidates that fit a VRAM budget.

Enumerates architectures over a grid, prunes those that cannot hold the required
context inside the budget, and ranks survivors by non-embedding parameters — a
*capacity proxy*, not a capability measurement. Treat the output as a shortlist of
hypotheses to train and measure, never as a result.

Example::

    python scripts/search_architectures.py --vram 16 --context 32768 --top 15 \
        --json experiments/architecture_search/pass1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.architecture.memory import QUANT_BYTES_PER_PARAM, DeploymentConfig
from qwen_distill.architecture.search import SearchConstraints, generate_grid, search

DEFAULT_HIDDEN = [2560, 3072, 3584, 4096, 4608, 5120]
DEFAULT_LAYERS = [24, 32, 40, 48, 56, 64]
DEFAULT_FFN_MULT = [2.0, 2.5, 3.0, 3.4]
DEFAULT_INTERVALS = [2, 3, 4, 6]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--vram", type=float, default=16.0, help="card size in GiB")
    parser.add_argument("--reserved", type=float, default=1.0, help="GiB withheld for driver/desktop")
    parser.add_argument("--context", type=int, default=32768, help="context the candidate must support")
    parser.add_argument("--quant", default="q4_k_m", choices=sorted(QUANT_BYTES_PER_PARAM))
    parser.add_argument("--embedding-quant", default="q6_k", choices=sorted(QUANT_BYTES_PER_PARAM))
    parser.add_argument("--hidden", type=int, nargs="+", default=DEFAULT_HIDDEN)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--ffn-mult", type=float, nargs="+", default=DEFAULT_FFN_MULT)
    parser.add_argument("--intervals", type=int, nargs="+", default=DEFAULT_INTERVALS)
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument(
        "--tie-embeddings", choices=["both", "yes", "no"], default="both",
        help="whether to consider tied input/output embeddings",
    )
    parser.add_argument("--min-tok-s", type=float, default=12.0, help="minimum bandwidth-ceiling tok/s")
    parser.add_argument("--bandwidth", type=float, default=448.0, help="reference GB/s for the ceiling")
    parser.add_argument("--top", type=int, default=20, help="rows to print")
    parser.add_argument("--json", type=Path, help="write all feasible candidates as JSON")
    parser.add_argument("--save-top-spec", type=Path, help="write the top candidate's spec here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    tie_options = {"both": (False, True), "yes": (True,), "no": (False,)}[args.tie_embeddings]
    constraints = SearchConstraints(
        vram_gib=args.vram,
        required_context=args.context,
        reserved_gib=args.reserved,
        deployment=DeploymentConfig(
            context_length=args.context,
            weight_quant=args.quant,
            embedding_quant=args.embedding_quant,
        ),
        min_tokens_per_second=args.min_tok_s,
        reference_bandwidth_gb_s=args.bandwidth,
    )

    specs = list(
        generate_grid(
            hidden_sizes=args.hidden,
            layer_counts=args.layers,
            ffn_multipliers=args.ffn_mult,
            full_attention_intervals=args.intervals,
            vocab_size=args.vocab_size,
            tie_word_embeddings=tie_options,
        )
    )
    results = search(specs, constraints)

    print(f"grid            : {len(specs)} candidates")
    print(f"constraint      : {constraints.usable_gib:.1f} GiB usable, "
          f"{args.context:,} ctx, {args.quant} weights")
    print(f"feasible        : {len(results)}")
    print("ranking         : non-embedding parameters (capacity proxy, NOT measured capability)\n")

    if not results:
        print("No candidate satisfies the constraint. Relax --context, --vram or --quant.")
        return 1

    header = (
        f"{'#':>3} {'architecture':<32}{'total':>9}{'non-emb':>9}"
        f"{'GiB':>7}{'maxctx':>9}{'tok/s':>7}{'GF/tok':>8}"
    )
    print(header)
    print("-" * len(header))
    for rank, candidate in enumerate(results[: args.top], start=1):
        row = candidate.summary_row()
        print(
            f"{rank:>3} {row['name']:<32}{row['total_params_h']:>9}{row['non_embedding_h']:>9}"
            f"{row['total_gib']:>7.2f}{row['max_context']:>9,}{row['tok_s_ceiling']:>7.1f}"
            f"{row['decode_gflops']:>8.1f}"
        )

    best = results[0]
    print(f"\ntop candidate   : {best.spec.name}")
    print(f"  hidden {best.spec.hidden_size}, {best.spec.num_hidden_layers} layers, "
          f"ffn {best.spec.intermediate_size}, interval {best.spec.full_attention_interval}, "
          f"tied={best.spec.tie_word_embeddings}")
    print(f"  {best.params.total / 1e9:.2f}B total / {best.params.non_embedding / 1e9:.2f}B non-embedding")
    print(f"  fits {best.memory.total_gib:.2f} GiB at {args.context:,} ctx; "
          f"max context {best.max_context:,}")
    print("\nNOTE: this is an analytical shortlist. Train and evaluate the top candidates")
    print("      before drawing any conclusion about capability.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "constraints": {
                "vram_gib": args.vram,
                "reserved_gib": args.reserved,
                "usable_gib": constraints.usable_gib,
                "required_context": args.context,
                "weight_quant": args.quant,
                "embedding_quant": args.embedding_quant,
                "min_tokens_per_second": args.min_tok_s,
                "reference_bandwidth_gb_s": args.bandwidth,
            },
            "grid_size": len(specs),
            "n_feasible": len(results),
            "ranking_objective": "non_embedding_parameters (capacity proxy)",
            "candidates": [c.summary_row() for c in results],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")

    if args.save_top_spec:
        args.save_top_spec.parent.mkdir(parents=True, exist_ok=True)
        best.spec.save(args.save_top_spec)
        print(f"wrote {args.save_top_spec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
