#!/usr/bin/env python3
"""Inspect a teacher checkpoint and emit a verified architecture report.

This is the script that turns "the model card says X" into "the checkpoint contains X".
Nothing in ``docs/ARCHITECTURE.md`` should be treated as confirmed until this has been
run against the real weights and its output committed under ``evaluations/baselines/``.

Examples::

    # metadata only (a few MB) from the Hub
    python scripts/inspect_teacher.py --repo-id Qwen/Qwen3.8-27B --config-only

    # full cross-check against a local download (reads headers only, not tensor data)
    python scripts/inspect_teacher.py --path /models/Qwen3.8-27B --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (side effect: sys.path)

from qwen_distill.architecture.memory import GIB, DeploymentConfig, estimate_memory
from qwen_distill.architecture.params import count_parameters, format_params
from qwen_distill.teacher.inspect import cross_check, inspect_hub, inspect_local


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", type=Path, help="local checkpoint directory")
    source.add_argument("--repo-id", help="Hugging Face repo id, e.g. Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", help="Hub revision (branch, tag or commit sha)")
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="read config/tokenizer only; skip safetensors headers (no weight download)",
    )
    parser.add_argument("--json", type=Path, help="write the full report as JSON to this path")
    parser.add_argument(
        "--save-spec", type=Path, help="write the recovered HybridArchSpec to this path"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.path:
        report = inspect_local(args.path, config_only=args.config_only)
    else:
        report = inspect_hub(args.repo_id, config_only=args.config_only, revision=args.revision)

    print(f"source              : {report.source}")
    print(f"model_type          : {report.model_type}")
    print(f"architectures       : {', '.join(report.architectures) or '(none listed)'}")
    print(f"torch_dtype         : {report.torch_dtype}")
    print(f"multimodal          : {report.is_multimodal}")
    if report.tokenizer_vocab_size:
        print(f"tokenizer vocab     : {report.tokenizer_vocab_size:,}")
    if report.reasoning_controls:
        print(f"reasoning controls  : {', '.join(report.reasoning_controls)}")
    else:
        print("reasoning controls  : (none detected)")

    if report.spec is not None:
        spec = report.spec
        print("\n--- architecture ---")
        print(f"hidden_size         : {spec.hidden_size}")
        print(f"num_hidden_layers   : {spec.num_hidden_layers}")
        print(f"intermediate_size   : {spec.intermediate_size}")
        print(f"vocab_size          : {spec.vocab_size:,}")
        print(f"tie_word_embeddings : {spec.tie_word_embeddings}")
        print(f"attention heads     : {spec.num_attention_heads} q / {spec.num_key_value_heads} kv"
              f" @ head_dim {spec.head_dim} (rope {spec.rope_dim})")
        print(f"deltanet heads      : {spec.linear_num_value_heads} v / {spec.linear_num_key_heads} k"
              f" @ {spec.linear_value_head_dim}/{spec.linear_key_head_dim}")
        print(f"layer layout        : {spec.num_linear_attention_layers} linear / "
              f"{spec.num_full_attention_layers} full  (interval {spec.full_attention_interval})")
        print(f"max_position_embed  : {spec.max_position_embeddings:,}")

        print("\n--- analytical parameter breakdown (text tower) ---")
        breakdown = count_parameters(spec)
        shares = breakdown.shares()
        for key, value in breakdown.as_dict().items():
            share = shares.get(key)
            suffix = f"  {share * 100:5.1f}%" if share is not None else ""
            print(f"  {key:<20}{format_params(value):>10}{suffix}")

        print("\n--- deployment envelope (analytical estimate) ---")
        for quant in ("bf16", "q8_0", "q4_k_m"):
            est = estimate_memory(spec, DeploymentConfig(context_length=32768, weight_quant=quant))
            fits = "fits" if est.fits_in(16.0) else "DOES NOT FIT"
            print(f"  {quant:<8} @32k ctx: weights {est.weights / GIB:6.2f} GiB, "
                  f"total {est.total_gib:6.2f} GiB -> {fits} in 16 GiB")

    if report.tensors:
        print("\n--- checkpoint tensors ---")
        print(f"  tensors read      : {len(report.tensors):,}")
        print(f"  mtp tensors       : {len(report.mtp_tensors)}")
        print(f"  vision tensors    : {len(report.vision_tensors)}")

    print("\n--- cross-check ---")
    for line in cross_check(report):
        print(f"  {line}")

    if report.warnings:
        print("\n--- warnings ---")
        for warning in report.warnings:
            print(f"  ! {warning}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    if args.save_spec and report.spec is not None:
        args.save_spec.parent.mkdir(parents=True, exist_ok=True)
        report.spec.save(args.save_spec)
        print(f"wrote {args.save_spec}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
