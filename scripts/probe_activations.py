#!/usr/bin/env python3
"""Find out which modules a model's activation memory actually goes to.

The Level-2 T4 run OOMed at ~24.8 GiB against a 4.53 GiB estimate. Stage-boundary CUDA
measurements said the estimate was wrong; they could not say *which term*. This can:
66% of the retained activations were inside the Gated DeltaNet mixers, across 5,244
saved tensors against 86 for all four attention layers combined.

**Runs on CPU, needs no GPU.** Tensor shapes and dtypes do not depend on the device, so
this is a pre-flight check — run it before renting hardware, not after an OOM.

Examples::

    python scripts/probe_activations.py --config configs/experiments/t4_level2_100m.yaml
    python scripts/probe_activations.py --config <config> --batch-size 2 --gradient-checkpointing
    python scripts/probe_activations.py --config <config> --scaling --json probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen_distill.training.activation_probe import (  # noqa: E402
    GIB,
    probe_activations,
    scaling_study,
)
from qwen_distill.training.config import ExperimentConfig  # noqa: E402

RULE = "=" * 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True, help="experiment config")
    parser.add_argument("--batch-size", type=int, help="override the config's batch size")
    parser.add_argument("--sequence-length", type=int, help="override the config's sequence length")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="measure with checkpointing on, whatever the config says")
    parser.add_argument("--scaling", action="store_true",
                        help="fit against batch size and extrapolate to larger batches")
    parser.add_argument("--json", type=Path, help="write the full report here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2

    config = ExperimentConfig.load(args.config)
    spec = config.model.resolve_spec()
    if spec is None:
        print("This config has no inline architecture, so there is nothing to probe.",
              file=sys.stderr)
        return 2

    batch = args.batch_size or config.training.batch_size
    seq = args.sequence_length or config.data.max_sequence_length
    checkpointing = args.gradient_checkpointing or config.training.gradient_checkpointing

    print(f"{RULE}\nACTIVATION ATTRIBUTION\n{RULE}\n")
    print(f"  config    : {args.config}")
    print(f"  layers    : {spec.num_hidden_layers} "
          f"({spec.num_linear_attention_layers} DeltaNet + "
          f"{spec.num_full_attention_layers} full attention)\n")

    payload: dict[str, object] = {"config": str(args.config)}

    if args.scaling:
        study = scaling_study(spec, sequence_length=seq, gradient_checkpointing=checkpointing)
        payload["scaling"] = study
        if not study["available"]:
            print(f"  scaling study failed: {study.get('error')}", file=sys.stderr)
            return 1
        for profile in study["measured"]:
            print(f"  batch {profile['batch_size']}: {profile['total_gib']:.2f} GiB retained")
        model = study["model"]
        print(f"\n  linear model: {model['intercept_bytes'] / GIB:.2f} GiB "
              f"+ {model['bytes_per_batch'] / GIB:.2f} GiB x batch\n")
        print(f"  {'batch':>7}{'activations':>14}")
        for size, gib in study["extrapolated_gib"].items():
            print(f"  {size:>7}{gib:>13.2f}G")
    else:
        profile = probe_activations(
            spec, batch_size=batch, sequence_length=seq,
            gradient_checkpointing=checkpointing,
        )
        payload["profile"] = profile.to_dict()
        if profile.error:
            print(f"  probe failed: {profile.error}", file=sys.stderr)
            return 1
        print(profile.render())
        dominant = profile.dominant()
        if dominant and dominant.bytes_retained > profile.total_bytes * 0.5:
            print(f"\n  {dominant.scope} holds "
                  f"{dominant.bytes_retained / profile.total_bytes * 100:.0f}% of all "
                  "retained activations.")
            if dominant.scope.startswith("deltanet"):
                print("  That is the reference torch_chunk_gated_delta_rule path: it "
                      "force-upcasts\n  to fp32 and its sequential loop retains "
                      "O(chunk^2) clones per chunk.\n  Gradient checkpointing is the "
                      "effective mitigation; installing `fla`\n  replaces the kernel "
                      "entirely.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
