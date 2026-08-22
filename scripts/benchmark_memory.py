#!/usr/bin/env python3
"""Measure real peak VRAM and compare it against the analytical estimate.

``docs/DEPLOYMENT_PLAN.md`` forbids publishing a deployment claim from weight size
alone. This script measures the real thing, in the three phases that matter:

* **load** — weights resident, nothing else;
* **prefill** — usually where peak memory actually occurs, on a long prompt;
* **decode** — steady-state generation.

Both PyTorch allocator statistics and ``nvidia-smi`` process usage are recorded,
because they differ: the allocator reports what tensors need, ``nvidia-smi`` reports
what the process holds including the CUDA context and allocator reserve. **A deployment
claim must use the larger, ``nvidia-smi`` number.**

The measured/estimated ratio is the calibration factor for
``qwen_distill.architecture.memory``. If it is consistently far from 1.0, the analytical
model is missing an overhead term and should be corrected rather than explained away.

Requires a CUDA GPU. On a CPU-only machine it reports that and exits non-zero rather
than emitting zeros that could be mistaken for measurements.

Example::

    python scripts/benchmark_memory.py --model /models/student --context 8192 32768 \\
        --json evaluations/deployment/student_memory.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.architecture.memory import DeploymentConfig, estimate_memory
from qwen_distill.architecture.spec import HybridArchSpec
from qwen_distill.utils.hardware import collect_hardware, measure_memory, reset_peak_memory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="checkpoint path or repo id")
    parser.add_argument("--context", type=int, nargs="+", default=[4096],
                        help="prompt lengths to measure, in tokens")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--quant", default="bf16", help="weight format, for the estimate")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hardware = collect_hardware()

    print("--- hardware ---")
    print(f"  platform      : {hardware.platform}")
    print(f"  cuda available: {hardware.cuda_available}")
    print(f"  gpu           : {hardware.gpu_name or '(none)'}")
    if hardware.total_vram_gib:
        print(f"  total VRAM    : {hardware.total_vram_gib:.2f} GiB")
    print(f"  driver        : {hardware.driver_version or 'unknown'}")
    for name in ("torch", "transformers"):
        print(f"  {name:<14}: {hardware.versions.get(name)}")

    if not hardware.cuda_available:
        print("\nNo CUDA device available: VRAM cannot be measured on this machine.")
        print("Run this on the 16 GB target hardware. Refusing to emit zeros as if")
        print("they were measurements.")
        return 2

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    measurements = []
    reset_peak_memory()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code,
        dtype=getattr(torch, args.dtype), device_map="cuda",
    ).eval()
    measurements.append(measure_memory("load"))
    print(f"\n  after load    : {measurements[-1].torch_allocated_gib:.2f} GiB allocated, "
          f"{measurements[-1].nvidia_smi_used_gib or float('nan'):.2f} GiB process")

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    try:
        spec = HybridArchSpec.from_hf_config(config.to_dict(), name=args.model)
    except (KeyError, ValueError):
        spec = None
        print("  (config is not a hybrid spec; skipping analytical comparison)")

    rows = []
    for context in args.context:
        reset_peak_memory()
        prompt_ids = torch.randint(
            0, min(tokenizer.vocab_size, 1000), (1, context), device="cuda"
        )
        with torch.no_grad():
            output = model(prompt_ids, use_cache=True)
        prefill = measure_memory(f"prefill@{context}")
        measurements.append(prefill)

        with torch.no_grad():
            model.generate(
                prompt_ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                past_key_values=output.past_key_values,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        decode = measure_memory(f"decode@{context}")
        measurements.append(decode)

        peak = max(prefill.torch_peak_reserved_gib, decode.torch_peak_reserved_gib)
        estimated = None
        if spec is not None:
            estimated = estimate_memory(
                spec,
                DeploymentConfig(
                    context_length=context + args.max_new_tokens,
                    weight_quant=args.quant,
                    embedding_quant=None,
                    kv_cache_dtype="bf16" if args.dtype == "bfloat16" else "fp16",
                ),
            ).total_gib
        rows.append(
            {
                "context": context,
                "peak_torch_reserved_gib": peak,
                "peak_nvidia_smi_gib": max(
                    prefill.nvidia_smi_used_gib or 0.0, decode.nvidia_smi_used_gib or 0.0
                ),
                "analytical_estimate_gib": estimated,
                "measured_over_estimated": (peak / estimated) if estimated else None,
            }
        )
        del prompt_ids, output
        torch.cuda.empty_cache()

    print(f"\n{'context':>9}{'peak(torch)':>14}{'peak(smi)':>12}{'estimate':>11}{'ratio':>8}")
    for row in rows:
        est = f"{row['analytical_estimate_gib']:.2f}" if row["analytical_estimate_gib"] else "-"
        ratio = f"{row['measured_over_estimated']:.3f}" if row["measured_over_estimated"] else "-"
        print(f"{row['context']:>9}{row['peak_torch_reserved_gib']:>14.2f}"
              f"{row['peak_nvidia_smi_gib']:>12.2f}{est:>11}{ratio:>8}")

    ratios = [r["measured_over_estimated"] for r in rows if r["measured_over_estimated"]]
    if ratios:
        mean_ratio = sum(ratios) / len(ratios)
        print(f"\ncalibration factor (measured/estimated): {mean_ratio:.3f}")
        if abs(mean_ratio - 1.0) > 0.15:
            print("ACTION: the analytical memory model is off by more than 15%.")
            print("        Correct qwen_distill.architecture.memory rather than explaining it away.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "dtype": args.dtype,
                    "quant_for_estimate": args.quant,
                    "hardware": hardware.to_dict(),
                    "measurements": [m.to_dict() for m in measurements],
                    "rows": rows,
                    "weight_bytes_note": (
                        "GiB = 1024^3 bytes throughout. Weight *file* size on disk and "
                        "loaded VRAM differ; both are reported separately."
                    ),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ) + "\n"
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
