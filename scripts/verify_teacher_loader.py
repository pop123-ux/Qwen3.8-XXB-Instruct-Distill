#!/usr/bin/env python3
"""Verify what a checkpoint actually loads as, under the intended software stack.

Answers Phase 1 questions 2-4 and 23 with direct evidence: the declared
``model_type``, the concrete class `transformers` resolves, whether
``trust_remote_code`` is required, and whether a real generation succeeds.

Class resolution needs only the config, so ``--config-only`` works on a metadata
download. ``--probe`` additionally loads weights and generates, which is the only
thing that proves the stack works — an import succeeding proves nothing.

Examples::

    python scripts/verify_teacher_loader.py --model Qwen/Qwen3.8-27B --config-only
    python scripts/verify_teacher_loader.py --model /models/Qwen3.8-27B --probe \
        --json evaluations/baselines/loader_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.teacher.loader import verify_loader
from qwen_distill.utils.offline import looks_local, offline_for


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Hugging Face repo id or local path")
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="allow checkpoint-provided modeling code (record when this was needed)",
    )
    parser.add_argument(
        "--config-only", action="store_true",
        help="resolve classes from the config only; do not load weights",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="load weights and run a real generation (the only proof the stack works)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--json", type=Path, help="write the structured report here")
    parser.add_argument(
        "--runtime-json", type=Path,
        default=Path("evaluations/baselines/teacher_runtime_report.json"),
        help="where --probe writes the Stage 2 runtime report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    local = looks_local(args.model)
    with offline_for(args.model) as offline:
        report = verify_loader(args.model, trust_remote_code=args.trust_remote_code)
    payload = report.to_dict()
    payload["offline_enforced"] = offline

    print("=== STAGE 1: METADATA VERIFICATION ===")
    if local:
        print("network access             : disabled (local path, offline mode enforced)")
    print(f"source                     : {report.source}")
    print(f"model_type                 : {report.model_type}")
    print(f"config class               : {report.config_class}")
    print(f"config module              : {report.config_module}")
    print(f"declared architectures     : {', '.join(report.declared_architectures) or '(none)'}")
    print(f"resolved model class       : {report.resolved_model_class}")
    print(f"resolved model module      : {report.resolved_model_module}")
    print(f"requires trust_remote_code : {report.requires_trust_remote_code}")
    print(f"uses native transformers   : {report.uses_native_transformers}")
    print(f"tokenizer class            : {report.tokenizer_class}")
    print(f"VERDICT                    : {report.verdict}")

    if report.remote_code_evidence:
        print("\nremote-code evidence:")
        for line in report.remote_code_evidence:
            print(f"  - {line}")

    print("\nversions:")
    for name, version in report.versions.items():
        print(f"  {name:<18}{version}")

    print("\n=== STAGE 2: RUNTIME VERIFICATION ===")
    if not args.probe or args.config_only:
        payload["runtime_verification"] = {
            "performed": False,
            "reason": "not requested (--probe omitted or --config-only set)",
        }
        print("  NOT PERFORMED.")
        print("  A config that resolves is NOT proof the checkpoint loads and generates.")
        print("  Weight loading, tensor-shape agreement and a real generation are")
        print("  unverified until this stage runs with the full weights:")
        print(f"    python scripts/verify_teacher_loader.py --model {args.model} --probe")

    if args.probe and not args.config_only:
        from qwen_distill.evaluation.runner import TransformersBackend

        print("  running a real generation against the weights...")
        backend = TransformersBackend(
            args.model, device=args.device, dtype=args.dtype,
            trust_remote_code=args.trust_remote_code, max_new_tokens=32,
        )
        probe = backend.probe()
        payload["probe"] = probe.to_dict()
        payload["runtime_verification"] = {"performed": True, "ok": probe.ok}
        print(f"  ok               : {probe.ok}")
        print(f"  model class      : {probe.model_class}")
        print(f"  generated tokens : {probe.generated_tokens}")
        print(f"  latency          : {probe.latency_s:.2f}s" if probe.latency_s else "  latency          : -")
        if probe.generated_text:
            print(f"  sample output    : {probe.generated_text[:120]!r}")
        if probe.error:
            print(f"  ERROR            : {probe.error}")
        print(f"  tokenizer class  : {probe.tokenizer_class}")
        print(f"  prompt sha256    : {(probe.rendered_prompt_sha256 or '-')[:16]}")
        if probe.cuda_available:
            print(f"  peak GPU memory  : {probe.peak_gpu_memory_gib:.2f} GiB ({probe.gpu_name})")
        else:
            print("  peak GPU memory  : UNAVAILABLE (no CUDA device)")

        args.runtime_json.parent.mkdir(parents=True, exist_ok=True)
        args.runtime_json.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "stage": "runtime",
                    "performed": True,
                    "probe": probe.to_dict(),
                    "stage1_verdict": report.verdict,
                },
                indent=2, ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {args.runtime_json}")

    if report.warnings:
        print("\nwarnings:")
        for warning in report.warnings:
            print(f"  ! {warning}")
    if report.errors:
        print("\nerrors:")
        for error in report.errors:
            print(f"  ! {error}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0 if not report.errors else 1


if __name__ == "__main__":
    sys.exit(main())
