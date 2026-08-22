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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    report = verify_loader(args.model, trust_remote_code=args.trust_remote_code)
    payload = report.to_dict()

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

    if args.probe and not args.config_only:
        from qwen_distill.evaluation.runner import TransformersBackend

        print("\n--- live generation probe ---")
        backend = TransformersBackend(
            args.model, device=args.device, dtype=args.dtype,
            trust_remote_code=args.trust_remote_code, max_new_tokens=32,
        )
        probe = backend.probe()
        payload["probe"] = probe.to_dict()
        print(f"  ok               : {probe.ok}")
        print(f"  model class      : {probe.model_class}")
        print(f"  generated tokens : {probe.generated_tokens}")
        print(f"  latency          : {probe.latency_s:.2f}s" if probe.latency_s else "  latency          : -")
        if probe.generated_text:
            print(f"  sample output    : {probe.generated_text[:120]!r}")
        if probe.error:
            print(f"  ERROR            : {probe.error}")

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
