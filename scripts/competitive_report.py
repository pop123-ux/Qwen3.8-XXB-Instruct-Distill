#!/usr/bin/env python3
"""Who we have to beat inside a 16 GB card, what it would cost them, and what we still
cannot say.

The objective is not a parameter count. It is: **be the strongest model a person can
actually run on a 16 GB GPU** (and separately, on a 12 GB one). This report puts the field
and our own candidates in the same frame so that "is this the right size?" is answered by
the envelope rather than assumed in advance.

Read the last section first. Every competitor figure in this repository is currently
**unverified** — some supplied by hand, some recalled from a language model's training
data — and none of it has been checked against a model card here. Until it is, the field is
a list of things to confirm, not a scoreboard. The report exits non-zero under ``--strict``
for exactly that reason.

Two different estimators are in play, deliberately:

* **our candidates** are modelled by ``qwen_distill.architecture.memory``, which knows the
  hybrid layout — that only 1 layer in 4 keeps a KV cache, and what the DeltaNet recurrent
  and conv states cost.
* **competitors** are modelled from a parameter count and, where known, a KV geometry.
  That is coarser, and it is coarser because we do not have their configs. Where a
  geometry is unknown the cache is reported as unknown rather than as zero.

Exit codes: ``0`` reported, ``1`` unverified figures remain and ``--strict`` was given,
``2`` the request could not be set up.

Examples::

    python scripts/competitive_report.py
    python scripts/competitive_report.py --context 8192 --quant q4_k_m
    python scripts/competitive_report.py --candidate 3072,28,10240,2,8 --candidate 4096,32,13824,3,12
    python scripts/competitive_report.py --strict --json field.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.analysis.competition import (
    MISSING_CAPABILITIES,
    TARGET_BENCHMARKS,
    envelope,
    reference_field,
    verification_backlog,
)
from qwen_distill.architecture.memory import DeploymentConfig, estimate_memory
from qwen_distill.architecture.params import count_parameters, format_params
from qwen_distill.architecture.spec import HybridArchSpec
from qwen_distill.architecture.transfer import student_from_teacher

#: Usable VRAM after the driver, CUDA context and allocator fragmentation. The 16 GB
#: figure is measured on the Level-2 T4 run; the 12 GB one is from vendor capacity and has
#: not been measured on a real 12 GB card.
BUDGETS = {"16 GB": 13.56, "12 GB": 10.76}

TEACHER = HybridArchSpec(name="qwen3.8-27b")


def parse_candidate(text: str) -> HybridArchSpec:
    """``hidden,layers,ffn[,kv_heads[,dn_key_heads]]`` -> a transferable student."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 3:
        raise ValueError(f"expected hidden,layers,ffn[,kv,dnk], got {text!r}")
    hidden, layers, ffn = (int(p) for p in parts[:3])
    kv = int(parts[3]) if len(parts) > 3 else None
    dnk = int(parts[4]) if len(parts) > 4 else None
    return student_from_teacher(
        TEACHER, name=f"h{hidden}-L{layers}", hidden_size=hidden,
        num_hidden_layers=layers, intermediate_size=ffn,
        num_key_value_heads=kv, linear_num_key_heads=dnk, tie_word_embeddings=True,
    )


def candidate_row(spec: HybridArchSpec, *, quant: str, context: int, budget: float) -> dict:
    estimate = estimate_memory(
        spec, DeploymentConfig(context_length=context, weight_quant=quant)
    )
    breakdown = count_parameters(spec)
    total = estimate.total_gib
    headroom = budget - total
    return {
        "name": spec.name,
        "parameters": breakdown.total,
        "active_parameters": breakdown.total,
        "weight_gib": estimate.weights / (1024 ** 3),
        "kv_gib": estimate.kv_cache / (1024 ** 3),
        "total_gib": total,
        "headroom_gib": headroom,
        "verdict": "DOES NOT FIT" if headroom < 0 else "TIGHT" if headroom < 1.5 else "FITS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quant", default="q4_k_m")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--budget", choices=(*BUDGETS, "both"), default="both")
    parser.add_argument("--candidate", action="append", default=[],
                        help="hidden,layers,ffn[,kv_heads[,dn_key_heads]]; repeatable")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 while unverified competitor figures remain")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    budgets = BUDGETS if args.budget == "both" else {args.budget: BUDGETS[args.budget]}
    field = reference_field()
    report: dict = {"quant": args.quant, "context": args.context, "budgets": {}}

    try:
        candidates = [parse_candidate(text) for text in args.candidate]
    except ValueError as exc:
        print(f"  {exc}", file=sys.stderr)
        return 2

    print(f"\n  DEPLOYMENT ENVELOPE   weights {args.quant}, context {args.context:,}, batch 1")
    for label, budget in budgets.items():
        print(f"\n  --- {label} (usable {budget:.2f} GiB) "
              f"{'-' * (46 - len(label))}")
        print(f"    {'model':<20}{'params':>9}{'weights':>10}{'KV':>10}{'total':>10}  verdict")
        rows = []
        for competitor in field.values():
            result = envelope(
                competitor, budget_gib=budget, quant=args.quant, context=args.context
            )
            kv = f"{result.kv_gib:.2f}" if result.kv_gib is not None else "unknown"
            weights = f"{result.weight_gib:.2f}" if result.weight_gib is not None else "unknown"
            total = f"{result.total_gib:.2f}" if result.total_gib is not None else "-"
            params = format_params(competitor.parameters) if competitor.parameters else "?"
            print(f"    {competitor.name:<20}{params:>9}{weights:>10}{kv:>10}{total:>10}  "
                  f"{result.verdict}")
            rows.append(result.to_dict())
        for spec in candidates:
            row = candidate_row(spec, quant=args.quant, context=args.context, budget=budget)
            print(f"    {row['name'] + ' (ours)':<20}{format_params(row['parameters']):>9}"
                  f"{row['weight_gib']:>10.2f}{row['kv_gib']:>10.2f}{row['total_gib']:>10.2f}  "
                  f"{row['verdict']}")
            rows.append(row)
        report["budgets"][label] = {"usable_gib": budget, "rows": rows}

    if candidates:
        print("\n    Our rows use the hybrid-aware estimator (1 KV layer in 4, plus DeltaNet")
        print("    recurrent and conv state). Competitor rows are parameter-count estimates")
        print("    with a KV geometry only where one is recorded. They are not equally exact.")

    print("\n  TARGET BOARD   the benchmarks the objective is stated in")
    print(f"    {'benchmark':<20}{'target':>8}  {'status':<16}measures")
    target = field["Qwen3.5-9B"]
    for name, benchmark in TARGET_BENCHMARKS.items():
        score = target.scores.get(name)
        value = f"{score.value:.1f}" if score else "-"
        status = "IMPLEMENTED" if benchmark.implemented else "not implemented"
        print(f"    {name:<20}{value:>8}  {status:<16}{', '.join(benchmark.capabilities)}")
    print(f"\n    Targets are {target.name}'s reported scores. They are UNVERIFIED "
          f"({target.source}).")
    print("    They are an aspiration to confirm, not a bar that can currently be cleared.")
    for gap in MISSING_CAPABILITIES.values():
        print(f"\n    ! {gap}")
    report["target_board"] = {
        "competitor": target.name,
        "scores": {n: s.to_dict() for n, s in target.scores.items()},
        "benchmarks": {n: b.__dict__ for n, b in TARGET_BENCHMARKS.items()},
    }

    backlog = verification_backlog(field)
    print(f"\n  BEFORE ANY CLAIM OF SUPERIORITY   {len(backlog)} item(s)")
    for item in backlog:
        print(f"    - {item}")
    report["verification_backlog"] = backlog

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\n  wrote {args.json}")

    if args.strict and backlog:
        print(f"\n  --strict: {len(backlog)} unverified item(s) remain.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
