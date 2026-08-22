#!/usr/bin/env python3
"""Check the analytical parameter and memory models against the reference implementation.

Phase 0 derived its formulas by reading ``transformers.models.qwen3_5``. This script
verifies them empirically:

* **parameters** — instantiates the architecture with `transformers` on the ``meta``
  device (shapes only, zero storage, so even 27B fits on a laptop) and compares
  component by component against ``count_parameters``;
* **cache shapes** — runs a real forward pass on a small model and compares the
  measured KV cache, DeltaNet recurrent state and conv state against the memory model.

Requires no GPU, no network, and no teacher checkpoint. A non-zero exit means the
analytical model has drifted from upstream and must be reconciled before any estimate
in this repository is trusted.

Example::

    python scripts/validate_analytical_model.py --teacher --json evaluations/baselines/validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.architecture.spec import HybridArchSpec
from qwen_distill.teacher.validate import SMALL_SPEC, validate_cache_shapes, validate_parameters


def print_result(result) -> None:
    status = "PASS" if result.passed else ("ERROR" if result.error else "FAIL")
    print(f"\n--- {result.name}: {status} ---")
    if result.error:
        print(f"  {result.error}")
        return
    if result.comparisons:
        print(f"  {'quantity':<24}{'measured':>18}{'analytical':>18}{'delta':>10}")
        for key, values in result.comparisons.items():
            flag = "OK" if values["delta"] == 0 else "DIFF"
            print(
                f"  {key:<24}{values['measured']:>18,}{values['analytical']:>18,}{flag:>10}"
            )
    for key, value in result.details.items():
        if isinstance(value, list) and len(value) > 12:
            value = f"{value[:8]} ... ({len(value)} entries)"
        print(f"  {key:<24}{value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--teacher", action="store_true",
        help="also validate the full 27B teacher spec (meta device; slower but no memory)",
    )
    parser.add_argument("--spec", type=Path, help="validate a saved HybridArchSpec JSON")
    parser.add_argument("--json", type=Path, help="write results here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = []

    print("Validating analytical model against transformers reference implementation.")
    results.append(validate_parameters(SMALL_SPEC))
    print_result(results[-1])

    results.append(validate_cache_shapes(SMALL_SPEC))
    print_result(results[-1])

    if args.teacher:
        results.append(validate_parameters(HybridArchSpec(name="teacher-published-spec")))
        print_result(results[-1])

    if args.spec:
        results.append(validate_parameters(HybridArchSpec.load(args.spec)))
        print_result(results[-1])

    passed = sum(1 for r in results if r.passed)
    print(f"\n{'=' * 62}")
    print(f"{passed}/{len(results)} checks passed")
    if passed != len(results):
        print("ACTION: the analytical model disagrees with the reference implementation.")
        print("        Reconcile src/qwen_distill/architecture/ before trusting any estimate.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"results": [r.to_dict() for r in results],
                        "n_passed": passed, "n_total": len(results)}, indent=2) + "\n"
        )
        print(f"wrote {args.json}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
