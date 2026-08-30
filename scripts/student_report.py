#!/usr/bin/env python3
"""Every audit the frozen student has, in one run.

    python scripts/student_report.py                    # everything, to stdout
    python scripts/student_report.py --section memory   # one section
    python scripts/student_report.py --json out.json    # machine-readable
    python scripts/student_report.py --ledger           # also append to the ledger

Nothing here trains, downloads or allocates a full model: the parameter audit runs on
meta tensors and the memory accounting is analytical, so the whole report runs on a
laptop in seconds. Numbers that are estimates say so, in the report and in the ledger.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen_distill.architecture.moe_student import (  # noqa: E402
    FROZEN_STUDENT,
    MTP_STATUS,
    audit,
    render_audit,
)
from qwen_distill.research.ablations import matrix  # noqa: E402
from qwen_distill.research.context import CONTEXT_REGIMES, CURRICULA  # noqa: E402
from qwen_distill.research.ledger import DEFAULT_LEDGER, Ledger  # noqa: E402
from qwen_distill.research.memory import (  # noqa: E402
    build_table,
    frontier,
    headline,
    render_table,
)

SECTIONS = ("architecture", "memory", "context", "ablations")


def _architecture() -> tuple[str, dict]:
    report = audit(FROZEN_STUDENT)
    return render_audit(report), report


def _memory() -> tuple[str, dict]:
    table = build_table()
    result = headline()
    text = render_table(table) + "\n\n  FINDING\n    " + _wrap(result["finding"])
    text += "\n\n  IMPLICATION\n    " + _wrap(result["implication"])
    fits = frontier()[:5]
    if fits:
        text += "\n\n    closest configurations that do fit:"
        for row in fits:
            text += (f"\n      experts {row['expert_quant']:<8} dense {row['dense_quant']:<8} "
                     f"embeddings {row['embedding_quant']:<8} -> "
                     f"{row['max_context']:,} tokens")
    return text, {"table": table.to_dict(), "headline": result, "frontier": frontier()}


def _wrap(text: str, width: int = 92, indent: str = "    ") -> str:
    import textwrap

    return ("\n" + indent).join(textwrap.wrap(text, width))


def _context() -> tuple[str, dict]:
    lines = ["  context regimes", ""]
    for regime in CONTEXT_REGIMES:
        lines.append(f"    {regime.name:<7} {regime.min_tokens:>8,} - {regime.max_tokens - 1:>8,}"
                     f"   probe: {regime.probe}")
    lines += ["", "  curricula (share of tokens, not steps)", ""]
    for arm, curriculum in CURRICULA.items():
        share = ", ".join(f"{length//1024}K:{fraction:.0%}"
                          for length, fraction in curriculum.token_share().items())
        lines.append(f"    {arm}  {curriculum.name:<26} {share}")
    return "\n".join(lines), {
        "regimes": [{"name": r.name, "min_tokens": r.min_tokens, "max_tokens": r.max_tokens,
                     "mechanism": r.mechanism, "probe": r.probe} for r in CONTEXT_REGIMES],
        "curricula": {arm: c.to_dict() for arm, c in CURRICULA.items()},
    }


def _ablations() -> tuple[str, dict]:
    data = matrix()
    lines = ["  ablation matrix", ""]
    for family, block in data["families"].items():
        lines.append(f"    {family} (control {block['control']})")
        for arm in block["arms"]:
            lines.append(f"      {arm['arm']}  {arm['name']}")
            lines.append(f"          falsified if: {_wrap(arm['falsified_if'], 80, '              ')}")
        lines.append("")
    return "\n".join(lines), data


BUILDERS = {"architecture": _architecture, "memory": _memory,
            "context": _context, "ablations": _ablations}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--section", choices=SECTIONS, action="append",
                        help="limit the report; repeatable. Default: all sections.")
    parser.add_argument("--json", type=Path, help="also write the machine-readable report here")
    parser.add_argument("--ledger", nargs="?", const=str(DEFAULT_LEDGER), default=None,
                        help="append the architecture audit and memory accounting to a ledger")
    args = parser.parse_args(argv)

    sections = args.section or list(SECTIONS)
    payload: dict[str, dict] = {}
    for name in sections:
        text, data = BUILDERS[name]()
        payload[name] = data
        print(f"\n{'=' * 96}\n  {name.upper()}\n{'=' * 96}")
        print(text)

    if "architecture" in sections:
        print(f"\n  MTP: {_wrap(MTP_STATUS)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\n  wrote {args.json}")

    if args.ledger:
        ledger = Ledger(args.ledger)
        if "architecture" in payload:
            ledger.measured("architecture_audit", f"{FROZEN_STUDENT.name} parameter audit",
                            payload["architecture"], tags=["frozen_student"])
        if "memory" in payload:
            ledger.estimated(
                "memory_accounting", f"{FROZEN_STUDENT.name} end-to-end VRAM",
                payload["memory"]["headline"],
                method="analytical: audited component counts x quantisation bytes, plus "
                       "measured KV/state formulas, activation model and 0.9 GiB runtime "
                       "overhead. No GPU measurement was taken.",
                tags=["16gb", "frozen_student"],
            )
        print(f"\n  appended to {ledger.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
