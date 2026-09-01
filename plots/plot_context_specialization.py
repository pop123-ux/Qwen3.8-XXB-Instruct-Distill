#!/usr/bin/env python3
"""Context-performance curves, one per training-length mixture.

The question: does the distribution of lengths seen during distillation move where the
curve breaks? Each curve is one arm of the B family; the x-axis is *evaluation* context,
which is a different quantity from training context and from the architectural maximum.

Reads ContextCurve artifacts. Without them it draws the stamped schematic — the axes and
the question, not an answer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import COLOURS, MARKERS, ROOT, save, schematic, style  # noqa: E402

CURVE_DIR = ROOT / "experiments" / "context_curves"


def main() -> int:
    style()
    import matplotlib.pyplot as plt

    from qwen_distill.research.context import CURRICULA, CURVE_LENGTHS, ContextCurve

    fig, ax = plt.subplots()
    curves = sorted(CURVE_DIR.glob("*.json")) if CURVE_DIR.exists() else []
    if curves:
        for index, path in enumerate(curves):
            curve = ContextCurve.from_dict(json.loads(path.read_text(encoding="utf-8")))
            ax.plot([p.sequence_length for p in curve.points],
                    [p.value for p in curve.points],
                    marker=MARKERS[index % len(MARKERS)],
                    color=COLOURS[index % len(COLOURS)],
                    label=f"{curve.context_arm}: {CURRICULA[curve.context_arm].name}"
                    if curve.context_arm in CURRICULA else curve.model)
        source = f"{CURVE_DIR.relative_to(ROOT)}/*.json ({len(curves)} curves)"
        ax.legend(title="training mixture")
    else:
        schematic(ax, "SCHEMATIC — no evaluation runs yet")
        ax.set_xlim(CURVE_LENGTHS[0], CURVE_LENGTHS[-1])
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.62,
                "One curve per training-length mixture (B0-B5).\n"
                "The claim under test is that the mixture moves the knee.",
                transform=ax.transAxes, ha="center", fontsize=8.5, color="#666666")
        source = (f"no curves in {CURVE_DIR.relative_to(ROOT)}; axes only. "
                  "Each arm writes one ContextCurve there.")

    ax.set_xscale("log", base=2)
    ax.set_xticks(CURVE_LENGTHS)
    ax.set_xticklabels([f"{c // 1024}K" for c in CURVE_LENGTHS])
    ax.set_xlabel("evaluation context length (tokens)")
    ax.set_ylabel("retrieval accuracy")
    ax.set_title("Context specialisation: does the training mixture move the knee?")
    save(fig, "context_specialization", paper=True, source=source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
