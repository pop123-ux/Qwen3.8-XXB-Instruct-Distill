#!/usr/bin/env python3
"""How closely the student reproduces the teacher's computation, per signal.

This is the paper's central measurement: not "does the student score well" but "does it do
the same work". Five signals, each with its own scale, each read from a real artifact when
one exists.

Today the only signal measurable without a GPU is the MoE reconstruction at
initialisation, which this script computes for real. The rest need a materialised student
and a loaded teacher, so they are drawn as a stamped schematic showing the shape of the
question rather than an answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import COLOURS, ROOT, save, schematic, style  # noqa: E402

#: Written by an initialisation audit; absent until one has run against the real teacher.
ALIGNMENT_ARTIFACT = ROOT / "experiments" / "alignment.json"

SIGNALS = ("logits", "hidden states", "attention maps", "DeltaNet state", "FFN / MoE")


def main() -> int:
    style()
    import matplotlib.pyplot as plt
    from common import require  # noqa: PLC0415

    fig, ax = plt.subplots()
    if ALIGNMENT_ARTIFACT.exists():
        data = require(ALIGNMENT_ARTIFACT, "behaviour alignment", "see docs/RESEARCH.md")
        names = list(data["signals"])
        values = [data["signals"][n] for n in names]
        source = f"{ALIGNMENT_ARTIFACT.relative_to(ROOT)} (experiment {data.get('id', '?')})"
    else:
        # No fabricated numbers: bars of equal height, stamped, showing only the axes.
        names, values = list(SIGNALS), [0.0] * len(SIGNALS)
        schematic(ax, "SCHEMATIC — needs a materialised student\nand a loaded teacher")
        source = ("no alignment artifact yet; axes only. Produce one with an initialisation "
                  "audit against the real teacher — see docs/RESEARCH.md")

    ax.barh(names[::-1], values[::-1], color=COLOURS[0], height=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("cosine similarity to the teacher's signal (1.0 = identical)")
    ax.set_title("Computational behaviour: which signals transfer, and how well")
    ax.text(0.02, -0.6,
            "DeltaNet state is not directly comparable (shape mismatch); it is measured at "
            "the hidden-size interface.\nSee BEHAVIORAL_DISTILLATION.md.",
            fontsize=6.5, color="#666666", transform=ax.get_yaxis_transform())
    save(fig, "behavior_alignment", paper=True, source=source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
