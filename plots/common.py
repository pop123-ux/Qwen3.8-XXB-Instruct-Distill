"""Shared plotting setup: one style, one save path, one rule about where data comes from.

The rule, which is the reason this module exists rather than a copy of matplotlib defaults
in seven scripts:

    **A figure never invents its numbers.** Every plot here either reads a real artifact
    produced by this repository — an audit, a memory table, a ledger entry — or it is
    explicitly a schematic and is stamped as one. There is no third case, and
    :func:`require` exists to make the absence of data a loud failure rather than a
    plausible-looking curve.

Style is deliberately plain: no gridlines fighting the data, no colour where a shape will
do, no chartjunk. The target is "the result is understandable at a glance in greyscale
print", not "this looks impressive".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = Path(__file__).resolve().parent / "outputs"
PAPER = OUTPUTS / "paper"
README = OUTPUTS / "readme"

#: The deployment constraint, drawn on every memory figure. Usable capacity on a real 16 GB
#: card, not the nominal number — see research/memory.py.
VRAM_LIMIT_GIB = 13.56
NOMINAL_VRAM_GIB = 16.0

#: A restrained sequence: dark neutral first, so a single-series figure is greyscale-safe.
COLOURS = ("#1a1a1a", "#c1553b", "#3a6ea5", "#6b8f3a", "#8b5fa8", "#a08020")
MARKERS = ("o", "s", "^", "D", "v", "P")


class MissingData(SystemExit):
    """Raised when a figure has no real artifact to draw. Exits 2 with what to run."""

    def __init__(self, what: str, how: str) -> None:
        super().__init__(2)
        self.what, self.how = what, how
        print(f"  no data for {what}.\n  produce it with:\n    {how}", file=sys.stderr)


def require(path: Path, what: str, how: str) -> dict[str, Any]:
    """Load a JSON artifact or fail loudly, naming the command that would create it."""
    if not Path(path).exists():
        raise MissingData(what, how)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def style() -> None:
    """Apply the house style. Called by every figure before it draws."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.figsize": (6.4, 4.0),
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
    })


def schematic(ax, note: str = "SCHEMATIC — no measured data") -> None:
    """Stamp a figure that shows a shape rather than a result.

    A schematic that is not labelled becomes a fabricated result the moment it is pasted
    into a document, so the label is part of drawing one.
    """
    ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center", va="center",
            fontsize=16, color="#c1553b", alpha=0.18, rotation=18, zorder=0,
            fontweight="bold")


def provenance(fig, source: str) -> None:
    """Footnote every figure with where its numbers came from."""
    fig.text(0.005, 0.002, source, fontsize=6, color="#666666", ha="left", va="bottom")


def save(fig, name: str, *, source: str, paper: bool = True) -> Path:
    """Write the figure and return its path. ``source`` is required, not optional."""
    provenance(fig, source)
    directory = PAPER if paper else README
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def student_facts() -> dict[str, Any]:
    """The canonical student's audited numbers, computed rather than typed in."""
    sys.path.insert(0, str(ROOT / "src"))
    from qwen_distill.architecture.moe_student import FROZEN_STUDENT, audit

    report = audit(FROZEN_STUDENT)
    return {"spec": FROZEN_STUDENT, "audit": report}
