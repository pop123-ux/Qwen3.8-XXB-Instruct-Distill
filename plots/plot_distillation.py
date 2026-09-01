#!/usr/bin/env python3
"""Recovery curves: loss against training tokens, one curve per objective.

Compares the research protocol's four baselines — CE-only, CE + logit KD, CE + layer KD,
CE + behaviour KD — which are arms A0, A2, A1 and A3.

No run has happened, so there is nothing to plot. This script exits 2 and names the ledger
entries it would read, rather than drawing curves that would be read as results.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import COLOURS, MARKERS, save, style  # noqa: E402

ARMS = {"A0": "CE only", "A2": "CE + logit KD", "A1": "CE + layer KD",
        "A3": "CE + behaviour KD"}


def main() -> int:
    style()
    import matplotlib.pyplot as plt

    from qwen_distill.research.ledger import Ledger

    ledger = Ledger()
    runs = {arm: ledger.entries(kind="training_run", arm=arm) for arm in ARMS}
    if not any(runs.values()):
        print("  no training runs in the ledger for arms " + ", ".join(ARMS) + ".",
              file=sys.stderr)
        print("  Each arm writes kind='training_run' entries carrying 'tokens' and 'loss'.\n"
              "  Run the arms, then re-run this script.", file=sys.stderr)
        return 2

    fig, ax = plt.subplots()
    for index, (arm, label) in enumerate(ARMS.items()):
        points = [(e["payload"]["tokens"], e["payload"]["loss"]) for e in runs[arm]
                  if "tokens" in e.get("payload", {}) and "loss" in e["payload"]]
        if not points:
            continue
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker=MARKERS[index],
                color=COLOURS[index], label=f"{arm}: {label}")
    ax.set_xscale("log")
    ax.set_xlabel("training tokens")
    ax.set_ylabel("validation loss (nats/token)")
    ax.set_title("Distillation recovery by objective")
    ax.legend()
    save(fig, "distillation_recovery", paper=True,
         source=f"experiments/ledger.jsonl, kind=training_run, arms {', '.join(ARMS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
