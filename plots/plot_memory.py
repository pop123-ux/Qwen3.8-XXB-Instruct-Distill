#!/usr/bin/env python3
"""The 16 GB constraint: context against peak VRAM, and the release Pareto frame.

The memory numbers are analytical — they come from ``research/memory.py`` and are labelled
as estimates on the figure, because an analytical total is not a measurement on hardware.
The quality axis of the Pareto plot stays empty until benchmarks exist; drawing the frame
without points is the honest version.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import COLOURS, MARKERS, VRAM_LIMIT_GIB, save, style  # noqa: E402


def main() -> int:
    style()
    import matplotlib.pyplot as plt

    from qwen_distill.architecture.moe_student import FROZEN_STUDENT, audit
    from qwen_distill.research.memory import (
        CONTEXT_LADDER,
        QUANT_LABELS,
        RELEASE_QUANTS,
        RuntimeConfig,
        account,
        build_table,
    )

    components = audit(FROZEN_STUDENT)["components"]
    table = build_table()

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.9))

    # --- context x VRAM -------------------------------------------------
    for index, quant in enumerate(RELEASE_QUANTS):
        rows = [r for r in table.rows if r["quant"] == quant]
        left.plot([r["context_length"] for r in rows], [r["total_gib"] for r in rows],
                  marker=MARKERS[index], color=COLOURS[index], label=QUANT_LABELS[quant])
    left.axhline(VRAM_LIMIT_GIB, color=COLOURS[1], linestyle="--", linewidth=1.1)
    left.text(CONTEXT_LADDER[0], VRAM_LIMIT_GIB + 0.35,
              f"{VRAM_LIMIT_GIB:.2f} GiB usable on a 16 GB card",
              fontsize=7.5, color=COLOURS[1])
    left.set_xscale("log", base=2)
    left.set_xticks(CONTEXT_LADDER)
    left.set_xticklabels([f"{c // 1024}K" for c in CONTEXT_LADDER])
    left.set_xlabel("context length (tokens)")
    left.set_ylabel("peak VRAM (GiB, estimated)")
    left.set_title("End-to-end VRAM, fully GPU-resident")
    left.legend(title="weights")

    # --- the breakdown at one operating point ---------------------------
    acc = account(FROZEN_STUDENT,
                  RuntimeConfig(context_length=32_768, expert_quant="q4_k_m",
                                dense_quant="q4_k_m", embedding_quant="q4_k_m"),
                  components)
    gib = acc.to_dict()["gib"]
    parts = [("weights", gib["weights"]), ("quant overhead", gib["quantisation_overhead"]),
             ("KV cache", gib["kv_cache"]),
             ("DeltaNet state", gib["recurrent_state"] + gib["conv_state"]),
             ("activations", gib["activations"]), ("runtime", gib["runtime_overhead"])]
    bottom = 0.0
    for index, (label, value) in enumerate(parts):
        right.bar(["Q4 @ 32K"], [value], bottom=bottom, label=label,
                  color=COLOURS[index % len(COLOURS)], width=0.42)
        bottom += value
    right.axhline(VRAM_LIMIT_GIB, color=COLOURS[1], linestyle="--", linewidth=1.1)
    right.set_ylabel("GiB")
    right.set_ylim(0, max(VRAM_LIMIT_GIB * 1.15, bottom * 1.1))
    right.set_title(f"What the {bottom:.2f} GiB is made of")
    right.legend(loc="upper right", fontsize=7)

    save(fig, "memory", paper=True,
         source="analytical estimate from research/memory.py (build_table, account); "
                "not a measured GPU value")

    # --- the Pareto frame, deliberately empty ---------------------------
    fig2, ax = plt.subplots()
    ax.axvspan(VRAM_LIMIT_GIB, VRAM_LIMIT_GIB * 1.6, color=COLOURS[1], alpha=0.07)
    ax.axvline(VRAM_LIMIT_GIB, color=COLOURS[1], linestyle="--", linewidth=1.2)
    ax.text(VRAM_LIMIT_GIB * 1.01, 0.5, "does not fit 16 GB", rotation=90,
            va="center", fontsize=8, color=COLOURS[1])
    for index, quant in enumerate(RELEASE_QUANTS):
        rows = [r for r in table.rows if r["quant"] == quant and r["context_length"] == 32_768]
        if rows:
            ax.axvline(rows[0]["total_gib"], color=COLOURS[index], linewidth=0.9, alpha=0.5)
            ax.text(rows[0]["total_gib"], 0.02, f" {QUANT_LABELS[quant]} @32K",
                    rotation=90, fontsize=7, color=COLOURS[index], va="bottom")
    ax.set_xlim(0, VRAM_LIMIT_GIB * 1.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("peak VRAM (GiB)")
    ax.set_ylabel("benchmark score")
    ax.set_title("16 GB Pareto frame — no benchmark results exist yet")
    ax.text(0.5, 0.55,
            "This project holds no benchmark numbers.\n"
            "Points appear here when experiments produce them.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#666666")
    save(fig2, "pareto_16gb", paper=True,
         source="VRAM axis analytical (research/memory.py); quality axis empty — "
                "no benchmark has been run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
