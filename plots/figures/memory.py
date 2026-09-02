"""Memory figures: analytical deployment accounting, and the measured training peaks.

Everything on the deployment side is **analytical** — it comes from the project's own
memory model in :mod:`qwen_distill.research.memory`, and no inference memory has been
measured on hardware. Every axis label and every title says "estimated" for that reason.
The one measured series in this module is A40 *training* memory, which is a different
quantity on different hardware and is never drawn on the same axes as a deployment
estimate.
"""
from __future__ import annotations

from common import (
    ANALYTICAL,
    COLOURS,
    MARKERS,
    MIXED,
    Profile,
    Provenance,
    figure,
    grid,
    repo_commit,
    save,
    style,
)
from data import discover_runs

#: Where the deployment estimate is broken down. Chosen because it is the largest context
#: on the ladder that the model fits at every release quantisation.
BREAKDOWN_CONTEXT = 32_768

#: Measured training peaks are discovered rather than listed. A new calibration or arm
#: appears on this figure the moment its record lands, which is the point of keeping the
#: run loader between the artifacts and the figure.

_COMPONENTS = (
    ("weights", "weights"),
    ("quantisation_overhead", "quantisation overhead"),
    ("kv_cache", "KV cache"),
    ("recurrent_state", "DeltaNet recurrent state"),
    ("conv_state", "convolution state"),
    ("activations", "activations"),
    ("runtime_overhead", "runtime / workspace"),
)


def _memory_model():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from qwen_distill.architecture.moe_student import FROZEN_STUDENT, audit
    from qwen_distill.research import memory as model

    return model, FROZEN_STUDENT, audit(FROZEN_STUDENT)["components"]


def _context_ticks(ax, ladder) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(ladder))
    ax.set_xticklabels([f"{c // 1024}K" for c in ladder])
    ax.set_xlabel("deployment context length (tokens)")


# ---------------------------------------------------------------------------
# F09
# ---------------------------------------------------------------------------
def context_memory_accounting(profile: Profile) -> list:
    """F09 — where the estimated deployment budget goes as context grows.

    Stacked, because the question is compositional: the weights are fixed and the
    context-dependent terms are what decide whether a length fits. A grouped bar chart
    would answer a different question.
    """
    style(profile)
    model, spec, components = _memory_model()
    ladder = model.CONTEXT_LADDER
    quant = model.RELEASE_QUANTS[0]

    accounts = []
    for context in ladder:
        config = model.RuntimeConfig(context_length=context, expert_quant=quant,
                                     dense_quant=quant, embedding_quant=quant)
        accounts.append(model.account(spec, config, components).to_dict()["gib"])

    fig, ax = figure(profile, width=1.05)
    bottoms = [0.0] * len(ladder)
    for index, (key, label) in enumerate(_COMPONENTS):
        values = [a[key] for a in accounts]
        ax.bar(range(len(ladder)), values, bottom=bottoms, width=0.66,
               color=COLOURS[index % len(COLOURS)],
               alpha=1.0 if index < len(COLOURS) else 0.55, label=label)
        bottoms = [b + v for b, v in zip(bottoms, values, strict=True)]
    ax.axhline(model.USABLE_GIB, color=COLOURS[1], linestyle="--", linewidth=1.2)

    ax.set_xticks(range(len(ladder)))
    ax.set_xticklabels([f"{c // 1024}K" for c in ladder])
    ax.set_xlabel("deployment context length (tokens)")
    ax.set_ylabel("estimated peak VRAM (GiB)")
    ax.set_ylim(0, max(max(bottoms) * 1.06, model.USABLE_GIB * 1.15))
    ax.set_title(f"Estimated deployment memory at {model.QUANT_LABELS[quant]}, "
                 "fully GPU-resident", pad=22)
    ax.legend(loc="upper left", ncol=2, fontsize=profile.font_size - 1.5)
    grid(ax, axis="y")

    twin = ax.twinx()
    twin.set_ylim(ax.get_ylim())
    twin.set_yticks([model.USABLE_GIB])
    twin.set_yticklabels([f"{model.USABLE_GIB:.2f} usable on a 16 GB card"],
                         color=COLOURS[1])
    twin.tick_params(axis="y", length=0, labelsize=profile.font_size - 1.5)
    for spine in twin.spines.values():
        spine.set_visible(False)
    twin.grid(False)

    # Unconditional across profiles: "this is an estimate" is truth-in-labelling, not
    # annotation density, and is never the thing a smaller profile drops.
    ax.text(0.0, 1.012,
            "ANALYTICAL — computed from qwen_distill.research.memory. "
            "No inference memory has been measured on hardware.",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=profile.font_size - 1.5, color=COLOURS[1])
    fig.tight_layout()
    return save(fig, "context_memory_accounting", profile=profile, provenance=Provenance(
        figure_id="F09",
        sources=("qwen_distill.research.memory:account",
                 "qwen_distill.architecture.moe_student:audit"),
        metrics=tuple(key for key, _ in _COMPONENTS),
        value_kind=ANALYTICAL, data_commit=repo_commit(),
        note=f"quantisation {quant}; usable capacity {model.USABLE_GIB} GiB "
             f"({model.MEASURED_TOTAL_GIB} measured total less "
             f"{model.RESERVED_GIB} reserved)",
        extra={"context_ladder": list(ladder),
               "totals_gib": [a["total"] for a in accounts]},
    ))


# ---------------------------------------------------------------------------
# F22
# ---------------------------------------------------------------------------
def context_vs_memory(profile: Profile) -> list:
    """F22 — context against peak VRAM, with the estimate and the measurement kept apart.

    Two panels rather than one axes with two line styles. The left panel is an inference
    estimate on a 16 GB card; the right is a measured training peak on a 48 GB A40 with a
    resident 4-bit teacher, an optimizer and gradients. They share a name and nothing else,
    and putting them on one axes would invite exactly the reading this project must not
    make.
    """
    style(profile)
    model, spec, components = _memory_model()
    table = model.build_table()
    ladder = model.CONTEXT_LADDER

    fig, (left, right) = figure(profile, ncols=2, width=0.82)

    # (a) analytical inference
    for index, quant in enumerate(model.RELEASE_QUANTS):
        rows = sorted((r for r in table.rows if r["quant"] == quant),
                      key=lambda r: r["context_length"])
        left.plot([r["context_length"] for r in rows], [r["total_gib"] for r in rows],
                  marker=MARKERS[index], color=COLOURS[index],
                  label=model.QUANT_LABELS[quant])
    left.axhline(model.USABLE_GIB, color=COLOURS[1], linestyle="--", linewidth=1.2)
    # Below the rule, not above: above it lands in the legend, and the curves are all
    # higher than the boundary so there is clear space underneath.
    left.text(ladder[0], model.USABLE_GIB - 0.25,
              f"{model.USABLE_GIB:.2f} GiB usable on a 16 GB card",
              fontsize=profile.font_size - 1.5, color=COLOURS[1], va="top")
    _context_ticks(left, ladder)
    left.set_ylabel("estimated peak VRAM (GiB)")
    left.set_title("a  Inference — ANALYTICAL, 16 GB target")
    left.legend(title="weight quantisation", loc="upper left")
    grid(left, axis="both")

    # (b) measured training
    measured = []
    for run in discover_runs():
        if run.memory.get("peak_allocated_gib") and run.sequence_length:
            role = "calibration" if run.run_class == "calibration" \
                else f"{run.last_step} steps"
            measured.append((run.sequence_length, run.memory["peak_allocated_gib"],
                             run.memory["peak_reserved_gib"], run.experiment_id,
                             run.run_class, run.memory.get("total_vram_gib"),
                             f"{run.short_label}\n{role}"))
    if not measured:
        from common import MissingData

        raise MissingData("measured training memory",
                          "run scripts/kd_run.py; the summary carries the memory probe")
    # Categorical, ordered by sequence length. Three runs share 1536 tokens, so a numeric
    # axis stacks them into one unreadable column; the ordered ticks keep the
    # context relationship legible while separating the objectives.
    measured.sort(key=lambda m: (m[0], m[1]))
    positions = list(range(len(measured)))
    width = 0.36
    right.bar([p - width / 2 for p in positions], [m[1] for m in measured], width=width,
              color=COLOURS[0], label="peak allocated")
    right.bar([p + width / 2 for p in positions], [m[2] for m in measured], width=width,
              color=COLOURS[2], label="peak reserved")
    capacity = next((m[5] for m in measured if m[5]), None)
    for position, entry in zip(positions, measured, strict=True):
        for offset, value in ((-width / 2, entry[1]), (width / 2, entry[2])):
            right.text(position + offset, value - 0.8, f"{value:.2f}", ha="center",
                       va="top", rotation=90, color="white",
                       fontsize=profile.font_size - 1.5)
    if capacity:
        right.axhline(capacity, color=COLOURS[1], linestyle="-", linewidth=1.1)
        twin = right.twinx()
        twin.set_ylim(0, capacity * 1.18)
        twin.set_yticks([capacity])
        twin.set_yticklabels([f"{capacity:.2f} A40 usable"], color=COLOURS[1])
        twin.tick_params(axis="y", length=0, labelsize=profile.font_size - 1.5)
        for spine in twin.spines.values():
            spine.set_visible(False)
        twin.grid(False)
    right.set_xticks(positions)
    right.set_xticklabels([f"{m[0]}\n{m[6]}" for m in measured],
                          fontsize=profile.font_size - 1.5)
    right.set_xlabel("training sequence length (tokens) and objective")
    right.set_ylabel("measured peak VRAM (GiB)")
    right.set_ylim(0, (capacity or max(m[2] for m in measured)) * 1.18)
    right.set_title("b  Training — MEASURED on an A40")
    # Below the axes: the bars reach the top of the panel, so every in-axes position
    # would sit on data, and above the axes collides with the panel title.
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.40), ncol=2,
                 fontsize=profile.font_size - 1.5, handlelength=1.2)
    grid(right, axis="y")

    fig.suptitle("Context against peak VRAM: an estimate and a measurement, "
                 "different quantities on different hardware",
                 y=1.02, x=0.005, ha="left", fontsize=profile.font_size + 1.0)
    fig.tight_layout()
    return save(fig, "context_vs_memory", profile=profile, provenance=Provenance(
        figure_id="F22",
        experiments=tuple(m[3] for m in measured),
        sources=("qwen_distill.research.memory:build_table",
                 *[f"experiments/{m[3]}/summary.json" for m in measured]),
        metrics=("total_gib", "memory.peak_allocated_gib", "memory.peak_reserved_gib"),
        value_kind=MIXED, data_commit=repo_commit(),
        note="panel a is analytical inference memory; panel b is measured A40 training "
             "memory. They are not comparable and are never plotted on one axes.",
        extra={"measured": [{"sequence_length": m[0], "peak_allocated_gib": m[1],
                             "peak_reserved_gib": m[2], "experiment": m[3]}
                            for m in measured],
               "analytical_quants": list(model.RELEASE_QUANTS)},
    ))
