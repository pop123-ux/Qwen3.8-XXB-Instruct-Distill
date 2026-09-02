"""Context-specialisation figures.

The declared curricula are real — they are the experimental design this project would run,
read out of :data:`qwen_distill.research.context.CURRICULA` rather than drawn by hand — but
no context arm has been trained or evaluated. So F20 is real (design), and the two figures
that would report a *result* refuse until the measurements exist.
"""
from __future__ import annotations

import json

from common import (
    COLOURS,
    DESIGN,
    MARKERS,
    ROOT,
    MissingData,
    Profile,
    Provenance,
    figure,
    grid,
    repo_commit,
    save,
    style,
)

CURVE_DIR = ROOT / "experiments" / "context_curves"


def _context_module():
    import sys

    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from qwen_distill.research import context

    return context


# ---------------------------------------------------------------------------
# F20
# ---------------------------------------------------------------------------
def training_context_distribution(profile: Profile) -> list:
    """F20 — the training-length distribution each curriculum declares.

    A design figure, not an observation: these are the mixtures the context study would
    train, and the point of drawing them is that they differ in shape, which is what makes
    RQ2 falsifiable. Fractions are token shares and sum to one per arm, which the figure
    checks rather than assumes.
    """
    style(profile)
    context = _context_module()
    curricula = context.CURRICULA
    lengths = sorted({stage.sequence_length
                      for curriculum in curricula.values()
                      for stage in curriculum.stages})

    problems = [arm for arm, curriculum in curricula.items()
                if abs(sum(s.fraction for s in curriculum.stages) - 1.0) > 1e-6]

    # Horizontal: six arms with multi-word names do not fit under vertical bars, and a
    # collided tick label is worse than a rotated axis.
    fig, ax = figure(profile, width=1.0, height=1.05)
    arms = list(curricula)
    positions = list(range(len(arms)))[::-1]
    lefts = [0.0] * len(arms)
    for index, length in enumerate(lengths):
        values = [
            sum(s.fraction for s in curricula[arm].stages if s.sequence_length == length)
            for arm in arms
        ]
        ax.barh(positions, values, left=lefts, height=0.62,
                color=COLOURS[index % len(COLOURS)], label=f"{length // 1024}K")
        for position, (value, left) in zip(positions, zip(values, lefts, strict=True),
                                           strict=True):
            if value >= 0.08:
                ax.text(left + value / 2, position, f"{value:.0%}", ha="center",
                        va="center", color="white", fontsize=profile.font_size - 1.5)
        lefts = [left + value for left, value in zip(lefts, values, strict=True)]

    ax.set_yticks(positions)
    ax.set_yticklabels([f"{arm}  {curricula[arm].name.replace('_', ' ')}" for arm in arms])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("share of training tokens")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Declared training-length distributions — the design, not a measurement",
                 pad=22)
    ax.legend(title="training sequence length", loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=len(lengths))
    grid(ax, axis="x")
    ax.text(0.0, 1.012,
            "DESIGN — no context arm has been trained. "
            + (f"Fractions do not sum to 1 for: {', '.join(problems)}." if problems
               else "Every arm's fractions sum to 1."),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=profile.font_size - 1.5, color="#555555")
    fig.tight_layout()
    return save(fig, "training_context_distribution", profile=profile,
                provenance=Provenance(
                    figure_id="F20",
                    sources=("qwen_distill.research.context:CURRICULA",),
                    metrics=("stages.sequence_length", "stages.fraction"),
                    value_kind=DESIGN, data_commit=repo_commit(),
                    note="declared curricula; none has been trained",
                    extra={"arms": {arm: {str(s.sequence_length): s.fraction
                                          for s in curriculum.stages}
                                    for arm, curriculum in curricula.items()},
                           "fraction_sum_problems": problems},
                ))


# ---------------------------------------------------------------------------
# F21
# ---------------------------------------------------------------------------
def capability_vs_context(profile: Profile) -> list:
    """F21 — one capability curve per training curriculum, across evaluation lengths."""
    style(profile)
    context = _context_module()
    curves = sorted(CURVE_DIR.glob("*.json")) if CURVE_DIR.exists() else []
    if not curves:
        raise MissingData(
            "context capability curves",
            f"train a context arm and evaluate it across "
            f"{context.CURVE_LENGTHS[0] // 1024}K-"
            f"{context.CURVE_LENGTHS[-1] // 1024}K; each arm writes one ContextCurve to "
            f"{CURVE_DIR.relative_to(ROOT)}/",
        )

    fig, ax = figure(profile, width=1.0)
    for index, path in enumerate(curves):
        curve = context.ContextCurve.from_dict(json.loads(path.read_text(encoding="utf-8")))
        label = (f"{curve.context_arm}: {context.CURRICULA[curve.context_arm].name}"
                 if curve.context_arm in context.CURRICULA else curve.model)
        ax.plot([p.sequence_length for p in curve.points], [p.value for p in curve.points],
                marker=MARKERS[index % len(MARKERS)], color=COLOURS[index % len(COLOURS)],
                label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(context.CURVE_LENGTHS))
    ax.set_xticklabels([f"{c // 1024}K" for c in context.CURVE_LENGTHS])
    ax.set_xlabel("evaluation context length (tokens)")
    ax.set_ylabel("evaluation metric")
    ax.set_title("Capability against evaluation context, one curve per training curriculum")
    ax.legend(title="training mixture")
    grid(ax, axis="both")
    fig.tight_layout()
    return save(fig, "capability_vs_context", profile=profile, provenance=Provenance(
        figure_id="F21",
        sources=tuple(str(p.relative_to(ROOT)) for p in curves),
        metrics=("points.sequence_length", "points.value"),
        data_commit=repo_commit(),
    ))


# ---------------------------------------------------------------------------
# F23
# ---------------------------------------------------------------------------
def context_efficiency(profile: Profile) -> list:
    """F23 — quality against the memory and compute each context length costs."""
    style(profile)
    curves = sorted(CURVE_DIR.glob("*.json")) if CURVE_DIR.exists() else []
    if not curves:
        raise MissingData(
            "context efficiency (quality against memory/compute per context length)",
            "F23 needs F21's quality measurements first; the memory axis alone is "
            "already drawn by F09 and F22",
        )
    raise MissingData(  # pragma: no cover - unreachable until curves and throughput exist
        "measured per-context throughput to pair with the quality curves",
        "run an inference benchmark writing experiments/deployment/*.json",
    )
