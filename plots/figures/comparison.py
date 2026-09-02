"""The controlled cross-objective comparison — the figures RQ1 is actually settled by.

Every figure here reads :func:`data.matched_arms`, which admits a run only when its
protocol equals the reference arm's in every field except the distillation objective. That
is what makes these comparisons controlled, and it is also why they refuse to draw today:
only one matched arm (Run 002, pure logit KD) exists. Run 001 is excluded automatically —
different sequence length, different objective weighting, different corpus — rather than by
anybody remembering to leave it out.

Thresholds
----------
:data:`VALIDATION_LOSS_THRESHOLD` and :data:`AGREEMENT_THRESHOLD` are fixed here, in
source, and are written into the provenance sidecar of every figure that uses them. They
were chosen on 2026-09-02, when Run 002 was the only completed arm, and sit at the level
Run 002 reached — so a later arm is measured against the control's achieved level. Moving
them after seeing a new arm would turn F14 into a figure about the threshold rather than
about the objectives.
"""
from __future__ import annotations

from common import (
    COLOURS,
    MARKERS,
    MEASURED,
    Profile,
    Provenance,
    figure,
    grid,
    save,
    style,
)
from data import OBJECTIVE_ORDER, ArmSet, matched_arms, require_arms

#: Held-out cross-entropy, nats/token. Declared 2026-09-02, before any second arm existed.
VALIDATION_LOSS_THRESHOLD = 5.0
#: Fraction of training positions where the student's argmax matches the teacher's.
AGREEMENT_THRESHOLD = 0.30

REFERENCE = "run002_logit_kd"


def _style_for(objective: str) -> tuple[str, str]:
    index = OBJECTIVE_ORDER.index(objective) if objective in OBJECTIVE_ORDER else 0
    return COLOURS[index % len(COLOURS)], MARKERS[index % len(MARKERS)]


def _armset(minimum: int, what: str) -> ArmSet:
    return require_arms(matched_arms(REFERENCE), minimum, what)


def _provenance(figure_id: str, armset: ArmSet, metrics: tuple[str, ...],
                **extra) -> Provenance:
    return Provenance(
        figure_id=figure_id, experiments=armset.experiments(), sources=armset.sources(),
        metrics=metrics, value_kind=MEASURED,
        data_commit=armset.reference.data_commit,
        note=f"matched protocol: {armset.reference.protocol()}",
        extra={"excluded": [{"experiment": e, "reason": r} for e, r in armset.excluded],
               **extra},
    )


def _budget_note(ax, armset: ArmSet, profile: Profile) -> None:
    budgets = {run.tokens_seen for _, run in armset.ordered()}
    text = (f"matched budget: {next(iter(budgets)):,} tokens"
            if len(budgets) == 1 else
            f"WARNING — token budgets differ across arms: "
            f"{', '.join(f'{b:,}' for b in sorted(budgets))}")
    ax.text(0.005, 1.012, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=profile.font_size - 1.5,
            color="#555555" if len(budgets) == 1 else COLOURS[1])


# ---------------------------------------------------------------------------
# F10
# ---------------------------------------------------------------------------
def matched_distillation_recovery(profile: Profile) -> list:
    """F10 — where each objective ends up at the matched budget.

    The headline comparison: endpoints, not trajectories, so the arms can be read against
    each other at a glance. The trajectories that produced these endpoints are F11-F13.
    """
    style(profile)
    armset = _armset(2, "matched distillation recovery across objectives")
    arms = armset.ordered()

    fig, (left, right) = figure(profile, ncols=2, width=0.8)
    for ax, extractor, ylabel, title, better in (
        (left, lambda run: run.validation_series()[1][-1],
         "final validation loss (nats / token)", "a  Held-out loss", "lower is better"),
        (right, lambda run: run.series("top1_agreement")[1][-1],
         "final teacher top-1 agreement", "b  Teacher imitation", "higher is better"),
    ):
        labels, values, colours = [], [], []
        for objective, run in arms:
            labels.append(run.label)
            values.append(extractor(run))
            colours.append(_style_for(objective)[0])
        bars = ax.bar(labels, values, color=colours, width=0.55)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.02,
                    f"{value:.3f}", ha="center", va="bottom",
                    fontsize=profile.font_size - 1.0)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.18)
        ax.set_title(f"{title} — {better}")
        grid(ax, axis="y")

    _budget_note(left, armset, profile)
    fig.suptitle("Matched distillation recovery at a common token budget",
                 y=1.03, x=0.005, ha="left", fontsize=profile.font_size + 1.0)
    fig.tight_layout()
    return save(fig, "matched_distillation_recovery", profile=profile,
                provenance=_provenance("F10", armset,
                                       ("validation_loss", "top1_agreement", "tokens_seen")))


# ---------------------------------------------------------------------------
# F11
# ---------------------------------------------------------------------------
def training_loss_by_objective(profile: Profile) -> list:
    """F11 — training-loss trajectories, one per objective.

    Different objectives compute different quantities, so these curves are not on a common
    scale and the figure says so. It shows the *shape* of optimisation, and the arms are
    ranked by F12, which is on a common scale.
    """
    style(profile)
    armset = _armset(2, "training-loss comparison across objectives")

    fig, ax = figure(profile, width=1.0)
    for objective, run in armset.ordered():
        colour, _ = _style_for(objective)
        xs, ys = run.series("loss")
        ax.plot(xs, ys, color=colour, linewidth=0.7, alpha=0.30)
        ax.plot(xs, _rolling(ys), color=colour, linewidth=1.8,
                label=f"{run.label} ({run.experiment_id})")
    ax.set_xlabel("training tokens")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0")
    ax.set_ylabel("training loss (objective units, nats / token)")
    ax.set_ylim(bottom=0)
    ax.set_title("Training loss by objective — shapes, not a ranking")
    ax.legend()
    grid(ax, axis="y")
    ax.text(0.99, 0.96,
            "different objectives optimise different quantities;\n"
            "these curves are not on a common scale",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=profile.font_size - 1.5, color=COLOURS[1])
    _budget_note(ax, armset, profile)
    fig.tight_layout()
    return save(fig, "training_loss_by_objective", profile=profile,
                provenance=_provenance("F11", armset, ("loss", "tokens_seen")))


def _rolling(values: list[float], window: int = 8) -> list[float]:
    return [sum(values[max(0, i - window + 1):i + 1])
            / len(values[max(0, i - window + 1):i + 1]) for i in range(len(values))]


# ---------------------------------------------------------------------------
# F12
# ---------------------------------------------------------------------------
def validation_loss_by_objective(profile: Profile) -> list:
    """F12 — held-out loss trajectories. The one comparison on a common scale."""
    style(profile)
    armset = _armset(2, "validation-loss comparison across objectives")

    fig, ax = figure(profile, width=1.0)
    for objective, run in armset.ordered():
        colour, marker = _style_for(objective)
        xs, ys = run.validation_series()
        ax.plot(xs, ys, color=colour, marker=marker, linewidth=1.4, markersize=5,
                label=f"{run.label} ({run.experiment_id})")
    ax.axhline(VALIDATION_LOSS_THRESHOLD, color="#999999", linestyle=":", linewidth=1.0)
    ax.text(0.005, VALIDATION_LOSS_THRESHOLD, f" threshold {VALIDATION_LOSS_THRESHOLD} "
            "(declared in advance)", transform=ax.get_yaxis_transform(), va="bottom",
            fontsize=profile.font_size - 1.5, color="#777777")
    ax.set_xlabel("training tokens")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0")
    ax.set_ylabel("validation loss (nats / token)")
    ax.set_title("Validation loss by objective — same held-out split, same scale")
    ax.legend()
    grid(ax, axis="y")
    _budget_note(ax, armset, profile)
    fig.tight_layout()
    return save(fig, "validation_loss_by_objective", profile=profile,
                provenance=_provenance("F12", armset, ("validation_loss", "tokens_seen"),
                                       validation_loss_threshold=VALIDATION_LOSS_THRESHOLD))


# ---------------------------------------------------------------------------
# F13
# ---------------------------------------------------------------------------
def teacher_imitation_by_objective(profile: Profile) -> list:
    """F13 — KD loss and top-1 agreement per objective, in two panels.

    Split rather than twin-axed: two scales on one axes make a crossing look meaningful
    when it is an artefact of how the axes were scaled.
    """
    style(profile)
    armset = _armset(2, "teacher-imitation comparison across objectives")

    fig, (left, right) = figure(profile, ncols=2, width=0.8, sharex=True)
    for objective, run in armset.ordered():
        colour, _ = _style_for(objective)
        label = f"{run.label} ({run.experiment_id})"
        try:
            xs, ys = run.series("kd_loss")
            left.plot(xs, _rolling(ys), color=colour, linewidth=1.8, label=label)
        except SystemExit:
            pass
        xs, ys = run.series("top1_agreement")
        right.plot(xs, _rolling(ys), color=colour, linewidth=1.8, label=label)
    right.axhline(AGREEMENT_THRESHOLD, color="#999999", linestyle=":", linewidth=1.0)

    for ax, ylabel, title in (
        (left, "KD loss — KL(teacher ‖ student) (nats / token)", "a  KD loss"),
        (right, "teacher top-1 agreement (fraction)", "b  Top-1 agreement"),
    ):
        ax.set_xlabel("training tokens")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        grid(ax, axis="y")
    left.set_ylim(bottom=0)
    right.set_ylim(bottom=0)
    fig.suptitle("Teacher imitation by objective — rolling means over 8 steps",
                 y=1.03, x=0.005, ha="left", fontsize=profile.font_size + 1.0)
    fig.tight_layout()
    return save(fig, "teacher_imitation_by_objective", profile=profile,
                provenance=_provenance("F13", armset,
                                       ("kd_loss", "top1_agreement", "tokens_seen"),
                                       agreement_threshold=AGREEMENT_THRESHOLD))


# ---------------------------------------------------------------------------
# F14
# ---------------------------------------------------------------------------
def convergence_efficiency(profile: Profile) -> list:
    """F14 — tokens to reach a threshold declared before any arm was compared.

    An arm that never reaches its threshold is drawn as "not reached within the budget"
    rather than extrapolated, because extrapolating a convergence claim is the failure this
    figure would otherwise introduce.
    """
    style(profile)
    armset = _armset(2, "convergence efficiency across objectives")
    arms = armset.ordered()

    def first_crossing(xs, ys, threshold, *, falling: bool):
        for x, y in zip(xs, ys, strict=True):
            if (y <= threshold) if falling else (y >= threshold):
                return x
        return None

    panels = [
        ("a  Tokens to validation loss "
         f"≤ {VALIDATION_LOSS_THRESHOLD}",
         lambda run: first_crossing(*run.validation_series(),
                                    VALIDATION_LOSS_THRESHOLD, falling=True)),
        ("b  Tokens to top-1 agreement "
         f"≥ {AGREEMENT_THRESHOLD}",
         lambda run: first_crossing(*run.series("top1_agreement"),
                                    AGREEMENT_THRESHOLD, falling=False)),
    ]
    fig, axes = figure(profile, ncols=2, width=0.8)
    reached: dict[str, dict[str, int | None]] = {}
    for ax, (title, extractor) in zip(axes, panels, strict=True):
        labels, values, colours = [], [], []
        for objective, run in arms:
            crossing = extractor(run)
            reached.setdefault(run.experiment_id, {})[title] = crossing
            labels.append(run.label)
            values.append(crossing)
            colours.append(_style_for(objective)[0])
        budget = max(run.tokens_seen for _, run in arms)
        drawn = [v if v is not None else 0 for v in values]
        bars = ax.bar(labels, drawn, color=colours, width=0.55)
        for bar, value in zip(bars, values, strict=True):
            if value is None:
                ax.text(bar.get_x() + bar.get_width() / 2, budget * 0.03,
                        "not reached\nwithin the budget", ha="center", va="bottom",
                        fontsize=profile.font_size - 1.5, color=COLOURS[1])
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, value + budget * 0.02,
                        f"{value:,}", ha="center", va="bottom",
                        fontsize=profile.font_size - 1.0)
        ax.axhline(budget, color="#999999", linestyle=":", linewidth=1.0)
        ax.set_ylabel("training tokens to threshold")
        ax.set_ylim(0, budget * 1.18)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0")
        ax.set_title(title)
        grid(ax, axis="y")

    fig.suptitle("Convergence efficiency — thresholds declared in "
                 "plots/figures/comparison.py before any arm was compared",
                 y=1.03, x=0.005, ha="left", fontsize=profile.font_size + 1.0)
    fig.tight_layout()
    return save(fig, "convergence_efficiency", profile=profile,
                provenance=_provenance("F14", armset,
                                       ("validation_loss", "top1_agreement", "tokens_seen"),
                                       validation_loss_threshold=VALIDATION_LOSS_THRESHOLD,
                                       agreement_threshold=AGREEMENT_THRESHOLD,
                                       tokens_to_threshold=reached))
