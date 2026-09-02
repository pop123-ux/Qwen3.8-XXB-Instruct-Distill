"""Run 002 — the pure logit-KD control at 1536 tokens, one trajectory per figure.

Every series comes from ``experiments/run002_logit_kd/metrics.jsonl``, all 128 per-step
records. The summary's first/final endpoints are never substituted for the trajectory:
the shape is the part these figures exist to show.

Run 002 is a *control*, not a capability result. 196,608 tokens of QLoRA on a frozen
quantised base moves adapters; it does not make the student good at anything, and no
figure here says otherwise.
"""
from __future__ import annotations

from common import (
    COLOURS,
    MEASURED,
    Profile,
    Provenance,
    figure,
    grid,
    save,
    style,
)
from data import load_run

RUN = "run002_logit_kd"
CALIBRATIONS = ("run002_calibration_1536", "run002_calibration")
#: Declared before any curve is drawn. The window is stated on the figure and in the
#: sidecar so a reader can tell the smoothed line from the measurements.
SMOOTH_WINDOW = 8
#: The project's predefined training-memory safety gate, in GiB.
SAFETY_GATE_GIB = 42.0


def _rolling_mean(values: list[float], window: int) -> list[float]:
    out = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start:index + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _tokens_axis(ax, xs: list[int]) -> None:
    """Training tokens along the bottom, in thousands."""
    ax.set_xlabel("training tokens")
    ax.set_xlim(0, max(xs) * 1.01)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0")


def _steps_axis(ax, profile: Profile, seq_len: int | None) -> None:
    """The same axis read as optimizer steps, along the top. Exact, since tokens = steps x
    sequence length at batch 1 with no accumulation."""
    if not seq_len:
        return
    top = ax.secondary_xaxis("top", functions=(lambda t: t / seq_len,
                                              lambda s: s * seq_len))
    top.set_xlabel("optimizer steps", fontsize=profile.font_size)
    top.tick_params(labelsize=profile.font_size - 0.5)


def _token_axis(ax, xs: list[int], profile: Profile, seq_len: int | None) -> None:
    _tokens_axis(ax, xs)
    _steps_axis(ax, profile, seq_len)


def _reference_ticks(ax, marks: list[tuple[float, str]], colour: str) -> None:
    """Name the horizontal reference lines on a right-hand axis instead of over the data.

    A text label placed against a rule inside the axes lands on whatever the data is doing
    there; a tick outside cannot collide with anything.
    """
    twin = ax.twinx()
    twin.set_ylim(ax.get_ylim())
    twin.set_yticks([value for value, _ in marks])
    twin.set_yticklabels([label for _, label in marks], color=colour)
    twin.tick_params(axis="y", length=0, labelsize=ax.xaxis.label.get_size() - 1.0)
    for spine in twin.spines.values():
        spine.set_visible(False)
    twin.grid(False)


def _provenance(figure_id: str, run, metrics: tuple[str, ...], **extra) -> Provenance:
    return Provenance(
        figure_id=figure_id, experiments=(run.experiment_id,),
        sources=run.source_paths(), metrics=metrics, value_kind=MEASURED,
        data_commit=run.data_commit, extra=extra,
    )


# ---------------------------------------------------------------------------
# F03
# ---------------------------------------------------------------------------
def training_loss(profile: Profile) -> list:
    """F03 — the optimised objective against the token budget."""
    style(profile)
    run = load_run(RUN)
    xs, ys = run.series("loss")
    smooth = _rolling_mean(ys, SMOOTH_WINDOW)

    fig, ax = figure(profile)
    ax.plot(xs, ys, color=COLOURS[0], linewidth=0.7, alpha=0.40, label="per step")
    ax.plot(xs, smooth, color=COLOURS[0], linewidth=1.8,
            label=f"rolling mean ({SMOOTH_WINDOW} steps)")
    _token_axis(ax, xs, profile, run.sequence_length)
    ax.set_ylabel("training loss (nats / token)")
    ax.set_ylim(0, max(ys) * 1.06)
    ax.set_title(f"Run 002 training loss — pure logit KD, {run.last_step} steps at "
                 f"{run.sequence_length} tokens")
    ax.legend(loc="upper right")
    grid(ax, axis="y")
    if profile.annotate:
        ax.text(0.99, 0.55,
                f"{ys[0]:.2f} → {ys[-1]:.2f} over {xs[-1]:,} tokens.\n"
                "A control, not a capability result.",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=profile.font_size - 1.2, color="#555555")
    fig.tight_layout()
    return save(fig, "run002_training_loss", profile=profile,
                provenance=_provenance("F03", run, ("loss", "tokens_seen", "step"),
                                       n_points=len(ys), smoothing_window=SMOOTH_WINDOW,
                                       first=ys[0], final=ys[-1]))


# ---------------------------------------------------------------------------
# F04
# ---------------------------------------------------------------------------
def validation_loss(profile: Profile) -> list:
    """F04 — held-out loss at the four steps the run actually evaluated.

    Four points, not 128. The line joins observations so the direction is readable; it is
    not an interpolation model, and the markers are what was measured.
    """
    style(profile)
    run = load_run(RUN)
    xs, ys = run.validation_series()

    fig, ax = figure(profile)
    ax.plot(xs, ys, color=COLOURS[0], marker="o", linewidth=1.4, markersize=5,
            label=f"{len(ys)} validation observations")
    for x, y in zip(xs, ys, strict=True):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=profile.font_size - 1.2)
    _token_axis(ax, xs, profile, run.sequence_length)
    ax.set_xlim(0, max(xs) * 1.06)
    ax.set_ylabel("validation loss (nats / token)")
    span = max(ys) - min(ys)
    ax.set_ylim(min(ys) - span * 0.18, max(ys) + span * 0.24)
    ax.set_title(f"Run 002 validation loss — evaluated every "
                 f"{run.config.get('training', {}).get('eval_every')} steps")
    ax.legend(loc="upper right")
    grid(ax, axis="y")
    if profile.annotate:
        ax.text(0.99, 0.60,
                "the line joins measurements; nothing is interpolated between them",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=profile.font_size - 1.2, color="#555555")
    fig.tight_layout()
    return save(fig, "run002_validation_loss", profile=profile,
                provenance=_provenance("F04", run, ("validation_loss", "step"),
                                       n_observations=len(ys),
                                       observed_at_steps=[r["step"] for r in run.validations],
                                       first=ys[0], final=ys[-1]))


# ---------------------------------------------------------------------------
# F05
# ---------------------------------------------------------------------------
def kd_loss(profile: Profile) -> list:
    """F05 — the teacher-imitation term.

    Run 002 sets ``kd_weight = 1.0``, so this term *is* the optimised objective. The figure
    checks that against the record rather than asserting it, and states the largest
    disagreement it found, because a silent divergence between the two would mean the run
    optimised something other than what its config says.
    """
    style(profile)
    run = load_run(RUN)
    xs, ys = run.series("kd_loss")
    _, total = run.series("loss")
    largest_gap = max(abs(a - b) for a, b in zip(ys, total, strict=True))
    smooth = _rolling_mean(ys, SMOOTH_WINDOW)

    fig, ax = figure(profile)
    ax.plot(xs, ys, color=COLOURS[2], linewidth=0.7, alpha=0.40, label="per step")
    ax.plot(xs, smooth, color=COLOURS[2], linewidth=1.8,
            label=f"rolling mean ({SMOOTH_WINDOW} steps)")
    _token_axis(ax, xs, profile, run.sequence_length)
    ax.set_ylabel("KD loss — KL(teacher ‖ student), T = "
                  f"{run.config.get('training', {}).get('kd_temperature')} (nats / token)")
    ax.set_ylim(0, max(ys) * 1.06)
    ax.set_title("Run 002 KD loss — the teacher-imitation objective")
    ax.legend(loc="upper right")
    grid(ax, axis="y")
    ax.text(0.99, 0.53,
            f"kd_weight = {run.config.get('training', {}).get('kd_weight')}, so this term is "
            f"the whole training loss\n(largest disagreement with F03 across "
            f"{len(ys)} steps: {largest_gap:.2e} nats)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=profile.font_size - 1.2, color="#555555")
    fig.tight_layout()
    return save(fig, "run002_kd_loss", profile=profile,
                provenance=_provenance("F05", run, ("kd_loss", "loss", "tokens_seen"),
                                       n_points=len(ys), smoothing_window=SMOOTH_WINDOW,
                                       first=ys[0], final=ys[-1],
                                       max_abs_difference_from_total_loss=largest_gap,
                                       kd_temperature=run.config.get("training", {})
                                       .get("kd_temperature")))


# ---------------------------------------------------------------------------
# F06
# ---------------------------------------------------------------------------
def top1_agreement(profile: Profile) -> list:
    """F06 — how often the student's argmax matched the teacher's, per training batch.

    An imitation diagnostic measured on the training batches. It is not accuracy, not a
    benchmark, and not evidence of capability; the axis label and the note both say so.
    """
    style(profile)
    run = load_run(RUN)
    xs, ys = run.series("top1_agreement")
    smooth = _rolling_mean(ys, SMOOTH_WINDOW)

    fig, ax = figure(profile)
    ax.plot(xs, ys, color=COLOURS[3], linewidth=0.7, alpha=0.40, label="per step")
    ax.plot(xs, smooth, color=COLOURS[3], linewidth=1.8,
            label=f"rolling mean ({SMOOTH_WINDOW} steps)")
    _token_axis(ax, xs, profile, run.sequence_length)
    ax.set_ylabel("teacher top-1 agreement (fraction of positions)")
    ax.set_ylim(0, max(max(ys) * 1.18, 0.05))
    ax.set_title("Run 002 teacher/student top-1 agreement, on the training batches")
    ax.legend(loc="upper left")
    grid(ax, axis="y")
    ax.text(0.99, 0.06,
            f"{ys[0]:.4f} → {ys[-1]:.4f}. An imitation diagnostic on the training "
            "batches,\nnot held-out accuracy and not a capability claim.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=profile.font_size - 1.2, color="#555555")
    fig.tight_layout()
    return save(fig, "run002_top1_agreement", profile=profile,
                provenance=_provenance("F06", run, ("top1_agreement", "tokens_seen"),
                                       n_points=len(ys), smoothing_window=SMOOTH_WINDOW,
                                       first=ys[0], final=ys[-1]))


# ---------------------------------------------------------------------------
# F07
# ---------------------------------------------------------------------------
def teacher_diagnostics(profile: Profile) -> list:
    """F07 — what the teacher's distribution looked like while it supervised.

    Two panels, because entropy and tail mass are two properties of the same object and
    share an x-axis; neither belongs on a loss figure. Both are measured at the KD
    temperature, which flattens the distribution and inflates tail mass — stated on the
    figure, since an uncaveated tail mass would misinform the offline-corpus decision.
    """
    style(profile)
    run = load_run(RUN)
    temperature = run.config.get("training", {}).get("kd_temperature")
    top_k = run.config.get("training", {}).get("kd_top_k")
    entropy_x, entropy_y = run.series("teacher_entropy")
    tail_x, tail_y = run.series("teacher_tail_mass")

    fig, (top, bottom) = figure(profile, nrows=2, sharex=True, height=0.72)
    for ax, xs, ys, colour, label in (
        (top, entropy_x, entropy_y, COLOURS[0], "teacher entropy"),
        (bottom, tail_x, tail_y, COLOURS[1], "teacher tail mass"),
    ):
        ax.plot(xs, ys, color=colour, linewidth=0.7, alpha=0.40)
        ax.plot(xs, _rolling_mean(ys, SMOOTH_WINDOW), color=colour, linewidth=1.8,
                label=label)
        grid(ax, axis="y")

    top.set_ylabel("entropy (nats)")
    top.set_title("a  Teacher output entropy")
    bottom.set_ylabel("tail mass (fraction)")
    bottom.set_title(f"b  Teacher tail mass beyond top-{top_k}")
    _tokens_axis(bottom, tail_x)
    top.set_xlim(bottom.get_xlim())
    _steps_axis(top, profile, run.sequence_length)

    fig.suptitle("Run 002 teacher signal — a property of the teacher, not of the student",
                 y=1.035, x=0.005, ha="left", fontsize=profile.font_size + 1.0)
    fig.text(0.005, 0.965,
             f"thin line: per step   ·   bold line: rolling mean over {SMOOTH_WINDOW} steps"
             f"   ·   both measured at the KD temperature T={temperature}, which flattens "
             "the distribution and inflates tail mass",
             fontsize=profile.font_size - 1.5, color="#555555", ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return save(fig, "run002_teacher_diagnostics", profile=profile,
                provenance=_provenance("F07", run,
                                       ("teacher_entropy", "teacher_tail_mass", "tokens_seen"),
                                       kd_temperature=temperature, kd_top_k=top_k,
                                       entropy_first=entropy_y[0], entropy_final=entropy_y[-1],
                                       tail_first=tail_y[0], tail_final=tail_y[-1]))


# ---------------------------------------------------------------------------
# F08
# ---------------------------------------------------------------------------
def training_memory(profile: Profile) -> list:
    """F08 — measured A40 **training** memory, and the gate that shaped the run.

    This is not a deployment measurement. The student's 16 GB target concerns inference on
    a different card with a quantised, adapter-merged model and no optimizer state; nothing
    on this figure bears on it, and the title says so.
    """
    style(profile)
    run = load_run(RUN)
    memory = run.memory
    if not memory.get("snapshots"):
        from common import MissingData

        raise MissingData(f"{run.experiment_id} memory snapshots",
                          "re-run with the memory probe enabled")
    capacity = memory["total_vram_gib"]

    fig, (left, right) = figure(profile, ncols=2, width=0.78)

    # (a) where the memory goes inside one training step
    stages = [s["stage"] for s in memory["snapshots"]]
    allocated = [s["max_allocated_gib"] for s in memory["snapshots"]]
    reserved = [s["max_reserved_gib"] for s in memory["snapshots"]]
    positions = range(len(stages))
    left.plot(positions, allocated, marker="o", color=COLOURS[0], label="peak allocated")
    left.plot(positions, reserved, marker="s", color=COLOURS[2], label="peak reserved")
    left.axhline(capacity, color=COLOURS[1], linestyle="-", linewidth=1.1)
    left.axhline(SAFETY_GATE_GIB, color=COLOURS[1], linestyle="--", linewidth=1.1)
    left.set_xticks(list(positions))
    left.set_xticklabels([s.replace("_", " ") for s in stages], rotation=38, ha="right")
    left.set_ylabel("GPU memory (GiB)")
    left.set_ylim(0, capacity * 1.16)
    left.set_title(f"a  Through one {run.sequence_length}-token step")
    left.legend(loc="lower right")
    grid(left, axis="y")
    _reference_ticks(left, [(SAFETY_GATE_GIB, f"{SAFETY_GATE_GIB:.1f} safety gate"),
                            (capacity, f"{capacity:.2f} A40 usable")], COLOURS[1])

    # (b) the peaks by sequence length, and why 2048 was refused
    points = []
    for experiment_id in (run.experiment_id, *CALIBRATIONS):
        other = load_run(experiment_id)
        other_memory = other.memory
        points.append((other.sequence_length, other_memory["peak_allocated_gib"],
                       other_memory["peak_reserved_gib"], other.experiment_id,
                       other.run_class))
    points.sort()
    gate_failed = [p for p in points if p[1] > SAFETY_GATE_GIB]
    labels = [f"{p[0]}\n{'Run 002' if p[4] == 'training' else 'calibration'}"
              for p in points]
    offsets = range(len(points))
    width = 0.36
    right.bar([o - width / 2 for o in offsets], [p[1] for p in points], width=width,
              color=COLOURS[0], label="peak allocated")
    right.bar([o + width / 2 for o in offsets], [p[2] for p in points], width=width,
              color=COLOURS[2], label="peak reserved")
    for offset, point in zip(offsets, points, strict=True):
        # Inside the bar and rotated: two adjacent labels above narrow bars collide, and a
        # collided number is worse than no number on a memory figure.
        right.text(offset - width / 2, point[1] - 0.8, f"{point[1]:.2f}", ha="center",
                   va="top", rotation=90, color="white",
                   fontsize=profile.font_size - 1.5)
        right.text(offset + width / 2, point[2] - 0.8, f"{point[2]:.2f}", ha="center",
                   va="top", rotation=90, color="white",
                   fontsize=profile.font_size - 1.5)
    right.axhline(capacity, color=COLOURS[1], linestyle="-", linewidth=1.1)
    right.axhline(SAFETY_GATE_GIB, color=COLOURS[1], linestyle="--", linewidth=1.1)
    right.set_xticks(list(offsets))
    right.set_xticklabels(labels)
    right.set_xlabel("training sequence length (tokens)")
    right.set_ylabel("GPU memory (GiB)")
    right.set_ylim(0, capacity * 1.16)
    right.set_title("b  Peaks by sequence length")
    if gate_failed:
        lengths = ", ".join(str(p[0]) for p in gate_failed)
        right.text(0.985, 0.985,
                   f"{lengths} exceeded the gate:\nthe run was not launched",
                   transform=right.transAxes, ha="right", va="top",
                   fontsize=profile.font_size - 1.5, color=COLOURS[1])
    grid(right, axis="y")
    _reference_ticks(right, [(SAFETY_GATE_GIB, f"{SAFETY_GATE_GIB:.1f} safety gate"),
                             (capacity, f"{capacity:.2f} A40 usable")], COLOURS[1])

    fig.suptitle(
        f"Run 002 TRAINING memory on {memory.get('device_name', 'the GPU')} — "
        "not a deployment result, and no bearing on the 16 GB inference target",
        y=1.02, x=0.005, ha="left", fontsize=profile.font_size + 1.0)
    fig.tight_layout()
    return save(fig, "run002_training_memory", profile=profile, provenance=Provenance(
        figure_id="F08",
        experiments=(run.experiment_id, *CALIBRATIONS),
        sources=(f"experiments/{run.experiment_id}/summary.json",
                 *[f"experiments/{c}/summary.json" for c in CALIBRATIONS]),
        metrics=("memory.snapshots.max_allocated_gib", "memory.snapshots.max_reserved_gib",
                 "memory.peak_allocated_gib", "memory.peak_reserved_gib",
                 "memory.total_vram_gib"),
        value_kind=MEASURED, data_commit=run.data_commit,
        note="training memory on an A40; the 42.0 GiB safety gate is the project's "
             "predefined launch condition",
        extra={"device": memory.get("device_name"), "usable_gib": capacity,
               "safety_gate_gib": SAFETY_GATE_GIB,
               "peaks": [{"sequence_length": p[0], "peak_allocated_gib": p[1],
                          "peak_reserved_gib": p[2], "experiment": p[3]} for p in points]},
    ))
