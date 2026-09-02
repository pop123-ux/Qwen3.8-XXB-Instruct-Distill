"""Architecture figures: what changed between the teacher and the student.

Every number is counted at plot time from the frozen specification and the pinned teacher
preset, so these figures cannot drift from the architecture they describe. Nothing here is
a measurement of a running system.
"""
from __future__ import annotations

from common import (
    AUDITED,
    COLOURS,
    DESIGN,
    Profile,
    Provenance,
    figure,
    grid,
    repo_commit,
    save,
    student_facts,
    style,
    teacher_facts,
)

BILLION = 1e9
_DELTANET = "#3a6ea5"
_ATTENTION = "#c1553b"


def _bar_labels(ax, bars, values, fmt="{:.2f}B", pad=0.015) -> None:
    span = max(values) or 1.0
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + span * pad, fmt.format(value),
                ha="center", va="bottom", fontsize=ax.figure.get_axes()[0].xaxis.label.get_size())


# ---------------------------------------------------------------------------
# F01
# ---------------------------------------------------------------------------
def model_compression(profile: Profile) -> list:
    """F01 — four architectural properties, one per panel.

    Cramming parameters, depth, sparsity and attention geometry into one axes would force
    a shared scale onto quantities with no common unit. Small multiples keep each property
    readable on its own terms while the row reads as one comparison.
    """
    style(profile)
    student = student_facts()
    teacher = teacher_facts()
    spec, report = student["spec"], student["audit"]
    t_spec, t_total = teacher["spec"], teacher["total"]
    s_total = report["exact_parameter_count"]
    s_active = report["active_parameters_per_token"]

    fig, axes = figure(profile, ncols=4, nrows=1, width=0.62, height=1.02)
    a, b, c, d = axes

    # (a) stored parameters
    values = [t_total / BILLION, s_total / BILLION]
    bars = a.bar(["teacher", "student"], values, color=[COLOURS[0], COLOURS[2]], width=0.55)
    _bar_labels(a, bars, values)
    a.set_ylabel("stored parameters (billions)")
    a.set_ylim(0, values[0] * 1.22)
    a.set_title("a  Stored parameters")
    if profile.annotate:
        a.text(0.5, 0.90, f"student is {s_total / t_total:.0%} of the teacher",
               transform=a.transAxes, ha="center", fontsize=profile.font_size - 1.2,
               color="#555555")

    # (b) active parameters per token
    values = [t_total / BILLION, s_active / BILLION]
    bars = b.bar(["teacher\n(dense)", "student\n(top-2 of 8)"], values,
                 color=[COLOURS[0], COLOURS[2]], width=0.55)
    _bar_labels(b, bars, values)
    b.set_ylabel("active parameters / token (billions)")
    b.set_ylim(0, values[0] * 1.22)
    b.set_title("b  Active per token")
    if profile.annotate:
        b.text(0.5, 0.90, "every teacher parameter is active;\n"
                          f"{s_active / s_total:.0%} of the student's is",
               transform=b.transAxes, ha="center", fontsize=profile.font_size - 1.2,
               color="#555555")

    # (c) depth, by block type
    t_attention = t_spec.num_full_attention_layers
    t_deltanet = t_spec.num_hidden_layers - t_attention
    s_attention = len(spec.attention_layer_indices)
    s_deltanet = len(spec.deltanet_layer_indices)
    labels = ["teacher", "student"]
    c.bar(labels, [t_deltanet, s_deltanet], color=_DELTANET, width=0.55, label="DeltaNet")
    c.bar(labels, [t_attention, s_attention], bottom=[t_deltanet, s_deltanet],
          color=_ATTENTION, width=0.55, label="full attention")
    for index, (deltanet, attention) in enumerate([(t_deltanet, t_attention),
                                                   (s_deltanet, s_attention)]):
        c.text(index, deltanet / 2, str(deltanet), ha="center", va="center",
               color="white", fontsize=profile.font_size - 0.5)
        c.text(index, deltanet + attention / 2, str(attention), ha="center", va="center",
               color="white", fontsize=profile.font_size - 0.5)
        c.text(index, deltanet + attention + 1.2, f"{deltanet + attention} layers",
               ha="center", fontsize=profile.font_size - 1.0)
    c.set_ylabel("layers")
    c.set_ylim(0, t_spec.num_hidden_layers * 1.22)
    c.set_title("c  Depth by block type")
    c.legend(loc="upper right", handlelength=1.1, borderaxespad=0.2)

    # (d) MoE sparsity: FFN width available against FFN width used per token
    teacher_ffn = t_spec.intermediate_size
    student_available = spec.num_experts * spec.moe_intermediate_size
    student_active = (spec.num_experts_per_tok * spec.moe_intermediate_size
                      + spec.shared_expert_intermediate_size)
    d.bar(["teacher\ndense FFN"], [teacher_ffn], color=COLOURS[0], width=0.55)
    d.bar(["student\nMoE"], [student_available], color="#c8cdd4", width=0.55,
          label=f"{spec.num_experts} experts available")
    d.bar(["student\nMoE"], [student_active], color=COLOURS[2], width=0.55,
          label=f"top-{spec.num_experts_per_tok} + shared, per token")
    d.text(0, teacher_ffn * 1.03, f"{teacher_ffn:,}", ha="center",
           fontsize=profile.font_size - 1.0)
    d.text(1, student_available * 1.03, f"{student_available:,}", ha="center",
           fontsize=profile.font_size - 1.0)
    d.text(1, student_active * 1.06, f"{student_active:,}", ha="center", color=COLOURS[2],
           fontsize=profile.font_size - 1.0)
    d.set_ylabel("FFN hidden units")
    d.set_ylim(0, teacher_ffn * 1.24)
    d.set_title("d  FFN width and MoE sparsity")
    d.legend(loc="upper right", handlelength=1.1, borderaxespad=0.2)

    for ax in axes:
        grid(ax, axis="y")

    fig.suptitle(
        f"{t_spec.name} → {spec.name}: same hidden size ({spec.hidden_size}), "
        f"different computational topology",
        y=1.015, fontsize=profile.font_size + 1.0, x=0.005, ha="left",
    )
    fig.tight_layout()
    return save(fig, "model_compression", profile=profile, provenance=Provenance(
        figure_id="F01",
        sources=("qwen_distill.architecture.moe_student:audit",
                 "qwen_distill.architecture.presets:get_spec('teacher')",
                 "qwen_distill.architecture.params:count_parameters"),
        metrics=("exact_parameter_count", "active_parameters_per_token",
                 "num_hidden_layers", "num_full_attention_layers", "num_experts",
                 "num_experts_per_tok", "moe_intermediate_size", "intermediate_size"),
        value_kind=AUDITED, data_commit=repo_commit(),
        note="counted from the frozen specification at plot time; no measurement involved",
        extra={"teacher_total_parameters": t_total, "student_total_parameters": s_total,
               "student_active_parameters_per_token": s_active},
    ))


# ---------------------------------------------------------------------------
# F02
# ---------------------------------------------------------------------------
def parameter_counts(profile: Profile) -> list:
    """F02 — the three parameter counts that get conflated, side by side.

    Stored parameters set VRAM. Active parameters set per-token compute. Reporting an MoE
    model with one number hides which of the two a claim is about, so all three are drawn
    and each is labelled with what it costs.
    """
    style(profile)
    student = student_facts()
    teacher = teacher_facts()
    report = student["audit"]
    values = [teacher["total"] / BILLION,
              report["exact_parameter_count"] / BILLION,
              report["active_parameters_per_token"] / BILLION]
    labels = ["teacher\nstored", "student\nstored", "student\nactive / token"]

    fig, ax = figure(profile, width=0.86)
    bars = ax.bar(labels, values, color=[COLOURS[0], COLOURS[2], COLOURS[3]], width=0.58)
    for bar, value, exact in zip(bars, values,
                                 [teacher["total"], report["exact_parameter_count"],
                                  report["active_parameters_per_token"]], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.02,
                f"{value:.3f}B", ha="center", va="bottom",
                fontsize=profile.font_size)
        if profile.annotate:
            ax.text(bar.get_x() + bar.get_width() / 2, value * 0.5, f"{exact:,}",
                    ha="center", va="center", rotation=90, color="white",
                    fontsize=profile.font_size - 1.5)
    ax.set_ylabel("parameters (billions)")
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_title("Parameter counts: stored is what costs VRAM, active is what costs compute")
    grid(ax, axis="y")
    fig.tight_layout()
    return save(fig, "parameter_counts", profile=profile, provenance=Provenance(
        figure_id="F02",
        sources=("qwen_distill.architecture.moe_student:audit",
                 "qwen_distill.architecture.presets:get_spec('teacher')"),
        metrics=("exact_parameter_count", "active_parameters_per_token"),
        value_kind=AUDITED, data_commit=repo_commit(),
        extra={"teacher_total": teacher["total"],
               "student_total": report["exact_parameter_count"],
               "student_active_per_token": report["active_parameters_per_token"]},
    ))


# ---------------------------------------------------------------------------
# F16
# ---------------------------------------------------------------------------
def layer_mapping(profile: Profile) -> list:
    """F16 — the teacher→student correspondence the implementation actually uses.

    Read off :func:`qwen_distill.architecture.moe_init.map_layers` and
    :func:`qwen_distill.distillation.behavioral.layer_spans`, not drawn by hand. The two
    together are the figure's point: layer KD supervises the anchors (the dots), while the
    behavioural objective supervises the spans (the bars), which is where the sixteen
    dropped teacher layers go.
    """
    style(profile)
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from qwen_distill.architecture.moe_init import map_layers
    from qwen_distill.architecture.moe_student import FROZEN_STUDENT
    from qwen_distill.distillation.behavioral import layer_spans

    mapping = map_layers(FROZEN_STUDENT, teacher_layers=64)
    spans = layer_spans(mapping.mapping, 64)
    students = sorted(mapping.mapping)

    fig, ax = figure(profile, width=1.02, height=1.5)
    for student in students:
        start, end = spans[student]
        ax.barh(student, end - start, left=start, height=0.72,
                color="#dfe3e8", edgecolor="none", zorder=1)
    for student in students:
        teacher = mapping.mapping[student]
        is_attention = mapping.student_types[student] == "full_attention"
        ax.plot([teacher + 0.5], [student], marker="s" if is_attention else "o",
                color=_ATTENTION if is_attention else _DELTANET, markersize=3.2, zorder=3)
    for dropped in mapping.removed_teacher_layers:
        ax.axvspan(dropped, dropped + 1, color=_ATTENTION, alpha=0.055, zorder=0)

    ax.set_xlim(0, 64)
    ax.set_ylim(-1, 48)
    ax.set_xlabel("teacher layer index (64 layers)")
    ax.set_ylabel("student layer index (48 layers)")
    ax.set_xticks(range(0, 65, 8))
    ax.set_yticks(range(0, 49, 8))
    ax.set_title(f"Layer mapping, strategy '{mapping.strategy}': "
                 f"{len(mapping.removed_teacher_layers)} teacher layers have no student anchor")
    grid(ax, axis="both")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=_DELTANET, markersize=4,
               label="DeltaNet anchor (layer-KD pair)"),
        Line2D([], [], marker="s", linestyle="none", color=_ATTENTION, markersize=4,
               label="full-attention anchor (layer-KD pair)"),
        Patch(facecolor="#dfe3e8", label="teacher span charged to that student layer"),
        Patch(facecolor=_ATTENTION, alpha=0.12, label="teacher layer with no anchor"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.99))
    if profile.annotate:
        ax.text(0.99, 0.02,
                "spans tile the teacher's depth: every teacher layer is charged to exactly "
                "one student layer",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=profile.font_size - 1.5, color="#555555")
    fig.tight_layout()
    return save(fig, "layer_mapping", profile=profile, provenance=Provenance(
        figure_id="F16",
        sources=("qwen_distill.architecture.moe_init:map_layers",
                 "qwen_distill.distillation.behavioral:layer_spans"),
        metrics=("mapping", "removed_teacher_layers", "student_types", "spans"),
        value_kind=DESIGN, data_commit=repo_commit(),
        note=f"strategy={mapping.strategy}; type-consistency problems: "
             f"{len(mapping.problems)}",
        extra={"removed_teacher_layers": mapping.removed_teacher_layers,
               "n_pairs": len(mapping.mapping)},
    ))
