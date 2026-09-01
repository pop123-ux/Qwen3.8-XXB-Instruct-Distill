#!/usr/bin/env python3
"""Teacher against student: parameters, active parameters, depth, expert sparsity.

Every number is computed at plot time from the frozen specification and the teacher preset.
Nothing is typed in, so the figure cannot drift from the architecture.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import COLOURS, save, student_facts, style  # noqa: E402


def main() -> int:
    style()
    import matplotlib.pyplot as plt

    from qwen_distill.architecture.params import count_parameters
    from qwen_distill.architecture.presets import get_spec

    facts = student_facts()
    spec, report = facts["spec"], facts["audit"]
    teacher = get_spec("teacher")
    teacher_total = count_parameters(teacher).total
    student_total = report["exact_parameter_count"]
    student_active = report["active_parameters_per_token"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.8))

    labels = ["teacher\n(stored)", "student\n(stored)", "student\n(active/token)"]
    values = [teacher_total / 1e9, student_total / 1e9, student_active / 1e9]
    bars = left.bar(labels, values, color=[COLOURS[0], COLOURS[1], COLOURS[2]], width=0.62)
    for bar, value in zip(bars, values, strict=True):
        left.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.2f}B",
                  ha="center", fontsize=8)
    left.set_ylabel("parameters (billions)")
    left.set_title("Stored parameters are what cost VRAM")
    left.set_ylim(0, teacher_total / 1e9 * 1.18)
    left.text(0.5, 0.92,
              f"student is {student_total / teacher_total:.0%} of the teacher; "
              f"{student_active / student_total:.0%} of it runs per token",
              transform=left.transAxes, ha="center", fontsize=7.5, color="#444444")

    components = report["components"]
    order = ["routed_experts", "deltanet", "embedding", "lm_head", "attention",
             "shared_expert", "router", "norms"]
    shown = [(k, components[k]) for k in order if components.get(k, 0) > 0]
    right.barh([k.replace("_", " ") for k, _ in shown][::-1],
               [v / 1e9 for _, v in shown][::-1], color=COLOURS[0], height=0.62)
    right.set_xlabel("parameters (billions)")
    right.set_title(f"Where the {student_total / 1e9:.2f}B sits")
    for index, (_, value) in enumerate(shown[::-1]):
        if value / student_total > 0.03:
            right.text(value / 1e9 + 0.05, index, f"{100 * value / student_total:.1f}%",
                       va="center", fontsize=7.5)

    fig.suptitle(
        f"{spec.name}: {spec.num_hidden_layers} layers "
        f"({len(spec.deltanet_layer_indices)} DeltaNet + {len(spec.attention_layer_indices)} "
        f"attention), {spec.num_experts} experts x {spec.moe_intermediate_size}, "
        f"top-{spec.num_experts_per_tok}",
        fontsize=9.5, y=1.02,
    )
    save(fig, "architecture", paper=True,
         source=f"computed from FROZEN_STUDENT and preset 'teacher' via audit(); "
                f"teacher {teacher_total:,}, student {student_total:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
