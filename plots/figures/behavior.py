"""Behavioural / state alignment — the paper's central measurement, once it exists.

The question is not "does the student score well" but "does it do the same work". Five
signals, each with its own scale, each needing a probe that runs the real teacher and the
materialised student over the same batch and records their agreement.

No such probe has been run, so this refuses. One of the five is additionally blocked at the
method level rather than by compute: the DeltaNet recurrent state has 16 key heads and 48
value heads on the student against the teacher's own shape, and comparing them needs a
projection that would itself be an untested modelling choice — see
:data:`qwen_distill.distillation.behavioral.DELTANET_STATE`. The refusal says so, because
"we have not measured it" and "this quantity is not defined yet" are different problems and
a reader planning the experiment needs to know which one they face.
"""
from __future__ import annotations

from common import ROOT, MissingData, Profile, Provenance, figure, grid, save, style

ALIGNMENT_ARTIFACT = ROOT / "experiments" / "alignment.json"

SIGNALS = (
    ("logit_similarity", "logits"),
    ("hidden_state_similarity", "hidden states"),
    ("attention_similarity", "attention maps"),
    ("deltanet_state_similarity", "DeltaNet recurrent state"),
    ("moe_reconstruction_error", "FFN / MoE reconstruction"),
)


def behavior_state_alignment(profile: Profile) -> list:
    """F15 — per-signal teacher/student agreement."""
    style(profile)
    if not ALIGNMENT_ARTIFACT.exists():
        raise MissingData(
            "teacher/student behavioural alignment",
            f"run an alignment probe that scores the real teacher and the materialised "
            f"student on one batch and writes {ALIGNMENT_ARTIFACT.relative_to(ROOT)} with "
            f"a 'signals' object keyed by {', '.join(k for k, _ in SIGNALS)}. Note that "
            f"deltanet_state_similarity is not defined yet — the shapes differ and the "
            f"projection is an open modelling choice (behavioral.DELTANET_STATE)",
        )
    import json

    data = json.loads(ALIGNMENT_ARTIFACT.read_text(encoding="utf-8"))
    signals = data["signals"]
    present = [(key, label) for key, label in SIGNALS if signals.get(key) is not None]
    if not present:
        raise MissingData("any behavioural alignment signal",
                          f"{ALIGNMENT_ARTIFACT.relative_to(ROOT)} carries no scored signal")
    missing = [label for key, label in SIGNALS if signals.get(key) is None]

    fig, ax = figure(profile, width=1.0)
    labels = [label for _, label in present][::-1]
    values = [signals[key] for key, _ in present][::-1]
    ax.barh(labels, values, color="#1a1a1a", height=0.6)
    for index, value in enumerate(values):
        ax.text(value + 0.01, index, f"{value:.3f}", va="center",
                fontsize=profile.font_size - 1.0)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("similarity to the teacher's signal (1.0 = identical)")
    ax.set_title("Which of the teacher's computations the student reproduces")
    grid(ax, axis="x")
    if missing:
        ax.text(0.0, 1.012, "not measured: " + ", ".join(missing),
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=profile.font_size - 1.5, color="#c1553b")
    fig.tight_layout()
    return save(fig, "behavior_state_alignment", profile=profile, provenance=Provenance(
        figure_id="F15",
        experiments=(data.get("experiment_id", "alignment"),),
        sources=(str(ALIGNMENT_ARTIFACT.relative_to(ROOT)),),
        metrics=tuple(key for key, _ in present),
        data_commit=data.get("git_commit"),
        extra={"unmeasured_signals": missing},
    ))
