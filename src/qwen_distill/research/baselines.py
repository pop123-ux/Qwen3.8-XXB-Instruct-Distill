"""The historical dense candidate, kept as a baseline rather than deleted.

Before the direction reset the project's leading candidate was a **dense, depth-only**
student: `h5120 L40`, 17.76B parameters, keeping every one of the teacher's widths and head
counts and changing only depth. It was not wrong, and it is not superseded evidence — it is
the natural control for the sparse MoE target, because the two differ in exactly two things:

    ==================  ====================  ====================
    property            dense_h5120_l40       frozen MoE student
    ==================  ====================  ====================
    layers              40                    48
    FFN                 dense, 17408 wide     8 experts x 768, top-2
    hidden size         5120                  5120
    vocabulary          248,320               248,320
    head_dim            256                   256
    hybrid pattern      3 DeltaNet : 1 attn   3 DeltaNet : 1 attn
    ==================  ====================  ====================

**The project's first scientific comparison is these two models**, distilled from the same
teacher under the same objective. Everything shared between them is shared exactly, so a
difference is attributable to sparsity and depth rather than to a dozen simultaneous
changes.

Why the dense candidate is a *good* control rather than merely an old one: its transfer
plan from the teacher is **533 tensor copies, 100% coverage, zero warnings** — no slicing,
no head subsetting, no width reduction anywhere. Every retained tensor is the teacher's bit
for bit. That removes the `slice`-baseline assumption (that a teacher's parameters are
ordered by importance, which nothing guarantees) from the comparison entirely, so a result
cannot be blamed on a questionable width-reduction heuristic.

Both candidates fit 16 GB: the dense baseline at 13.18 GiB at 32K, the corrected sparse
student at 10.21 GiB at the same context. The sparse student is the smaller of the two after
the expert-budget correction, so the comparison is no longer "more parameters, fewer active"
— it is fewer of both, which is a cleaner question.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..architecture.presets import get_spec
from ..architecture.spec import HybridArchSpec
from ..architecture.transfer import student_from_teacher

#: Teacher preset name, so this module and the presets registry cannot disagree.
TEACHER_PRESET = "teacher"


@dataclass(frozen=True)
class Baseline:
    """A retained candidate and what it is a baseline *for*."""

    name: str
    spec: HybridArchSpec
    role: str
    status: str
    evidence: str

    @property
    def parameters(self) -> int:
        from ..architecture.params import count_parameters

        return count_parameters(self.spec).total

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "role": self.role, "status": self.status,
            "evidence": self.evidence, "parameters": self.parameters,
            "hidden_size": self.spec.hidden_size,
            "num_hidden_layers": self.spec.num_hidden_layers,
            "intermediate_size": self.spec.intermediate_size,
            "num_key_value_heads": self.spec.num_key_value_heads,
        }


def dense_h5120_l40() -> HybridArchSpec:
    """The historical candidate, derived from the teacher rather than hard-coded.

    Deriving it means it inherits the teacher's head dimensions, conv kernel, hybrid period
    and vocabulary by construction, so it stays transfer-compatible even if the teacher
    preset is corrected.
    """
    return student_from_teacher(
        get_spec(TEACHER_PRESET),
        name="dense_h5120_l40",
        num_hidden_layers=40,
    )


def baselines() -> dict[str, Baseline]:
    return {
        "dense_h5120_l40": Baseline(
            name="dense_h5120_l40",
            spec=dense_h5120_l40(),
            role="control for the sparse MoE student: same teacher, same width, same "
                 "vocabulary, differing only in depth (40 vs 48) and in dense vs sparse FFN",
            status="historical candidate — retained as a baseline, not a target",
            evidence="transfers from the teacher as 533 pure tensor copies, 100% coverage, "
                     "zero warnings, no width reduction; fits 16 GB at 13.18 GiB at 32K",
        ),
    }


def comparison() -> dict[str, Any]:
    """The first scientific comparison, stated so it can be run rather than described."""
    from ..architecture.moe_student import FROZEN_STUDENT, audit

    dense = baselines()["dense_h5120_l40"]
    sparse = audit(FROZEN_STUDENT)
    return {
        "name": "dense L40 against sparse-MoE L48",
        "baseline": dense.to_dict(),
        "candidate": {
            "name": FROZEN_STUDENT.name,
            "parameters": sparse["exact_parameter_count"],
            "active_parameters_per_token": sparse["active_parameters_per_token"],
            "num_hidden_layers": FROZEN_STUDENT.num_hidden_layers,
            "num_experts": FROZEN_STUDENT.num_experts,
            "num_experts_per_tok": FROZEN_STUDENT.num_experts_per_tok,
        },
        "held_constant": [
            "teacher and teacher revision", "hidden size 5120", "vocabulary 248,320",
            "head_dim 256", "hybrid pattern 3 DeltaNet : 1 full attention",
            "distillation objective and token budget",
        ],
        "varies": ["depth 40 vs 48", "dense FFN vs 8-expert top-2 MoE"],
        "question": (
            "Does routing 13.01B stored parameters at 9.61B active per token beat 17.76B "
            "dense at the same width, from the same teacher, under the same objective? The "
            "sparse student is smaller on both counts, so a win would be a win on capability "
            "per parameter and not merely on capacity."
        ),
        "why_it_is_fair": (
            "The dense candidate transfers as pure copies with no width reduction, so its "
            "result cannot be attributed to a slicing heuristic. Both students inherit every "
            "teacher field that a transfer cannot reduce."
        ),
        "status": "not run — neither model has been distilled",
    }
