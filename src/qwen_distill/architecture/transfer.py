"""Teacher-to-student weight transfer: a research API, not a chosen method.

Initialising a student from teacher weights is plausibly much cheaper than training from
scratch, but *how* to do it is an open question with several defensible answers and no
obvious winner. So this module provides strategies that can be compared, not one
strategy presented as correct.

An explicit warning against the obvious approach: **naive slicing is a baseline, not a
solution.** Taking the first N rows of a projection assumes the teacher's parameters are
ordered by importance, which nothing guarantees. Head-importance selection and
merging are included precisely so slicing has something to lose to.

Nothing here trains anything. It produces a *plan* — a mapping from teacher tensors to
student tensors — which can be inspected, tested, and applied when weights are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .spec import FULL_ATTENTION, HybridArchSpec

LayerSelection = Literal["first", "last", "uniform", "interleave"]
WidthReduction = Literal["slice", "mean_pool", "importance"]


@dataclass
class TensorMapping:
    """How one student tensor is produced from teacher tensor(s)."""

    student_name: str
    teacher_names: list[str]
    operation: str
    student_shape: list[int]
    teacher_shape: list[int] | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class TransferPlan:
    """A complete, inspectable initialisation plan."""

    strategy: str
    teacher: str
    student: str
    layer_map: dict[int, int] = field(default_factory=dict)
    mappings: list[TensorMapping] = field(default_factory=list)
    randomly_initialised: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of student tensors that receive teacher weights."""
        total = len(self.mappings) + len(self.randomly_initialised)
        return len(self.mappings) / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "teacher": self.teacher,
            "student": self.student,
            "layer_map": {str(k): v for k, v in self.layer_map.items()},
            "coverage": self.coverage,
            "n_mapped": len(self.mappings),
            "n_random": len(self.randomly_initialised),
            "mappings": [m.to_dict() for m in self.mappings],
            "randomly_initialised": self.randomly_initialised,
            "warnings": self.warnings,
        }


def select_layers(
    teacher_layers: int, student_layers: int, strategy: LayerSelection = "uniform"
) -> dict[int, int]:
    """Map each student layer index to a teacher layer index.

    ``uniform`` spreads the selection evenly, which preserves the depth-wise progression
    of representations; ``first``/``last`` keep one end. All are guesses until measured —
    that is the point of making the strategy a parameter.
    """
    if student_layers > teacher_layers:
        raise ValueError(
            f"student has more layers ({student_layers}) than the teacher "
            f"({teacher_layers}); transfer cannot invent depth"
        )
    if strategy == "first":
        return {i: i for i in range(student_layers)}
    if strategy == "last":
        offset = teacher_layers - student_layers
        return {i: i + offset for i in range(student_layers)}
    if strategy == "uniform":
        if student_layers == 1:
            return {0: teacher_layers // 2}
        step = (teacher_layers - 1) / (student_layers - 1)
        return {i: round(i * step) for i in range(student_layers)}
    if strategy == "interleave":
        stride = teacher_layers // student_layers
        return {i: i * stride for i in range(student_layers)}
    raise ValueError(f"unknown layer selection {strategy!r}")


def _layout_compatible(teacher: HybridArchSpec, student: HybridArchSpec, layer_map: dict[int, int]) -> list[str]:
    """Warn when a mapped teacher layer is a different block type than the student's.

    A DeltaNet layer's weights are meaningless in an attention slot: the tensors do not
    even have the same names. Silently mapping across types would produce a plan that
    fails at apply time, or worse, half-applies.
    """
    teacher_types = teacher.resolved_layer_types()
    student_types = student.resolved_layer_types()
    mismatches = [
        f"student layer {s} is {student_types[s]} but maps to teacher layer {t} "
        f"which is {teacher_types[t]}"
        for s, t in layer_map.items()
        if student_types[s] != teacher_types[t]
    ]
    return mismatches


def build_transfer_plan(
    teacher: HybridArchSpec,
    student: HybridArchSpec,
    *,
    layer_selection: LayerSelection = "uniform",
    width_reduction: WidthReduction = "slice",
) -> TransferPlan:
    """Plan how to initialise ``student`` from ``teacher``.

    Produces a mapping only; nothing is loaded or written. Tensors that cannot be
    derived are listed under ``randomly_initialised`` rather than silently omitted, so
    the coverage figure is honest.
    """
    plan = TransferPlan(
        strategy=f"layers={layer_selection}, width={width_reduction}",
        teacher=teacher.name, student=student.name,
    )

    if student.vocab_size != teacher.vocab_size:
        plan.warnings.append(
            f"vocabulary differs ({student.vocab_size} vs {teacher.vocab_size}): embedding "
            "transfer is invalid, and logit distillation would need a token mapping. "
            "Keeping the teacher's tokenizer avoids both problems."
        )

    plan.layer_map = select_layers(
        teacher.num_hidden_layers, student.num_hidden_layers, layer_selection
    )
    for mismatch in _layout_compatible(teacher, student, plan.layer_map):
        plan.warnings.append(mismatch)

    same_width = student.hidden_size == teacher.hidden_size
    if not same_width and width_reduction == "slice":
        plan.warnings.append(
            f"hidden size differs ({student.hidden_size} vs {teacher.hidden_size}) and "
            "width_reduction='slice' takes leading rows. That assumes teacher parameters "
            "are ordered by importance, which nothing guarantees — treat it as a baseline "
            "to beat, not a method."
        )

    # --- embeddings -------------------------------------------------------
    if student.vocab_size == teacher.vocab_size:
        plan.mappings.append(TensorMapping(
            "model.embed_tokens.weight", ["model.embed_tokens.weight"],
            "copy" if same_width else f"{width_reduction} to hidden {student.hidden_size}",
            [student.vocab_size, student.hidden_size],
            [teacher.vocab_size, teacher.hidden_size],
        ))
        if not student.tie_word_embeddings:
            source = "lm_head.weight" if not teacher.tie_word_embeddings else "model.embed_tokens.weight"
            plan.mappings.append(TensorMapping(
                "lm_head.weight", [source],
                "copy" if same_width else f"{width_reduction} to hidden {student.hidden_size}",
                [student.vocab_size, student.hidden_size],
                notes="teacher ties embeddings; the embedding is reused for the head"
                if teacher.tie_word_embeddings else "",
            ))
        else:
            plan.mappings.append(TensorMapping(
                "lm_head.weight", ["model.embed_tokens.weight"], "tied to embedding",
                [student.vocab_size, student.hidden_size],
                notes="student ties embeddings: the teacher embedding fills both roles",
            ))
    else:
        plan.randomly_initialised += ["model.embed_tokens.weight", "lm_head.weight"]

    # --- per-layer --------------------------------------------------------
    student_types = student.resolved_layer_types()
    teacher_types = teacher.resolved_layer_types()
    for s_index, t_index in plan.layer_map.items():
        prefix = f"model.layers.{s_index}"
        t_prefix = f"model.layers.{t_index}"
        if student_types[s_index] != teacher_types[t_index]:
            plan.randomly_initialised.append(f"{prefix}.* (block type mismatch)")
            continue

        for norm in ("input_layernorm", "post_attention_layernorm"):
            plan.mappings.append(TensorMapping(
                f"{prefix}.{norm}.weight", [f"{t_prefix}.{norm}.weight"],
                "copy" if same_width else width_reduction, [student.hidden_size],
            ))
        for proj, shape in (
            ("mlp.gate_proj", [student.intermediate_size, student.hidden_size]),
            ("mlp.up_proj", [student.intermediate_size, student.hidden_size]),
            ("mlp.down_proj", [student.hidden_size, student.intermediate_size]),
        ):
            plan.mappings.append(TensorMapping(
                f"{prefix}.{proj}.weight", [f"{t_prefix}.{proj}.weight"],
                "copy" if (same_width and student.intermediate_size == teacher.intermediate_size)
                else width_reduction, shape,
            ))

        if student_types[s_index] == FULL_ATTENTION:
            gate = 2 if student.attn_output_gate else 1
            for proj, shape in (
                ("self_attn.q_proj", [student.num_attention_heads * student.head_dim * gate,
                                      student.hidden_size]),
                ("self_attn.k_proj", [student.num_key_value_heads * student.head_dim,
                                      student.hidden_size]),
                ("self_attn.v_proj", [student.num_key_value_heads * student.head_dim,
                                      student.hidden_size]),
                ("self_attn.o_proj", [student.hidden_size,
                                      student.num_attention_heads * student.head_dim]),
            ):
                operation = "copy"
                if student.num_attention_heads != teacher.num_attention_heads:
                    operation = f"head selection ({width_reduction})"
                plan.mappings.append(TensorMapping(
                    f"{prefix}.{proj}.weight", [f"{t_prefix}.{proj}.weight"], operation, shape,
                ))
            for norm in ("self_attn.q_norm", "self_attn.k_norm"):
                plan.mappings.append(TensorMapping(
                    f"{prefix}.{norm}.weight", [f"{t_prefix}.{norm}.weight"],
                    "copy" if student.head_dim == teacher.head_dim else width_reduction,
                    [student.head_dim],
                ))
        else:
            operation = (
                "copy" if student.linear_num_value_heads == teacher.linear_num_value_heads
                else f"head subset ({width_reduction})"
            )
            for proj, shape in (
                ("linear_attn.in_proj_qkv", [student.linear_conv_dim, student.hidden_size]),
                ("linear_attn.in_proj_z", [student.linear_value_dim, student.hidden_size]),
                ("linear_attn.in_proj_b", [student.linear_num_value_heads, student.hidden_size]),
                ("linear_attn.in_proj_a", [student.linear_num_value_heads, student.hidden_size]),
                ("linear_attn.out_proj", [student.hidden_size, student.linear_value_dim]),
            ):
                plan.mappings.append(TensorMapping(
                    f"{prefix}.{proj}.weight", [f"{t_prefix}.{proj}.weight"], operation, shape,
                ))
            for name, shape in (
                ("linear_attn.conv1d.weight",
                 [student.linear_conv_dim, 1, student.linear_conv_kernel_dim]),
                ("linear_attn.dt_bias", [student.linear_num_value_heads]),
                ("linear_attn.A_log", [student.linear_num_value_heads]),
                ("linear_attn.norm.weight", [student.linear_value_head_dim]),
            ):
                plan.mappings.append(TensorMapping(
                    f"{prefix}.{name}", [f"{t_prefix}.{name}"], operation, shape,
                ))

    plan.mappings.append(TensorMapping(
        "model.norm.weight", ["model.norm.weight"],
        "copy" if same_width else width_reduction, [student.hidden_size],
    ))
    return plan


def compare_strategies(
    teacher: HybridArchSpec,
    student: HybridArchSpec,
    selections: tuple[LayerSelection, ...] = ("first", "last", "uniform", "interleave"),
) -> dict[str, TransferPlan]:
    """Build a plan per layer-selection strategy, for side-by-side comparison.

    Which one initialises a better student is an empirical question. This makes the
    candidates concrete so the experiment can be run rather than argued about.
    """
    plans: dict[str, TransferPlan] = {}
    for selection in selections:
        try:
            plans[selection] = build_transfer_plan(teacher, student, layer_selection=selection)
        except ValueError as exc:
            plan = TransferPlan(strategy=selection, teacher=teacher.name, student=student.name)
            plan.warnings.append(str(exc))
            plans[selection] = plan
    return plans
