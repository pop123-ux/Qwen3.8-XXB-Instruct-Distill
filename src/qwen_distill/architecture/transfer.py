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

LayerSelection = Literal["first", "last", "uniform", "interleave", "group"]
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
    teacher_layers: int,
    student_layers: int,
    strategy: LayerSelection = "uniform",
    *,
    group_size: int = 1,
) -> dict[int, int]:
    """Map each student layer index to a teacher layer index.

    ``uniform`` spreads the selection evenly, which preserves the depth-wise progression
    of representations; ``first``/``last`` keep one end. All are guesses until measured —
    that is the point of making the strategy a parameter.

    ``group`` is ``uniform`` applied to whole *hybrid groups* rather than to individual
    layers, and it exists because the others cannot both span the depth and respect the
    layout. In a period-``g`` hybrid (here ``g = full_attention_interval``) a layer's
    position within its group determines its block type, so a selection that spans the
    teacher's depth at a non-integer stride lands student layers on teacher layers of the
    wrong type — measured at 64 -> 48: ``uniform`` puts 8 of 28 layers on the wrong block
    type, while ``interleave`` degenerates to ``first`` because ``64 // 48 == 1``. Moving
    the selection up to the group and copying position-for-position within it keeps every
    student layer on a teacher layer of its own type *and* keeps the depth-wise spread.
    Whether that initialises a better student is still an empirical question; what it
    stops being is structurally broken.
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
    if strategy == "group":
        if group_size < 1:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if teacher_layers % group_size or student_layers % group_size:
            raise ValueError(
                f"group selection needs whole groups: {teacher_layers} teacher and "
                f"{student_layers} student layers must both be divisible by "
                f"group_size={group_size}. A partial group would put its layers at the "
                "wrong position in the hybrid cycle, which is the failure this strategy "
                "exists to avoid."
            )
        group_map = select_layers(
            teacher_layers // group_size, student_layers // group_size, "uniform"
        )
        return {
            s * group_size + offset: group_map[s] * group_size + offset
            for s in range(student_layers // group_size)
            for offset in range(group_size)
        }
    raise ValueError(f"unknown layer selection {strategy!r}")


def _group_size(teacher: HybridArchSpec, student: HybridArchSpec) -> int:
    """The hybrid period both models must share for group-aligned selection.

    A layer's block type is decided by its position in the period, so aligning groups is
    only meaningful when the two models have the same period. Different periods are a
    different layout, not a scaled one, and the error says so rather than producing a
    map that ``_layout_compatible`` would then reject tensor by tensor.
    """
    if teacher.full_attention_interval != student.full_attention_interval:
        raise ValueError(
            f"group selection needs a shared hybrid period, but the teacher's "
            f"full_attention_interval is {teacher.full_attention_interval} and the "
            f"student's is {student.full_attention_interval}. Group alignment cannot "
            "reconcile different layouts; choose a student with the teacher's interval, "
            "or a selection strategy that does not claim layout preservation."
        )
    return teacher.full_attention_interval


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
        teacher.num_hidden_layers,
        student.num_hidden_layers,
        layer_selection,
        group_size=_group_size(teacher, student) if layer_selection == "group" else 1,
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
    selections: tuple[LayerSelection, ...] = ("first", "last", "uniform", "interleave", "group"),
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

def student_from_teacher(
    teacher: HybridArchSpec,
    *,
    name: str = "student",
    hidden_size: int | None = None,
    num_hidden_layers: int | None = None,
    intermediate_size: int | None = None,
    num_key_value_heads: int | None = None,
    linear_num_key_heads: int | None = None,
    tie_word_embeddings: bool | None = None,
    max_position_embeddings: int | None = None,
) -> HybridArchSpec:
    """A student the teacher can actually be transferred into.

    Every field a transfer cannot reduce is inherited rather than offered as a knob:
    ``head_dim``, the DeltaNet head dimensions, the conv kernel, the hybrid period and —
    the important one — the vocabulary. Choosing a student's vocabulary independently is
    what makes logit distillation need a token mapping and makes embedding transfer
    meaningless, so this signature does not let it happen by accident.

    Head *counts* are knobs, but only through the group-defining ones: give
    ``num_key_value_heads`` and the query heads follow at the teacher's GQA ratio; give
    ``linear_num_key_heads`` and the value heads follow at the teacher's ratio. That keeps
    both ratios fixed by construction instead of by a later check.
    """
    gqa = teacher.num_attention_heads // teacher.num_key_value_heads
    dn_ratio = teacher.linear_num_value_heads // teacher.linear_num_key_heads
    kv_heads = num_key_value_heads or teacher.num_key_value_heads
    key_heads = linear_num_key_heads or teacher.linear_num_key_heads

    layers = num_hidden_layers or teacher.num_hidden_layers
    if layers % teacher.full_attention_interval:
        raise ValueError(
            f"num_hidden_layers={layers} is not a whole number of "
            f"{teacher.full_attention_interval}-layer hybrid groups. A partial group puts "
            "its layers at the wrong position in the cycle and cannot be transferred into."
        )
    return HybridArchSpec(
        name=name,
        hidden_size=hidden_size or teacher.hidden_size,
        num_hidden_layers=layers,
        intermediate_size=intermediate_size or teacher.intermediate_size,
        vocab_size=teacher.vocab_size,
        tie_word_embeddings=(
            teacher.tie_word_embeddings if tie_word_embeddings is None else tie_word_embeddings
        ),
        num_attention_heads=kv_heads * gqa,
        num_key_value_heads=kv_heads,
        head_dim=teacher.head_dim,
        attention_bias=teacher.attention_bias,
        partial_rotary_factor=teacher.partial_rotary_factor,
        attn_output_gate=teacher.attn_output_gate,
        linear_num_value_heads=key_heads * dn_ratio,
        linear_num_key_heads=key_heads,
        linear_key_head_dim=teacher.linear_key_head_dim,
        linear_value_head_dim=teacher.linear_value_head_dim,
        linear_conv_kernel_dim=teacher.linear_conv_kernel_dim,
        full_attention_interval=teacher.full_attention_interval,
        max_position_embeddings=max_position_embeddings or teacher.max_position_embeddings,
        provenance=f"derived from {teacher.name!r} by qwen_distill.architecture.transfer",
    )
