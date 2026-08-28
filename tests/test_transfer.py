"""Tests for the teacher-to-student weight transfer research API.

These pin behaviour, not a chosen method: which initialisation strategy is best is an
open empirical question. What must not happen is a plan that silently maps incompatible
tensors — that would fail at apply time, or half-apply, which is worse.
"""

from __future__ import annotations

import json

import pytest

from qwen_distill.architecture.spec import HybridArchSpec
from qwen_distill.architecture.transfer import (
    build_transfer_plan,
    compare_strategies,
    select_layers,
    student_from_teacher,
)

TEACHER = HybridArchSpec(name="teacher")  # 64 layers, 3:1 hybrid


def student(layers: int = 48, **overrides) -> HybridArchSpec:
    base = dict(
        name="student", hidden_size=5120, num_hidden_layers=layers, intermediate_size=17408,
        vocab_size=248320, num_attention_heads=24, num_key_value_heads=4, head_dim=256,
        linear_num_key_heads=16, linear_num_value_heads=48, tie_word_embeddings=True,
    )
    base.update(overrides)
    return HybridArchSpec(**base)


# --- layer selection ------------------------------------------------------
def test_first_and_last_select_contiguous_blocks():
    assert select_layers(64, 4, "first") == {0: 0, 1: 1, 2: 2, 3: 3}
    assert select_layers(64, 4, "last") == {0: 60, 1: 61, 2: 62, 3: 63}


def test_uniform_spans_the_full_depth():
    mapping = select_layers(64, 4, "uniform")
    assert mapping[0] == 0
    assert mapping[3] == 63


def test_interleave_uses_a_constant_stride():
    mapping = select_layers(64, 8, "interleave")
    strides = {mapping[i + 1] - mapping[i] for i in range(7)}
    assert len(strides) == 1


def test_cannot_transfer_into_a_deeper_student():
    with pytest.raises(ValueError, match="cannot invent depth"):
        select_layers(24, 48, "uniform")


def test_unknown_selection_is_rejected():
    with pytest.raises(ValueError, match="unknown layer selection"):
        select_layers(64, 8, "telepathy")


# --- layout compatibility -------------------------------------------------
def test_uniform_selection_breaks_the_hybrid_layout():
    """A real finding, pinned so it cannot be lost.

    Both models use a period-4 layout (3 DeltaNet : 1 attention). Spreading 48 student
    layers over 64 teacher layers gives a non-integer stride, so student layers land on
    teacher layers of the *wrong block type*. Those tensors do not even share names.
    """
    plan = build_transfer_plan(TEACHER, student(48), layer_selection="uniform")
    assert plan.warnings
    assert any("block type" in w or "is linear_attention but maps" in w for w in plan.warnings)
    assert plan.coverage < 1.0


def test_period_preserving_selections_keep_the_layout_intact():
    for selection in ("first", "last", "interleave"):
        plan = build_transfer_plan(TEACHER, student(48), layer_selection=selection)
        assert plan.coverage == 1.0, selection
        assert not any("block type" in w for w in plan.warnings), selection


def test_mismatched_layers_are_listed_as_random_not_silently_mapped():
    plan = build_transfer_plan(TEACHER, student(48), layer_selection="uniform")
    assert plan.randomly_initialised
    assert all("block type mismatch" in name for name in plan.randomly_initialised)


# --- plan content ---------------------------------------------------------
def test_plan_covers_every_component_of_a_matched_layer():
    plan = build_transfer_plan(TEACHER, student(48), layer_selection="interleave")
    names = " ".join(m.student_name for m in plan.mappings)
    for expected in ("embed_tokens", "mlp.gate_proj", "mlp.down_proj", "input_layernorm",
                     "linear_attn.in_proj_qkv", "linear_attn.A_log", "self_attn.q_proj",
                     "model.norm.weight"):
        assert expected in names, expected


def test_tied_student_reuses_the_teacher_embedding_for_the_head():
    plan = build_transfer_plan(TEACHER, student(48, tie_word_embeddings=True))
    head = next(m for m in plan.mappings if m.student_name == "lm_head.weight")
    assert head.teacher_names == ["model.embed_tokens.weight"]
    assert "tied" in head.operation


def test_vocabulary_mismatch_blocks_embedding_transfer_and_warns_about_kd():
    plan = build_transfer_plan(TEACHER, student(48, vocab_size=32000))
    assert "model.embed_tokens.weight" in plan.randomly_initialised
    assert any("logit distillation" in w for w in plan.warnings)


def test_slicing_is_labelled_a_baseline_not_a_method():
    """Guard against naive slicing being mistaken for a solution."""
    plan = build_transfer_plan(
        TEACHER, student(48, hidden_size=4096, num_attention_heads=16),
        width_reduction="slice",
    )
    assert any("baseline to beat" in w for w in plan.warnings)


def test_gated_student_q_proj_is_doubled_in_the_plan():
    plan = build_transfer_plan(TEACHER, student(48), layer_selection="interleave")
    q = next(m for m in plan.mappings if m.student_name.endswith("self_attn.q_proj.weight"))
    assert q.student_shape[0] == 24 * 256 * 2


def test_compare_strategies_returns_one_plan_each():
    plans = compare_strategies(TEACHER, student(48))
    assert set(plans) == {"first", "last", "uniform", "interleave", "group"}
    assert all(p.mappings or p.warnings for p in plans.values())


def test_compare_strategies_records_failures_rather_than_raising():
    plans = compare_strategies(TEACHER, student(128))   # deeper than the teacher
    assert all(p.warnings for p in plans.values())


def test_plan_is_json_serialisable():
    json.dumps(build_transfer_plan(TEACHER, student(48)).to_dict())


# --- group-aligned selection ---------------------------------------------
def test_group_selection_copies_position_within_the_group():
    """Group alignment is uniform over groups, identity inside them.

    The within-group offset *is* the block type, so preserving it is the whole
    mechanism. 64 -> 28 with period 4 selects 7 of 16 teacher groups whole.
    """
    mapping = select_layers(64, 28, "group", group_size=4)
    assert len(mapping) == 28
    for student_layer, teacher_layer in mapping.items():
        assert student_layer % 4 == teacher_layer % 4
    assert mapping[0] == 0 and mapping[27] == 63          # spans the full depth
    assert len({t // 4 for t in mapping.values()}) == 7   # whole groups, none repeated


def test_group_selection_is_the_only_strategy_that_spans_depth_without_breaking_layout():
    """The measured finding this strategy exists for, pinned across four depths.

    ``first``/``last`` keep the layout but see one end of the teacher. ``uniform`` and
    ``interleave`` span the depth but land layers on the wrong block type. Only ``group``
    does both, so this asserts both halves — a future change that quietly reverts to a
    per-layer stride would keep coverage but lose the span, and vice versa.
    """
    for layers in (16, 24, 28, 32):
        plan = build_transfer_plan(TEACHER, student(layers), layer_selection="group")
        assert plan.coverage == 1.0, layers
        assert not any(" which is " in w for w in plan.warnings), layers
        assert min(plan.layer_map.values()) == 0, layers
        assert max(plan.layer_map.values()) == 63, layers

        spread = build_transfer_plan(TEACHER, student(layers), layer_selection="uniform")
        assert spread.coverage < 1.0, layers   # the failure `group` avoids

        kept = build_transfer_plan(TEACHER, student(layers), layer_selection="first")
        assert max(kept.layer_map.values()) < 63, layers   # the span `group` keeps


def test_group_selection_refuses_partial_groups():
    with pytest.raises(ValueError, match="whole groups"):
        select_layers(64, 30, "group", group_size=4)


def test_group_selection_refuses_a_different_hybrid_period():
    """Aligning groups across different periods would be meaningless, not merely lossy."""
    plans = compare_strategies(
        TEACHER, student(32, full_attention_interval=8), selections=("group",)
    )
    assert any("shared hybrid period" in w for w in plans["group"].warnings)


def test_group_and_first_agree_when_the_student_is_a_prefix_of_the_teacher():
    """A sanity anchor: with every group selected there is nothing to choose between."""
    assert select_layers(64, 64, "group", group_size=4) == select_layers(64, 64, "first")


# --- deriving a transferable student --------------------------------------
def test_the_student_inherits_every_field_a_transfer_cannot_reduce():
    """Chief among them the vocabulary.

    Choosing a student's vocabulary independently is what makes logit distillation need a
    token mapping and makes embedding transfer meaningless, so it is not a parameter at
    all rather than a parameter with a good default.
    """
    derived = student_from_teacher(TEACHER, hidden_size=2560, num_hidden_layers=16)
    assert derived.vocab_size == TEACHER.vocab_size
    assert derived.head_dim == TEACHER.head_dim
    assert derived.linear_key_head_dim == TEACHER.linear_key_head_dim
    assert derived.linear_value_head_dim == TEACHER.linear_value_head_dim
    assert derived.linear_conv_kernel_dim == TEACHER.linear_conv_kernel_dim
    assert derived.full_attention_interval == TEACHER.full_attention_interval
    assert derived.partial_rotary_factor == TEACHER.partial_rotary_factor
    assert "student_from_teacher" not in derived.provenance  # names the module, not itself
    assert "transfer" in derived.provenance


def test_head_counts_move_only_through_the_group_defining_ones():
    """Both ratios stay fixed by construction rather than by a later check."""
    derived = student_from_teacher(TEACHER, num_key_value_heads=2, linear_num_key_heads=8)
    assert derived.num_attention_heads == 2 * (
        TEACHER.num_attention_heads // TEACHER.num_key_value_heads
    )
    assert derived.linear_num_value_heads == 8 * (
        TEACHER.linear_num_value_heads // TEACHER.linear_num_key_heads
    )


def test_a_depth_that_is_not_whole_groups_is_refused():
    with pytest.raises(ValueError, match="whole number of"):
        student_from_teacher(TEACHER, num_hidden_layers=30)


def test_a_derived_student_transfers_with_no_warnings_about_structure():
    derived = student_from_teacher(TEACHER, num_hidden_layers=28, num_key_value_heads=2,
                                   linear_num_key_heads=8)
    plan = build_transfer_plan(TEACHER, derived, layer_selection="group")
    assert plan.coverage == 1.0
    assert not any("vocabulary differs" in w for w in plan.warnings)
    assert not any(" which is " in w for w in plan.warnings)
