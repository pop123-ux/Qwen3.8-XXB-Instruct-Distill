"""The frozen student specification and its parameter audit.

The spec is frozen by the project brief, so these tests are not design choices under
review — they are a tripwire. If a future edit changes the hidden size, the expert count,
the hybrid pattern or the resulting parameter count, that edit has silently replaced the
research target and these tests fail.

The audit tests pin the *measured* parameter count, including the finding that the target
does not weigh 19B. That number is reported, not corrected: the architecture is fixed and
the label is what is inaccurate.
"""
from __future__ import annotations

import json

import pytest

from qwen_distill.architecture.moe_student import (
    FROZEN_STUDENT,
    MOE_MODEL_TYPE,
    MTP_STATUS,
    STUDENT_ID,
    TEACHER_FFN_INTERMEDIATE,
    TEACHER_ID,
    TEACHER_KV_HEADS,
    TEACHER_LAYERS,
    MoEStudentSpec,
    audit,
    build_config,
    render_audit,
    tiny_fixture,
)
from qwen_distill.architecture.spec import FULL_ATTENTION, LINEAR_ATTENTION


# ---------------------------------------------------------------------------
# the frozen numbers
# ---------------------------------------------------------------------------
def test_frozen_spec_matches_the_brief_field_for_field():
    s = FROZEN_STUDENT
    assert s.name == STUDENT_ID == "qwen38_19b_h5120_l48_moe"
    assert (s.hidden_size, s.num_hidden_layers, s.vocab_size) == (5120, 48, 248320)
    assert s.max_position_embeddings == 262144
    assert s.tie_word_embeddings is False
    assert s.rms_norm_eps == 1e-6
    assert (s.num_attention_heads, s.num_key_value_heads, s.head_dim) == (24, 2, 256)
    assert s.attention_bias is False and s.attention_dropout == 0
    assert (s.partial_rotary_factor, s.rope_theta) == (0.25, 10_000_000)
    assert s.rope_dim == 64
    assert (s.linear_num_key_heads, s.linear_num_value_heads) == (16, 48)
    assert s.linear_key_head_dim == s.linear_value_head_dim == 128
    assert s.linear_conv_kernel_dim == 4
    assert (s.num_experts, s.num_experts_per_tok) == (24, 2)
    assert s.moe_intermediate_size == s.shared_expert_intermediate_size == 768
    assert s.router_aux_loss_coef == 0.001 and s.router_jitter is False
    assert s.mtp_num_hidden_layers == 1 and s.distill_from_teacher_mtp is True


def test_teacher_constants_are_the_measured_ones():
    assert TEACHER_ID == "Qwen/Qwen3.8-27B"
    assert (TEACHER_LAYERS, TEACHER_KV_HEADS, TEACHER_FFN_INTERMEDIATE) == (64, 4, 17408)


def test_hybrid_pattern_is_three_deltanet_then_one_attention_twelve_times():
    types = FROZEN_STUDENT.layer_types()
    assert types[:4] == [LINEAR_ATTENTION] * 3 + [FULL_ATTENTION]
    assert len(types) == 48
    assert types == ([LINEAR_ATTENTION] * 3 + [FULL_ATTENTION]) * 12
    assert types.count(LINEAR_ATTENTION) == 36 and types.count(FULL_ATTENTION) == 12
    assert FROZEN_STUDENT.attention_layer_indices == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
    assert FROZEN_STUDENT.num_groups == 12 and FROZEN_STUDENT.group_size == 4


def test_compression_is_depth_only_with_no_width_reduction():
    assert FROZEN_STUDENT.num_hidden_layers / TEACHER_LAYERS == 0.75
    assert FROZEN_STUDENT.hidden_size == 5120, "the teacher's width is preserved"
    assert FROZEN_STUDENT.vocab_size == 248320, "the teacher's vocabulary is preserved"
    assert FROZEN_STUDENT.num_key_value_heads * 2 == TEACHER_KV_HEADS


# ---------------------------------------------------------------------------
# validation catches the ways it could be broken
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "override, message",
    [
        ({"num_hidden_layers": 50}, "hybrid groups"),
        ({"num_attention_heads": 25}, "divisible"),
        ({"linear_num_value_heads": 47}, "divisible"),
        ({"num_experts_per_tok": 25}, "cannot exceed"),
        ({"head_dim": 6, "partial_rotary_factor": 0.5}, "rotary"),
    ],
)
def test_invalid_specs_are_rejected_at_construction(override, message):
    with pytest.raises(ValueError, match=message):
        MoEStudentSpec(**override)


def test_a_valid_variation_is_still_allowed():
    """Validation must not be so strict that the ablations in the roadmap are unbuildable."""
    assert MoEStudentSpec(num_hidden_layers=44).num_groups == 11


# ---------------------------------------------------------------------------
# it round-trips into a real transformers config
# ---------------------------------------------------------------------------
def test_config_is_the_registered_moe_text_architecture():
    pytest.importorskip("transformers")
    cfg = build_config(FROZEN_STUDENT)
    assert cfg.model_type == MOE_MODEL_TYPE == "qwen3_5_moe_text"
    assert cfg.hidden_size == 5120 and cfg.num_hidden_layers == 48
    assert cfg.layer_types == FROZEN_STUDENT.layer_types()
    assert cfg.num_experts == 24 and cfg.num_experts_per_tok == 2
    assert cfg.router_aux_loss_coef == 0.001


def test_partial_rotary_travels_inside_rope_parameters():
    """It is not a top-level config field on this architecture; putting it there would be
    silently dropped and the model would use full rotary."""
    pytest.importorskip("transformers")
    cfg = build_config(FROZEN_STUDENT)
    assert cfg.rope_parameters["partial_rotary_factor"] == 0.25
    assert cfg.rope_parameters["rope_theta"] == 10_000_000


def test_spec_serialises_for_the_ledger():
    d = FROZEN_STUDENT.to_dict()
    assert d["attention_layer_indices"] == FROZEN_STUDENT.attention_layer_indices
    assert d["rope_dim"] == 64
    json.loads(json.dumps(d))


# ---------------------------------------------------------------------------
# the parameter audit
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def report():
    pytest.importorskip("transformers")
    return audit()


def test_exact_parameter_count_is_reported_not_rounded_to_the_label(report):
    """The frozen target is named 19B and weighs 22.07B. The architecture is not adjusted to
    make the label true; the difference is reported."""
    assert report["exact_parameter_count"] == 22_072_134_528
    assert report["difference_from_19B"] == 3_072_134_528
    assert report["difference_from_19B"] > 0


def test_non_embedding_count_explains_where_the_19B_label_came_from(report):
    """19.53B non-embedding is the plausible origin of the name, and saying so is more
    useful than either defending or hiding the gap."""
    assert report["non_embedding_parameters"] == 19_529_337_728
    assert abs(report["non_embedding_parameters"] - 19e9) / 19e9 < 0.03


def test_sparsity_actually_buys_something(report):
    """Under 10B active per token against 22B stored: that ratio is the entire argument for
    choosing MoE over a dense model of the same quality."""
    assert report["active_parameters_per_token"] == 9_615_051_648
    assert report["active_parameters_per_token"] / report["exact_parameter_count"] < 0.45


def test_component_breakdown_is_complete_and_sums_to_the_total(report):
    components = report["components"]
    assert sum(components.values()) == report["exact_parameter_count"]
    assert components["other"] == 0, "an unclassified tensor appeared in the audit"
    for name in ("embedding", "lm_head", "attention", "deltanet", "routed_experts",
                 "shared_expert", "router", "norms"):
        assert components[name] > 0, f"{name} contributed no parameters"


def test_routed_experts_dominate_the_parameter_budget(report):
    """61% of the model is expert weights, of which 2 of 24 run per token. Any parameter- or
    memory-reduction work that is not aimed at the experts is aimed at the wrong 38%."""
    c = report["components"]
    total = report["exact_parameter_count"]
    assert c["routed_experts"] / total > 0.6
    assert c["embedding"] == c["lm_head"], "untied heads should be the same size"
    assert c["router"] / total < 0.001


def test_mtp_contributes_nothing_because_it_is_not_built(report):
    assert report["components"].get("mtp", 0) == 0
    assert "DECLARED, NOT BUILT" in MTP_STATUS
    assert report["mtp_status"] == MTP_STATUS


def test_audit_is_serialisable_and_renderable(report):
    json.loads(json.dumps(report))
    text = render_audit(report)
    assert "22,072,134,528" in text
    assert "+3,072,134,528" in text
    assert "36 DeltaNet + 12 full attention" in text


def test_audit_follows_the_spec_it_is_given():
    """The audit must be a measurement of a model, not a lookup table of the frozen one."""
    pytest.importorskip("transformers")
    small = audit(tiny_fixture())
    assert small["exact_parameter_count"] < 2_000_000
    assert sum(small["components"].values()) == small["exact_parameter_count"]
