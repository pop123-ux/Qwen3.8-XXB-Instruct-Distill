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
    PARAMETER_BUDGET,
    REJECTED,
    STUDENT_ID,
    TEACHER_FFN_INTERMEDIATE,
    TEACHER_ID,
    TEACHER_KV_HEADS,
    TEACHER_LAYERS,
    MoEStudentSpec,
    audit,
    build_config,
    parameter_model,
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
    assert (s.num_experts, s.num_experts_per_tok) == (8, 2)
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
    assert cfg.num_experts == 8 and cfg.num_experts_per_tok == 2
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


def test_exact_parameter_count_is_the_corrected_one(report):
    """The corrected count, pinned. 13.01B, down from the 22.07B first implementation that
    could not be deployed. The name still says 19B and the architecture is not adjusted to
    make a label true in either direction — the difference is reported."""
    assert report["exact_parameter_count"] == 13_008_505_728
    assert report["difference_from_19B"] == -5_991_494_272


def test_the_correction_did_not_touch_per_token_capacity(report):
    """The whole argument for the correction: it removed stored-but-unused experts, not
    capacity. Active parameters moved by 0.04%, entirely from the smaller router."""
    assert report["active_parameters_per_token"] == 9_611_119_488
    previous_active = 9_615_051_648
    drift = abs(report["active_parameters_per_token"] - previous_active) / previous_active
    assert drift < 0.001, "the correction should not have changed per-token capacity"


def test_the_rejected_configuration_is_on_the_record():
    """Deleting the failed architecture would delete the evidence for the current one."""
    rejected = {entry["config"]: entry for entry in REJECTED}
    failed = rejected["num_experts=24, moe_intermediate_size=768"]
    assert failed["total_parameters"] == 22_072_134_528
    assert failed["active_parameters"] == 9_615_051_648
    assert "does not fit" in failed["why_rejected"].lower()
    # Every rejected entry must say what measurement rejected it.
    for entry in REJECTED:
        assert len(entry["why_rejected"]) > 80
        assert entry["total_parameters"] > 0


def test_the_parameter_budget_is_a_hard_ceiling(report):
    """The bound exists so a future edit adding experts, widening the FFN or untying
    something cannot silently reintroduce a model that does not deploy."""
    assert report["exact_parameter_count"] <= PARAMETER_BUDGET, (
        f"the student is {report['exact_parameter_count']:,} parameters, over the "
        f"{PARAMETER_BUDGET:,} deployment budget. Adding parameters here is only correct "
        "if the 16 GB feasibility table in research/memory.py still passes."
    )
    assert PARAMETER_BUDGET < 26_895_998_464, "the budget must stay under the teacher"


def test_the_student_is_a_real_compression_of_the_teacher(report):
    """22.07B against a 26.90B teacher was an 18% reduction, which is not a distillation
    result. 13.01B is 48% of the teacher."""
    teacher = 26_895_998_464
    assert report["exact_parameter_count"] < teacher
    assert report["exact_parameter_count"] / teacher < 0.55


def test_the_closed_form_and_the_instantiated_model_agree(report):
    """Two independent derivations — one from the spec's arithmetic, one by building the
    model and summing tensors. An error in either shows up here as a mismatch."""
    model = parameter_model(FROZEN_STUDENT)
    assert model["total"] == report["exact_parameter_count"]
    assert model["active_per_token"] == report["active_parameters_per_token"]
    for bucket in ("embedding", "lm_head", "attention", "deltanet",
                   "routed_experts", "shared_expert", "router", "norms"):
        assert model[bucket] == report["components"][bucket], f"{bucket} disagrees"


def test_total_depends_on_the_expert_product_and_active_on_the_active_width():
    """The two invariants the correction turned on. Splitting a fixed expert budget between
    count and width is free for memory and is not free for per-token capacity."""
    from dataclasses import replace

    base = parameter_model(FROZEN_STUDENT)
    # Same E x W product, different split: identical total.
    wider = parameter_model(replace(FROZEN_STUDENT, num_experts=4,
                                    moe_intermediate_size=1536))
    assert wider["routed_experts"] == base["routed_experts"]
    # ... and strictly more active parameters, because active scales with K x W.
    assert wider["active_per_token"] > base["active_per_token"]
    # Doubling the count at the same width doubles only the stored experts.
    doubled = parameter_model(replace(FROZEN_STUDENT, num_experts=16))
    assert doubled["routed_experts"] == 2 * base["routed_experts"]
    assert doubled["active_per_token"] - base["active_per_token"] < 5_000_000


def test_component_breakdown_is_complete_and_sums_to_the_total(report):
    components = report["components"]
    assert sum(components.values()) == report["exact_parameter_count"]
    assert components["other"] == 0, "an unclassified tensor appeared in the audit"
    for name in ("embedding", "lm_head", "attention", "deltanet", "routed_experts",
                 "shared_expert", "router", "norms"):
        assert components[name] > 0, f"{name} contributed no parameters"


def test_routed_experts_no_longer_dominate_the_parameter_budget(report):
    """They were 61.6% of the model and are now 34.8%. That shift is the correction: the
    experts stopped being the thing the VRAM budget was mostly spent on."""
    c = report["components"]
    total = report["exact_parameter_count"]
    assert 0.30 < c["routed_experts"] / total < 0.40
    assert c["embedding"] == c["lm_head"], "untied heads should be the same size"
    assert c["router"] / total < 0.001


def test_stored_experts_are_still_the_largest_single_component(report):
    """Reduced, not eliminated: the MoE is intact and is still where a future memory
    reduction would have to look first."""
    c = report["components"]
    assert c["routed_experts"] == max(c.values())
    assert c["routed_experts"] > c["deltanet"]


def test_mtp_contributes_nothing_because_it_is_not_built(report):
    assert report["components"].get("mtp", 0) == 0
    assert "DECLARED, NOT BUILT" in MTP_STATUS
    assert report["mtp_status"] == MTP_STATUS


def test_audit_is_serialisable_and_renderable(report):
    json.loads(json.dumps(report))
    text = render_audit(report)
    assert "13,008,505,728" in text
    assert "-5,991,494,272" in text
    assert "36 DeltaNet + 12 full attention" in text


def test_audit_follows_the_spec_it_is_given():
    """The audit must be a measurement of a model, not a lookup table of the frozen one."""
    pytest.importorskip("transformers")
    small = audit(tiny_fixture())
    assert small["exact_parameter_count"] < 2_000_000
    assert sum(small["components"].values()) == small["exact_parameter_count"]
