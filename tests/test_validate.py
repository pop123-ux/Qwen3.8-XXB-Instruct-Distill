"""Tests that check the analytical model against the reference implementation.

These are the highest-value tests in the repository: they are what turns
"we read the source carefully" into "we verified it". If they fail, an upstream shape
changed and every memory and FLOP estimate here is suspect.
"""

from __future__ import annotations

from conftest import requires_stack

from qwen_distill.architecture.params import count_parameters, mtp_params
from qwen_distill.architecture.spec import HybridArchSpec
from qwen_distill.teacher.validate import (
    SMALL_SPEC,
    validate_cache_shapes,
    validate_parameters,
)


@requires_stack
def test_small_spec_parameters_match_transformers_exactly():
    result = validate_parameters(SMALL_SPEC)
    assert result.error is None, result.error
    assert result.comparisons["total"]["delta"] == 0
    assert result.details["unmatched_parameters"] == 0
    assert result.passed


@requires_stack
def test_every_component_matches_not_just_the_total():
    """A total can match by cancelling errors; components must match individually."""
    result = validate_parameters(SMALL_SPEC)
    for name, values in result.comparisons.items():
        assert values["delta"] == 0, f"{name}: {values}"


@requires_stack
def test_teacher_spec_parameters_match_transformers_exactly():
    """The full 27B structure, built on the meta device (no storage allocated)."""
    result = validate_parameters(HybridArchSpec(name="teacher"))
    assert result.error is None, result.error
    assert result.passed
    assert result.comparisons["total"]["measured"] == 26_895_998_464
    assert result.details["model_class"] == "Qwen3_5ForCausalLM"
    assert result.details["n_full_attention"] == 16


@requires_stack
def test_cache_shapes_match_a_real_forward_pass():
    result = validate_cache_shapes(SMALL_SPEC)
    assert result.error is None, result.error
    assert result.passed
    for name, values in result.comparisons.items():
        assert values["delta"] == 0, f"{name}: {values}"


@requires_stack
def test_kv_cache_exists_only_on_full_attention_layers():
    """The property the whole long-context argument rests on."""
    result = validate_cache_shapes(SMALL_SPEC)
    assert result.details["kv_only_on_full_attention"]
    assert result.details["layers_with_kv_cache"] == result.details["expected_kv_layers"]


@requires_stack
def test_recurrent_state_shape_is_heads_by_k_by_v():
    result = validate_cache_shapes(SMALL_SPEC)
    assert result.details["recurrent_state_shape"] == [
        1, SMALL_SPEC.linear_num_value_heads,
        SMALL_SPEC.linear_key_head_dim, SMALL_SPEC.linear_value_head_dim,
    ]


@requires_stack
def test_conv_state_shape_is_conv_dim_by_kernel():
    result = validate_cache_shapes(SMALL_SPEC)
    assert result.details["conv_state_shape"] == [
        1, SMALL_SPEC.linear_conv_dim, SMALL_SPEC.linear_conv_kernel_dim
    ]


@requires_stack
def test_recurrent_state_does_not_grow_with_sequence_length():
    """Measured, not asserted from the formula."""
    short = validate_cache_shapes(SMALL_SPEC, sequence_length=8)
    long = validate_cache_shapes(SMALL_SPEC, sequence_length=64)
    assert (
        short.comparisons["recurrent_state_bytes"]["measured"]
        == long.comparisons["recurrent_state_bytes"]["measured"]
    )
    # ...while the KV cache does grow.
    assert (
        long.comparisons["kv_cache_bytes"]["measured"]
        > short.comparisons["kv_cache_bytes"]["measured"]
    )


def test_mtp_head_is_a_small_fraction_of_the_model():
    """MTP adds ~1.6% at 27B: cheap enough that keeping it is a real option."""
    spec = HybridArchSpec(name="teacher")
    share = mtp_params(spec) / count_parameters(spec).total
    assert 0.01 < share < 0.03


def test_mtp_scales_with_layer_count():
    spec = HybridArchSpec(name="teacher")
    assert mtp_params(spec, 2) > mtp_params(spec, 1)
