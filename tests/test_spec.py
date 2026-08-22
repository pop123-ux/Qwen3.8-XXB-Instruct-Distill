"""Tests for the architecture spec, anchored on upstream's own expansion rules."""

from __future__ import annotations

import pytest

from qwen_distill.architecture.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    HybridArchSpec,
    build_layer_types,
)


def test_layer_type_expansion_matches_upstream_rule():
    """Upstream: ``"linear_attention" if bool((i + 1) % interval) else "full_attention"``."""
    types = build_layer_types(8, 4)
    assert types == [
        LINEAR_ATTENTION, LINEAR_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION,
        LINEAR_ATTENTION, LINEAR_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION,
    ]


def test_full_attention_is_last_layer_of_each_group():
    """Every group ends in full attention, so the last layer is always full attention."""
    for layers, interval in ((64, 4), (48, 4), (32, 2), (60, 3)):
        types = build_layer_types(layers, interval)
        assert types[-1] == FULL_ATTENTION
        assert len(types) == layers


def test_teacher_layout_is_48_linear_16_full():
    spec = HybridArchSpec(name="teacher")
    assert spec.num_hidden_layers == 64
    assert spec.num_linear_attention_layers == 48
    assert spec.num_full_attention_layers == 16
    assert spec.num_linear_attention_layers == 3 * spec.num_full_attention_layers


def test_derived_deltanet_dimensions():
    spec = HybridArchSpec(name="teacher")
    assert spec.linear_key_dim == 16 * 128 == 2048
    assert spec.linear_value_dim == 48 * 128 == 6144
    # conv_dim = key_dim * 2 + value_dim
    assert spec.linear_conv_dim == 2 * 2048 + 6144 == 10240


def test_rope_dim_is_partial():
    """head_dim 256 with partial_rotary_factor 0.25 gives the documented RoPE dim 64."""
    assert HybridArchSpec(name="teacher").rope_dim == 64


def test_explicit_layer_types_override_interval():
    spec = HybridArchSpec(
        name="custom",
        num_hidden_layers=4,
        layer_types=[FULL_ATTENTION, LINEAR_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION],
    )
    assert spec.num_full_attention_layers == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_attention_heads": 25, "num_key_value_heads": 4},  # not divisible
        {"linear_num_value_heads": 47, "linear_num_key_heads": 16},  # not divisible
        {"partial_rotary_factor": 0.0},  # out of range
        {"num_hidden_layers": 0},
    ],
)
def test_invalid_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        HybridArchSpec(name="bad", **kwargs)


def test_all_linear_layout_is_rejected():
    """A model with no exact-recall layer should not silently pass the search."""
    with pytest.raises(ValueError):
        HybridArchSpec(name="no-attn", num_hidden_layers=4, layer_types=[LINEAR_ATTENTION] * 4)


def test_round_trip_through_hf_config():
    spec = HybridArchSpec(name="teacher")
    restored = HybridArchSpec.from_hf_config({"text_config": spec.to_hf_text_config()})
    for f in ("hidden_size", "num_hidden_layers", "intermediate_size", "vocab_size",
              "num_attention_heads", "num_key_value_heads", "head_dim",
              "linear_num_value_heads", "linear_num_key_heads"):
        assert getattr(restored, f) == getattr(spec, f), f
    assert restored.resolved_layer_types() == spec.resolved_layer_types()


def test_round_trip_through_dict(tmp_path):
    spec = HybridArchSpec(name="teacher")
    path = tmp_path / "spec.json"
    spec.save(path)
    assert HybridArchSpec.load(path).to_dict() == spec.to_dict()
