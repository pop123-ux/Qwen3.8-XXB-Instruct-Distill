"""VRAM-estimation tests.

The property that matters most for this project is that the *linear* layers cost
nothing per token: that is what makes a large context affordable, and it is what
distinguishes this architecture from a plain transformer.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from qwen_distill.architecture.memory import (
    GIB,
    DeploymentConfig,
    conv_state_bytes,
    estimate_memory,
    kv_cache_bytes,
    max_context_within,
    recurrent_state_bytes,
    weight_bytes,
)
from qwen_distill.architecture.spec import HybridArchSpec

TEACHER = HybridArchSpec(name="teacher")


def test_kv_cache_counts_only_full_attention_layers():
    """16 of 64 layers hold a KV cache; a dense equivalent would hold all 64."""
    cfg = DeploymentConfig(context_length=8192, kv_cache_dtype="fp16")
    expected = 2 * TEACHER.num_key_value_heads * TEACHER.head_dim * 16 * 8192 * 2
    assert kv_cache_bytes(TEACHER, cfg) == expected


def test_kv_cache_scales_linearly_with_context():
    a = kv_cache_bytes(TEACHER, DeploymentConfig(context_length=8192))
    b = kv_cache_bytes(TEACHER, DeploymentConfig(context_length=16384))
    assert b == 2 * a


def test_recurrent_state_is_constant_in_context():
    """The defining property of the hybrid design."""
    short = recurrent_state_bytes(TEACHER, DeploymentConfig(context_length=1024))
    long = recurrent_state_bytes(TEACHER, DeploymentConfig(context_length=262144))
    assert short == long


def test_conv_state_is_constant_in_context():
    short = conv_state_bytes(TEACHER, DeploymentConfig(context_length=1024))
    long = conv_state_bytes(TEACHER, DeploymentConfig(context_length=262144))
    assert short == long


def test_recurrent_state_shape_matches_reference_implementation():
    """State is (num_v_heads, head_k_dim, head_v_dim) per linear layer, fp32."""
    cfg = DeploymentConfig(recurrent_state_dtype="fp32", batch_size=1)
    expected = 48 * 128 * 128 * TEACHER.num_linear_attention_layers * 4
    assert recurrent_state_bytes(TEACHER, cfg) == expected


def test_states_scale_with_batch_size():
    one = recurrent_state_bytes(TEACHER, DeploymentConfig(batch_size=1))
    four = recurrent_state_bytes(TEACHER, DeploymentConfig(batch_size=4))
    assert four == 4 * one


def test_quantisation_reduces_weight_memory_monotonically():
    order = ["bf16", "int8", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m"]
    sizes = [
        weight_bytes(TEACHER, DeploymentConfig(weight_quant=q, embedding_quant=None))
        for q in order
    ]
    assert sizes == sorted(sizes, reverse=True)


def test_embedding_quant_override_is_applied():
    """A 248k vocab makes the embedding precision choice material."""
    uniform = weight_bytes(TEACHER, DeploymentConfig(weight_quant="q4_k_m", embedding_quant=None))
    mixed = weight_bytes(TEACHER, DeploymentConfig(weight_quant="q4_k_m", embedding_quant="q6_k"))
    assert mixed > uniform


def test_teacher_does_not_fit_16gib_at_4bit():
    """The premise of the project, stated as an executable assertion."""
    est = estimate_memory(TEACHER, DeploymentConfig(context_length=32768, weight_quant="q4_k_m"))
    assert not est.fits_in(16.0)
    assert est.weights / GIB > 15.0


def test_total_is_the_sum_of_its_parts():
    est = estimate_memory(TEACHER, DeploymentConfig())
    assert est.total == (
        est.weights + est.kv_cache + est.recurrent_state
        + est.conv_state + est.activations + est.runtime_overhead
    )


def test_max_context_within_is_monotone_in_budget():
    small = HybridArchSpec(name="s", hidden_size=3072, num_hidden_layers=32, intermediate_size=8192)
    cfg = DeploymentConfig(weight_quant="q4_k_m")
    assert max_context_within(small, 12.0, cfg) <= max_context_within(small, 16.0, cfg)


def test_max_context_returns_zero_when_weights_alone_overflow():
    cfg = DeploymentConfig(weight_quant="bf16")
    assert max_context_within(TEACHER, 16.0, cfg) == 0


def test_max_context_is_capped_at_max_position_embeddings():
    tiny = HybridArchSpec(
        name="tiny", hidden_size=1024, num_hidden_layers=8, intermediate_size=2048,
        num_attention_heads=4, vocab_size=32000, max_position_embeddings=8192,
    )
    assert max_context_within(tiny, 40.0, DeploymentConfig()) == 8192


def test_max_context_boundary_actually_fits():
    """The returned context must fit and one token more must not (below the cap)."""
    spec = HybridArchSpec(name="m", hidden_size=4096, num_hidden_layers=48, intermediate_size=12288)
    cfg = DeploymentConfig(weight_quant="q4_k_m")
    ctx = max_context_within(spec, 15.0, cfg)
    assert ctx > 0
    assert estimate_memory(spec, replace(cfg, context_length=ctx)).fits_in(15.0)
    if ctx < spec.max_position_embeddings:
        assert not estimate_memory(spec, replace(cfg, context_length=ctx + 1)).fits_in(15.0)


def test_unknown_quantisation_is_rejected():
    with pytest.raises(ValueError, match="unknown weight quantisation"):
        weight_bytes(TEACHER, DeploymentConfig(weight_quant="q2_secret"))


def test_unknown_kv_dtype_is_rejected():
    with pytest.raises(ValueError, match="unknown dtype"):
        kv_cache_bytes(TEACHER, DeploymentConfig(kv_cache_dtype="fp3"))
