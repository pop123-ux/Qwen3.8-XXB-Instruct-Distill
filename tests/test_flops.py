"""FLOP and bandwidth accounting tests."""

from __future__ import annotations

from qwen_distill.architecture.flops import (
    bandwidth_bound_tokens_per_second,
    decode_bytes_per_token,
    decode_flops_per_token,
    format_flops,
    prefill_flops,
)
from qwen_distill.architecture.spec import HybridArchSpec

TEACHER = HybridArchSpec(name="teacher")


def test_only_attention_scores_grow_with_context():
    a = decode_flops_per_token(TEACHER, 0)
    b = decode_flops_per_token(TEACHER, 65536)
    assert b.full_attention_scores > a.full_attention_scores
    for field in ("mlp", "full_attention_proj", "linear_attention_proj",
                  "linear_attention_state", "lm_head"):
        assert getattr(a, field) == getattr(b, field), field


def test_attention_score_flops_scale_linearly_with_context():
    a = decode_flops_per_token(TEACHER, 16384).full_attention_scores
    b = decode_flops_per_token(TEACHER, 32768).full_attention_scores
    assert b == 2 * a


def test_mlp_dominates_decode_flops_at_short_context():
    f = decode_flops_per_token(TEACHER, 0)
    assert f.mlp > f.total / 2


def test_prefill_is_superlinear_in_sequence_length():
    """Doubling the sequence more than doubles prefill cost, via the quadratic term."""
    a = prefill_flops(TEACHER, 8192)
    b = prefill_flops(TEACHER, 16384)
    assert b > 2 * a


def test_prefill_of_one_token_has_no_quadratic_term():
    assert prefill_flops(TEACHER, 1) == decode_flops_per_token(TEACHER, 0).total


def test_decode_bytes_shrink_with_quantisation():
    assert decode_bytes_per_token(TEACHER, "q4_k_m") < decode_bytes_per_token(TEACHER, "bf16")


def test_bandwidth_ceiling_scales_with_bandwidth():
    slow = bandwidth_bound_tokens_per_second(TEACHER, 320.0)
    fast = bandwidth_bound_tokens_per_second(TEACHER, 640.0)
    assert abs(fast - 2 * slow) < 1e-6


def test_smaller_model_has_higher_throughput_ceiling():
    small = HybridArchSpec(name="s", hidden_size=3072, num_hidden_layers=32, intermediate_size=8192)
    assert bandwidth_bound_tokens_per_second(small, 448.0) > bandwidth_bound_tokens_per_second(
        TEACHER, 448.0
    )


def test_format_flops():
    assert format_flops(64.4e9) == "64.40G"
    assert format_flops(1.5e12) == "1.50T"
