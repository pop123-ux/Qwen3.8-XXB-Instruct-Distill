"""Parameter-accounting tests.

The anchor test is :func:`test_teacher_spec_reproduces_published_27b`: the published
Qwen3.8-27B configuration must reproduce a ~27B parameter count from our formulas.
If that test fails, every memory and FLOP estimate in the repository is suspect.
"""

from __future__ import annotations

from qwen_distill.architecture.params import (
    count_parameters,
    format_params,
    full_attention_params,
    linear_attention_params,
    mlp_params,
)
from qwen_distill.architecture.spec import HybridArchSpec


def test_teacher_spec_reproduces_published_27b():
    """The published config must land within 2% of the advertised 27B."""
    total = count_parameters(HybridArchSpec(name="teacher")).total
    assert 26.4e9 < total < 27.6e9, f"got {format_params(total)}"


def test_mlp_dominates_teacher_parameters():
    """MLP is the largest single component: the primary compression lever."""
    breakdown = count_parameters(HybridArchSpec(name="teacher"))
    shares = breakdown.shares()
    assert shares["mlp"] > 0.6
    assert shares["mlp"] == max(
        shares[k] for k in ("mlp", "full_attention", "linear_attention", "embedding")
    )


def test_gated_attention_q_proj_is_doubled():
    """q_proj emits 2x n_heads*head_dim because the second half is the output gate.

    Dropping the gate would under-count a 27B model by roughly a billion parameters,
    so this is pinned explicitly.
    """
    spec = HybridArchSpec(name="teacher")
    h, hd = spec.hidden_size, spec.head_dim
    q = h * spec.num_attention_heads * hd * 2
    kv = 2 * (h * spec.num_key_value_heads * hd)
    o = spec.num_attention_heads * hd * h
    assert full_attention_params(spec) == q + kv + o + 2 * hd


def test_deltanet_includes_z_b_a_gates_and_conv():
    spec = HybridArchSpec(name="teacher")
    h = spec.hidden_size
    expected = (
        h * spec.linear_conv_dim              # in_proj_qkv
        + h * spec.linear_value_dim           # in_proj_z
        + 2 * (h * spec.linear_num_value_heads)  # in_proj_b, in_proj_a
        + spec.linear_value_dim * h           # out_proj
        + spec.linear_conv_dim * spec.linear_conv_kernel_dim  # depthwise conv1d
        + 2 * spec.linear_num_value_heads     # dt_bias, A_log
        + spec.linear_value_head_dim          # RMSNormGated
    )
    assert linear_attention_params(spec) == expected


def test_mlp_is_three_matrices():
    spec = HybridArchSpec(name="teacher")
    assert mlp_params(spec) == 3 * spec.hidden_size * spec.intermediate_size


def test_tying_embeddings_removes_exactly_the_head():
    untied = count_parameters(HybridArchSpec(name="a", tie_word_embeddings=False))
    tied = count_parameters(HybridArchSpec(name="b", tie_word_embeddings=True))
    assert tied.lm_head == 0
    assert untied.total - tied.total == untied.embedding
    assert untied.non_embedding == tied.non_embedding


def test_non_embedding_excludes_lookup_tables():
    b = count_parameters(HybridArchSpec(name="teacher"))
    assert b.non_embedding == b.total - b.embedding - b.lm_head


def test_breakdown_shares_sum_to_one():
    b = count_parameters(HybridArchSpec(name="teacher"))
    shares = b.shares()
    leaves = ("embedding", "lm_head", "final_norm", "layer_norms",
              "mlp", "full_attention", "linear_attention")
    assert abs(sum(shares[k] for k in leaves) - 1.0) < 1e-9


def test_format_params():
    assert format_params(26_895_998_464) == "26.90B"
    assert format_params(1_500_000) == "1.50M"
    assert format_params(512) == "512"
