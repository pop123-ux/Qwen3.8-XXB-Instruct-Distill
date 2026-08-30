"""Structural proof that the frozen student is a *buildable* model, not a config that
happens to validate.

A configuration test can pass while the architecture is impossible — a head count that
divides on paper but produces a shape error in the attention kernel, a routing width that
the expert module rejects, a hybrid pattern the cache cannot represent. The only way to
rule that out before a 22B materialisation is to instantiate a model of the same
architecture family, at a size that fits in memory here, and actually run it.

``tiny_fixture()`` is that model: identical in every structural respect to the frozen
target and smaller in every size. Each test below names the frozen-target property it is
standing in for.
"""
from __future__ import annotations

import pytest

from qwen_distill.architecture.moe_student import (
    FROZEN_STUDENT,
    MTP_STATUS,
    build_config,
    tiny_fixture,
)
from qwen_distill.architecture.spec import FULL_ATTENTION, LINEAR_ATTENTION

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny():
    return tiny_fixture()


@pytest.fixture(scope="module")
def model(tiny):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(build_config(tiny))
    m.eval()
    return m


# ---------------------------------------------------------------------------
# the fixture is a faithful stand-in
# ---------------------------------------------------------------------------
def test_fixture_shares_the_frozen_topology(tiny):
    """Same hybrid pattern, same group size, same ordering — only fewer groups."""
    assert tiny.group_size == FROZEN_STUDENT.group_size
    assert tiny.deltanet_per_group == FROZEN_STUDENT.deltanet_per_group
    assert tiny.attention_per_group == FROZEN_STUDENT.attention_per_group
    assert tiny.layer_types()[: tiny.group_size] == FROZEN_STUDENT.layer_types()[: tiny.group_size]
    assert tiny.partial_rotary_factor == FROZEN_STUDENT.partial_rotary_factor
    assert tiny.tie_word_embeddings is FROZEN_STUDENT.tie_word_embeddings is False
    assert tiny.num_experts_per_tok == FROZEN_STUDENT.num_experts_per_tok
    # DeltaNet value:key head ratio is 3:1 in both.
    assert (
        tiny.linear_num_value_heads // tiny.linear_num_key_heads
        == FROZEN_STUDENT.linear_num_value_heads // FROZEN_STUDENT.linear_num_key_heads
    )
    # Both are grouped-query, not multi-head and not multi-query.
    for spec in (tiny, FROZEN_STUDENT):
        assert 1 < spec.num_key_value_heads < spec.num_attention_heads


def test_fixture_is_small_enough_to_run(model):
    assert sum(p.numel() for p in model.parameters()) < 2_000_000


# ---------------------------------------------------------------------------
# the model builds with the intended blocks in the intended places
# ---------------------------------------------------------------------------
def test_hybrid_blocks_land_where_the_pattern_says(model, tiny):
    types = tiny.layer_types()
    for i, layer in enumerate(model.model.layers):
        if types[i] == FULL_ATTENTION:
            assert hasattr(layer, "self_attn"), f"layer {i} should be full attention"
            assert not hasattr(layer, "linear_attn")
        else:
            assert types[i] == LINEAR_ATTENTION
            assert hasattr(layer, "linear_attn"), f"layer {i} should be DeltaNet"
            assert not hasattr(layer, "self_attn")
    assert [i for i, t in enumerate(types) if t == FULL_ATTENTION] == tiny.attention_layer_indices


def test_full_attention_block_is_gated_and_grouped(model, tiny):
    """`q_proj` is twice the query width because the output gate is fused into it.

    This is the structural evidence for the frozen spec's ``output gate enabled``: the
    runtime has no ``attn_output_gate`` flag, so the only way to know the gate exists is
    the projection width.
    """
    attn = model.model.layers[tiny.attention_layer_indices[0]].self_attn
    q_width = tiny.num_attention_heads * tiny.head_dim
    kv_width = tiny.num_key_value_heads * tiny.head_dim
    assert attn.q_proj.weight.shape == (2 * q_width, tiny.hidden_size)
    assert attn.k_proj.weight.shape == (kv_width, tiny.hidden_size)
    assert attn.v_proj.weight.shape == (kv_width, tiny.hidden_size)
    assert attn.o_proj.weight.shape == (tiny.hidden_size, q_width)
    assert attn.q_proj.bias is None and attn.k_proj.bias is None  # attention_bias=False
    assert attn.q_norm.weight.shape == (tiny.head_dim,)


def test_deltanet_block_has_its_gates_and_conv(model, tiny):
    dn = model.model.layers[tiny.deltanet_layer_indices[0]].linear_attn
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "conv1d"):
        assert hasattr(dn, name), f"DeltaNet is missing {name}"
    assert dn.conv1d.weight.shape[-1] == tiny.linear_conv_kernel_dim
    # b and a are per-value-head decay/beta gates.
    assert dn.in_proj_b.weight.shape == (tiny.linear_num_value_heads, tiny.hidden_size)
    assert dn.in_proj_a.weight.shape == (tiny.linear_num_value_heads, tiny.hidden_size)
    assert dn.A_log.shape == (tiny.linear_num_value_heads,)


def test_moe_block_has_routed_experts_a_router_and_a_gated_shared_expert(model, tiny):
    mlp = model.model.layers[0].mlp
    e = mlp.experts
    assert e.gate_up_proj.shape == (tiny.num_experts, 2 * tiny.moe_intermediate_size, tiny.hidden_size)
    assert e.down_proj.shape == (tiny.num_experts, tiny.hidden_size, tiny.moe_intermediate_size)
    assert mlp.gate.weight.shape == (tiny.num_experts, tiny.hidden_size)
    assert mlp.gate.top_k == tiny.num_experts_per_tok
    assert mlp.shared_expert.gate_proj.weight.shape == (
        tiny.shared_expert_intermediate_size,
        tiny.hidden_size,
    )
    # The shared expert is gated, so it is "1 active expert" with a learned scalar, not a
    # plain residual addition.
    assert mlp.shared_expert_gate.weight.shape == (1, tiny.hidden_size)


def test_every_layer_is_sparse(model):
    """No dense FFN survives anywhere: all 8 layers route."""
    assert all(hasattr(layer.mlp, "experts") for layer in model.model.layers)


def test_lm_head_is_untied_from_the_embedding(model, tiny):
    assert model.lm_head.weight.shape == (tiny.vocab_size, tiny.hidden_size)
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()


# ---------------------------------------------------------------------------
# it runs
# ---------------------------------------------------------------------------
def test_forward_produces_finite_logits(model, tiny):
    ids = torch.randint(0, tiny.vocab_size, (2, 24))
    out = model(input_ids=ids)
    assert out.logits.shape == (2, 24, tiny.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_backward_reaches_every_parameter(model, tiny):
    """Including the experts: a routing bug that starves an expert shows up here as a
    ``None`` gradient, which is the failure this test exists to catch."""
    model.zero_grad(set_to_none=True)
    ids = torch.randint(0, tiny.vocab_size, (2, 32))
    out = model(input_ids=ids, labels=ids, output_router_logits=True)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == [], f"no gradient reached: {missing}"
    model.zero_grad(set_to_none=True)


def test_router_emits_one_logit_tensor_per_layer_and_an_aux_loss(model, tiny):
    ids = torch.randint(0, tiny.vocab_size, (1, 16))
    out = model(input_ids=ids, labels=ids, output_router_logits=True)
    assert len(out.router_logits) == tiny.num_hidden_layers
    assert out.router_logits[0].shape[-1] == tiny.num_experts
    assert out.aux_loss is not None and torch.isfinite(out.aux_loss)


def test_expert_selection_is_top_k_and_normalised(model, tiny):
    """The router's contract, verified on the real module rather than assumed."""
    hidden = torch.randn(16, tiny.hidden_size)
    logits, scores, indices = model.model.layers[0].mlp.gate(hidden)
    assert logits.shape == (16, tiny.num_experts)
    assert indices.shape == (16, tiny.num_experts_per_tok)
    assert scores.shape == (16, tiny.num_experts_per_tok)
    # Exactly top-k distinct experts per token, weights summing to one.
    assert all(len(set(row.tolist())) == tiny.num_experts_per_tok for row in indices)
    assert torch.allclose(scores.sum(-1), torch.ones(16), atol=1e-5)
    assert (indices < tiny.num_experts).all()


def test_shared_expert_contributes_on_every_token(model, tiny):
    """Zeroing the shared expert must change the block's output for all tokens, which is
    what distinguishes it from a 25th routed expert."""
    mlp = model.model.layers[0].mlp
    hidden = torch.randn(1, 12, tiny.hidden_size)
    with torch.no_grad():
        before = mlp(hidden)
        saved = mlp.shared_expert.down_proj.weight.clone()
        mlp.shared_expert.down_proj.weight.zero_()
        after = mlp(hidden)
        mlp.shared_expert.down_proj.weight.copy_(saved)
    changed = (before - after).abs().amax(-1) > 0
    assert changed.all(), "the shared expert did not affect every token"


def test_routed_experts_contribute_selectively(model, tiny):
    """Zeroing one expert changes only the tokens routed to it — proof the top-k path is
    live and sparse rather than every expert running on every token."""
    mlp = model.model.layers[0].mlp
    torch.manual_seed(3)
    hidden = torch.randn(1, 64, tiny.hidden_size)
    flat = hidden.view(-1, tiny.hidden_size)
    with torch.no_grad():
        _, _, indices = mlp.gate(flat)
        target = int(indices[:, 0].mode().values)
        routed = (indices == target).any(-1)
        before = mlp(hidden)
        saved = mlp.experts.down_proj.data[target].clone()
        mlp.experts.down_proj.data[target].zero_()
        after = mlp(hidden)
        mlp.experts.down_proj.data[target].copy_(saved)
    changed = (before - after).abs().view(-1, tiny.hidden_size).amax(-1) > 0
    assert routed.any(), "no token routed to the modal expert"
    assert torch.equal(changed, routed), "expert influence does not match the routing mask"


def test_deltanet_and_attention_both_move_the_residual_stream(model, tiny):
    """Each mixer type is load-bearing; a silently no-op block would pass a shape test."""
    hidden = torch.randn(1, 20, tiny.hidden_size)
    position_ids = torch.arange(20).unsqueeze(0)
    embeds = model.model.embed_tokens(torch.randint(0, tiny.vocab_size, (1, 20)))
    with torch.no_grad():
        pos_emb = model.model.rotary_emb(embeds, position_ids)
        attn_layer = model.model.layers[tiny.attention_layer_indices[0]]
        dn_layer = model.model.layers[tiny.deltanet_layer_indices[0]]
        a_out = attn_layer(hidden, position_embeddings=pos_emb, attention_mask=None)
        d_out = dn_layer(hidden, position_embeddings=pos_emb, attention_mask=None)
    for label, out in (("attention", a_out), ("deltanet", d_out)):
        tensor = out[0] if isinstance(out, tuple) else out
        assert tensor.shape == hidden.shape
        assert torch.isfinite(tensor).all()
        assert not torch.allclose(tensor, hidden), f"{label} block was a no-op"


def test_incremental_decoding_matches_a_full_forward(model, tiny):
    """The hybrid cache has to carry two different things — KV for 2 layers and a recurrent
    DeltaNet state for 6. If either is mishandled, step-by-step decoding diverges from the
    single-pass forward. This is the strongest available check that the recurrent state is
    real and correctly threaded."""
    torch.manual_seed(7)
    ids = torch.randint(0, tiny.vocab_size, (1, 12))
    with torch.no_grad():
        full = model(input_ids=ids).logits
        cache = None
        steps = []
        for t in range(ids.shape[1]):
            out = model(input_ids=ids[:, t : t + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            steps.append(out.logits)
        stepwise = torch.cat(steps, dim=1)
    assert torch.allclose(full, stepwise, atol=2e-4), (
        f"cached decoding diverged; max delta {(full - stepwise).abs().max():.3e}"
    )


def test_generation_runs_end_to_end(model, tiny):
    ids = torch.randint(0, tiny.vocab_size, (1, 8))
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=6, do_sample=False)
    assert out.shape == (1, 14)


# ---------------------------------------------------------------------------
# MTP: recorded honestly, not faked
# ---------------------------------------------------------------------------
def test_mtp_is_declared_but_not_built(model):
    """The frozen spec asks for one MTP layer; this runtime builds none. The test pins the
    gap so a future transformers release that *does* build it fails loudly here instead of
    silently changing the parameter count."""
    assert FROZEN_STUDENT.mtp_num_hidden_layers == 1
    assert "DECLARED, NOT BUILT" in MTP_STATUS
    assert [n for n, _ in model.named_modules() if "mtp" in n.lower()] == []
    assert not hasattr(build_config(FROZEN_STUDENT), "mtp_num_hidden_layers")
