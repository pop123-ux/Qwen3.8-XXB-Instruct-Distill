"""The three reductions that turn the dense 64-layer teacher into the sparse 48-layer
student, each tested against the property it is supposed to have — and, where a claim is
quantitative, against a measured number rather than an assertion that it "works".

The FFN tests run the *real* ``Qwen3_5MoeSparseMoeBlock`` after initialising it, because a
decomposition that only ever exists as a list of channel indices has not been shown to
survive contact with the routing rule.
"""
from __future__ import annotations

import pytest

from qwen_distill.architecture import moe_init as mi
from qwen_distill.architecture.moe_student import (
    FROZEN_STUDENT,
    TEACHER_FFN_INTERMEDIATE,
    TEACHER_LAYERS,
    TINY_TEACHER_FFN_INTERMEDIATE,
    build_config,
    tiny_fixture,
)
from qwen_distill.architecture.spec import FULL_ATTENTION, LINEAR_ATTENTION

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

TI = TINY_TEACHER_FFN_INTERMEDIATE


@pytest.fixture(scope="module")
def tiny():
    return tiny_fixture()


@pytest.fixture(scope="module")
def teacher_ffn(tiny):
    """A stand-in dense FFN with the teacher's orientation and a realistic width ratio."""
    g = torch.Generator().manual_seed(11)
    h = tiny.hidden_size
    return (
        torch.randn(TI, h, generator=g) / h**0.5,
        torch.randn(TI, h, generator=g) / h**0.5,
        torch.randn(h, TI, generator=g) / TI**0.5,
    )


@pytest.fixture(scope="module")
def hidden(tiny):
    g = torch.Generator().manual_seed(12)
    return torch.randn(1, 256, tiny.hidden_size, generator=g)


@pytest.fixture
def block(tiny):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(build_config(tiny))
    model.eval()
    return model.model.layers[0].mlp


# ---------------------------------------------------------------------------
# 64 -> 48 layers
# ---------------------------------------------------------------------------
def test_group_mapping_never_puts_a_block_on_the_wrong_type():
    """The failure this exists to prevent: deleting every fourth teacher layer rotates the
    hybrid pattern, so DeltaNet weights land in attention slots."""
    m = mi.map_layers(FROZEN_STUDENT, teacher_layers=TEACHER_LAYERS)
    assert m.strategy == "group"
    assert len(m.mapping) == FROZEN_STUDENT.num_hidden_layers == 48
    assert len(m.removed_teacher_layers) == TEACHER_LAYERS - 48 == 16
    assert m.problems == []
    assert m.block_types_preserved
    for s, t in m.mapping.items():
        assert m.student_types[s] == m.teacher_types[t]


def test_naive_stride_deletion_is_what_the_group_mapping_avoids():
    """Pins the counter-example, so the design choice is evidenced and not merely asserted."""
    student_types = FROZEN_STUDENT.layer_types()
    teacher_types = mi._hybrid_types(TEACHER_LAYERS, 3, 1)
    naive = [t for t in range(TEACHER_LAYERS) if t % 4 != 3][:48]
    mismatches = sum(1 for s, t in enumerate(naive) if student_types[s] != teacher_types[t])
    assert mismatches > 0, "if this is zero the naive baseline is fine and the group map is moot"
    # For contrast, the group mapping has none.
    assert mi.map_layers(FROZEN_STUDENT, teacher_layers=TEACHER_LAYERS).problems == []


def test_group_mapping_spans_the_full_teacher_depth():
    """Both ends of the teacher must be represented: the embedding-adjacent layers and the
    output-adjacent layers do different jobs, and dropping either tail changes the model."""
    m = mi.map_layers(FROZEN_STUDENT, teacher_layers=TEACHER_LAYERS)
    kept = sorted(set(m.mapping.values()))
    assert kept[0] == 0
    assert kept[-1] == TEACHER_LAYERS - 1
    assert sorted(m.mapping) == list(range(48))


def test_mapping_is_injective_and_order_preserving():
    m = mi.map_layers(FROZEN_STUDENT, teacher_layers=TEACHER_LAYERS)
    targets = [m.mapping[s] for s in sorted(m.mapping)]
    assert len(set(targets)) == len(targets), "two student layers share a teacher layer"
    assert targets == sorted(targets), "the mapping reorders the teacher's depth"


def test_importance_mapping_keeps_the_highest_scoring_groups():
    scores = {g: float(g % 5) for g in range(TEACHER_LAYERS // 4)}
    m = mi.map_layers(FROZEN_STUDENT, teacher_layers=TEACHER_LAYERS,
                      strategy="importance", importance=scores)
    assert m.strategy == "importance"
    assert m.problems == []
    kept_groups = sorted({t // 4 for t in m.mapping.values()})
    dropped_groups = [g for g in scores if g not in kept_groups]
    assert min(scores[g] for g in kept_groups) >= max(scores[g] for g in dropped_groups)


def test_importance_mapping_refuses_to_invent_a_score():
    with pytest.raises(ValueError, match="measured"):
        mi.map_layers(FROZEN_STUDENT, teacher_layers=TEACHER_LAYERS, strategy="importance")


def test_mapping_rejects_depths_that_are_not_whole_groups():
    with pytest.raises(ValueError, match="whole"):
        mi.map_layers(FROZEN_STUDENT, teacher_layers=62)


def test_mapping_serialises_for_the_ledger():
    d = mi.map_layers(FROZEN_STUDENT, teacher_layers=TEACHER_LAYERS).to_dict()
    assert d["n_removed"] == 16 and d["block_types_preserved"] is True
    assert d["mapping"]["0"] == 0
    import json

    json.loads(json.dumps(d))


# ---------------------------------------------------------------------------
# dense FFN -> 8 experts
# ---------------------------------------------------------------------------
def test_decomposition_partitions_rather_than_duplicating_the_teacher(tiny, teacher_ffn):
    """The explicitly forbidden shortcut is copying the whole teacher FFN into every
    expert. Each expert must hold a *different*, narrower slice."""
    g, u, d = teacher_ffn
    plan = mi.plan_ffn_decomposition(tiny, importance=mi.channel_importance(g, u, d),
                                     teacher_intermediate=TI)
    assert plan.expert_intermediate == tiny.moe_intermediate_size < TI
    sets = [frozenset(c) for c in plan.expert_channels]
    assert len(set(sets)) == len(sets), "two experts received identical channels"
    for a in sets:
        assert len(a) <= tiny.moe_intermediate_size
        assert a != frozenset(range(TI)), "an expert received the entire teacher FFN"
    # Overlap between experts comes only from padding, never from wholesale duplication.
    pairwise = [len(a & b) / len(a) for i, a in enumerate(sets) for b in sets[i + 1:]]
    assert max(pairwise) < 0.5


def test_decomposition_is_not_random_initialisation(tiny, teacher_ffn):
    """Every channel an expert holds must be a real teacher channel index."""
    g, u, d = teacher_ffn
    plan = mi.plan_ffn_decomposition(tiny, importance=mi.channel_importance(g, u, d),
                                     teacher_intermediate=TI)
    for channels in plan.expert_channels + [plan.shared_channels]:
        assert all(0 <= c < TI for c in channels)
    assert plan.coverage > 0.9


def test_shared_expert_receives_the_strongest_channels(tiny, teacher_ffn):
    """It runs on every token, so it should carry what every token needs."""
    g, u, d = teacher_ffn
    importance = mi.channel_importance(g, u, d)
    plan = mi.plan_ffn_decomposition(tiny, importance=importance, teacher_intermediate=TI)
    top = set(torch.argsort(importance, descending=True)[: len(plan.shared_channels)].tolist())
    assert set(plan.shared_channels) == top


def test_routed_experts_get_a_comparable_share_of_strong_channels(tiny, teacher_ffn):
    """Round-robin in descending importance: no expert may be dead on arrival."""
    g, u, d = teacher_ffn
    importance = mi.channel_importance(g, u, d)
    plan = mi.plan_ffn_decomposition(tiny, importance=importance, teacher_intermediate=TI)
    totals = [float(importance[torch.tensor(c)].sum()) for c in plan.expert_channels]
    assert min(totals) / max(totals) > 0.8, f"experts are unevenly strong: {totals}"


def test_activation_based_importance_differs_from_weight_energy(tiny, teacher_ffn):
    """The activation path must actually use the activations, not silently fall back."""
    g, u, d = teacher_ffn
    acts = torch.rand(64, TI)
    assert not torch.allclose(
        mi.channel_importance(g, u, d), mi.channel_importance(g, u, d, activations=acts)
    )


def test_no_importance_falls_back_to_a_labelled_contiguous_split(tiny):
    plan = mi.plan_ffn_decomposition(tiny, importance=None, teacher_intermediate=TI)
    assert plan.method == "contiguous_partition"


def test_active_width_is_the_documented_fraction_of_the_teacher():
    """The bound the whole FFN section is written around, stated as a test so it cannot
    drift silently."""
    plan = mi.plan_ffn_decomposition(FROZEN_STUDENT, importance=None)
    assert plan.active_width == 2 * 768 + 768 == 2304
    assert plan.teacher_intermediate == TEACHER_FFN_INTERMEDIATE == 17408
    ratio = plan.active_width / TEACHER_FFN_INTERMEDIATE
    assert 0.12 < ratio < 0.14


def test_channel_coverage_is_the_price_paid_for_fitting_sixteen_gb():
    """The expert-budget correction's real cost, measured rather than glossed.

    With 24 experts the decomposition could hold every one of the teacher's 17,408 FFN
    channels somewhere. With 8 it holds 6,912 of them — 39.7%. The remaining 60.3% are
    dropped at initialisation and have to be learned rather than transferred.

    What did *not* change is what any single token sees: active width is still 2,304, so
    the reconstruction bound is unmoved. The loss is in how much teacher FFN the router has
    to choose between, not in per-token capacity."""
    from dataclasses import replace

    plan = mi.plan_ffn_decomposition(FROZEN_STUDENT, importance=None)
    assert plan.coverage == pytest.approx(6912 / 17408, rel=1e-3)
    assert 0.39 < plan.coverage < 0.40
    # Coverage is fixed by the expert budget, not by how it is split between count and
    # width: the same total parameters buy the same coverage either way.
    resplit = mi.plan_ffn_decomposition(
        replace(FROZEN_STUDENT, num_experts=24, moe_intermediate_size=256), importance=None
    )
    assert resplit.coverage == pytest.approx(plan.coverage, rel=1e-3)
    # ... but the resplit carries strictly less per-token capacity, which is why the
    # correction cut the count and kept the width.
    assert resplit.active_width < plan.active_width


# ---------------------------------------------------------------------------
# the decomposition, applied to the real block
# ---------------------------------------------------------------------------
def _plan_and_apply(tiny, block, teacher_ffn, **kw):
    g, u, d = teacher_ffn
    plan = mi.plan_ffn_decomposition(tiny, importance=mi.channel_importance(g, u, d),
                                     teacher_intermediate=TI)
    weights = mi.build_moe_weights(plan, g, u, d, spec=tiny, **kw)
    written = mi.apply_moe_weights(block, weights)
    return plan, written


def test_applying_the_plan_writes_every_tensor_in_the_block(tiny, block, teacher_ffn):
    """An initialisation that leaves tensors at their random defaults is a silent partial
    transfer; ``apply_moe_weights`` raises rather than allowing it."""
    _, written = _plan_and_apply(tiny, block, teacher_ffn)
    assert set(written) == {n for n, _ in block.named_parameters()}


def test_experts_are_fused_in_the_order_the_runtime_chunks_them(tiny, block, teacher_ffn):
    """``gate_up_proj`` is chunked into (gate, up); swapping the halves is a silent bug that
    no shape check would catch."""
    g, u, d = teacher_ffn
    plan, _ = _plan_and_apply(tiny, block, teacher_ffn)
    width = tiny.moe_intermediate_size
    idx = torch.tensor(plan.expert_channels[0])
    assert torch.allclose(block.experts.gate_up_proj[0][:width], g[idx])
    assert torch.allclose(block.experts.gate_up_proj[0][width:], u[idx])


def test_gate_compensation_restores_the_teacher_output_scale(tiny, block, teacher_ffn):
    """Measured, not argued: the convex routing weights and the sigmoid gate halve the
    block's output, and the compensation cancels exactly that factor."""
    g, u, d = teacher_ffn
    h = torch.randn(1, 128, tiny.hidden_size, generator=torch.Generator().manual_seed(5))
    _plan_and_apply(tiny, block, teacher_ffn, compensate=False)
    without = mi.measure_block_reconstruction(block, g, u, d, h)
    _plan_and_apply(tiny, block, teacher_ffn, compensate=True)
    with_ = mi.measure_block_reconstruction(block, g, u, d, h)
    assert with_["norm_ratio"] == pytest.approx(2 * without["norm_ratio"], rel=1e-3)
    assert with_["relative_norm_error"] < without["relative_norm_error"]
    # Scaling is orthogonal to direction, so the cosine must be untouched.
    assert with_["cosine_similarity"] == pytest.approx(without["cosine_similarity"], rel=1e-4)


def test_initialised_block_beats_a_random_block_on_every_metric(tiny, block, teacher_ffn):
    """The claim that this is teacher transfer rather than dressed-up random init."""
    g, u, d = teacher_ffn
    h = torch.randn(1, 128, tiny.hidden_size, generator=torch.Generator().manual_seed(6))
    random_score = mi.measure_block_reconstruction(block, g, u, d, h)
    _plan_and_apply(tiny, block, teacher_ffn)
    transferred = mi.measure_block_reconstruction(block, g, u, d, h)
    assert transferred["cosine_similarity"] > random_score["cosine_similarity"] + 0.3
    assert transferred["relative_norm_error"] < random_score["relative_norm_error"]
    assert transferred["mse"] < random_score["mse"]


def test_reconstruction_is_lossy_by_construction(tiny, block, teacher_ffn):
    """A near-zero error would mean the measurement is wrong, not that the method is
    perfect: 2304 of 17408 active channels cannot reproduce a dense FFN."""
    g, u, d = teacher_ffn
    h = torch.randn(1, 128, tiny.hidden_size, generator=torch.Generator().manual_seed(7))
    _plan_and_apply(tiny, block, teacher_ffn)
    r = mi.measure_block_reconstruction(block, g, u, d, h)
    assert 0.0 < r["cosine_similarity"] < 0.99
    assert r["relative_norm_error"] > 0.05


def test_oracle_router_bounds_the_initialised_router(tiny, block, teacher_ffn):
    """The oracle is an upper bound; if the real router beat it the bound is not a bound."""
    g, u, d = teacher_ffn
    h = torch.randn(256, tiny.hidden_size, generator=torch.Generator().manual_seed(8))
    plan, _ = _plan_and_apply(tiny, block, teacher_ffn)
    oracle = mi.measure_ffn_reconstruction(plan, g, u, d, h, top_k=tiny.num_experts_per_tok)
    actual = mi.measure_block_reconstruction(block, g, u, d, h.unsqueeze(0))
    assert oracle["relative_norm_error"] <= actual["relative_norm_error"] + 1e-6
    assert oracle["cosine_similarity"] >= actual["cosine_similarity"] - 1e-6


def test_block_reconstruction_report_is_serialisable(tiny, block, teacher_ffn, hidden):
    import json

    g, u, d = teacher_ffn
    _plan_and_apply(tiny, block, teacher_ffn)
    json.loads(json.dumps(mi.measure_block_reconstruction(block, g, u, d, hidden)))


def test_apply_rejects_a_shape_mismatch(tiny, block, teacher_ffn):
    g, u, d = teacher_ffn
    plan = mi.plan_ffn_decomposition(tiny, importance=None, teacher_intermediate=TI)
    weights = mi.build_moe_weights(plan, g, u, d, spec=tiny)
    weights["gate.weight"] = torch.zeros(tiny.num_experts + 1, tiny.hidden_size)
    with pytest.raises(ValueError, match="block wants"):
        mi.apply_moe_weights(block, weights)


# ---------------------------------------------------------------------------
# 4 -> 2 KV heads
# ---------------------------------------------------------------------------
def test_kv_merge_halves_the_projection_and_pairs_adjacent_heads():
    head_dim = 8
    w = torch.arange(4 * head_dim * 16, dtype=torch.float32).reshape(4 * head_dim, 16)
    merged = mi.merge_kv_heads(w, teacher_heads=4, student_heads=2, head_dim=head_dim)
    assert merged.shape == (2 * head_dim, 16)
    grouped = w.reshape(4, head_dim, 16)
    expected = (grouped[0] + grouped[1]) / 2
    assert torch.allclose(merged.reshape(2, head_dim, 16)[0], expected)


@pytest.mark.parametrize("method", ["mean", "weighted", "first"])
def test_kv_merge_methods_are_all_available_and_distinct(method):
    torch.manual_seed(2)
    w = torch.randn(4 * 8, 16)
    merged = mi.merge_kv_heads(w, teacher_heads=4, student_heads=2, head_dim=8, method=method)
    assert merged.shape == (16, 16)
    assert torch.isfinite(merged).all()
    mean = mi.merge_kv_heads(w, teacher_heads=4, student_heads=2, head_dim=8, method="mean")
    if method != "mean":
        assert not torch.allclose(merged, mean)


def test_first_head_selection_discards_rather_than_merges():
    w = torch.randn(4 * 8, 16)
    merged = mi.merge_kv_heads(w, teacher_heads=4, student_heads=2, head_dim=8, method="first")
    assert torch.allclose(merged[:8], w[:8])
    assert torch.allclose(merged[8:], w[16:24])


def test_kv_merge_rejects_an_uneven_grouping():
    with pytest.raises(ValueError, match="evenly"):
        mi.merge_kv_heads(torch.randn(3 * 8, 16), teacher_heads=3, student_heads=2, head_dim=8)


def test_kv_merge_error_is_measured_and_finite():
    torch.manual_seed(4)
    k = torch.randn(4 * 8, 32) / 32**0.5
    v = torch.randn(4 * 8, 32) / 32**0.5
    h = torch.randn(128, 32)
    report = mi.measure_kv_merge(k, v, h, teacher_heads=4, student_heads=2, head_dim=8)
    for key in ("k_mse", "v_mse", "k_cosine", "v_cosine", "k_relative_error"):
        assert key in report
    # Mean-merging reproduces the mean-folded teacher exactly; this is the identity that
    # makes "mean" the baseline, and it is the reference the other methods are scored against.
    assert report["k_mse"] < 1e-10
    assert report["k_cosine"] > 0.999


def test_weighted_merge_moves_away_from_the_mean_baseline():
    torch.manual_seed(4)
    k = torch.randn(4 * 8, 32) / 32**0.5
    v = torch.randn(4 * 8, 32) / 32**0.5
    h = torch.randn(128, 32)
    weighted = mi.measure_kv_merge(k, v, h, teacher_heads=4, student_heads=2,
                                   head_dim=8, method="weighted")
    assert weighted["k_mse"] > 1e-10
    assert weighted["method"] == "weighted"


# ---------------------------------------------------------------------------
# router initialisation
# ---------------------------------------------------------------------------
def test_exactly_uniform_router_kills_most_experts():
    """The measured failure that sets :data:`mi.DEFAULT_ROUTER_SCALE`. Identical logits make
    ``topk`` break ties by index, so two experts take every token and the rest never receive
    a gradient. Entropy stays at its maximum throughout, which is why entropy alone is not a
    sufficient health check."""
    torch.manual_seed(0)
    hidden = torch.randn(2048, FROZEN_STUDENT.hidden_size)
    w = mi.init_router(FROZEN_STUDENT.num_experts, FROZEN_STUDENT.hidden_size, scale=0.0)
    r = mi.measure_router_balance(w, hidden, top_k=FROZEN_STUDENT.num_experts_per_tok)
    assert r["dead_experts"] == FROZEN_STUDENT.num_experts - FROZEN_STUDENT.num_experts_per_tok
    assert r["max_load_share"] == pytest.approx(0.5)
    assert r["entropy_fraction_of_uniform"] == pytest.approx(1.0)


def test_default_router_scale_leaves_no_dead_experts_and_near_maximal_entropy():
    torch.manual_seed(0)
    hidden = torch.randn(2048, FROZEN_STUDENT.hidden_size)
    w = mi.init_router(FROZEN_STUDENT.num_experts, FROZEN_STUDENT.hidden_size)
    r = mi.measure_router_balance(w, hidden, top_k=FROZEN_STUDENT.num_experts_per_tok)
    assert r["dead_experts"] == 0
    assert r["overloaded_experts"] == 0
    assert r["entropy_fraction_of_uniform"] > 0.99
    uniform = 1.0 / FROZEN_STUDENT.num_experts
    assert r["max_load_share"] < 1.5 * uniform
    assert sum(r["load_share"]) == pytest.approx(1.0)


def test_router_init_is_deterministic_for_a_seed():
    a = mi.init_router(24, 128, seed=3)
    b = mi.init_router(24, 128, seed=3)
    c = mi.init_router(24, 128, seed=4)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_router_balance_report_is_serialisable():
    import json

    w = mi.init_router(6, 32)
    json.loads(json.dumps(mi.measure_router_balance(w, torch.randn(64, 32))))


def test_initialised_block_router_is_balanced_in_situ(tiny, block, teacher_ffn, hidden):
    """End to end: after the real block is initialised, its own router spreads tokens."""
    _plan_and_apply(tiny, block, teacher_ffn)
    flat = hidden.reshape(-1, tiny.hidden_size)
    r = mi.measure_router_balance(block.gate.weight.detach(), flat,
                                  top_k=tiny.num_experts_per_tok)
    assert r["dead_experts"] == 0
    assert r["entropy_fraction_of_uniform"] > 0.99


def test_shared_expert_gate_starts_neutral(tiny, block, teacher_ffn):
    """A zero gate is sigmoid 0.5, which is the value the compensation cancels; a random
    gate would make the compensation wrong by an unknown factor."""
    _plan_and_apply(tiny, block, teacher_ffn)
    assert torch.equal(block.shared_expert_gate.weight,
                       torch.zeros_like(block.shared_expert_gate.weight))


# ---------------------------------------------------------------------------
# the reductions compose
# ---------------------------------------------------------------------------
def test_teacher_and_student_hybrid_layouts_are_the_ones_recorded():
    """64 = 48 DeltaNet + 16 attention, 48 = 36 DeltaNet + 12 attention."""
    teacher = mi._hybrid_types(TEACHER_LAYERS, 3, 1)
    assert teacher.count(LINEAR_ATTENTION) == 48
    assert teacher.count(FULL_ATTENTION) == 16
    student = FROZEN_STUDENT.layer_types()
    assert student.count(LINEAR_ATTENTION) == 36
    assert student.count(FULL_ATTENTION) == 12
    assert len(student) == 48
