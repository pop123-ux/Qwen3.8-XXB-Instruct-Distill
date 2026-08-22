"""Architecture-search tests."""

from __future__ import annotations

from qwen_distill.architecture.search import (
    SearchConstraints,
    evaluate_candidate,
    generate_grid,
    search,
)
from qwen_distill.architecture.spec import HybridArchSpec


def test_teacher_is_rejected_by_the_16gib_constraint():
    """The search must not admit the model the project exists to shrink."""
    candidate = evaluate_candidate(HybridArchSpec(name="teacher"), SearchConstraints())
    assert not candidate.feasible
    assert "budget" in (candidate.rejected_reason or "") or "context" in (
        candidate.rejected_reason or ""
    )


def test_grid_generates_only_valid_specs():
    specs = list(generate_grid([3072, 4096], [32, 48], [2.5, 3.4], [2, 4]))
    assert specs
    for spec in specs:
        spec.validate()  # must not raise
        assert spec.num_attention_heads % spec.num_key_value_heads == 0


def test_grid_ffn_multiplier_is_applied():
    (spec,) = list(generate_grid([4096], [32], [3.0], [4]))
    assert spec.intermediate_size == 12288


def test_grid_respects_tie_flag():
    specs = list(generate_grid([4096], [32], [3.0], [4], tie_word_embeddings=(True,)))
    assert all(s.tie_word_embeddings for s in specs)
    assert all(s.name.endswith("-tied") for s in specs)


def test_feasible_candidates_fit_the_budget_and_context():
    constraints = SearchConstraints(vram_gib=16.0, required_context=32768)
    results = search(generate_grid([3072, 4096, 5120], [32, 48, 64], [2.5, 3.4], [4]), constraints)
    assert results
    for candidate in results:
        assert candidate.memory.total_gib <= constraints.usable_gib
        assert candidate.max_context >= constraints.required_context


def test_results_are_ranked_by_non_embedding_capacity():
    results = search(generate_grid([3072, 4096, 4608], [32, 48], [2.5, 3.0], [4]))
    counts = [c.params.non_embedding for c in results]
    assert counts == sorted(counts, reverse=True)


def test_tighter_budget_admits_fewer_candidates():
    specs = list(generate_grid([3072, 4096, 4608, 5120], [32, 48, 64], [2.5, 3.0, 3.4], [4]))
    roomy = search(specs, SearchConstraints(vram_gib=16.0))
    tight = search(specs, SearchConstraints(vram_gib=10.0))
    assert len(tight) < len(roomy)


def test_longer_required_context_admits_fewer_candidates():
    specs = list(generate_grid([3072, 4096, 4608, 5120], [32, 48, 64], [2.5, 3.0, 3.4], [4]))
    short = search(specs, SearchConstraints(required_context=8192))
    long = search(specs, SearchConstraints(required_context=131072))
    assert len(long) <= len(short)


def test_keep_infeasible_returns_everything_with_reasons():
    specs = list(generate_grid([5120], [64], [3.4], [4]))
    results = search(specs, SearchConstraints(), keep_infeasible=True)
    assert len(results) == len(specs)
    assert all(c.rejected_reason for c in results if not c.feasible)


def test_aspect_ratio_guard_rejects_pathological_shapes():
    """A 1-layer 8192-wide model is not a serious candidate."""
    spec = HybridArchSpec(
        name="pathological", hidden_size=8192, num_hidden_layers=4,
        intermediate_size=1024, num_attention_heads=8, vocab_size=8192,
    )
    candidate = evaluate_candidate(spec, SearchConstraints())
    assert not candidate.feasible
    assert "aspect ratio" in (candidate.rejected_reason or "")


def test_summary_row_is_serialisable():
    import json

    results = search(generate_grid([4096], [48], [3.0], [4]), SearchConstraints())
    assert results
    json.dumps(results[0].summary_row())


def test_constraints_reserve_headroom():
    assert SearchConstraints(vram_gib=16.0, reserved_gib=1.0).usable_gib == 15.0
