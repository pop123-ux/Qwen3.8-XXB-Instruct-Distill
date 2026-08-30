"""End-to-end VRAM accounting against the 16 GB ceiling.

The load-bearing tests are :func:`test_the_student_fits_sixteen_gb_at_every_release_quant`
and :func:`test_inactive_experts_are_not_free`. The first pins the deployment claim so it
cannot silently stop being true; the second pins the reason the previous architecture
failed, which was counting an MoE against its active parameters rather than its stored
ones.
"""
from __future__ import annotations

import json

import pytest

from qwen_distill.architecture.memory import GIB
from qwen_distill.architecture.moe_student import FROZEN_STUDENT, audit
from qwen_distill.research.memory import (
    CONTEXT_LADDER,
    DOES_NOT_FIT,
    NOMINAL_VRAM_GIB,
    RELEASE_HEADROOM_GIB,
    RELEASE_QUANTS,
    USABLE_GIB,
    MemoryAccount,
    RuntimeConfig,
    account,
    build_table,
    conv_state_bytes,
    frontier,
    headline,
    kv_cache_bytes,
    max_context,
    offload_note,
    recurrent_state_bytes,
    render_table,
    weight_bytes,
)

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def components():
    return audit(FROZEN_STUDENT)["components"]


# ---------------------------------------------------------------------------
# the individual terms
# ---------------------------------------------------------------------------
def test_kv_cache_covers_only_the_twelve_attention_layers():
    """The hybrid layout's whole deployment argument. If this ever counts 48 layers the
    long-context numbers quadruple and the release plan is wrong."""
    config = RuntimeConfig(context_length=1)
    per_token = kv_cache_bytes(FROZEN_STUDENT, config)
    assert per_token == 2 * 2 * 256 * 12 * 2  # K+V, 2 kv heads, dim 256, 12 layers, fp16
    assert per_token == 24_576, "24 KiB per token"
    dense_equivalent = per_token * 4  # if all 48 layers cached
    assert dense_equivalent == 98_304


def test_kv_cache_is_linear_in_context():
    a = kv_cache_bytes(FROZEN_STUDENT, RuntimeConfig(context_length=32_768))
    b = kv_cache_bytes(FROZEN_STUDENT, RuntimeConfig(context_length=65_536))
    assert b == 2 * a
    assert kv_cache_bytes(FROZEN_STUDENT, RuntimeConfig(context_length=262_144)) == 6 * GIB


def test_recurrent_state_is_constant_in_context():
    """Verified against the running model: the state shape does not change with sequence
    length. This is the property that makes 36 of 48 layers free at long context."""
    short = recurrent_state_bytes(FROZEN_STUDENT, RuntimeConfig(context_length=2_048))
    long_ = recurrent_state_bytes(FROZEN_STUDENT, RuntimeConfig(context_length=262_144))
    assert short == long_
    # 36 layers x 48 value heads x 128 x 128 x fp32
    assert short == 36 * 48 * 128 * 128 * 4
    assert short / GIB < 0.11


def test_conv_state_uses_the_runtimes_own_conv_dim_formula():
    """conv_dim = 2*key_dim + value_dim, confirmed against the built module."""
    state = conv_state_bytes(FROZEN_STUDENT, RuntimeConfig(context_length=4_096))
    key_dim, value_dim = 16 * 128, 48 * 128
    assert state == (2 * key_dim + value_dim) * 4 * 36 * 4


def test_all_state_together_is_smaller_than_one_octave_of_kv():
    total_state = (recurrent_state_bytes(FROZEN_STUDENT, RuntimeConfig())
                   + conv_state_bytes(FROZEN_STUDENT, RuntimeConfig()))
    assert total_state < kv_cache_bytes(FROZEN_STUDENT, RuntimeConfig(context_length=8_192))


def test_weights_are_bucketed_by_component_not_uniformly(components):
    """Per-component quantisation is a real lever and a uniform model would hide it."""
    cheap = weight_bytes(components, RuntimeConfig(expert_quant="q3_k_m", dense_quant="q6_k"))
    uniform = weight_bytes(components, RuntimeConfig(expert_quant="q6_k", dense_quant="q6_k"))
    assert cheap["experts"] < uniform["experts"]
    assert cheap["dense"] == uniform["dense"]


def test_weight_counts_come_from_the_audit(components):
    """Memory and parameter tables cannot disagree if they share one source."""
    acc = account(components=components)
    assert acc.parameters == sum(components.values()) == 13_008_505_728


def test_inactive_experts_are_not_free(components):
    """The mistake that produced the rejected architecture. 9.61B of 13.01B parameters are
    active per token, but all 13.01B are resident, and the weight total must reflect that.
    Sizing an MoE against its active count is how a model that cannot deploy looks
    deployable on paper."""
    acc = account(components=components)
    active = 9_611_119_488
    assert acc.parameters > active
    q4 = acc.weight_total
    active_only = int(active * (4.9 / 8))
    assert q4 > active_only, "weights were sized against active parameters"
    # Every expert bucket is counted in full.
    assert acc.weights["experts"] == int(
        (components["routed_experts"] + components["shared_expert"]) * (4.9 / 8)
    )


def test_unknown_quantisation_is_refused(components):
    with pytest.raises(ValueError, match="unknown quantisation"):
        weight_bytes(components, RuntimeConfig(expert_quant="q2_vibes"))


# ---------------------------------------------------------------------------
# the total
# ---------------------------------------------------------------------------
def test_total_includes_every_term(components):
    acc = account(components=components)
    parts = (acc.weight_total, acc.quantisation_overhead, acc.kv_cache, acc.recurrent_state,
             acc.conv_state, acc.activations, acc.runtime_overhead)
    assert all(p > 0 for p in parts), "a term contributed nothing, which means it is missing"
    assert acc.total == sum(parts)


def test_quantisation_overhead_is_counted_on_top_of_the_nominal_bits(components):
    """A nominal 4.9 bits per parameter is a file-size number. Treating it as the VRAM
    number is a standard way to be wrong by a third of a gigabyte."""
    acc = account(components=components)
    assert acc.quantisation_overhead == int(acc.weight_total * 0.03)
    assert acc.quantisation_overhead / GIB > 0.2
    bare = account(config=RuntimeConfig(quantisation_overhead_fraction=0.0),
                   components=components)
    assert bare.total < acc.total


def test_runtime_overhead_is_not_optional(components):
    """~0.9 GiB before a single model tensor is allocated. Omitting it is the second most
    common way a memory plan is wrong."""
    acc = account(components=components)
    assert acc.runtime_overhead == pytest.approx(0.9 * GIB, rel=1e-6)


def test_fp32_logits_over_a_248k_vocabulary_are_counted(components):
    """A megabyte per sequence, and frequently forgotten."""
    acc = account(components=components)
    logits = FROZEN_STUDENT.vocab_size * 4
    assert acc.activations > logits


# ---------------------------------------------------------------------------
# the finding
# ---------------------------------------------------------------------------
def test_the_student_fits_sixteen_gb_at_every_release_quant(components):
    """The headline deployment claim, pinned. Fully GPU-resident, quantisation overhead and
    runtime included, every stored expert counted."""
    for quant in RELEASE_QUANTS:
        acc = account(config=RuntimeConfig(context_length=4_096, expert_quant=quant,
                                           dense_quant=quant, embedding_quant=quant),
                      components=components)
        assert acc.verdict() != DOES_NOT_FIT, f"{quant} no longer fits at 4K"
        assert acc.headroom_gib() >= RELEASE_HEADROOM_GIB
    assert headline()["fits_at_any_release_quant"] is True


def test_each_release_quant_reaches_a_useful_context(components):
    """Fitting at 4,096 tokens would not be a deployable long-context model. The claim is
    about the context each precision actually reaches with headroom kept in reserve."""
    reach = headline()["max_context_by_quant"]
    assert reach["Q4"] >= 131_072
    assert reach["Q5"] >= 65_536
    assert reach["Q6"] >= 32_768


def test_the_full_window_needs_a_quantised_kv_cache(components):
    """262,144 tokens is 6 GiB of fp16 KV on its own. Reported as its own row rather than
    folded into the headline, because KV quantisation costs retrieval accuracy."""
    full = headline()["full_window_262k"]
    assert full["Q4"]["fp16_kv_gib"] > USABLE_GIB
    assert full["Q4"]["fits_with_fp8_kv"] is True
    assert full["Q4"]["fp8_kv_gib"] < full["Q4"]["fp16_kv_gib"]
    assert "retrieval accuracy" in headline()["full_window_note"]


def test_the_headline_states_that_active_parameters_never_reduce_vram():
    note = headline()["sparsity_note"]
    assert "never reduce VRAM" in note
    assert "resident" in note


def test_planning_against_the_nominal_sixteen_gib_is_still_the_wrong_ceiling():
    """The correction fixed the model, not the arithmetic trap. A real card reports
    14.56 GiB, and the last gigabyte belongs to the rest of the system."""
    assert USABLE_GIB < NOMINAL_VRAM_GIB
    assert NOMINAL_VRAM_GIB - USABLE_GIB > 2.0


def test_the_rejected_architecture_would_still_fail_this_table(components):
    """Guards the correction itself: restore the 24-expert budget and the table must go
    red again, or this suite is not actually testing feasibility."""
    inflated = dict(components)
    inflated["routed_experts"] = 13_589_544_960
    for quant in RELEASE_QUANTS:
        acc = account(config=RuntimeConfig(context_length=4_096, expert_quant=quant,
                                           dense_quant=quant, embedding_quant=quant),
                      components=inflated)
        assert acc.verdict() == DOES_NOT_FIT, (
            f"the 22.07B architecture appears to fit at {quant}, which contradicts the "
            "measurement that rejected it"
        )


def test_max_context_reports_zero_when_nothing_fits(components):
    """A model that cannot fit at the shortest context must report 0, not the shortest."""
    assert max_context(config=RuntimeConfig(expert_quant="q8_0", dense_quant="q8_0",
                                            embedding_quant="q8_0"),
                       usable_gib=2.0) == 0
    assert max_context(config=RuntimeConfig(expert_quant="q4_k_m", dense_quant="q4_k_m",
                                            embedding_quant="q4_k_m")) >= 131_072


def test_release_precisions_are_on_the_frontier(components):
    """Before the correction nothing on the frontier used a release precision. Now the
    release set is reachable, which is what makes a Q4/Q5 release plan real."""
    fits = frontier()
    assert fits, "nothing fits at all, which would be a different finding"
    assert any(f["uses_release_quant_only"] for f in fits)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def test_table_covers_the_required_matrix():
    table = build_table()
    assert len(table.rows) == len(RELEASE_QUANTS) * len(CONTEXT_LADDER)
    assert {r["quant"] for r in table.rows} == set(RELEASE_QUANTS)
    assert {r["context_length"] for r in table.rows} == set(CONTEXT_LADDER)


def test_table_saves_and_states_the_offload_rule(tmp_path):
    table = build_table()
    path = table.save(tmp_path / "memory.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["gpu_resident_only"] is True
    assert "must never be used to claim 16 GB compliance" in data["offload_note"]
    assert "offload" in offload_note().lower()


def test_rendered_table_shows_every_term_and_the_verdict():
    text = render_table()
    for header in ("weights", "KV", "state", "acts", "runtime", "TOTAL", "headroom"):
        assert header in text
    assert DOES_NOT_FIT in text
    assert "fully GPU-resident" in text


def test_account_serialises(components):
    data = account(components=components).to_dict()
    json.loads(json.dumps(data))
    assert data["gpu_resident_only"] is True
    assert set(data["gib"]) >= {"weights", "kv_cache", "recurrent_state", "conv_state",
                                "activations", "runtime_overhead", "total"}


def test_verdicts_are_ordered_by_headroom():
    acc = MemoryAccount(spec_name="x", config=RuntimeConfig(),
                        weights={"experts": 1}, kv_cache=1, recurrent_state=1,
                        conv_state=1, activations=1, runtime_overhead=1)
    assert acc.verdict(usable_gib=100.0) == "FIT"
    assert acc.verdict(usable_gib=0.5) == "BORDERLINE"
    assert acc.verdict(usable_gib=0.0) == DOES_NOT_FIT
