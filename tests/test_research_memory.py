"""End-to-end VRAM accounting against the 16 GB ceiling.

The load-bearing test here is :func:`test_frozen_student_does_not_fit_sixteen_gb`. It pins
a finding that is inconvenient — the frozen research target exceeds the deployment
constraint — so that no later change can quietly turn the answer into "fits" without the
change being visible in this file.
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
    """Experts are 61.6% of the model, so a per-component quantisation is the lever with the
    most mass. A uniform model would hide it."""
    cheap = weight_bytes(components, RuntimeConfig(expert_quant="q3_k_m", dense_quant="q6_k"))
    uniform = weight_bytes(components, RuntimeConfig(expert_quant="q6_k", dense_quant="q6_k"))
    assert cheap["experts"] < uniform["experts"]
    assert cheap["dense"] == uniform["dense"]
    assert cheap["experts"] > cheap["dense"] + cheap["embeddings"]


def test_weight_counts_come_from_the_audit(components):
    """Memory and parameter tables cannot disagree if they share one source."""
    acc = account(components=components)
    assert acc.parameters == sum(components.values()) == 22_072_134_528


def test_unknown_quantisation_is_refused(components):
    with pytest.raises(ValueError, match="unknown quantisation"):
        weight_bytes(components, RuntimeConfig(expert_quant="q2_vibes"))


# ---------------------------------------------------------------------------
# the total
# ---------------------------------------------------------------------------
def test_total_includes_every_term(components):
    acc = account(components=components)
    parts = (acc.weight_total, acc.kv_cache, acc.recurrent_state,
             acc.conv_state, acc.activations, acc.runtime_overhead)
    assert all(p > 0 for p in parts), "a term contributed nothing, which means it is missing"
    assert acc.total == sum(parts)


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
def test_frozen_student_does_not_fit_sixteen_gb(components):
    """The headline result, pinned. The frozen architecture exceeds the deployment
    constraint at every release precision and every context length — including 2,048
    tokens, before long context is even a factor."""
    for quant in RELEASE_QUANTS:
        for length in (CONTEXT_LADDER[0], CONTEXT_LADDER[-1]):
            acc = account(config=RuntimeConfig(context_length=length, expert_quant=quant,
                                               dense_quant=quant, embedding_quant=quant),
                          components=components)
            assert acc.verdict() == DOES_NOT_FIT, f"{quant} at {length} now fits — verify why"
    assert headline()["fits_at_any_release_quant"] is False


def test_the_shortfall_is_small_enough_to_be_worth_reporting_precisely(components):
    """Under half a gigabyte at the best case. That is a design decision away from fitting,
    which is why the number is reported rather than rounded to 'too big'."""
    h = headline()
    assert 0 < h["shortfall_gib"] < 1.0
    assert h["best_case_release_quant_gib"] > USABLE_GIB


def test_planning_against_the_nominal_sixteen_gib_gives_the_wrong_answer():
    """The specific arithmetic error this module exists to prevent: against a nominal
    16.0 GiB, Q4 appears to reach 64K; against a real card it never fits at all."""
    h = headline()
    naive = {row["quant"]: row["max_context"] for row in h["naive_nominal_16gib_result"]}
    assert naive["q4_k_m"] >= 65_536
    assert h["fits_at_any_release_quant"] is False
    assert USABLE_GIB < NOMINAL_VRAM_GIB


def test_fits_begin_below_the_release_precisions(components):
    fits = frontier()
    assert fits, "nothing fits at all, which would be a different finding"
    assert all(not f["uses_release_quant_only"] for f in fits)
    assert fits[0]["expert_quant"] == "q3_k_m"


def test_max_context_reports_zero_when_nothing_fits(components):
    assert max_context(config=RuntimeConfig(expert_quant="q4_k_m", dense_quant="q4_k_m")) == 0
    assert max_context(config=RuntimeConfig(expert_quant="q3_k_m", dense_quant="q3_k_m",
                                            embedding_quant="q3_k_m")) > 0


def test_a_smaller_expert_budget_is_what_would_close_the_gap(components):
    """Named in the implication text, and checked here so the advice is arithmetic rather
    than intuition: halving the routed-expert count frees more than the shortfall."""
    h = headline()
    halved = dict(components)
    halved["routed_experts"] //= 2
    acc = account(config=RuntimeConfig(context_length=2_048, expert_quant="q4_k_m",
                                       dense_quant="q4_k_m", embedding_quant="q4_k_m"),
                  components=halved)
    assert acc.headroom_gib() > 0
    assert h["shortfall_gib"] < (sum(components.values()) - sum(halved.values())) * 0.6125 / GIB


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
