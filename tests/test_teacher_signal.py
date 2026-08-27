"""Tests for where the teacher distribution comes from.

Kept apart from the loss tests on purpose: this is the axis where a run can turn into SFT
without anything looking wrong, so the interesting assertions are about what is refused
and about the provenance that survives into the run record.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from qwen_distill.architecture.spec import HybridArchSpec  # noqa: E402
from qwen_distill.distillation.kd_loss import TeacherSignal, kd_divergence  # noqa: E402
from qwen_distill.distillation.teacher_signal import (  # noqa: E402
    OnlineTeacher,
    ReplaySignals,
    build_provider,
    estimate_offline_corpus_gib,
)

TINY = dict(
    hidden_size=64, num_hidden_layers=4, intermediate_size=128, vocab_size=256,
    num_attention_heads=2, num_key_value_heads=1, head_dim=32,
    linear_num_key_heads=1, linear_num_value_heads=2,
    linear_key_head_dim=32, linear_value_head_dim=32,
    full_attention_interval=4, tie_word_embeddings=True, max_position_embeddings=256,
)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    spec = HybridArchSpec(name="tiny", **TINY)
    fields = {k: v for k, v in spec.to_hf_text_config().items() if k != "model_type"}
    return AutoModelForCausalLM.from_config(AutoConfig.for_model("qwen3_5_text", **fields))


def test_the_online_teacher_produces_a_usable_signal(model):
    teacher = OnlineTeacher(model=model, top_k=8, temperature=1.0)
    ids = torch.randint(0, 256, (2, 16))
    signal = teacher.signal_for(ids)

    assert signal.k == 8
    assert signal.top_values.shape == (2, 16, 8)
    assert signal.logsumexp.shape == (2, 16)
    assert signal.logsumexp_temperature == 1.0
    divergence, _ = kd_divergence(torch.randn(2, 16, 256), signal, tail="bucket")
    assert divergence.item() >= 0


def test_the_signal_carries_no_gradient(model):
    """The teacher is not being trained, and holding its graph would double the memory."""
    teacher = OnlineTeacher(model=model, top_k=4)
    signal = teacher.signal_for(torch.randint(0, 256, (1, 8)))
    assert not signal.top_values.requires_grad
    assert not signal.logsumexp.requires_grad


def test_a_dense_teacher_is_available_when_the_vocabulary_is_small_enough(model):
    teacher = OnlineTeacher(model=model, top_k=None)
    signal = teacher.signal_for(torch.randint(0, 256, (1, 8)))
    assert signal.is_dense
    assert signal.logits.shape == (1, 8, 256)


def test_the_model_is_left_in_the_mode_it_arrived_in(model):
    teacher = OnlineTeacher(model=model, top_k=4)
    model.train()
    teacher.signal_for(torch.randint(0, 256, (1, 8)))
    assert model.training

    model.eval()
    teacher.signal_for(torch.randint(0, 256, (1, 8)))
    assert not model.training


def test_provenance_reaches_the_signal(model):
    """A signal with no provenance is not reproducible: the same repo id has served
    different weights over time."""
    teacher = OnlineTeacher(
        model=model, top_k=4, teacher_model="Qwen/Qwen3.8-27B", teacher_revision="abc123"
    )
    signal = teacher.signal_for(torch.randint(0, 256, (1, 8)))
    assert signal.metadata["teacher_model"] == "Qwen/Qwen3.8-27B"
    assert signal.metadata["teacher_revision"] == "abc123"
    assert signal.metadata["source"] == "online"


# --- replay ---------------------------------------------------------------
def test_replay_returns_captured_signals_in_order():
    signals = [TeacherSignal(logits=torch.randn(1, 4, 8)) for _ in range(3)]
    provider = ReplaySignals(signals=signals)
    ids = torch.zeros(1, 4, dtype=torch.long)
    assert provider.signal_for(ids) is signals[0]
    assert provider.signal_for(ids) is signals[1]
    assert provider.signal_for(ids) is signals[2]
    assert provider.signal_for(ids) is signals[0]  # wraps


def test_replay_refuses_a_signal_that_does_not_cover_the_batch():
    """A misaligned signal would train the student against another batch's teacher —
    silently, and with a perfectly normal-looking loss curve."""
    provider = ReplaySignals(signals=[TeacherSignal(logits=torch.randn(1, 4, 8))])
    with pytest.raises(ValueError, match="another batch's teacher"):
        provider.signal_for(torch.zeros(2, 16, dtype=torch.long))


def test_replay_with_nothing_to_replay_says_so():
    with pytest.raises(IndexError, match="no captured signals"):
        ReplaySignals(signals=[]).signal_for(torch.zeros(1, 2, dtype=torch.long))


# --- construction ---------------------------------------------------------
def test_the_offline_reader_refuses_with_the_reason_it_is_missing():
    with pytest.raises(NotImplementedError, match="on-disk corpus layout"):
        build_provider("offline")


def test_an_online_provider_without_a_model_is_refused():
    with pytest.raises(ValueError, match="needs a loaded teacher model"):
        build_provider("online")


def test_an_unknown_provider_is_refused():
    with pytest.raises(ValueError, match="unknown teacher signal provider"):
        build_provider("telepathy")


def test_offline_storage_is_sized_for_the_decision_it_informs():
    """3.9 GB per 10M tokens at k=64 is the number the online/offline choice turns on."""
    assert estimate_offline_corpus_gib(10_000_000, 64) == pytest.approx(3.61, abs=0.01)
    assert estimate_offline_corpus_gib(10_000_000, 128) > estimate_offline_corpus_gib(
        10_000_000, 64
    )
