"""Tests for the knowledge-distillation objective.

The central risk is not a crash but a redefinition: a KD implementation that quietly
optimises something other than "match the teacher's distribution" produces a training
curve that looks fine and a comparison against SFT that means nothing. So the anchor test
here is exact equivalence between the sparse path at ``k = vocab`` and the dense path, and
the behavioural test is that the two tail treatments differ in precisely the way the
module claims they do.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from qwen_distill.distillation.kd_loss import (  # noqa: E402
    IGNORE_INDEX,
    KDSignalError,
    TeacherSignal,
    capture_signal,
    distillation_loss,
    kd_divergence,
    signal_bytes_per_token,
)

BATCH, POSITIONS, VOCAB = 2, 6, 40


@pytest.fixture
def logits():
    torch.manual_seed(0)
    return torch.randn(BATCH, POSITIONS, VOCAB) * 2, torch.randn(BATCH, POSITIONS, VOCAB)


# --- the anchor -----------------------------------------------------------
def test_sparse_at_full_k_is_the_dense_objective(logits):
    """If these disagree, the sparse path is optimising something else."""
    teacher, student = logits
    dense, dense_diagnostics = kd_divergence(student, TeacherSignal(logits=teacher))
    sparse, sparse_diagnostics = kd_divergence(student, capture_signal(teacher, top_k=VOCAB))

    assert dense.item() == pytest.approx(sparse.item(), abs=1e-5)
    assert dense_diagnostics["teacher_entropy"] == pytest.approx(
        sparse_diagnostics["teacher_entropy"], abs=1e-5
    )
    assert sparse_diagnostics["tail_mass"] == pytest.approx(0.0, abs=1e-6)


def test_renormalize_at_full_k_is_also_the_dense_objective(logits):
    """Renormalising over the whole vocabulary is the whole vocabulary."""
    teacher, student = logits
    dense, _ = kd_divergence(student, TeacherSignal(logits=teacher))
    renormalized, _ = kd_divergence(
        student, capture_signal(teacher, top_k=VOCAB), tail="renormalize"
    )
    assert dense.item() == pytest.approx(renormalized.item(), abs=1e-5)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0, 4.0])
def test_sparse_matches_dense_at_every_temperature(logits, temperature):
    teacher, student = logits
    dense, _ = kd_divergence(student, TeacherSignal(logits=teacher), temperature=temperature)
    sparse, _ = kd_divergence(
        student,
        capture_signal(teacher, top_k=VOCAB, temperature=temperature),
        temperature=temperature,
    )
    assert dense.item() == pytest.approx(sparse.item(), abs=1e-4)


# --- it is a divergence ---------------------------------------------------
def test_self_distillation_is_zero(logits):
    teacher, _ = logits
    divergence, _ = kd_divergence(teacher, TeacherSignal(logits=teacher))
    assert divergence.item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("tail", ["bucket", "renormalize"])
def test_divergence_is_never_negative(tail):
    torch.manual_seed(7)
    for _ in range(50):
        teacher = torch.randn(BATCH, POSITIONS, VOCAB) * 3
        student = torch.randn(BATCH, POSITIONS, VOCAB) * 3
        divergence, _ = kd_divergence(student, capture_signal(teacher, top_k=5), tail=tail)
        assert divergence.item() >= -1e-6


# --- the reason the logsumexp is stored -----------------------------------
def test_bucket_penalises_mass_outside_the_top_k_and_renormalize_cannot_see_it(logits):
    """The measured difference between the two objectives.

    A student that dumps probability on a token the teacher's top-k excludes is worse by
    any reading of "match the teacher". ``bucket`` says so; ``renormalize`` is blind to it
    because that mass never enters its k-simplex. This is what the extra fp32 per token
    buys, stated as a test rather than an argument.
    """
    teacher, student = logits
    signal = capture_signal(teacher, top_k=4)
    outside = next(
        v for v in range(VOCAB) if v not in signal.top_indices[0, 0].tolist()
    )
    dumped = student.clone()
    dumped[..., outside] += 8.0

    bucket_before, _ = kd_divergence(student, signal, tail="bucket")
    bucket_after, _ = kd_divergence(dumped, signal, tail="bucket")
    renormalized_before, _ = kd_divergence(student, signal, tail="renormalize")
    renormalized_after, _ = kd_divergence(dumped, signal, tail="renormalize")

    assert bucket_after.item() > bucket_before.item() * 1.5
    assert renormalized_after.item() == pytest.approx(renormalized_before.item(), abs=1e-5)


def test_tail_mass_falls_as_k_grows(logits):
    """The diagnostic that answers 'is k big enough' from data rather than convention."""
    teacher, student = logits
    masses = [
        kd_divergence(student, capture_signal(teacher, top_k=k))[1]["tail_mass"]
        for k in (1, 2, 4, 8, 16, VOCAB)
    ]
    assert masses == sorted(masses, reverse=True)
    assert masses[0] > 0.1
    assert masses[-1] == pytest.approx(0.0, abs=1e-6)


# --- what it refuses ------------------------------------------------------
def test_bucket_without_a_logsumexp_is_refused_not_approximated(logits):
    teacher, student = logits
    signal = capture_signal(teacher, top_k=4)
    stripped = TeacherSignal(top_values=signal.top_values, top_indices=signal.top_indices)
    with pytest.raises(KDSignalError, match="logsumexp"):
        kd_divergence(student, stripped, tail="bucket")


def test_a_logsumexp_from_another_temperature_is_refused(logits):
    teacher, student = logits
    signal = capture_signal(teacher, top_k=4, temperature=1.0)
    with pytest.raises(KDSignalError, match="temperature"):
        kd_divergence(student, signal, tail="bucket", temperature=2.0)


def test_half_a_sparse_signal_is_refused(logits):
    teacher, student = logits
    signal = capture_signal(teacher, top_k=4)
    with pytest.raises(KDSignalError, match="top_values and top_indices"):
        kd_divergence(student, TeacherSignal(top_indices=signal.top_indices), tail="renormalize")


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_non_positive_temperature_is_rejected(logits, temperature):
    teacher, student = logits
    with pytest.raises(ValueError, match="temperature must be positive"):
        kd_divergence(student, TeacherSignal(logits=teacher), temperature=temperature)


# --- the combined loss ----------------------------------------------------
def test_alpha_zero_is_exactly_cross_entropy(logits):
    """The SFT control must come out of this code path, not a parallel one."""
    teacher, student = logits
    labels = torch.randint(0, VOCAB, (BATCH, POSITIONS))
    output = distillation_loss(
        student, labels, TeacherSignal(logits=teacher), alpha=0.0
    )
    reference = torch.nn.functional.cross_entropy(
        student[:, :-1].reshape(-1, VOCAB).float(), labels[:, 1:].reshape(-1)
    )
    assert output.total.item() == pytest.approx(reference.item(), abs=1e-6)
    assert output.cross_entropy.item() == pytest.approx(reference.item(), abs=1e-6)


def test_alpha_one_is_pure_kd(logits):
    teacher, student = logits
    labels = torch.randint(0, VOCAB, (BATCH, POSITIONS))
    output = distillation_loss(student, labels, TeacherSignal(logits=teacher), alpha=1.0)
    assert output.total.item() == pytest.approx(output.kd.item(), abs=1e-6)
    assert output.cross_entropy.item() > 0  # still reported, just not optimised


def test_alpha_mixes_the_two_terms(logits):
    teacher, student = logits
    labels = torch.randint(0, VOCAB, (BATCH, POSITIONS))
    output = distillation_loss(student, labels, TeacherSignal(logits=teacher), alpha=0.3)
    expected = 0.3 * output.kd.item() + 0.7 * output.cross_entropy.item()
    assert output.total.item() == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_alpha_outside_the_unit_interval_is_rejected(logits, alpha):
    teacher, student = logits
    labels = torch.randint(0, VOCAB, (BATCH, POSITIONS))
    with pytest.raises(ValueError, match="alpha"):
        distillation_loss(student, labels, TeacherSignal(logits=teacher), alpha=alpha)


def test_ignored_labels_are_excluded_from_both_terms(logits):
    """Prompt positions carry no supervision, and must not dilute either loss."""
    teacher, student = logits
    labels = torch.randint(0, VOCAB, (BATCH, POSITIONS))
    labels[:, :3] = IGNORE_INDEX
    output = distillation_loss(student, labels, TeacherSignal(logits=teacher), alpha=0.5)
    # positions 1 and 2 of the shifted labels are ignored, leaving 3 of 5 per sequence
    assert output.n_scored == BATCH * 3


def test_everything_masked_gives_a_finite_zero_loss(logits):
    teacher, student = logits
    labels = torch.full((BATCH, POSITIONS), IGNORE_INDEX)
    output = distillation_loss(student, labels, TeacherSignal(logits=teacher), alpha=0.5)
    assert output.n_scored == 0
    assert torch.isfinite(output.total)
    assert output.total.item() == pytest.approx(0.0, abs=1e-9)


def test_the_loss_is_differentiable_in_the_student(logits):
    teacher, student = logits
    student = student.clone().requires_grad_(True)
    labels = torch.randint(0, VOCAB, (BATCH, POSITIONS))
    output = distillation_loss(student, labels, capture_signal(teacher, top_k=8), alpha=0.7)
    output.total.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert student.grad.abs().sum() > 0


def test_shifting_can_be_turned_off_for_pre_aligned_inputs(logits):
    teacher, student = logits
    labels = torch.randint(0, VOCAB, (BATCH, POSITIONS))
    shifted = distillation_loss(student, labels, TeacherSignal(logits=teacher), alpha=1.0)
    manual = distillation_loss(
        student[:, :-1], labels[:, 1:], TeacherSignal(logits=teacher[:, :-1]),
        alpha=1.0, shift=False,
    )
    assert shifted.total.item() == pytest.approx(manual.total.item(), abs=1e-6)


# --- diagnostics and capture ----------------------------------------------
def test_top1_agreement_is_one_when_the_student_is_the_teacher(logits):
    teacher, _ = logits
    _, diagnostics = kd_divergence(teacher, capture_signal(teacher, top_k=4))
    assert diagnostics["top1_agreement"] == pytest.approx(1.0)


def test_capture_takes_the_logsumexp_before_truncating(logits):
    """After truncation the normaliser is unrecoverable, so the order matters."""
    teacher, _ = logits
    signal = capture_signal(teacher, top_k=3)
    assert torch.allclose(signal.logsumexp, torch.logsumexp(teacher.float(), dim=-1), atol=1e-5)
    assert signal.k == 3
    assert signal.top_indices.dtype == torch.int32


@pytest.mark.parametrize("top_k", [0, -1, VOCAB + 1])
def test_capture_rejects_an_impossible_k(logits, top_k):
    teacher, _ = logits
    with pytest.raises(ValueError):
        capture_signal(teacher, top_k=top_k)


def test_storage_cost_is_reported_for_sizing_an_offline_corpus():
    assert signal_bytes_per_token(64) == 64 * 6 + 4
    assert signal_bytes_per_token(64) * 10_000_000 / 1e9 == pytest.approx(3.88, abs=0.01)


def test_log_record_carries_the_parts_not_just_the_total(logits):
    teacher, student = logits
    labels = torch.randint(0, VOCAB, (BATCH, POSITIONS))
    record = distillation_loss(
        student, labels, capture_signal(teacher, top_k=8), alpha=0.5
    ).to_log()
    assert {"loss", "kd_loss", "ce_loss", "teacher_entropy", "top1_agreement",
            "teacher_tail_mass", "n_scored"} <= set(record)


def test_moving_a_signal_never_casts_its_token_indices(logits):
    """Indices are token identities. A positional dtype would turn them into floats and
    corrupt the gather they exist for, with no error anywhere."""
    teacher, _ = logits
    signal = capture_signal(teacher, top_k=4)
    moved = signal.to(dtype=torch.float16)
    assert moved.top_indices.dtype == signal.top_indices.dtype
    assert moved.top_values.dtype == torch.float16
    assert moved.logsumexp.dtype == torch.float16
    assert torch.equal(moved.top_indices, signal.top_indices)
