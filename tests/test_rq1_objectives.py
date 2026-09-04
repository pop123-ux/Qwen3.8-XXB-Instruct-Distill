from __future__ import annotations

import pytest
from conftest import HAS_TORCH

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="requires torch")


def _states(n_layers: int, *, width: int = 8, scale: float = 1.0):
    import torch

    base = torch.arange(2 * 3 * width, dtype=torch.float32).reshape(2, 3, width) / 100
    return tuple((base + scale * i * 0.07).clone().requires_grad_(True) for i in range(n_layers + 1))


def test_uniform_fdd_schedule_matches_paper_rule_for_48_to_64() -> None:
    from qwen_distill.distillation.rq1_objectives import uniform_fdd_schedule

    student, teacher = uniform_fdd_schedule(48, 64, 4)
    assert student == [9, 18, 27, 36]
    assert teacher == [12, 24, 36, 48]


def test_adjacent_residual_does_not_absorb_removed_teacher_layers() -> None:
    import torch

    from qwen_distill.distillation.rq1_objectives import adjacent_residual_loss_chunked

    student = _states(2)
    teacher = list(_states(4))
    # Make the teacher layer skipped between the two anchors enormous. An adjacent target
    # for student layer 1 uses teacher 3->4 only, so teacher boundary 2 must not affect it.
    teacher[2] = teacher[2] + 1000
    mapping = {0: 0, 1: 3}
    loss = adjacent_residual_loss_chunked(
        student, tuple(teacher), mapping, normalise=False, chunk_pairs=1
    )
    assert loss.output.mode == "adjacent_residual"
    assert loss.output.n_pairs == 2
    assert all(g is not None for g in loss.grads)


def test_pointwise_plus_span_is_sum_of_registered_components() -> None:
    from qwen_distill.distillation.rq1_objectives import anchored_transition_loss_chunked

    student = _states(4)
    teacher = _states(8, scale=1.3)
    mapping = {0: 0, 1: 1, 2: 6, 3: 7}
    out = anchored_transition_loss_chunked(
        student, teacher, mapping, transition="span", normalise=True,
        pointwise_weight=1.0, transition_weight=1.0, chunk_pairs=2,
    )
    c = out.output.components
    assert out.output.total == pytest.approx(c["pointwise_total"] + c["span_total"], rel=1e-6)
    out.backward()
    assert any(s.grad is not None for s in student)


def test_pointwise_plus_adjacent_is_not_labelled_fdd() -> None:
    from qwen_distill.distillation.rq1_objectives import anchored_transition_loss_chunked

    student = _states(4)
    teacher = _states(8, scale=1.2)
    mapping = {0: 0, 1: 1, 2: 6, 3: 7}
    out = anchored_transition_loss_chunked(
        student, teacher, mapping, transition="adjacent", chunk_pairs=2
    )
    assert out.output.mode == "pointwise_plus_adjacent"
    assert "adjacent_total" in out.output.components
    assert "fdd" not in out.output.mode.lower()


def test_fdd_identical_prediction_dynamics_is_near_zero() -> None:
    import torch

    from qwen_distill.distillation.rq1_objectives import fdd_prediction_dynamics_chunked

    # Different depths, but the selected schedules are populated with exactly corresponding
    # prediction-space states and the final states match as well.
    student = list(_states(9, width=6))
    teacher = list(_states(14, width=6, scale=0.4))
    s_idx = [1, 2, 3, 4]
    t_idx = [2, 4, 6, 8]
    # n_layers=4 on depths 9/14 gives Q=1/2 -> schedules above.
    for rank, (s, t) in enumerate(zip(s_idx, t_idx, strict=True), start=1):
        value = torch.randn_like(student[s]) + rank * 0.2
        student[s] = value.clone().requires_grad_(True)
        teacher[t] = value.clone()
    final = torch.randn_like(student[-1])
    student[-1] = final.clone().requires_grad_(True)
    teacher[-1] = final.clone()

    head_s = torch.nn.Linear(6, 11, bias=False)
    head_t = torch.nn.Linear(6, 11, bias=False)
    head_t.weight.data.copy_(head_s.weight.data)
    for p in head_s.parameters():
        p.requires_grad_(False)
    for p in head_t.parameters():
        p.requires_grad_(False)

    loss = fdd_prediction_dynamics_chunked(
        tuple(student), tuple(teacher), head_s, head_t,
        sampled_layers=4, token_chunk=2, trajectory_temperature=1.0,
        output_temperature=2.0,
    )
    assert loss.output.components["trajectory_kl"] == pytest.approx(0.0, abs=1e-6)
    assert loss.output.components["derivative_cosine"] == pytest.approx(0.0, abs=1e-5)
    assert loss.output.components["output_kd"] == pytest.approx(0.0, abs=1e-6)
    assert loss.output.total == pytest.approx(0.0, abs=1e-5)


def test_fdd_produces_hidden_state_gradients_with_frozen_heads() -> None:
    import torch

    from qwen_distill.distillation.rq1_objectives import fdd_prediction_dynamics_chunked

    student = _states(9, width=6)
    teacher = _states(14, width=6, scale=1.4)
    head_s = torch.nn.Linear(6, 13, bias=False)
    head_t = torch.nn.Linear(6, 13, bias=False)
    for p in list(head_s.parameters()) + list(head_t.parameters()):
        p.requires_grad_(False)
    loss = fdd_prediction_dynamics_chunked(
        student, teacher, head_s, head_t, sampled_layers=4, token_chunk=2
    )
    assert loss.sources
    assert all(g is not None for g in loss.grads)
    loss.backward()
    assert any(s.grad is not None for s in student)


def test_fdd_refuses_a_trainable_lm_head() -> None:
    import torch

    from qwen_distill.distillation.rq1_objectives import fdd_prediction_dynamics_chunked

    student = _states(9, width=4)
    teacher = _states(14, width=4)
    trainable = torch.nn.Linear(4, 7, bias=False)
    frozen = torch.nn.Linear(4, 7, bias=False)
    for p in frozen.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="frozen student LM head"):
        fdd_prediction_dynamics_chunked(student, teacher, trainable, frozen)
