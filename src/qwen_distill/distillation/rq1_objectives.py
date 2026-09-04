"""Objective kernels for the controlled RQ1 comparison.

This module does not own a training loop. It supplies the research-only losses used by the
canonical RQ1 launcher while preserving the validated generic trainer. Every loss computes
its gradient first with respect to detached stand-ins for student hidden states, then returns
those gradients for a single traversal of the student's real graph. This is the same memory
strategy already validated for Run 003's chunked layer KD.

Terminology matters:

* ``adjacent_residual`` is our internal hidden-state finite-difference ablation.
* ``span_delta`` is the topology-aware telescoping residual contribution from Run 004-M.
* ``fdd`` implements the defining prediction-space pieces of Gong et al. (ACL 2025): output
  KL, LM-head trajectory KL, and cosine alignment of first differences between adjacent
  selected prediction-space states. A raw residual delta must never be called FDD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .behavioral import (
    BehavioralLossOutput,
    ChunkedBehavioralLoss,
    _pair_term,
    behavioral_loss_chunked,
)


@dataclass
class RQ1LossOutput:
    """Trainer-compatible aggregate plus named research components."""

    total: float
    magnitude: float
    direction: float
    mode: str
    n_pairs: int
    student_norm: float = 1.0
    teacher_norm: float = 1.0
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": float(self.total),
            "magnitude": float(self.magnitude),
            "direction": float(self.direction),
            "mode": self.mode,
            "n_pairs": self.n_pairs,
            "student_norm": self.student_norm,
            "teacher_norm": self.teacher_norm,
            "components": dict(self.components),
        }


@dataclass
class RQ1ChunkedLoss:
    """A research objective whose gradients are ready for one student-graph traversal."""

    output: RQ1LossOutput
    sources: list[Any] = field(default_factory=list)
    grads: list[Any] = field(default_factory=list)

    def backward(self) -> None:
        import torch

        live = [(s, g) for s, g in zip(self.sources, self.grads, strict=True) if s.requires_grad]
        if live:
            torch.autograd.backward([s for s, _ in live], [g for _, g in live])


def _merge_sources(parts: list[ChunkedBehavioralLoss | RQ1ChunkedLoss]):
    """Merge already-scaled hidden-state gradients without traversing the model graph."""
    ordered: list[Any] = []
    grads: list[Any] = []
    by_id: dict[int, int] = {}
    for part in parts:
        for source, grad in zip(part.sources, part.grads, strict=True):
            key = id(source)
            if key in by_id:
                grads[by_id[key]] = grads[by_id[key]] + grad
            else:
                by_id[key] = len(ordered)
                ordered.append(source)
                grads.append(grad)
    return ordered, grads


def adjacent_residual_loss_chunked(
    student_hidden,
    teacher_hidden,
    mapping: dict[int, int],
    *,
    direction_weight: float = 1.0,
    normalise: bool = True,
    chunk_pairs: int = 4,
    loss_scale: float = 1.0,
    backward: Any = None,
) -> ChunkedBehavioralLoss:
    """Match one student residual transition to the adjacent mapped teacher transition.

    For mapped layer ``m(s)`` this compares::

        h_s[s+1] - h_s[s]
        h_t[m(s)+1] - h_t[m(s)]

    Removed teacher layers are therefore *not* absorbed. This makes the objective the clean
    adjacent-vs-span abstraction control for the telescoping span target.
    """
    import torch

    if chunk_pairs < 1:
        raise ValueError("chunk_pairs must be >= 1")
    pairs = tuple(sorted(mapping))
    if not pairs:
        raise ValueError("mapping is empty")
    if backward is None:
        def backward(tensor):  # noqa: E306
            tensor.backward()

    touched = sorted({i for s in pairs for i in (s, s + 1)})
    view = list(student_hidden)
    for i in touched:
        if i >= len(view):
            raise ValueError(f"student boundary {i} is out of range")
        leaf = student_hidden[i].detach()
        leaf.requires_grad_(True)
        view[i] = leaf

    mag_sum = dir_sum = 0.0
    per_layer: dict[int, float] = {}
    s_norms: list[float] = []
    t_norms: list[float] = []
    n_pairs = len(pairs)
    n_chunks = 0
    for start in range(0, n_pairs, chunk_pairs):
        terms = []
        for s in pairs[start:start + chunk_pairs]:
            t = mapping[s]
            if t + 1 >= len(teacher_hidden):
                raise ValueError(f"teacher boundary {t + 1} is out of range")
            student = view[s + 1] - view[s]
            teacher = teacher_hidden[t + 1] - teacher_hidden[t]
            magnitude, direction, s_norm, t_norm = _pair_term(
                student, teacher, s, normalise=normalise, mask=None
            )
            terms.append(magnitude + direction_weight * direction)
            mag_sum += float(magnitude.detach())
            dir_sum += float(direction.detach())
            per_layer[s] = float(magnitude.detach())
            s_norms.append(s_norm)
            t_norms.append(t_norm)
        chunk = torch.stack(terms).sum() / n_pairs
        if loss_scale != 1.0:
            chunk = chunk * loss_scale
        backward(chunk)
        n_chunks += 1

    output = BehavioralLossOutput(
        total=mag_sum / n_pairs + direction_weight * dir_sum / n_pairs,
        magnitude=mag_sum / n_pairs,
        direction=dir_sum / n_pairs,
        mode="adjacent_residual",
        n_pairs=n_pairs,
        per_layer=per_layer,
        student_norm=sum(s_norms) / len(s_norms),
        teacher_norm=sum(t_norms) / len(t_norms),
    )
    sources, grads = [], []
    for i in touched:
        leaf = view[i]
        sources.append(student_hidden[i])
        grads.append(leaf.grad if leaf.grad is not None else torch.zeros_like(leaf))
    return ChunkedBehavioralLoss(
        output=output, sources=sources, grads=grads,
        n_chunks=n_chunks, chunk_pairs=chunk_pairs,
    )


def anchored_transition_loss_chunked(
    student_hidden,
    teacher_hidden,
    mapping: dict[int, int],
    *,
    transition: str,
    pointwise_weight: float = 1.0,
    transition_weight: float = 1.0,
    direction_weight: float = 1.0,
    normalise: bool = True,
    chunk_pairs: int = 4,
    loss_scale: float = 1.0,
    backward: Any = None,
) -> RQ1ChunkedLoss:
    """Absolute-state anchor plus either adjacent or topology-span transition supervision."""
    if transition not in {"adjacent", "span"}:
        raise ValueError("transition must be 'adjacent' or 'span'")
    point = behavioral_loss_chunked(
        student_hidden, teacher_hidden, mapping,
        mode="pointwise", direction_weight=direction_weight, normalise=normalise,
        chunk_pairs=chunk_pairs, loss_scale=loss_scale * pointwise_weight,
        backward=backward,
    )
    if transition == "span":
        trans = behavioral_loss_chunked(
            student_hidden, teacher_hidden, mapping,
            mode="delta", direction_weight=direction_weight, normalise=normalise,
            chunk_pairs=chunk_pairs, loss_scale=loss_scale * transition_weight,
            backward=backward,
        )
    else:
        trans = adjacent_residual_loss_chunked(
            student_hidden, teacher_hidden, mapping,
            direction_weight=direction_weight, normalise=normalise,
            chunk_pairs=chunk_pairs, loss_scale=loss_scale * transition_weight,
            backward=backward,
        )
    sources, grads = _merge_sources([point, trans])
    p, t = point.output, trans.output
    return RQ1ChunkedLoss(
        output=RQ1LossOutput(
            total=pointwise_weight * float(p.total) + transition_weight * float(t.total),
            magnitude=pointwise_weight * float(p.magnitude) + transition_weight * float(t.magnitude),
            direction=pointwise_weight * float(p.direction) + transition_weight * float(t.direction),
            mode=f"pointwise_plus_{transition}",
            n_pairs=p.n_pairs + t.n_pairs,
            student_norm=p.student_norm,
            teacher_norm=p.teacher_norm,
            components={
                "pointwise_total": float(p.total),
                f"{transition}_total": float(t.total),
                "pointwise_weight": pointwise_weight,
                "transition_weight": transition_weight,
            },
        ),
        sources=sources,
        grads=grads,
    )


def uniform_fdd_schedule(student_layers: int, teacher_layers: int, n_layers: int = 4):
    """Uniform intermediate-layer schedule described by Gong et al. for FDD.

    For total depth ``L`` and desired sampled-layer count ``LD``, their experimental setup
    uses ``Q=floor(L/(LD+1))`` and indexes ``{Q, 2Q, ..., LD*Q}``.
    """
    if n_layers < 2:
        raise ValueError("FDD needs at least two sampled layers to define a derivative")
    if student_layers <= n_layers or teacher_layers <= n_layers:
        raise ValueError("model depth must exceed the number of sampled FDD layers")
    sq = student_layers // (n_layers + 1)
    tq = teacher_layers // (n_layers + 1)
    if sq < 1 or tq < 1:
        raise ValueError("FDD sampling interval collapsed to zero")
    return ([sq * i for i in range(1, n_layers + 1)],
            [tq * i for i in range(1, n_layers + 1)])


def _head_log_probs(head, hidden, temperature: float):
    import torch.nn.functional as F

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return F.log_softmax(head(hidden).float() / temperature, dim=-1)


def fdd_prediction_dynamics_chunked(
    student_hidden,
    teacher_hidden,
    student_head,
    teacher_head,
    *,
    sampled_layers: int = 4,
    alpha: float = 1.0,
    beta: float = 1.0,
    output_kd_weight: float = 1.0,
    trajectory_temperature: float = 1.0,
    output_temperature: float = 2.0,
    token_chunk: int = 16,
    loss_scale: float = 1.0,
    backward: Any = None,
) -> RQ1ChunkedLoss:
    """Prediction-space feature dynamics distillation, memory-chunked over tokens.

    The implementation follows the defining FDD structure while documenting this project's
    adaptation: four uniformly sampled intermediate layers by default, alpha=beta=1, each
    model's own frozen LM head, full shared vocabulary, and the project's registered output
    KD temperature. No tuned-lens adapters are introduced because they would add another
    trainable method component to the comparator.
    """
    import torch
    import torch.nn.functional as F

    if token_chunk < 1:
        raise ValueError("token_chunk must be >= 1")
    if backward is None:
        def backward(tensor):  # noqa: E306
            tensor.backward()
    if any(p.requires_grad for p in student_head.parameters()):
        raise ValueError("FDD chunking assumes the canonical frozen student LM head")
    if any(p.requires_grad for p in teacher_head.parameters()):
        raise ValueError("teacher LM head must be frozen")

    s_idx, t_idx = uniform_fdd_schedule(
        len(student_hidden) - 1, len(teacher_hidden) - 1, sampled_layers
    )
    # The final normalized hidden state reproduces the model's ordinary output head and
    # supplies conventional output KD without requiring a second traversal of model logits.
    s_final, t_final = len(student_hidden) - 1, len(teacher_hidden) - 1
    touched = sorted(set(s_idx + [s_final]))
    view = list(student_hidden)
    for i in touched:
        leaf = student_hidden[i].detach()
        leaf.requires_grad_(True)
        view[i] = leaf

    shape = student_hidden[0].shape
    if len(shape) != 3 or teacher_hidden[0].shape[:2] != shape[:2]:
        raise ValueError("FDD expects aligned [batch, sequence, hidden] teacher/student states")
    batch, seq = shape[:2]
    total_positions = batch * seq
    sums = {"output_kd": 0.0, "trajectory_kl": 0.0, "derivative_cosine": 0.0}

    for start in range(0, seq, token_chunk):
        end = min(seq, start + token_chunk)
        positions = batch * (end - start)
        fraction = positions / total_positions

        with torch.no_grad():
            t_logs = [
                _head_log_probs(teacher_head, teacher_hidden[i][:, start:end], trajectory_temperature)
                for i in t_idx
            ]
            t_out = _head_log_probs(
                teacher_head, teacher_hidden[t_final][:, start:end], output_temperature
            )
        s_logs = [
            _head_log_probs(student_head, view[i][:, start:end], trajectory_temperature)
            for i in s_idx
        ]
        s_out = _head_log_probs(
            student_head, view[s_final][:, start:end], output_temperature
        )

        traj_terms = [
            F.kl_div(s.reshape(-1, s.shape[-1]), t.reshape(-1, t.shape[-1]),
                     reduction="batchmean", log_target=True)
            for s, t in zip(s_logs, t_logs, strict=True)
        ]
        trajectory = torch.stack(traj_terms).mean()

        derivative_terms = []
        for j in range(1, sampled_layers):
            s_delta = s_logs[j] - s_logs[j - 1]
            t_delta = t_logs[j] - t_logs[j - 1]
            derivative_terms.append(
                1.0 - F.cosine_similarity(
                    s_delta.reshape(-1, s_delta.shape[-1]),
                    t_delta.reshape(-1, t_delta.shape[-1]), dim=-1
                ).mean()
            )
        derivative = torch.stack(derivative_terms).mean()
        output_kd = F.kl_div(
            s_out.reshape(-1, s_out.shape[-1]),
            t_out.reshape(-1, t_out.shape[-1]),
            reduction="batchmean", log_target=True,
        ) * (output_temperature ** 2)

        chunk_loss = fraction * (
            output_kd_weight * output_kd + alpha * trajectory + beta * derivative
        )
        sums["output_kd"] += fraction * float(output_kd.detach())
        sums["trajectory_kl"] += fraction * float(trajectory.detach())
        sums["derivative_cosine"] += fraction * float(derivative.detach())
        if loss_scale != 1.0:
            chunk_loss = chunk_loss * loss_scale
        backward(chunk_loss)

    sources, grads = [], []
    for i in touched:
        leaf = view[i]
        sources.append(student_hidden[i])
        grads.append(leaf.grad if leaf.grad is not None else torch.zeros_like(leaf))
    total = output_kd_weight * sums["output_kd"] + alpha * sums["trajectory_kl"] + beta * sums["derivative_cosine"]
    return RQ1ChunkedLoss(
        output=RQ1LossOutput(
            total=total,
            magnitude=sums["trajectory_kl"],
            direction=sums["derivative_cosine"],
            mode="fdd_prediction_dynamics",
            n_pairs=sampled_layers,
            components={
                **sums,
                "alpha": alpha,
                "beta": beta,
                "output_kd_weight": output_kd_weight,
                "sampled_layers": float(sampled_layers),
                "student_schedule": s_idx,
                "teacher_schedule": t_idx,
                "token_chunk": float(token_chunk),
            },
        ),
        sources=sources,
        grads=grads,
    )
