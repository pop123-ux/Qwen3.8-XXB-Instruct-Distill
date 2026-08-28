"""The knowledge-distillation objective itself: matching a teacher's distribution.

SFT trains on what the teacher *sampled*. KD trains on what the teacher *believed*, which
is strictly more information per token — a teacher that puts 0.4 on "increase" and 0.35 on
"raise" teaches something a single sampled token cannot.

The awkward part is that the teacher's belief is a vector over 248,320 tokens, and storing
one per training token is not affordable. Top-k is the standard answer, and it has a trap
this module exists to close:

    **Top-k logits alone do not determine the teacher's distribution.**

``softmax`` needs the sum over the *whole* vocabulary, and the top ``k`` values do not
contain it. An implementation that renormalises the top-k and calls the result "the
teacher's distribution" has silently redefined the objective: it stops penalising the
student for putting mass on the 248,256 tokens the teacher rejected. That is not a
detail — it is the difference between "be like the teacher" and "rank the teacher's
shortlist the way the teacher does".

So a sparse signal carries one extra number per token, the full-vocabulary
``logsumexp``, and with it the tail mass ``1 - sum(top-k)`` is exact. At k=64 that is
64 int32 indices + 64 fp16 values + 1 fp32 = 388 bytes per token — about 3.9 GB per 10M
tokens, which is affordable to store and to move.

Both tail treatments are implemented because the comparison is worth measuring rather
than asserting, and ``tail_mass`` is reported on every step so the choice of ``k`` is
answered by data instead of by convention.

Nothing here knows or cares whether the teacher ran a moment ago or a month ago: a
:class:`TeacherSignal` is a :class:`TeacherSignal`. That is what lets the online and
offline decision be made on cost, later, without touching the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

TailTreatment = Literal["bucket", "renormalize"]

#: ``labels`` positions to exclude from both losses, matching Hugging Face's convention.
IGNORE_INDEX = -100

#: Floor for probabilities entering a logarithm. Below fp16's smallest normal (6.1e-05)
#: but far above fp32 underflow, so it clamps genuine zeros without distorting small
#: real probabilities.
EPSILON = 1e-9


class KDSignalError(ValueError):
    """A teacher signal that cannot express the requested objective, and why."""


@dataclass
class TeacherSignal:
    """What the teacher believed at each position, dense or top-k.

    Alignment convention, which every caller must match: entry ``[b, t]`` is the
    teacher's prediction *for the token after position t*, exactly like
    ``student_logits[b, t]``. Producing the signal from the same ``input_ids`` the student
    sees gives this for free.
    """

    #: Dense: the full logit vector, ``(batch, positions, vocab)``. Exact at any
    #: temperature, and the only form available when the teacher is running live.
    logits: torch.Tensor | None = None

    #: Sparse: the top-k teacher *logits*, ``(batch, positions, k)``.
    top_values: torch.Tensor | None = None
    #: Sparse: which vocabulary entries those are, ``(batch, positions, k)``.
    top_indices: torch.Tensor | None = None
    #: Sparse: ``logsumexp`` over the **whole** vocabulary of ``logits / temperature``,
    #: ``(batch, positions)``. Without it the tail mass is unrecoverable.
    logsumexp: torch.Tensor | None = None
    #: The temperature ``logsumexp`` was computed at. Scaling logits changes the
    #: normaliser non-linearly, so a value captured at T=1 is wrong for any other T.
    logsumexp_temperature: float = 1.0

    #: Free-form provenance: which teacher, which revision, which k.
    metadata: dict[str, Any] | None = None

    @property
    def is_dense(self) -> bool:
        return self.logits is not None

    @property
    def k(self) -> int | None:
        return None if self.top_values is None else int(self.top_values.shape[-1])

    def validate(self, *, tail: TailTreatment, temperature: float) -> None:
        """Raise if this signal cannot serve the requested objective."""
        if self.is_dense:
            return
        if self.top_values is None or self.top_indices is None:
            raise KDSignalError(
                "a sparse teacher signal needs both top_values and top_indices; "
                f"got top_values={self.top_values is not None}, "
                f"top_indices={self.top_indices is not None}"
            )
        if self.top_values.shape != self.top_indices.shape:
            raise KDSignalError(
                f"top_values {tuple(self.top_values.shape)} and top_indices "
                f"{tuple(self.top_indices.shape)} must have the same shape"
            )
        if tail != "bucket":
            return
        if self.logsumexp is None:
            raise KDSignalError(
                "tail='bucket' needs the full-vocabulary logsumexp, which top-k logits do "
                "not contain. Either capture it alongside the top-k (one fp32 per token) "
                "or use tail='renormalize', which discards the tail mass — a different "
                "objective, not a cheaper approximation of this one."
            )
        if abs(self.logsumexp_temperature - temperature) > 1e-9:
            raise KDSignalError(
                f"logsumexp was captured at temperature {self.logsumexp_temperature} but "
                f"the loss is being computed at {temperature}. The normaliser is not a "
                "simple function of temperature, so reusing it would misstate the tail "
                "mass. Recapture at this temperature, or set the loss to "
                f"temperature={self.logsumexp_temperature}."
            )

    def to(self, device: Any = None, dtype: Any = None) -> TeacherSignal:
        """Move to a device, and cast **only** the floating-point fields.

        Deliberately not a passthrough to ``Tensor.to(*args)``: ``top_indices`` holds token
        identities, and a positional dtype would silently turn them into floats and
        corrupt the gather they exist for. Device moves apply to everything; dtype does
        not touch the indices.
        """

        def move(tensor: torch.Tensor | None, *, cast: bool) -> torch.Tensor | None:
            if tensor is None:
                return None
            if device is not None:
                tensor = tensor.to(device)
            if cast and dtype is not None:
                tensor = tensor.to(dtype)
            return tensor

        return TeacherSignal(
            logits=move(self.logits, cast=True),
            top_values=move(self.top_values, cast=True),
            top_indices=move(self.top_indices, cast=False),
            logsumexp=move(self.logsumexp, cast=True),
            logsumexp_temperature=self.logsumexp_temperature,
            metadata=self.metadata,
        )


@dataclass
class KDLossOutput:
    """The loss, its parts, and the diagnostics that say whether KD is doing anything.

    The parts are reported separately because a combined number cannot distinguish "KD is
    working" from "alpha is small and this is SFT with extra steps".
    """

    total: torch.Tensor
    kd: torch.Tensor
    cross_entropy: torch.Tensor
    #: Mean teacher entropy in nats. Near zero means a near-deterministic teacher, whose
    #: distribution carries little more than its argmax — KD would then be close to SFT.
    #: For a sparse signal this is the entropy of the top-k plus a *single* tail bucket,
    #: so it is a lower bound: the tail's internal entropy was discarded at capture and
    #: cannot be recovered.
    teacher_entropy: float
    #: Fraction of scored positions where the student's argmax matches the teacher's.
    top1_agreement: float
    #: Mean teacher probability *outside* the stored top-k. The empirical answer to
    #: "is k large enough"; 1.0 would mean the stored support holds none of the mass.
    tail_mass: float
    #: Positions that contributed to the loss, after masking.
    n_scored: int

    def to_log(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "kd_loss": float(self.kd.detach()),
            "ce_loss": float(self.cross_entropy.detach()),
            "teacher_entropy": self.teacher_entropy,
            "top1_agreement": self.top1_agreement,
            "teacher_tail_mass": self.tail_mass,
            "n_scored": self.n_scored,
        }


def _shift(
    student_logits: torch.Tensor, labels: torch.Tensor | None, signal: TeacherSignal
) -> tuple[torch.Tensor, torch.Tensor | None, TeacherSignal]:
    """Drop the last position and the first label, the standard causal-LM alignment."""
    shifted_signal = TeacherSignal(
        logits=None if signal.logits is None else signal.logits[:, :-1],
        top_values=None if signal.top_values is None else signal.top_values[:, :-1],
        top_indices=None if signal.top_indices is None else signal.top_indices[:, :-1],
        logsumexp=None if signal.logsumexp is None else signal.logsumexp[:, :-1],
        logsumexp_temperature=signal.logsumexp_temperature,
        metadata=signal.metadata,
    )
    return student_logits[:, :-1], None if labels is None else labels[:, 1:], shifted_signal


def kd_divergence(
    student_logits: torch.Tensor,
    signal: TeacherSignal,
    *,
    temperature: float = 1.0,
    tail: TailTreatment = "bucket",
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean KL(teacher || student) over the masked positions, plus diagnostics.

    Already aligned: no shifting happens here. ``student_logits`` and ``signal`` must both
    be predictions for the same positions.

    The classical ``T**2`` factor is applied so the gradient magnitude does not collapse
    as temperature rises — without it, raising T quietly turns the KD term off.
    """
    import torch

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    signal.validate(tail=tail, temperature=temperature)

    if mask is None:
        mask = student_logits.new_ones(student_logits.shape[:-1], dtype=torch.bool)
    n_scored = int(mask.sum())
    if n_scored == 0:
        zero = student_logits.float().sum() * 0.0
        return zero, {"teacher_entropy": 0.0, "top1_agreement": 0.0, "tail_mass": 0.0, "n_scored": 0}

    if signal.is_dense:
        student = student_logits.float()
        student_logp = torch.log_softmax(student / temperature, dim=-1)
        teacher_logp = torch.log_softmax(signal.logits.float() / temperature, dim=-1)
        teacher_p = teacher_logp.exp()
        per_position = (teacher_p * (teacher_logp - student_logp)).sum(-1)
        entropy = -(teacher_p * teacher_logp).sum(-1)
        teacher_top1 = signal.logits.argmax(-1)
        tail_mass = torch.zeros_like(entropy)
    else:
        student = student_logits
        values = signal.top_values.float()
        indices = signal.top_indices.long()
        # log_softmax(x)[i] == x[i] - logsumexp(x), so the k student log-probabilities can
        # be had without materialising a (batch, positions, 248320) float32 intermediate.
        # At the teacher's vocabulary that intermediate is ~2 GB per 2048-token batch, and
        # this is a KD loop that has to fit beside a teacher.
        student_lse = torch.logsumexp(student.float() / temperature, dim=-1, keepdim=True)
        student_logp_k = student.gather(-1, indices).float() / temperature - student_lse

        if tail == "renormalize":
            # Both sides are renormalised over the *same* k indices, so this is a genuine
            # KL on the k-simplex. Renormalising only the teacher would leave the
            # student's full-vocabulary normaliser in the expression, which is neither
            # this objective nor the bucket one.
            teacher_logp_k = torch.log_softmax(values / temperature, dim=-1)
            teacher_p_k = teacher_logp_k.exp()
            student_logp_k = torch.log_softmax(
                student.gather(-1, indices).float() / temperature, dim=-1
            )
            per_position = (teacher_p_k * (teacher_logp_k - student_logp_k)).sum(-1)
            entropy = -(teacher_p_k * teacher_logp_k).sum(-1)
            tail_mass = torch.zeros_like(entropy)
        else:
            teacher_logp_k = values / temperature - signal.logsumexp.float().unsqueeze(-1)
            teacher_p_k = teacher_logp_k.exp()
            teacher_tail = (1.0 - teacher_p_k.sum(-1)).clamp(min=0.0, max=1.0)
            student_tail = (1.0 - student_logp_k.exp().sum(-1)).clamp(min=EPSILON, max=1.0)
            head = (teacher_p_k * (teacher_logp_k - student_logp_k)).sum(-1)
            bucket = teacher_tail * (
                teacher_tail.clamp_min(EPSILON).log() - student_tail.log()
            )
            # 0 * log(0/q) is 0, not NaN: a teacher with no tail mass says nothing about it.
            per_position = head + torch.where(
                teacher_tail > EPSILON, bucket, torch.zeros_like(bucket)
            )
            entropy = -(teacher_p_k * teacher_logp_k).sum(-1) - torch.where(
                teacher_tail > EPSILON,
                teacher_tail * teacher_tail.clamp_min(EPSILON).log(),
                torch.zeros_like(teacher_tail),
            )
            tail_mass = teacher_tail
        teacher_top1 = indices.gather(-1, values.argmax(-1, keepdim=True)).squeeze(-1)

    divergence = (per_position * mask).sum() / n_scored * (temperature**2)
    agreement = ((student_logits.argmax(-1) == teacher_top1) & mask).sum() / n_scored
    diagnostics = {
        "teacher_entropy": float((entropy * mask).sum() / n_scored),
        "top1_agreement": float(agreement),
        "tail_mass": float((tail_mass * mask).sum() / n_scored),
        "n_scored": n_scored,
    }
    return divergence, diagnostics


def distillation_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    signal: TeacherSignal,
    *,
    alpha: float = 1.0,
    temperature: float = 1.0,
    tail: TailTreatment = "bucket",
    shift: bool = True,
    ignore_index: int = IGNORE_INDEX,
) -> KDLossOutput:
    """``alpha * KD + (1 - alpha) * CE``, with both terms always reported.

    ``alpha=1.0`` is pure KD, ``alpha=0.0`` is pure SFT — and the second is what the
    project needs as its control, so it is reachable through the same code path rather
    than a separate one that might differ in masking or shifting.
    """
    import torch

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    if shift:
        student_logits, labels, signal = _shift(student_logits, labels, signal)

    mask = labels != ignore_index
    divergence, diagnostics = kd_divergence(
        student_logits, signal, temperature=temperature, tail=tail, mask=mask
    )

    flat_logits = student_logits.reshape(-1, student_logits.shape[-1]).float()
    cross_entropy = torch.nn.functional.cross_entropy(
        flat_logits, labels.reshape(-1), ignore_index=ignore_index
    )
    if not torch.isfinite(cross_entropy):  # every position masked out
        cross_entropy = flat_logits.sum() * 0.0

    total = alpha * divergence + (1.0 - alpha) * cross_entropy
    return KDLossOutput(
        total=total,
        kd=divergence,
        cross_entropy=cross_entropy,
        teacher_entropy=diagnostics["teacher_entropy"],
        top1_agreement=diagnostics["top1_agreement"],
        tail_mass=diagnostics["tail_mass"],
        n_scored=diagnostics["n_scored"],
    )


# ---------------------------------------------------------------------------
# producing a signal
# ---------------------------------------------------------------------------
def capture_signal(
    logits: torch.Tensor, *, top_k: int | None = None, temperature: float = 1.0
) -> TeacherSignal:
    """Turn full teacher logits into a signal, dense or top-k with an exact tail.

    This is deliberately the *only* way a sparse signal is built, so the online path (a
    teacher in memory) and the offline path (a teacher that ran last month) produce
    byte-comparable artifacts. The ``logsumexp`` is taken over the full vocabulary before
    anything is discarded — after truncation it is gone for good.
    """
    import torch

    if top_k is None:
        return TeacherSignal(logits=logits.detach())
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    vocab = logits.shape[-1]
    if top_k > vocab:
        raise ValueError(f"top_k={top_k} exceeds the vocabulary ({vocab})")

    scaled = logits.detach().float() / temperature
    values, indices = torch.topk(logits.detach().float(), top_k, dim=-1)
    return TeacherSignal(
        top_values=values,
        top_indices=indices.to(torch.int32),
        logsumexp=torch.logsumexp(scaled, dim=-1),
        logsumexp_temperature=temperature,
        metadata={"top_k": top_k, "vocab_size": vocab},
    )


def signal_bytes_per_token(top_k: int, *, index_bytes: int = 4, value_bytes: int = 2) -> int:
    """Storage cost of one token's sparse signal, for sizing an offline corpus.

    Defaults match what :func:`capture_signal` produces when written as int32 indices and
    fp16 values, plus one fp32 ``logsumexp``.
    """
    return top_k * (index_bytes + value_bytes) + 4
