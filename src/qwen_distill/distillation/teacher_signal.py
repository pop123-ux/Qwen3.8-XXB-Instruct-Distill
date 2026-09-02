"""Where a teacher distribution comes from, kept separate from what is done with it.

:mod:`.kd_loss` deliberately does not know whether the teacher ran a moment ago or a month
ago. This module holds the difference, because it is a *cost* decision rather than an
objective one, and the project should be able to change its mind about it without
touching the loss:

``OnlineTeacher``
    the teacher is resident and answers every batch. Exact, no storage, no staleness — and
    it wants ~15 GB at 4-bit for the 27B teacher, on top of the student, its gradients and
    its optimizer state. On one GPU that is the binding constraint.

offline
    the teacher runs once over a fixed corpus and writes top-k signals; training then
    needs no teacher at all. At k=64 that is 388 bytes per token (3.9 GB per 10M tokens),
    it can be produced on rented hardware and consumed on a free T4, and the same corpus
    can train many students. The cost is that the corpus is fixed in advance: no
    on-policy data, no changing the tokenisation, no raising k without regenerating.

Both produce the identical artifact through :func:`~.kd_loss.capture_signal`, so a run can
switch between them and the only thing that changes is the bill. The offline *reader* is
not implemented here: it needs a corpus layout that should be decided against a measured
tail mass, not guessed. :class:`SignalProvider` is the seam it will slot into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from .kd_loss import TeacherSignal, capture_signal, signal_bytes_per_token

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


class SignalProvider(Protocol):
    """Supplies the teacher's distribution for a batch of student inputs."""

    def signal_for(self, input_ids: torch.Tensor) -> TeacherSignal: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass
class OnlineTeacher:
    """A resident teacher, queried once per batch under ``no_grad``.

    ``top_k`` is not only a storage question here. The teacher's logits are
    ``batch x positions x 248320``; at 2048 positions that is a gigabyte in fp16 before
    the student has produced its own. Truncating to top-k with the full-vocabulary
    ``logsumexp`` keeps the objective exact (see :mod:`.kd_loss`) while holding the
    intermediate to ``k`` columns, so it is worth doing even when nothing is stored.
    """

    model: Any
    top_k: int | None = 64
    temperature: float = 1.0
    #: Which model produced this, for the run record. A signal with no provenance is not
    #: reproducible: the same repo id has served different weights over time.
    teacher_model: str | None = None
    teacher_revision: str | None = None

    def signal_for(self, input_ids: torch.Tensor) -> TeacherSignal:
        import torch

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                # Move the ids the way ``real_teacher.teacher_logits`` does. The trainer
                # already builds its batch on the training device, so this is a no-op
                # there; a caller holding CPU ids — the smoke test does — would otherwise
                # fail inside the embedding's index_select with a device mismatch, and the
                # two routes to the same teacher must not disagree about whose job this is.
                device = getattr(self.model, "device", None)
                if device is not None:
                    input_ids = input_ids.to(device)
                logits = self.model(input_ids=input_ids).logits
            signal = capture_signal(logits, top_k=self.top_k, temperature=self.temperature)
        finally:
            if was_training:
                self.model.train()
        signal.metadata = {**(signal.metadata or {}), **self.describe()}
        return signal

    def describe(self) -> dict[str, Any]:
        return {
            "source": "online",
            "top_k": self.top_k,
            "temperature": self.temperature,
            "teacher_model": self.teacher_model,
            "teacher_revision": self.teacher_revision,
        }


@dataclass
class ReplaySignals:
    """Signals captured earlier in this process, keyed by batch index.

    Not a substitute for a real offline corpus — it holds everything in memory. It exists
    so the KD path can be exercised end to end without a teacher resident, which is what
    the Stage-0 pilot needs and what the tests use.
    """

    signals: list[TeacherSignal]
    position: int = 0

    def signal_for(self, input_ids: torch.Tensor) -> TeacherSignal:
        if not self.signals:
            raise IndexError("no captured signals to replay")
        signal = self.signals[self.position % len(self.signals)]
        self.position += 1
        expected = tuple(input_ids.shape[:2])
        actual = (
            tuple(signal.logits.shape[:2]) if signal.is_dense
            else tuple(signal.top_values.shape[:2])
        )
        if actual != expected:
            raise ValueError(
                f"replayed signal covers {actual} but the batch is {expected}; a "
                "misaligned signal would train the student against another batch's teacher"
            )
        return signal

    def describe(self) -> dict[str, Any]:
        return {"source": "replay", "n_signals": len(self.signals)}


def estimate_offline_corpus_gib(n_tokens: int, top_k: int) -> float:
    """Storage for an offline signal corpus, for deciding whether it is affordable."""
    return n_tokens * signal_bytes_per_token(top_k) / (1024**3)


def build_provider(
    kind: str,
    *,
    model: Any = None,
    top_k: int | None = 64,
    temperature: float = 1.0,
    **metadata: Any,
) -> SignalProvider:
    """Construct a provider by name, failing loudly on the unimplemented one."""
    if kind == "online":
        if model is None:
            raise ValueError("an online teacher provider needs a loaded teacher model")
        return OnlineTeacher(
            model=model,
            top_k=top_k,
            temperature=temperature,
            teacher_model=metadata.get("teacher_model"),
            teacher_revision=metadata.get("teacher_revision"),
        )
    if kind == "offline":
        raise NotImplementedError(
            "the offline signal reader is not implemented. The loss and the capture "
            "format are ready (top-k logits + full-vocabulary logsumexp); what is missing "
            "is the on-disk corpus layout, which should be chosen once a real run has "
            "reported its tail mass at a candidate k rather than guessed beforehand."
        )
    raise ValueError(f"unknown teacher signal provider {kind!r}; known: 'online', 'offline'")
