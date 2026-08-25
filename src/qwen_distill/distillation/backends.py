"""Teacher backends for generation, and a mock that can never be reached by accident.

The rule that shapes this module: **a fake teacher must never stand in for a real one
without being asked for.** A silent fallback would produce a dataset that looks like
teacher output, trains a student, and is worth nothing — and the failure would surface
weeks later as "distillation doesn't work". So :func:`make_backend` dispatches on an
explicit name, an unavailable real backend raises rather than degrades, and every
generated record carries the backend that produced it.

The mock is deterministic on purpose. Given the same prompt and mode it yields the same
response and the same token counts, so a test can assert on exact values and a resume
can be checked for producing identical output to an uninterrupted run.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .reasoning_modes import ReasoningMode, resolve_mode

#: Name that must be passed explicitly to get synthetic data. Deliberately unlovely.
MOCK_BACKEND = "mock"
TRANSFORMERS_BACKEND = "transformers"


@dataclass
class TeacherResponse:
    """One teacher generation, with its cost broken out.

    Token counts are separated because the project's central question is what reasoning
    costs. ``thinking + answer == total`` always holds here; a backend that cannot split
    them must say so rather than guess.
    """

    prompt: str
    thinking: str
    answer: str
    prompt_tokens: int
    thinking_tokens: int
    answer_tokens: int
    finish_reason: str = "stop"
    latency_s: float = 0.0
    error: str | None = None
    backend: str = MOCK_BACKEND
    token_counting_method: str = "unknown"

    @property
    def total_tokens(self) -> int:
        return self.thinking_tokens + self.answer_tokens

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.answer.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "thinking": self.thinking, "answer": self.answer,
            "prompt_tokens": self.prompt_tokens,
            "thinking_tokens": self.thinking_tokens,
            "answer_tokens": self.answer_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "latency_s": self.latency_s,
            "backend": self.backend,
            "token_counting_method": self.token_counting_method,
            "error": self.error,
        }


class TeacherBackend(Protocol):
    """What generation needs from a teacher, real or otherwise."""

    name: str

    def describe(self) -> dict[str, Any]:
        """Identity and settings, recorded in the run manifest."""
        ...

    def generate(self, prompt: str, *, mode: ReasoningMode) -> TeacherResponse:
        ...


@dataclass
class MockTeacher:
    """A deterministic synthetic teacher. **Never** produces usable training data.

    Exists so the whole pipeline — generation, sharding, manifests, resume, the student
    loader, evaluation — can be tested end to end without a 27B model or a GPU. Its
    outputs are transparently synthetic so that a dataset built from it cannot be
    mistaken for the real thing: every record it writes is marked, and
    :attr:`is_synthetic` is carried into the manifest.

    It simulates the property that matters for reasoning-cost work: higher effort modes
    produce more thinking tokens, and `thinking_disabled` produces none.
    """

    name: str = MOCK_BACKEND
    seed: int = 0
    #: Thinking tokens produced per mode, as a multiple of the base length. Mirrors the
    #: ordering a real teacher should show, so a sweep has something to exercise.
    effort_scale: dict[str, int] = field(default_factory=lambda: {
        "thinking_disabled": 0, "low": 1, "medium": 3, "xhigh": 8,
    })
    #: Prompts whose hash falls in this fraction fail, so failure handling is testable.
    failure_rate: float = 0.0
    latency_per_token_s: float = 0.0
    is_synthetic: bool = True

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "is_synthetic": True,
            "warning": (
                "MOCK TEACHER — outputs are synthetic and must never be used as real "
                "training data or reported as teacher behaviour"
            ),
            "seed": self.seed,
            "effort_scale": self.effort_scale,
            "failure_rate": self.failure_rate,
        }

    def _digest(self, prompt: str) -> int:
        payload = f"{self.seed}:{prompt}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def generate(self, prompt: str, *, mode: ReasoningMode) -> TeacherResponse:
        started = time.perf_counter()
        digest = self._digest(prompt)

        if self.failure_rate > 0 and (digest % 1000) / 1000.0 < self.failure_rate:
            return TeacherResponse(
                prompt=prompt, thinking="", answer="",
                prompt_tokens=len(prompt.split()), thinking_tokens=0, answer_tokens=0,
                finish_reason="error", backend=self.name,
                error="simulated teacher failure",
                token_counting_method="whitespace (mock)",
            )

        base = 4 + (digest % 5)
        scale = self.effort_scale.get(mode.name, 1)
        thinking_words = base * scale
        answer_words = 3 + (digest >> 8) % 6

        thinking = " ".join(f"step{i}" for i in range(thinking_words))
        answer = "MOCK " + " ".join(f"tok{i}" for i in range(answer_words))
        latency = self.latency_per_token_s * (thinking_words + answer_words)
        if self.latency_per_token_s:
            time.sleep(min(latency, 0.01))

        return TeacherResponse(
            prompt=prompt, thinking=thinking, answer=answer,
            prompt_tokens=len(prompt.split()),
            thinking_tokens=thinking_words, answer_tokens=len(answer.split()),
            finish_reason="stop", backend=self.name,
            latency_s=time.perf_counter() - started,
            # Whitespace, not a tokenizer: the mock has no tokenizer and says so rather
            # than implying its counts are comparable to real ones.
            token_counting_method="whitespace (mock)",
        )


@dataclass
class TransformersTeacher:
    """The real teacher, over `transformers`.

    Deliberately not implemented in this phase: loading Qwen3.8-27B needs ~50 GB of
    VRAM, and the infrastructure has to be finished and tested before any of that is
    spent. The class exists so the interface is fixed and callable, and so selecting it
    fails with an explanation rather than falling back to synthetic data.
    """

    name: str = TRANSFORMERS_BACKEND
    model: str = ""
    revision: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 2048
    temperature: float = 0.0
    is_synthetic: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name, "is_synthetic": False, "model": self.model,
            "revision": self.revision, "device": self.device, "dtype": self.dtype,
            "max_new_tokens": self.max_new_tokens, "temperature": self.temperature,
        }

    def load(self) -> None:
        raise NotImplementedError(
            "The real teacher backend is not wired up yet. Phase 3A builds the "
            "generation infrastructure and deliberately stops before spending GPU time: "
            "loading Qwen3.8-27B needs roughly 50 GB of VRAM. When it is implemented it "
            "will reuse evaluation.runner.TransformersBackend, which already handles "
            "template application and thinking/answer token splitting."
        )

    def generate(self, prompt: str, *, mode: ReasoningMode) -> TeacherResponse:
        self.load()  # raises; never reached
        raise AssertionError("unreachable")


def make_backend(name: str, **kwargs: Any) -> TeacherBackend:
    """Build a backend by explicit name.

    There is no default and no fallback. Asking for a real backend that cannot run is an
    error, not an invitation to substitute the mock — a synthetic dataset that looks
    real is the most expensive failure available here.
    """
    key = str(name).strip().lower()
    if key == MOCK_BACKEND:
        return MockTeacher(**kwargs)
    if key in (TRANSFORMERS_BACKEND, "hf", "huggingface"):
        return TransformersTeacher(**kwargs)
    raise ValueError(
        f"unknown teacher backend {name!r}. Known: {MOCK_BACKEND!r} (synthetic, for "
        f"tests only), {TRANSFORMERS_BACKEND!r} (real, not yet implemented). "
        "The mock is never selected implicitly."
    )


def resolve_backend_and_mode(
    backend_name: str, mode_name: str | None, **backend_kwargs: Any
) -> tuple[TeacherBackend, ReasoningMode]:
    """Validate both together, before anything expensive starts."""
    return make_backend(backend_name, **backend_kwargs), resolve_mode(mode_name)
