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

from .real_teacher import (
    DEFAULT_TEACHER_MODEL,
    TeacherLoadPlan,
    TeacherNotLoaded,
    generate_once,
    load_verified_teacher,
    teacher_logits,
)
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

    A thin adapter: everything that actually touches weights lives in
    :mod:`.real_teacher`, so this module stays readable as "the interface, the mock, and
    the rule that the mock is never reached by accident".

    Two behaviours differ deliberately from ``evaluation.runner.TransformersBackend``,
    which covers the same ground for surveys:

    * **A load that leaves tensors missing is fatal here.** `transformers` returns a
      freshly-initialised model and prints a report rather than raising, so a 27B teacher
      can "load" with random weights and generate fluent nonsense.
    * **A chat template that rejects a reasoning mode is fatal here.** The survey backend
      re-renders without the controls, which is right when the question is whether a
      control does anything. For teacher data it would label records with a mode the
      prompt never carried.
    """

    name: str = TRANSFORMERS_BACKEND
    model: str = DEFAULT_TEACHER_MODEL
    revision: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int = 0
    quantization: str | None = None
    max_memory: dict[str, str] | None = None
    offload_folder: str | None = None
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    local_path: str | None = None
    system_prompt: str | None = None
    strict_architecture: bool = True
    is_synthetic: bool = False
    #: Populated by :meth:`load`. ``None`` until then, and every weight-touching method
    #: raises rather than loading implicitly — a 50 GB load is not something to trigger
    #: as a side effect.
    loaded: Any = None

    def plan(self) -> TeacherLoadPlan:
        return TeacherLoadPlan(
            model=self.model, revision=self.revision, dtype=self.dtype,
            device_map=None if self.device == "cpu" else self.device,
            quantization=self.quantization, max_memory=self.max_memory,
            offload_folder=self.offload_folder, trust_remote_code=self.trust_remote_code,
            attn_implementation=self.attn_implementation, local_path=self.local_path,
        )

    def describe(self) -> dict[str, Any]:
        """Identity and settings. Before loading this is the *request*; after loading it
        is what was actually loaded, hashes and all."""
        if self.loaded is not None:
            return {**self.loaded.describe(), "backend": self.name,
                    "generation": self._generation_settings()}
        return {
            "backend": self.name, "is_synthetic": False, "loaded": False,
            "plan": self.plan().to_dict(), "generation": self._generation_settings(),
            "note": "not loaded yet; identity hashes appear after load()",
        }

    def _generation_settings(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens, "temperature": self.temperature,
            "top_p": self.top_p, "top_k": self.top_k, "seed": self.seed,
            "system_prompt": self.system_prompt,
        }

    def load(self) -> Any:
        """Load weights and tokenizer, verifying that they actually arrived."""
        if self.loaded is None:
            self.loaded = load_verified_teacher(
                self.plan(), strict_architecture=self.strict_architecture
            )
        return self.loaded

    def unload(self) -> None:
        if self.loaded is not None:
            self.loaded.unload()
            self.loaded = None

    def _require_loaded(self) -> Any:
        if self.loaded is None or self.loaded.model is None:
            raise TeacherNotLoaded(
                "the teacher is not loaded. Call load() explicitly — this backend does not "
                "load ~50 GB of weights as a side effect of another call."
            )
        return self.loaded

    def generate(self, prompt: str, *, mode: ReasoningMode) -> TeacherResponse:
        loaded = self._require_loaded()
        try:
            result = generate_once(
                loaded, prompt, mode=mode, system_prompt=self.system_prompt,
                max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                top_p=self.top_p, top_k=self.top_k, seed=self.seed,
            )
        except Exception as exc:  # noqa: BLE001 - a failed generation is a record, not a crash
            return TeacherResponse(
                prompt=prompt, thinking="", answer="", prompt_tokens=0,
                thinking_tokens=0, answer_tokens=0, finish_reason="error",
                backend=self.name, error=f"{type(exc).__name__}: {exc}",
                token_counting_method="none (generation failed)",
            )
        return TeacherResponse(
            prompt=prompt, thinking=result.thinking, answer=result.answer,
            prompt_tokens=result.prompt_tokens,
            thinking_tokens=result.thinking_tokens,
            answer_tokens=result.answer_tokens,
            finish_reason=result.finish_reason, latency_s=result.latency_s,
            backend=self.name, token_counting_method=result.token_counting_method,
        )

    # -- distillation --------------------------------------------------
    def logits(self, input_ids: Any) -> Any:
        """Full teacher logits for ids the student will see, aligned position for position."""
        return teacher_logits(self._require_loaded(), input_ids)

    def signal_provider(self, *, top_k: int | None = 64, temperature: float = 1.0) -> Any:
        """An :class:`~.teacher_signal.OnlineTeacher` over these weights.

        Returned rather than reimplemented: the KD path already has one signal format and
        one capture function, and the online/offline decision is deliberately a property of
        the provider rather than of the loss.
        """
        from .teacher_signal import OnlineTeacher

        loaded = self._require_loaded()
        return OnlineTeacher(
            model=loaded.model, top_k=top_k, temperature=temperature,
            teacher_model=self.model, teacher_revision=self.revision,
        )


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
        f"tests only), {TRANSFORMERS_BACKEND!r} (real). "
        "The mock is never selected implicitly."
    )


def resolve_backend_and_mode(
    backend_name: str, mode_name: str | None, **backend_kwargs: Any
) -> tuple[TeacherBackend, ReasoningMode]:
    """Validate both together, before anything expensive starts."""
    return make_backend(backend_name, **backend_kwargs), resolve_mode(mode_name)
