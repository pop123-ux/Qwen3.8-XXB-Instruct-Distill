"""Model-agnostic generation and measurement for evaluation runs.

Records what the project actually needs to compare a teacher and a student:
accuracy *and* the cost of getting there. Every generation reports prompt tokens,
thinking tokens, answer tokens, latency and time-to-first-token, because a model
that matches accuracy at a third of the tokens is a better model and the harness
must be able to say so.

Backends are deliberately thin. The important rule, from ``docs/EVALUATION_PLAN.md``:
**an import succeeding is not evidence a backend works.** :meth:`Backend.probe`
performs a real generation and reports what happened, and
``scripts/evaluate.py --probe-only`` is how you check a backend before trusting a run.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .tasks import Task

#: Tags Qwen-family chat templates use to delimit reasoning traces.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

#: Sentinel distinguishing "not yet looked up" from "looked up, not present".
_UNSET = object()


def split_thinking(text: str) -> tuple[str, str]:
    """Split generated text into ``(thinking, answer)``.

    Handles the three shapes that occur in practice:

    * ``<think>...</think>answer`` — the normal complete case;
    * ``...</think>answer`` — the template already emitted the opening tag, so the
      model's own output starts *inside* the reasoning block;
    * no tags at all — everything is the answer.

    An unterminated ``<think>`` (generation hit the token limit mid-reasoning) yields
    an empty answer, which is the correct reading: the model never answered.
    """
    if THINK_CLOSE in text:
        head, _, answer = text.partition(THINK_CLOSE)
        thinking = head.split(THINK_OPEN, 1)[-1] if THINK_OPEN in head else head
        return thinking.strip(), answer.strip()
    if THINK_OPEN in text:
        return text.split(THINK_OPEN, 1)[-1].strip(), ""
    return "", text.strip()


@dataclass
class GenerationResult:
    """One model response, with the cost of producing it."""

    task_id: str
    category: str
    difficulty: str
    prompt_tokens: int
    thinking_tokens: int
    answer_tokens: int
    total_generated_tokens: int
    latency_s: float
    time_to_first_token_s: float | None
    thinking_text: str
    answer_text: str
    correct: bool | None
    reasoning_effort: str | None = None
    finish_reason: str | None = None
    error: str | None = None

    @property
    def tokens_per_second(self) -> float:
        if self.latency_s <= 0:
            return 0.0
        return self.total_generated_tokens / self.latency_s

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tokens_per_second"] = self.tokens_per_second
        return data


@dataclass
class BackendProbe:
    """Evidence that a backend can actually run this architecture."""

    backend: str
    ok: bool
    model_class: str | None = None
    generated_text: str | None = None
    generated_tokens: int | None = None
    latency_s: float | None = None
    versions: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Backend(Protocol):
    """Minimal interface an evaluation backend must provide."""

    name: str

    def probe(self) -> BackendProbe:
        """Run a real generation and report whether it worked."""

    def generate(self, prompt: str, *, system_prompt: str | None, **kwargs) -> GenerationResult:
        """Generate a response and measure its cost."""


class TransformersBackend:
    """Reference backend: `transformers` generate, on CPU or GPU.

    Slow, but it is the correctness reference. When another backend disagrees with
    this one on a long-context retrieval task, this one is right.
    """

    name = "transformers"

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        seed: int = 0,
        reasoning_effort: str | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking
        self._model = None
        self._tokenizer = None
        self._think_close_cache: Any = _UNSET

    # -- loading -------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(self.seed)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code
        )
        dtype = None if self.dtype == "auto" else getattr(torch, self.dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            dtype=dtype,
            device_map=self.device if self.device != "cpu" else None,
        ).eval()

    def apply_template(self, prompt: str, system_prompt: str | None) -> str:
        """Render the chat template, passing reasoning controls when supported.

        The controls are passed as template kwargs because that is how Qwen exposes
        them. If the template ignores an unknown kwarg, the run still succeeds — and
        ``scripts/benchmark_reasoning.py`` is what detects that the control had no
        effect, by comparing token counts across settings.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
        except TypeError:
            # Template does not accept these kwargs; fall back and let the caller see
            # identical token counts across settings, which is itself a finding.
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    # -- generation ----------------------------------------------------
    def generate(
        self, prompt: str, *, system_prompt: str | None = None, task: Task | None = None
    ) -> GenerationResult:
        import torch

        self.load()
        text = self.apply_template(prompt, system_prompt)
        inputs = self._tokenizer(text, return_tensors="pt")
        if self._model.device.type != "cpu":
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        do_sample = self.temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update(temperature=self.temperature, top_p=self.top_p)
            if self.top_k is not None:
                gen_kwargs["top_k"] = self.top_k

        torch.manual_seed(self.seed)
        started = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)
        latency = time.perf_counter() - started

        new_tokens = output[0][prompt_tokens:]
        generated = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        thinking, answer = split_thinking(generated)
        thinking_tokens, answer_tokens = self._count_split_tokens(new_tokens, thinking, answer)

        return GenerationResult(
            task_id=task.task_id if task else "adhoc",
            category=task.category if task else "adhoc",
            difficulty=task.difficulty if task else "unknown",
            prompt_tokens=prompt_tokens,
            thinking_tokens=thinking_tokens,
            answer_tokens=answer_tokens,
            total_generated_tokens=int(new_tokens.shape[-1]),
            latency_s=latency,
            time_to_first_token_s=None,  # not separable without streaming
            thinking_text=thinking,
            answer_text=answer,
            correct=task.score(answer) if task else None,
            reasoning_effort=self.reasoning_effort,
            finish_reason="length"
            if int(new_tokens.shape[-1]) >= self.max_new_tokens
            else "stop",
        )


    def _think_close_id(self) -> int | None:
        """Token id of ``</think>``, when the tokenizer encodes it as a single token."""
        if self._think_close_cache is not _UNSET:
            return self._think_close_cache
        resolved = None
        try:
            ids = self._tokenizer.convert_tokens_to_ids(THINK_CLOSE)
            if isinstance(ids, int) and ids >= 0 and ids != self._tokenizer.unk_token_id:
                resolved = ids
        except Exception:  # noqa: BLE001 - tokenizer-specific; absence is not an error
            resolved = None
        self._think_close_cache = resolved
        return resolved

    def _count_split_tokens(self, new_tokens, thinking: str, answer: str) -> tuple[int, int]:
        """Split the generated token ids at ``</think>`` and count each side exactly.

        Counting by re-tokenising the decoded strings is lossy — whitespace and
        special-token handling mean the parts need not sum to the whole. Splitting on
        the token ids is exact, so it is preferred whenever ``</think>`` is a single
        token (as it is in Qwen-family tokenizers). Re-tokenisation remains as a
        fallback, and is only approximate.
        """
        total = int(new_tokens.shape[-1])
        close_id = self._think_close_id()
        if close_id is not None:
            positions = (new_tokens == close_id).nonzero()
            if positions.numel() > 0:
                cut = int(positions[0].item()) if positions.dim() == 1 else int(positions[0][0].item())
                # Tokens before </think> are reasoning; everything after it is the answer.
                return cut, max(0, total - cut - 1)
            # No closing tag: either the model never reasoned, or it ran out of budget.
            return (total, 0) if thinking and not answer else (0, total)

        def approx(text: str) -> int:
            if not text:
                return 0
            return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

        return approx(thinking), approx(answer)

    # -- probing -------------------------------------------------------
    def probe(self) -> BackendProbe:
        """Actually generate something. Import success proves nothing."""
        from ..teacher.loader import collect_versions

        probe = BackendProbe(backend=self.name, ok=False, versions=collect_versions())
        try:
            self.load()
            probe.model_class = type(self._model).__name__
            started = time.perf_counter()
            result = self.generate("Reply with the single word: OK", task=None)
            probe.latency_s = time.perf_counter() - started
            probe.generated_text = (result.answer_text or result.thinking_text)[:200]
            probe.generated_tokens = result.total_generated_tokens
            probe.ok = result.total_generated_tokens > 0
            if not probe.ok:
                probe.notes.append("backend loaded but produced zero tokens")
        except Exception as exc:  # noqa: BLE001 - the failure is the result
            probe.error = f"{type(exc).__name__}: {exc}"
        return probe


def run_tasks(
    backend: Backend,
    tasks: Iterable[Task],
    *,
    output_path: Path | None = None,
    limit: int | None = None,
    progress: bool = True,
) -> list[GenerationResult]:
    """Run ``tasks`` through ``backend``, optionally streaming JSONL to disk.

    Results are written incrementally so a long run that dies partway still leaves
    usable data behind.
    """
    task_list = list(tasks)
    if limit is not None:
        task_list = task_list[:limit]

    results: list[GenerationResult] = []
    handle = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("w")

    try:
        for index, task in enumerate(task_list, start=1):
            try:
                result = backend.generate(
                    task.prompt, system_prompt=task.system_prompt, task=task
                )
            except Exception as exc:  # noqa: BLE001 - record and continue
                result = GenerationResult(
                    task_id=task.task_id, category=task.category, difficulty=task.difficulty,
                    prompt_tokens=0, thinking_tokens=0, answer_tokens=0,
                    total_generated_tokens=0, latency_s=0.0, time_to_first_token_s=None,
                    thinking_text="", answer_text="", correct=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            if handle is not None:
                handle.write(json.dumps(result.to_dict()) + "\n")
                handle.flush()
            if progress:
                mark = {True: "ok", False: "WRONG", None: "-"}[result.correct]
                print(
                    f"[{index}/{len(task_list)}] {task.task_id:<18} {task.difficulty:<10} "
                    f"think={result.thinking_tokens:>5} ans={result.answer_tokens:>5} "
                    f"{result.latency_s:>6.2f}s  {mark}"
                )
    finally:
        if handle is not None:
            handle.close()
    return results
