"""Measure how reasoning controls actually behave.

Two independent checks, because they can disagree and the disagreement is informative:

1. :func:`compare_rendered_prompts` renders the chat template at each reasoning
   setting and diffs the results. This needs **only the tokenizer** — no weights, no
   GPU — and it decides the template-level question: does selecting a setting change
   the prompt at all? If two settings render byte-identical prompts, one of them is a
   no-op *by construction*, which is exactly the behaviour reported for Qwen3.8's
   ``medium``. That is a proof, not an inference.

2. :func:`sweep_reasoning_settings` generates at each setting and compares measured
   token counts. A setting can change the prompt yet barely change behaviour, or
   (with a model trained on control tokens) change behaviour without changing the
   prompt. Only generation settles that.

Run (1) first: it is free, and if it already shows two settings are identical, (2)
only confirms it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .metrics import RunSummary, summarise
from .runner import Backend, GenerationResult, run_tasks
from .tasks import Task

#: Settings to sweep. ``None`` means "send no control", i.e. the model's own default —
#: which is the setting a user gets without asking, and therefore the one that matters
#: most for the overthinking claim.
DEFAULT_SETTINGS: tuple[str | None, ...] = (None, "low", "medium", "xhigh")


@dataclass
class PromptRendering:
    """The prompt one reasoning setting produces."""

    setting: str | None
    rendered: str
    sha256: str
    n_chars: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "setting": self.setting,
            "sha256": self.sha256,
            "n_chars": self.n_chars,
            "rendered_excerpt": self.rendered[:600],
            "error": self.error,
        }


@dataclass
class TemplateComparison:
    """Which reasoning settings actually change the prompt."""

    renderings: list[PromptRendering] = field(default_factory=list)
    #: Groups of settings that render identically. A group with >1 member means those
    #: settings are indistinguishable at the template level.
    identical_groups: list[list[str]] = field(default_factory=list)
    n_distinct: int = 0

    @property
    def has_noop_settings(self) -> bool:
        return any(len(group) > 1 for group in self.identical_groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "renderings": [r.to_dict() for r in self.renderings],
            "identical_groups": self.identical_groups,
            "n_distinct": self.n_distinct,
            "has_noop_settings": self.has_noop_settings,
        }


def compare_rendered_prompts(
    tokenizer,
    prompt: str = "What is 15 * 7?",
    settings: tuple[str | None, ...] = DEFAULT_SETTINGS,
    *,
    system_prompt: str | None = None,
    enable_thinking_off: bool = True,
) -> TemplateComparison:
    """Render the chat template at each setting and group identical results.

    Requires only a tokenizer with a chat template, so this runs against a
    metadata-only checkpoint download.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    comparison = TemplateComparison()
    candidates: list[tuple[str, dict[str, Any]]] = []
    for setting in settings:
        label = setting if setting is not None else "(default)"
        kwargs: dict[str, Any] = {} if setting is None else {"reasoning_effort": setting}
        candidates.append((label, kwargs))
    if enable_thinking_off:
        candidates.append(("thinking_disabled", {"enable_thinking": False}))

    for label, kwargs in candidates:
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
            error = None
        except TypeError as exc:
            # The template rejected the kwarg outright: also a finding worth recording.
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            error = f"template does not accept {list(kwargs)}: {exc}"
        except Exception as exc:  # noqa: BLE001
            comparison.renderings.append(
                PromptRendering(label, "", "", 0, error=f"{type(exc).__name__}: {exc}")
            )
            continue
        comparison.renderings.append(
            PromptRendering(
                setting=label,
                rendered=rendered,
                sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                n_chars=len(rendered),
                error=error,
            )
        )

    groups: dict[str, list[str]] = {}
    for rendering in comparison.renderings:
        if rendering.sha256:
            groups.setdefault(rendering.sha256, []).append(str(rendering.setting))
    comparison.identical_groups = [g for g in groups.values() if len(g) > 1]
    comparison.n_distinct = len(groups)
    return comparison


@dataclass
class ReasoningSweep:
    """Measured behaviour across reasoning settings."""

    per_setting: dict[str, RunSummary] = field(default_factory=dict)
    raw: dict[str, list[GenerationResult]] = field(default_factory=dict)

    def token_ratios(self, reference: str) -> dict[str, float | None]:
        """Mean thinking tokens at each setting, relative to ``reference``."""
        base = self.per_setting.get(reference)
        if base is None or base.overall.mean_thinking_tokens <= 0:
            return {k: None for k in self.per_setting}
        return {
            k: v.overall.mean_thinking_tokens / base.overall.mean_thinking_tokens
            for k, v in self.per_setting.items()
        }

    def indistinguishable_settings(self, tolerance: float = 0.05) -> list[tuple[str, str]]:
        """Setting pairs whose mean thinking-token counts differ by less than ``tolerance``.

        Evidence that a control has little or no effect in practice. Combine with
        :func:`compare_rendered_prompts` to distinguish "the template ignores it" from
        "the template changes but the model does not care".
        """
        pairs: list[tuple[str, str]] = []
        names = list(self.per_setting)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ta = self.per_setting[a].overall.mean_thinking_tokens
                tb = self.per_setting[b].overall.mean_thinking_tokens
                scale = max(ta, tb)
                if scale <= 0:
                    if ta == tb:
                        pairs.append((a, b))
                elif abs(ta - tb) / scale < tolerance:
                    pairs.append((a, b))
        return pairs

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_setting": {k: v.to_dict() for k, v in self.per_setting.items()},
            "indistinguishable_settings": self.indistinguishable_settings(),
        }


def sweep_reasoning_settings(
    make_backend,
    tasks: list[Task],
    settings: tuple[str | None, ...] = DEFAULT_SETTINGS,
    *,
    output_dir=None,
    progress: bool = True,
) -> ReasoningSweep:
    """Run ``tasks`` once per reasoning setting and summarise each.

    ``make_backend`` is a callable taking ``reasoning_effort`` and returning a
    :class:`Backend`, so the caller controls model path, device and sampling while this
    function varies only the reasoning control.
    """
    sweep = ReasoningSweep()
    for setting in settings:
        label = setting if setting is not None else "(default)"
        if progress:
            print(f"\n=== reasoning_effort = {label} ===")
        backend: Backend = make_backend(setting)
        path = None
        if output_dir is not None:
            safe = str(label).strip("()").replace(" ", "_")
            path = output_dir / f"generations_{safe}.jsonl"
        results = run_tasks(backend, tasks, output_path=path, progress=progress)
        sweep.raw[label] = results
        sweep.per_setting[label] = summarise(results)
    return sweep
