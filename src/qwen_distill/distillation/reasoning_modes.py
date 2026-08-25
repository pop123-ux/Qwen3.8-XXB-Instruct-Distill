"""Reasoning modes the *verified* teacher template actually accepts.

Read off `vendor/qwen38-metadata/chat_template.jinja`, not from documentation or
secondary sources:

* the template accepts exactly ``xhigh``, ``medium`` and ``low``;
* anything else raises inside the template — including ``high``, which reads like it
  should work and does not;
* ``xhigh`` is the default when nothing is passed;
* thinking is suppressed by ``enable_thinking=False``, a separate control from
  ``reasoning_effort``.

Two distinctions this module refuses to blur:

**Prompt-level control is not behavioural control.** Setting ``reasoning_effort=low``
changes the rendered prompt. Whether the model then reasons less is an empirical
question that needs generation to answer, and nothing here should be read as claiming
it. :data:`PROMPT_LEVEL_ONLY` says so explicitly.

**``medium`` is not a no-op.** A widely repeated secondary claim says it is. The real
template refutes it: ``medium`` renders a distinct — in fact the shortest — prompt,
because it injects no reasoning instruction while the default injects the long ``xhigh``
one. Those differ precisely because the default is ``xhigh``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Values the template's `reasoning_effort` branch accepts. Order is low-to-high effort.
TEMPLATE_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "xhigh")

#: What the template uses when `reasoning_effort` is not supplied.
DEFAULT_EFFORT = "xhigh"

#: A separate mode, reached through `enable_thinking=False` rather than through
#: `reasoning_effort`. Kept in the same namespace because callers think of it as one
#: axis, but it is dispatched differently — see :func:`template_kwargs`.
THINKING_DISABLED = "thinking_disabled"

#: Every mode a caller may name.
SUPPORTED_MODES: tuple[str, ...] = (THINKING_DISABLED, *TEMPLATE_EFFORT_LEVELS)

#: Modes that look plausible and are NOT accepted, with why. Naming them explicitly
#: turns a confusing template error into an answerable one.
REJECTED_MODES: dict[str, str] = {
    "high": (
        "the template accepts xhigh, medium and low only — 'high' raises inside it. "
        "This is the one that catches people out; use 'xhigh' for maximum effort."
    ),
    "none": "use 'thinking_disabled' to suppress reasoning, or 'low' for minimal effort",
    "off": "use 'thinking_disabled'",
    "disabled": "use 'thinking_disabled'",
    "default": f"the template's default is {DEFAULT_EFFORT!r}; name it explicitly",
    "max": f"use {DEFAULT_EFFORT!r}",
}

#: What a mode does and does not establish. The distinction matters for every claim
#: this project will later make about reasoning cost.
PROMPT_LEVEL_ONLY = (
    "A reasoning mode changes the rendered prompt. Whether the model reasons less as a "
    "result is a measured question, not a template guarantee: prompt-level control is "
    "not behavioural control."
)


@dataclass(frozen=True)
class ReasoningMode:
    """A validated mode, and how it reaches the chat template."""

    name: str
    #: `reasoning_effort` value, or None when the mode is not an effort level.
    reasoning_effort: str | None
    #: `enable_thinking` value, or None to leave the template's default alone.
    enable_thinking: bool | None
    description: str

    @property
    def reasoning_enabled(self) -> bool:
        return self.enable_thinking is not False

    def template_kwargs(self) -> dict[str, Any]:
        """Exactly the kwargs to pass to ``apply_chat_template``."""
        kwargs: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reasoning_effort": self.reasoning_effort,
            "enable_thinking": self.enable_thinking,
            "reasoning_enabled": self.reasoning_enabled,
        }


MODES: dict[str, ReasoningMode] = {
    THINKING_DISABLED: ReasoningMode(
        name=THINKING_DISABLED, reasoning_effort=None, enable_thinking=False,
        description="reasoning suppressed via enable_thinking=False",
    ),
    "low": ReasoningMode(
        name="low", reasoning_effort="low", enable_thinking=None,
        description="minimum effort branch the template offers",
    ),
    "medium": ReasoningMode(
        name="medium", reasoning_effort="medium", enable_thinking=None,
        description=(
            "injects no reasoning instruction, which makes it the SHORTEST rendered "
            "prompt — not a no-op, because the default it replaces is xhigh"
        ),
    ),
    "xhigh": ReasoningMode(
        name="xhigh", reasoning_effort="xhigh", enable_thinking=None,
        description="maximum effort; the template's default when nothing is passed",
    ),
}


class UnsupportedReasoningMode(ValueError):
    """A mode the verified template does not accept."""


def resolve_mode(name: str | None) -> ReasoningMode:
    """Validate a mode name against the verified template.

    ``None`` resolves to the template's own default rather than to "no reasoning", which
    is what the template does and the opposite of what the name might suggest.
    """
    if name is None:
        return MODES[DEFAULT_EFFORT]
    key = str(name).strip().lower()
    if key in MODES:
        return MODES[key]
    if key in REJECTED_MODES:
        raise UnsupportedReasoningMode(
            f"reasoning mode {name!r} is not supported: {REJECTED_MODES[key]}"
        )
    raise UnsupportedReasoningMode(
        f"unknown reasoning mode {name!r}. The verified template supports: "
        f"{', '.join(SUPPORTED_MODES)}."
    )


def is_supported(name: str | None) -> bool:
    try:
        resolve_mode(name)
    except UnsupportedReasoningMode:
        return False
    return True


def sweep_modes(include_disabled: bool = True) -> tuple[ReasoningMode, ...]:
    """Every mode, ordered for a reasoning-cost sweep: cheapest first."""
    names = SUPPORTED_MODES if include_disabled else TEMPLATE_EFFORT_LEVELS
    return tuple(MODES[name] for name in names)
