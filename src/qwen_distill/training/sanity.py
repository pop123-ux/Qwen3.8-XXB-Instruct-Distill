"""Cheap checks that catch the failure Level 2 hid behind a good number.

Level 2 reached validation BPB 1.270 against an 8.0 baseline and generated
`"and and and and"`. The loss curve looked healthy the whole way. A metric that cannot
distinguish "learned language" from "learned the unigram distribution" will not warn you,
so something has to look at the actual output.

This is deliberately **not** a benchmark. It answers one question — *is the model
obviously broken?* — and the failure modes it looks for are the ones a byte-level model
at this scale actually produces:

* the same token forever (Level 2's exact failure);
* a handful of distinct characters, i.e. collapse to a tiny vocabulary;
* empty or whitespace-only output;
* a short cycle repeating (`abcabcabc`), which a raw repetition count misses;
* verbatim reproduction of training text, i.e. memorisation rather than modelling.

Passing these does **not** mean the model is good. Failing them means it is broken, which
is a much cheaper thing to learn early.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

#: Fixed prompts, used unchanged across every checkpoint so generations are comparable
#: over time. Short and ordinary: a model that has learned English should continue them
#: with something English-shaped, whatever the content.
SANITY_PROMPTS: tuple[str, ...] = (
    "The ",
    "In the beginning ",
    "It was ",
    "Once upon a time ",
    "The most important ",
    "When the sun ",
)

#: Thresholds. Set to catch the obviously-broken, not to grade quality — a real model
#: clears these by a wide margin and a degenerate one fails them unambiguously.
MAX_TOP_TOKEN_SHARE = 0.5      # >50% of words being one word is Level 2's failure
MIN_DISTINCT_CHARS = 8
MIN_DISTINCT_WORD_RATIO = 0.15
MAX_CYCLE_SHARE = 0.6          # a short cycle covering most of the output
MEMORISATION_WINDOW = 120      # characters that must match verbatim to count


@dataclass
class GenerationCheck:
    """One prompt's continuation, and what is wrong with it."""

    prompt: str
    completion: str
    n_chars: int = 0
    n_words: int = 0
    distinct_chars: int = 0
    distinct_word_ratio: float = 0.0
    top_token: str | None = None
    top_token_share: float = 0.0
    longest_cycle: str | None = None
    cycle_share: float = 0.0
    memorised: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def degenerate(self) -> bool:
        return bool(self.problems)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["degenerate"] = self.degenerate
        return data


def _longest_cycle(text: str, *, max_period: int = 40) -> tuple[str | None, float]:
    """Find a short repeating unit covering much of the text.

    `"and and and"` is caught by token counting, but `"the cat the cat the cat"` is not —
    no single token dominates. A cycle check catches both.
    """
    compact = " ".join(text.split())
    if len(compact) < 8:
        return None, 0.0
    best: tuple[str | None, float] = (None, 0.0)
    for period in range(1, min(max_period, len(compact) // 2) + 1):
        unit = compact[:period]
        repeats = 1
        while compact[repeats * period: (repeats + 1) * period] == unit:
            repeats += 1
        covered = repeats * period / len(compact)
        if repeats >= 3 and covered > best[1]:
            best = (unit, covered)
    return best


def check_generation(
    prompt: str, completion: str, *, training_text: str | None = None
) -> GenerationCheck:
    """Score one continuation against the known degenerate failure modes."""
    check = GenerationCheck(prompt=prompt, completion=completion)
    stripped = completion.strip()
    check.n_chars = len(completion)
    check.distinct_chars = len(set(completion))

    if not stripped:
        check.problems.append("empty or whitespace-only output")
        return check

    words = re.findall(r"[A-Za-z']+", completion.lower())
    check.n_words = len(words)
    if words:
        counts = Counter(words)
        token, count = counts.most_common(1)[0]
        check.top_token = token
        check.top_token_share = count / len(words)
        check.distinct_word_ratio = len(counts) / len(words)

        if check.top_token_share > MAX_TOP_TOKEN_SHARE:
            check.problems.append(
                f"{check.top_token_share:.0%} of words are {token!r} — this is the "
                "Level-2 failure mode (predicting the most frequent token forever)"
            )
        if check.distinct_word_ratio < MIN_DISTINCT_WORD_RATIO and len(words) >= 10:
            check.problems.append(
                f"only {len(counts)} distinct words in {len(words)} — vocabulary collapse"
            )
    else:
        check.problems.append("no alphabetic words in the output")

    if check.distinct_chars < MIN_DISTINCT_CHARS:
        check.problems.append(
            f"only {check.distinct_chars} distinct characters — collapsed output"
        )

    unit, share = _longest_cycle(completion)
    check.longest_cycle, check.cycle_share = unit, share
    if share > MAX_CYCLE_SHARE and unit:
        check.problems.append(
            f"a {len(unit)}-character cycle {unit!r} covers {share:.0%} of the output"
        )

    if training_text and len(stripped) >= MEMORISATION_WINDOW:
        window = stripped[:MEMORISATION_WINDOW]
        if window in training_text:
            check.memorised = True
            check.problems.append(
                "the first 120 characters appear verbatim in the training text — this is "
                "reproduction, not modelling"
            )
    return check


@dataclass
class SanityReport:
    """All prompts at one checkpoint."""

    checkpoint: str
    step: int | None = None
    checks: list[GenerationCheck] = field(default_factory=list)
    error: str | None = None

    @property
    def n_degenerate(self) -> int:
        return sum(1 for c in self.checks if c.degenerate)

    @property
    def passed(self) -> bool:
        """Passing means *not obviously broken*. It does not mean good."""
        return bool(self.checks) and self.n_degenerate == 0 and not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint, "step": self.step,
            "passed": self.passed, "n_degenerate": self.n_degenerate,
            "checks": [c.to_dict() for c in self.checks], "error": self.error,
            "interpretation": (
                "These checks detect obvious degeneracy only. Passing does NOT establish "
                "language capability; it establishes that the model is not producing the "
                "failure Level 2 produced."
            ),
        }

    def render(self) -> str:
        lines = [f"generation sanity: {self.checkpoint}"]
        if self.step is not None:
            lines.append(f"  step: {self.step}")
        if self.error:
            return "\n".join(lines + [f"  ERROR: {self.error}"])
        for check in self.checks:
            marker = "FAIL" if check.degenerate else "ok  "
            lines.append(f"\n  [{marker}] {check.prompt!r}")
            lines.append(f"         -> {check.completion[:88]!r}")
            if check.n_words:
                lines.append(
                    f"         {check.n_words} words, {check.distinct_word_ratio:.0%} "
                    f"distinct, top token {check.top_token!r} at "
                    f"{check.top_token_share:.0%}"
                )
            for problem in check.problems:
                lines.append(f"         ! {problem}")
        verdict = "NOT OBVIOUSLY BROKEN" if self.passed else "DEGENERATE"
        lines.append(f"\n  VERDICT: {verdict} "
                     f"({self.n_degenerate}/{len(self.checks)} prompts degenerate)")
        if self.passed:
            lines.append("  This says the model is not producing Level 2's failure. It "
                         "does not say the\n  model is good — that needs validation BPB "
                         "on held-out text and a real evaluation.")
        return "\n".join(lines)


def run_sanity_checks(
    model,
    *,
    prompts: tuple[str, ...] = SANITY_PROMPTS,
    max_new_tokens: int = 96,
    device: str = "cpu",
    training_text: str | None = None,
    checkpoint: str = "",
    step: int | None = None,
) -> SanityReport:
    """Generate greedily from each prompt and score the result.

    Greedy, so two runs at the same checkpoint give the same text and a change between
    checkpoints is a change in the model rather than in the sampler.
    """
    from .validate_checkpoint import generate_bytes

    report = SanityReport(checkpoint=checkpoint, step=step)
    try:
        for prompt in prompts:
            completion = generate_bytes(
                model, prompt, max_new_tokens=max_new_tokens, device=device
            )
            report.checks.append(
                check_generation(prompt, completion, training_text=training_text)
            )
    except Exception as exc:  # noqa: BLE001 - the failure is the result
        report.error = f"{type(exc).__name__}: {exc}"
    return report
