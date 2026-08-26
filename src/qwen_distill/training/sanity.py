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
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

#: Bump whenever :data:`SANITY_PROMPTS` changes. Two reports at different versions were
#: generated from different prompts, so their pass rates are not the same measurement —
#: recorded in every report so an old one stays interpretable instead of silently
#: comparable.
PROMPT_SET_VERSION = "2.0"

#: Fixed prompts, used unchanged across every checkpoint so generations are comparable
#: over time. Short and ordinary: a model that has learned English should continue them
#: with something English-shaped, whatever the content.
#:
#: Two lengths, deliberately. The short prompts (``"The "``, ``"It was "``) leave the model
#: almost unconstrained and are where Level 2's collapse to a single token showed up
#: fastest. The longer ones added in v2.0 supply several words of real syntactic context —
#: a determiner, a preposition, a tense — so a model that has learned local structure has
#: something to continue and one that has only learned unigram frequencies has nowhere to
#: hide. ``"Yesterday, I"`` additionally sets a past tense and a first person, and
#: ``"In the middle of the"`` ends mid-phrase on a determiner, where the only
#: grammatical continuation is a noun.
SANITY_PROMPTS: tuple[str, ...] = (
    # v1.0 — short and nearly unconstrained
    "The ",
    "In the beginning ",
    "It was ",
    "Once upon a time ",
    "The most important ",
    "When the sun ",
    # v2.0 — several words of syntactic context
    "The beginning of the story was",
    "It was a",
    "In the middle of the",
    "The most important thing",
    "Yesterday, I",
)

#: Thresholds. Set to catch the obviously-broken, not to grade quality — a real model
#: clears these by a wide margin and a degenerate one fails them unambiguously.
MAX_TOP_TOKEN_SHARE = 0.5      # >50% of words being one word is Level 2's failure
MIN_DISTINCT_CHARS = 8
MIN_DISTINCT_WORD_RATIO = 0.15
MAX_CYCLE_SHARE = 0.6          # a short cycle covering most of the output

#: Repeated-n-gram fraction: of all n-word windows, what share are windows seen before.
#: Level 2R's failure mode was phrase-level repetition — "the street was standing before
#: the street was standing before" — which clears every threshold above: no single token
#: dominates, the distinct-word ratio stays well over 0.15, and there is no exact
#: character cycle because the repeat is interrupted. Measuring it needs its own metric.
#:
#: This is REPORTED on every generation and only escalates to a problem at
#: :data:`MAX_REPEATED_NGRAM_SHARE`, which is set high deliberately: some repetition is
#: normal English ("of the", "and the"), and a detector that fires on it is a detector
#: nobody reads. The number matters more than the verdict.
REPETITION_NGRAM_SIZES: tuple[int, ...] = (3, 4)
MAX_REPEATED_NGRAM_SHARE = 0.5
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
    #: Repeated-n-gram share keyed by n. Level 2R's actual failure mode, and invisible to
    #: every other field here.
    repeated_ngrams: dict[int, float] = field(default_factory=dict)
    memorised: bool = False
    problems: list[str] = field(default_factory=list)

    #: Tokens actually produced, counted from the generated ids rather than from the
    #: decoded text. A byte-level model emits sequences that are not valid UTF-8, and
    #: ``decode`` replaces those, so the character count is not a token count.
    n_generated_tokens: int | None = None
    #: What ``max_new_tokens`` was set to. Fewer tokens produced than requested means
    #: generation stopped early, which is information, not noise.
    n_requested_tokens: int | None = None
    #: When this continuation was produced, UTC ISO-8601.
    generated_at: str | None = None

    @property
    def degenerate(self) -> bool:
        return bool(self.problems)

    @property
    def stopped_early(self) -> bool:
        return (
            self.n_generated_tokens is not None
            and self.n_requested_tokens is not None
            and self.n_generated_tokens < self.n_requested_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["degenerate"] = self.degenerate
        data["stopped_early"] = self.stopped_early
        return data


def repeated_ngram_share(text: str, n: int = 4) -> float:
    """Fraction of ``n``-word windows that have already been seen in this text.

    0.0 means every window is new; 0.5 means half the text is re-treading phrases it
    already produced. Word-level rather than character-level, so it survives the
    punctuation and line breaks that defeat an exact cycle check.

    Returns 0.0 when the text is shorter than one window — with nothing to repeat,
    "no repetition detected" would be a claim nobody measured.
    """
    words = re.findall(r"[A-Za-z']+", text.lower())
    if len(words) < n + 1:
        return 0.0
    windows = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    seen: set[tuple[str, ...]] = set()
    repeats = 0
    for window in windows:
        if window in seen:
            repeats += 1
        else:
            seen.add(window)
    return repeats / len(windows)


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
    prompt: str,
    completion: str,
    *,
    training_text: str | None = None,
    n_generated_tokens: int | None = None,
    n_requested_tokens: int | None = None,
    generated_at: str | None = None,
) -> GenerationCheck:
    """Score one continuation against the known degenerate failure modes.

    The provenance arguments are recorded, never used in scoring: a generation that
    stopped early is not thereby degenerate, and a report has to carry enough to be
    reproduced from.
    """
    check = GenerationCheck(
        prompt=prompt, completion=completion,
        n_generated_tokens=n_generated_tokens,
        n_requested_tokens=n_requested_tokens,
        generated_at=generated_at or _utc_now(),
    )
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

    check.repeated_ngrams = {
        n: round(repeated_ngram_share(completion, n), 4) for n in REPETITION_NGRAM_SIZES
    }
    worst_n = max(check.repeated_ngrams, key=lambda n: check.repeated_ngrams[n], default=None)
    if worst_n is not None and check.repeated_ngrams[worst_n] > MAX_REPEATED_NGRAM_SHARE:
        check.problems.append(
            f"{check.repeated_ngrams[worst_n]:.0%} of {worst_n}-word windows repeat an "
            f"earlier one — phrase-level looping"
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

    #: Exactly how these generations were produced. Recorded because greedy decoding at
    #: 96 tokens on CPU and sampled decoding at 512 on GPU are different measurements,
    #: and a report that does not say which is not reproducible.
    settings: dict[str, Any] = field(default_factory=dict)
    #: Which prompt set produced this report. Two reports at different versions used
    #: different prompts; their pass rates are not comparable.
    prompt_set_version: str = PROMPT_SET_VERSION
    generated_at: str = field(default_factory=_utc_now)
    #: Whether a training corpus was supplied for the memorisation check. Without one,
    #: ``memorised`` is False everywhere because nothing was checked — which must not
    #: read as "no memorisation found".
    memorisation_checked: bool = False

    @property
    def n_degenerate(self) -> int:
        return sum(1 for c in self.checks if c.degenerate)

    @property
    def passed(self) -> bool:
        """Passing means *not obviously broken*. It does not mean good."""
        return bool(self.checks) and self.n_degenerate == 0 and not self.error

    @property
    def total_generated_tokens(self) -> int:
        return sum(c.n_generated_tokens or 0 for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint, "step": self.step,
            "passed": self.passed, "n_degenerate": self.n_degenerate,
            "n_prompts": len(self.checks),
            "prompt_set_version": self.prompt_set_version,
            "generated_at": self.generated_at,
            "settings": self.settings,
            "memorisation_checked": self.memorisation_checked,
            "total_generated_tokens": self.total_generated_tokens,
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
        lines.append(f"  prompt set: v{self.prompt_set_version} ({len(self.checks)} prompts)")
        if self.settings:
            lines.append(
                "  settings: "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.settings.items()))
            )
        lines.append(f"  generated at: {self.generated_at}")
        if not self.memorisation_checked:
            lines.append(
                "  memorisation: NOT CHECKED (no --training-text given; 'memorised: no' "
                "below means nothing was compared)"
            )
        if self.error:
            return "\n".join(lines + [f"  ERROR: {self.error}"])
        for check in self.checks:
            marker = "FAIL" if check.degenerate else "ok  "
            lines.append(f"\n  [{marker}] {check.prompt!r}")
            lines.append(f"         -> {check.completion[:88]!r}")
            if check.n_generated_tokens is not None:
                early = " (stopped early)" if check.stopped_early else ""
                lines.append(
                    f"         {check.n_generated_tokens} of "
                    f"{check.n_requested_tokens} tokens{early}, "
                    f"{check.n_chars} chars"
                )
            if check.n_words:
                lines.append(
                    f"         {check.n_words} words, {check.distinct_word_ratio:.0%} "
                    f"distinct, top token {check.top_token!r} at "
                    f"{check.top_token_share:.0%}"
                )
            if check.repeated_ngrams:
                lines.append(
                    "         repeated n-grams: "
                    + ", ".join(f"{n}-gram {v:.0%}"
                                for n, v in sorted(check.repeated_ngrams.items()))
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

    Every generation records what produced it — prompt, text, checkpoint, decoding
    settings, token count and timestamp — so a report can be reproduced or found wanting
    later. Level 2's degenerate generations were recorded as prose in a README; that was
    enough to notice the problem and not enough to re-run the check.
    """
    from .validate_checkpoint import generate_bytes_detailed

    report = SanityReport(
        checkpoint=checkpoint,
        step=step,
        settings={
            "decoding": "greedy",
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "device": device,
            "tokenisation": "byte-level (vocab 256)",
        },
        memorisation_checked=bool(training_text),
        # A caller that supplied its own prompts did not run the versioned set, and a
        # report claiming v2.0 for arbitrary prompts would make two incomparable runs
        # look comparable.
        prompt_set_version=(
            PROMPT_SET_VERSION if tuple(prompts) == SANITY_PROMPTS else "custom"
        ),
    )
    try:
        for prompt in prompts:
            started = _utc_now()
            completion, generated_ids = generate_bytes_detailed(
                model, prompt, max_new_tokens=max_new_tokens, device=device
            )
            report.checks.append(
                check_generation(
                    prompt, completion, training_text=training_text,
                    n_generated_tokens=len(generated_ids),
                    n_requested_tokens=max_new_tokens,
                    generated_at=started,
                )
            )
    except Exception as exc:  # noqa: BLE001 - the failure is the result
        report.error = f"{type(exc).__name__}: {exc}"
    return report
