"""Task definitions for the reasoning-efficiency development set.

This is a small, hand-written development set — **not** a benchmark. Its purpose is
to measure how a model's reasoning cost scales with task difficulty, which is the
central claim this project makes about the teacher and hopes to improve in the
student. It is deliberately:

* **small** — it runs cheaply and often, on one GPU;
* **stratified by difficulty** — the whole point is the *shape* of the cost curve, so
  ``trivial`` items must be present and must be genuinely trivial;
* **exactly verifiable where possible** — every item carries a checker, so correctness
  is decided mechanically rather than by eyeballing generations;
* **written from scratch** — no benchmark items are copied in, so this set cannot leak
  a public test set into training data.

For actual capability numbers, use a real benchmark through
:mod:`qwen_distill.evaluation.runner`. This set answers "how many tokens did it spend,
and did it still get it right", not "how good is this model".
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

Difficulty = Literal["trivial", "easy", "moderate", "hard", "very_hard"]

#: Ordered from cheapest to most expensive; used to sort report rows.
DIFFICULTY_ORDER: tuple[Difficulty, ...] = ("trivial", "easy", "moderate", "hard", "very_hard")

Category = Literal[
    "arithmetic", "general_knowledge", "instruction_following", "coding",
    "debugging", "mathematics", "science", "reasoning", "long_context",
]


def _normalise(text: str) -> str:
    return re.sub(r"[\s,]+", " ", text.strip().lower()).strip(" .")


def exact_match(expected: str) -> Callable[[str], bool]:
    """Checker: the normalised answer equals ``expected``."""

    def check(output: str) -> bool:
        return _normalise(output) == _normalise(expected)

    return check


def contains(*expected: str) -> Callable[[str], bool]:
    """Checker: every ``expected`` string appears in the answer (case-insensitive)."""

    def check(output: str) -> bool:
        lowered = output.lower()
        return all(e.lower() in lowered for e in expected)

    return check


def final_number(expected: float, tolerance: float = 1e-6) -> Callable[[str], bool]:
    """Checker: the last number in the answer equals ``expected``.

    Models often restate the question before answering, so the *last* number is a
    more reliable target than the first.
    """

    def check(output: str) -> bool:
        matches = re.findall(r"-?\d+(?:\.\d+)?", output.replace(",", ""))
        if not matches:
            return False
        try:
            return abs(float(matches[-1]) - expected) <= tolerance
        except ValueError:
            return False

    return check


@dataclass(frozen=True)
class Task:
    """One evaluation item."""

    task_id: str
    category: Category
    difficulty: Difficulty
    prompt: str
    #: Returns True when the model's answer is correct. ``None`` means the item is
    #: recorded but not scored (useful for open-ended reasoning-cost probes).
    checker: Callable[[str], bool] | None = None
    system_prompt: str | None = None
    #: Rough guide to how many tokens a *good* answer needs. Not enforced; used to
    #: contextualise measured token counts in reports.
    reference_answer_tokens: int | None = None
    notes: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def score(self, output: str) -> bool | None:
        if self.checker is None:
            return None
        try:
            return self.checker(output)
        except Exception:  # noqa: BLE001 - a broken checker must not kill a run
            return False


def _reasoning_dev_set() -> list[Task]:
    """The stratified development set.

    Trivial items exist to expose overthinking: a model that spends thousands of
    thinking tokens on ``15 * 7`` is doing something wrong, and this set makes that
    visible as a number.
    """
    return [
        # --- trivial: correct answers should be near-immediate --------------
        Task("triv-arith-1", "arithmetic", "trivial", "What is 15 * 7?",
             final_number(105), reference_answer_tokens=5),
        Task("triv-arith-2", "arithmetic", "trivial", "What is 144 divided by 12?",
             final_number(12), reference_answer_tokens=5),
        Task("triv-arith-3", "arithmetic", "trivial", "What is 2 + 2?",
             final_number(4), reference_answer_tokens=3),
        Task("triv-know-1", "general_knowledge", "trivial",
             "What is the capital city of Japan? Answer with the city name only.",
             exact_match("Tokyo"), reference_answer_tokens=3),
        Task("triv-know-2", "general_knowledge", "trivial",
             "How many days are in a standard (non-leap) year? Answer with a number only.",
             final_number(365), reference_answer_tokens=3),
        Task("triv-instr-1", "instruction_following", "trivial",
             "Reply with exactly the word: ACKNOWLEDGED", exact_match("ACKNOWLEDGED"),
             reference_answer_tokens=3),

        # --- easy ----------------------------------------------------------
        Task("easy-arith-1", "arithmetic", "easy",
             "A shop sells pens for 3 units each and notebooks for 7 units each. "
             "What is the total cost of 4 pens and 3 notebooks?",
             final_number(33), reference_answer_tokens=40),
        Task("easy-code-1", "coding", "easy",
             "Write a Python function `is_even(n)` that returns True when n is even. "
             "Return only the code.",
             contains("def is_even", "%"), reference_answer_tokens=40),
        Task("easy-instr-1", "instruction_following", "easy",
             "List exactly three primary colours, one per line, with no other text.",
             contains("red", "blue"), reference_answer_tokens=15),
        Task("easy-sci-1", "science", "easy",
             "What gas do plants absorb from the atmosphere during photosynthesis? "
             "Answer with the gas name only.",
             contains("carbon dioxide"), reference_answer_tokens=5),

        # --- moderate ------------------------------------------------------
        Task("mod-debug-1", "debugging", "moderate",
             "This Python function should return the average of a list but crashes on "
             "an empty list:\n\n"
             "def average(xs):\n    return sum(xs) / len(xs)\n\n"
             "Explain the bug in one sentence and give the corrected function.",
             contains("def average"), reference_answer_tokens=120),
        Task("mod-math-1", "mathematics", "moderate",
             "Solve for x: 3x + 7 = 2x + 19. Give the numeric value of x.",
             final_number(12), reference_answer_tokens=60),
        Task("mod-reason-1", "reasoning", "moderate",
             "Alice is older than Bob. Carol is younger than Bob. "
             "Who is the youngest? Answer with the name only.",
             exact_match("Carol"), reference_answer_tokens=30),
        Task("mod-code-1", "coding", "moderate",
             "Write a Python function `merge_sorted(a, b)` that merges two sorted lists "
             "into one sorted list in O(n+m) time, without calling sorted(). Return only code.",
             contains("def merge_sorted"), reference_answer_tokens=180),

        # --- hard ----------------------------------------------------------
        Task("hard-math-1", "mathematics", "hard",
             "A number leaves remainder 2 when divided by 3, remainder 3 when divided by 5, "
             "and remainder 2 when divided by 7. What is the smallest positive such number?",
             final_number(23), reference_answer_tokens=400),
        Task("hard-reason-1", "reasoning", "hard",
             "Three boxes are labelled 'apples', 'oranges' and 'mixed'. Every label is wrong. "
             "You may draw one fruit from one box. Which box do you draw from to deduce all "
             "three contents? Answer with the label on that box.",
             contains("mixed"), reference_answer_tokens=350),
        Task("hard-code-1", "coding", "hard",
             "Write a Python function `longest_common_subsequence(a, b)` returning the length "
             "of the LCS of two strings, using O(len(a)*len(b)) time and O(min(len(a),len(b))) "
             "space. Return only code.",
             contains("def longest_common_subsequence"), reference_answer_tokens=500),

        # --- very hard: long reasoning is correct here, not a failure -------
        Task("vhard-math-1", "mathematics", "very_hard",
             "Prove that the square root of 2 is irrational. Give a complete proof.",
             contains("contradiction"), reference_answer_tokens=800),
        Task("vhard-reason-1", "reasoning", "very_hard",
             "You have 12 identical-looking balls; exactly one has a different weight "
             "(you do not know whether heavier or lighter). Using a balance scale at most "
             "three times, describe a strategy that always identifies the odd ball and "
             "whether it is heavy or light.",
             None, reference_answer_tokens=1200,
             notes="Open-ended; scored qualitatively. Included to probe the upper end of the cost curve."),
    ]


def reasoning_dev_set() -> list[Task]:
    """Return the stratified reasoning-efficiency development set."""
    return _reasoning_dev_set()


def needle_in_haystack(
    context_tokens: int,
    *,
    needle: str = "The access code for the vault is 74619.",
    question: str = "What is the access code for the vault? Answer with the number only.",
    depth: float = 0.5,
    filler: str = "The quarterly report notes routine operational activity with no incidents. ",
) -> Task:
    """Build a long-context retrieval probe with the needle at a given depth.

    ``depth`` is the fractional position of the needle in the context (0.0 = start,
    1.0 = end). Position matters: hybrid models can behave very differently for
    information held in the recurrent state versus in the attention window, so
    retrieval must be probed at several depths rather than one.

    ``context_tokens`` is approximate — filler is measured in words, and the exact
    token count depends on the tokenizer.
    """
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must be in [0, 1], got {depth}")
    words_per_repeat = len(filler.split())
    repeats = max(1, int(context_tokens / max(words_per_repeat, 1)))
    before = int(repeats * depth)
    body = filler * before + needle + " " + filler * (repeats - before)
    return Task(
        task_id=f"niah-{context_tokens}-d{int(depth * 100)}",
        category="long_context",
        difficulty="moderate",
        prompt=f"{body}\n\n{question}",
        checker=final_number(74619),
        reference_answer_tokens=5,
        metadata={"approx_context_tokens": str(context_tokens), "depth": str(depth)},
        notes="Synthetic retrieval probe; approximate context length.",
    )


def long_context_suite(
    context_lengths: tuple[int, ...] = (1024, 4096, 16384),
    depths: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> list[Task]:
    """Needle-in-haystack probes across context lengths and needle depths."""
    return [needle_in_haystack(n, depth=d) for n in context_lengths for d in depths]


def tasks_by_difficulty(tasks: list[Task]) -> dict[str, list[Task]]:
    """Group tasks by difficulty, in ascending difficulty order."""
    grouped: dict[str, list[Task]] = {d: [] for d in DIFFICULTY_ORDER}
    for task in tasks:
        grouped.setdefault(task.difficulty, []).append(task)
    return {k: v for k, v in grouped.items() if v}
