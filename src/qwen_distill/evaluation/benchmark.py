"""A benchmark suite as a versioned, checksummed artifact.

The commitment this makes is not about which benchmarks to run — that is deliberately
undecided. It is that once a suite is published, its prompts are **immutable**. A suite
whose contents drift cannot support the comparison it exists for: "the student improved"
and "the questions got easier" become indistinguishable, and nothing in the results says
which happened.

So a suite carries a digest over its own prompts, every result records that digest, and
comparing results across a changed suite is refused rather than silently averaged.

Categories are declared but empty. Downloading benchmark data is out of scope here, and
inventing questions to fill a category would produce a suite that measures nothing while
looking complete.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..distillation.provenance import RunManifest, sha256_json, sha256_text, utc_now

#: Capability areas a full evaluation should eventually cover. Naming them now shapes
#: the harness; choosing datasets for them is a later decision.
CATEGORIES: tuple[str, ...] = (
    "general_knowledge", "math", "reasoning", "coding",
    "instruction_following", "long_context",
)

SUITE_VERSION = "1.0"


@dataclass
class BenchmarkItem:
    """One immutable evaluation question."""

    id: str
    prompt: str
    category: str = "reasoning"
    difficulty: str = "medium"
    #: Accepted answers for exact-match grading. Empty means this item needs a grader
    #: that does not exist yet, and it is reported as ungraded rather than as wrong.
    answers: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gradable(self) -> bool:
        return bool(self.answers)

    def grade(self, output: str) -> bool | None:
        """Exact match after normalisation, or ``None`` when the item is not gradable.

        ``None`` matters: an ungraded item must not be counted as a failure, or accuracy
        silently becomes a measure of how much of the suite has answer keys.
        """
        if not self.answers:
            return None
        normalised = " ".join(output.lower().split()).strip(" .")
        return any(
            " ".join(answer.lower().split()).strip(" .") == normalised
            for answer in self.answers
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["answers"] = list(self.answers)
        data["gradable"] = self.gradable
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkItem:
        return cls(
            id=str(data["id"]), prompt=str(data["prompt"]),
            category=str(data.get("category", "reasoning")),
            difficulty=str(data.get("difficulty", "medium")),
            answers=tuple(data.get("answers", ())),
            metadata=data.get("metadata", {}) or {},
        )

    @property
    def prompt_sha256(self) -> str:
        return sha256_text(self.prompt)


@dataclass
class BenchmarkSuite:
    """A named, versioned, checksummed set of questions."""

    name: str
    version: str = SUITE_VERSION
    description: str = ""
    items: list[BenchmarkItem] = field(default_factory=list)
    #: Where the questions came from and whether they could be in training data. See
    #: docs/EVALUATION_PROTOCOL.md — an undocumented suite is not usable evidence.
    contamination_notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    @property
    def digest(self) -> str:
        """Over ids and prompt content, so any edit changes it."""
        return sha256_json([[item.id, item.prompt_sha256] for item in self.items])

    @property
    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.category] = counts.get(item.category, 0) + 1
        return counts

    def validate(self) -> list[str]:
        problems: list[str] = []
        seen: set[str] = set()
        for item in self.items:
            if item.id in seen:
                problems.append(f"duplicate item id {item.id!r}")
            seen.add(item.id)
            if not item.prompt.strip():
                problems.append(f"item {item.id!r} has an empty prompt")
        unknown = sorted({i.category for i in self.items} - set(CATEGORIES))
        if unknown:
            problems.append(f"unknown categories: {', '.join(unknown)}")
        if not self.contamination_notes:
            problems.append(
                "no contamination notes: a suite with no statement about whether its "
                "questions could appear in training data is not usable evidence"
            )
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "version": self.version, "description": self.description,
            "digest": self.digest, "n_items": len(self.items),
            "categories": self.categories,
            "contamination_notes": self.contamination_notes,
            "created_at": self.created_at,
            "items": [item.to_dict() for item in self.items],
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def read(cls, path: str | Path) -> BenchmarkSuite:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        suite = cls(
            name=data["name"], version=data.get("version", SUITE_VERSION),
            description=data.get("description", ""),
            items=[BenchmarkItem.from_dict(i) for i in data.get("items", [])],
            contamination_notes=data.get("contamination_notes", []),
            created_at=data.get("created_at", utc_now()),
        )
        recorded = data.get("digest")
        if recorded and recorded != suite.digest:
            raise ValueError(
                f"benchmark suite {suite.name!r} has been modified since it was written: "
                f"recorded digest {recorded[:12]}, computed {suite.digest[:12]}. A suite "
                "must be immutable once published, or results across it are not comparable."
            )
        return suite


@dataclass
class BenchmarkRun:
    """Results of one model against one suite, tied to both."""

    suite_name: str
    suite_version: str
    suite_digest: str
    model: str
    model_revision: str | None = None
    checkpoint: str | None = None
    reasoning_mode: str | None = None
    seed: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        from .paired import summarise_side

        return summarise_side(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name, "suite_version": self.suite_version,
            "suite_digest": self.suite_digest, "model": self.model,
            "model_revision": self.model_revision, "checkpoint": self.checkpoint,
            "reasoning_mode": self.reasoning_mode, "seed": self.seed,
            "n_results": len(self.results), "summary": self.summary(),
            "manifest": self.manifest, "results": self.results,
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target


def new_run(suite: BenchmarkSuite, *, model: str, model_revision: str | None = None,
            checkpoint: str | None = None, reasoning_mode: str | None = None,
            seed: int = 0) -> BenchmarkRun:
    """Start a run, binding it to the suite's digest at this moment."""
    return BenchmarkRun(
        suite_name=suite.name, suite_version=suite.version, suite_digest=suite.digest,
        model=model, model_revision=model_revision, checkpoint=checkpoint,
        reasoning_mode=reasoning_mode, seed=seed,
        manifest=RunManifest(
            kind="benchmark",
            teacher={"model": model, "revision": model_revision,
                     "is_pinned": model_revision is not None},
            reasoning_mode=reasoning_mode,
            dataset={"suite": suite.name, "version": suite.version,
                     "digest": suite.digest, "n_items": len(suite.items)},
        ).to_dict(),
    )


def comparable(a: BenchmarkRun, b: BenchmarkRun) -> tuple[bool, str | None]:
    """Whether two runs may be compared at all.

    Refusing is the point. Two results over different question sets can always be put in
    a table next to each other, and the table will be wrong.
    """
    if a.suite_name != b.suite_name:
        return False, f"different suites: {a.suite_name!r} vs {b.suite_name!r}"
    if a.suite_digest != b.suite_digest:
        return False, (
            f"same suite name but different contents ({a.suite_digest[:12]} vs "
            f"{b.suite_digest[:12]}): the questions changed between these runs"
        )
    return True, None
