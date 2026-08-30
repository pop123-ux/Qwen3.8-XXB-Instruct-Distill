"""A JSONL experiment ledger — append-only, dependency-free, readable with ``cat``.

Deliberately not a database. A research project of this size needs exactly three things
from its record-keeping: that a result cannot be silently edited after the fact, that
every entry says where it came from, and that the file survives being copied to a
different machine by someone who does not have the codebase. A JSONL file does all three;
a database server does none of them without extra work.

Two rules give it teeth.

**Append-only.** :meth:`Ledger.record` only appends. There is no update and no delete. A
result that turns out to be wrong is superseded by a later entry that names it in
``supersedes``, so the retraction is itself part of the record. This is the difference
between a ledger and a scratchpad.

**Provenance is required, not optional.** Every entry must say whether its numbers were
*measured here*, *reported by a third party*, or *estimated from a model*. Mixing those
three is the single most common way a research artifact becomes untrustworthy, so
:data:`PROVENANCE` is a closed set and an unknown value is refused. Estimates additionally
have to carry the method that produced them.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

#: How a number came to exist. Closed set — an entry may not invent a fourth.
MEASURED = "measured_here"
REPORTED = "reported_by_third_party"
ESTIMATED = "estimated"
PROVENANCE = (MEASURED, REPORTED, ESTIMATED)

EntryKind = Literal[
    "architecture_audit", "initialisation", "training_run", "evaluation",
    "memory_accounting", "ablation_result", "comparison", "note", "retraction",
]

DEFAULT_LEDGER = Path("experiments/ledger.jsonl")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def environment() -> dict[str, Any]:
    """What was running when an entry was written. Cheap enough to attach to everything."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }
    for module in ("torch", "transformers"):
        try:
            env[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 — a missing library is a fact about the run
            env[module] = None
    try:
        import torch

        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
        else:
            env["gpu"] = None
    except Exception:  # noqa: BLE001
        env["gpu"] = None
    return env


@dataclass
class Entry:
    """One immutable record."""

    kind: str
    title: str
    provenance: str
    payload: dict[str, Any] = field(default_factory=dict)
    arm: str = ""
    #: For ESTIMATED entries: how the number was produced. Required.
    method: str = ""
    #: For REPORTED entries: where it was published. Required.
    source: str = ""
    supersedes: str | None = None
    tags: list[str] = field(default_factory=list)
    timestamp: str = ""
    env: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCE:
            raise ValueError(
                f"provenance must be one of {PROVENANCE}, got {self.provenance!r}. "
                "There is no fourth option: a number is either measured here, reported "
                "elsewhere, or estimated."
            )
        if self.provenance == ESTIMATED and not self.method:
            raise ValueError(
                f"{self.title!r}: an estimate must carry the method that produced it, "
                "otherwise it is indistinguishable from a measurement in the record"
            )
        if self.provenance == REPORTED and not self.source:
            raise ValueError(
                f"{self.title!r}: a third-party number must carry its source. Unsourced "
                "competitor numbers are exactly what this field exists to prevent."
            )
        if self.provenance == MEASURED and self.source:
            raise ValueError(
                f"{self.title!r}: a measured-here entry must not cite an external source; "
                "that combination reads as a measurement but is a citation"
            )
        self.timestamp = self.timestamp or datetime.now(timezone.utc).isoformat()
        self.env = self.env or environment()
        self.id = self.id or self._digest()

    def _digest(self) -> str:
        body = json.dumps(
            {"kind": self.kind, "title": self.title, "arm": self.arm,
             "payload": self.payload, "timestamp": self.timestamp},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Ledger:
    """Append-only JSONL. One entry per line, newest last."""

    def __init__(self, path: str | Path = DEFAULT_LEDGER) -> None:
        self.path = Path(path)

    # -- writing --------------------------------------------------------
    def record(self, entry: Entry) -> Entry:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry.to_dict(), sort_keys=True, default=str)
        if "\n" in line:  # pragma: no cover — json.dumps escapes newlines
            raise ValueError("an entry serialised to more than one line")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return entry

    def measured(self, kind: str, title: str, payload: dict[str, Any], **kw: Any) -> Entry:
        return self.record(Entry(kind=kind, title=title, provenance=MEASURED,
                                 payload=payload, **kw))

    def reported(self, kind: str, title: str, payload: dict[str, Any], *,
                 source: str, **kw: Any) -> Entry:
        return self.record(Entry(kind=kind, title=title, provenance=REPORTED,
                                 payload=payload, source=source, **kw))

    def estimated(self, kind: str, title: str, payload: dict[str, Any], *,
                  method: str, **kw: Any) -> Entry:
        return self.record(Entry(kind=kind, title=title, provenance=ESTIMATED,
                                 payload=payload, method=method, **kw))

    def retract(self, entry_id: str, reason: str) -> Entry:
        """Supersede an earlier entry. The original line stays; the record shows both."""
        if not self.get(entry_id):
            raise KeyError(f"cannot retract unknown entry {entry_id!r}")
        return self.record(Entry(
            kind="retraction", title=f"retracts {entry_id}", provenance=MEASURED,
            payload={"reason": reason}, supersedes=entry_id,
        ))

    # -- reading --------------------------------------------------------
    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.path}:{number} is not valid JSON: {exc}") from exc

    def entries(self, *, kind: str | None = None, arm: str | None = None,
                provenance: str | None = None, include_superseded: bool = False,
                ) -> list[dict[str, Any]]:
        rows = list(self)
        if not include_superseded:
            retracted = {r["supersedes"] for r in rows if r.get("supersedes")}
            rows = [r for r in rows if r["id"] not in retracted]
        if kind:
            rows = [r for r in rows if r["kind"] == kind]
        if arm:
            rows = [r for r in rows if r.get("arm") == arm]
        if provenance:
            rows = [r for r in rows if r["provenance"] == provenance]
        return rows

    def get(self, entry_id: str) -> dict[str, Any] | None:
        for row in self:
            if row["id"] == entry_id:
                return row
        return None

    def latest(self, kind: str, arm: str = "") -> dict[str, Any] | None:
        rows = self.entries(kind=kind, arm=arm or None)
        return rows[-1] if rows else None

    # -- reporting ------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        rows = list(self)
        live = self.entries()
        by_kind: dict[str, int] = {}
        by_provenance: dict[str, int] = {}
        for row in live:
            by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
            by_provenance[row["provenance"]] = by_provenance.get(row["provenance"], 0) + 1
        return {
            "path": str(self.path),
            "entries": len(rows),
            "live_entries": len(live),
            "superseded": len(rows) - len(live),
            "by_kind": dict(sorted(by_kind.items())),
            "by_provenance": dict(sorted(by_provenance.items())),
            "measured_fraction": (by_provenance.get(MEASURED, 0) / len(live)) if live else 0.0,
            "arms_with_results": sorted({r["arm"] for r in live if r.get("arm")}),
        }

    def render(self) -> str:
        s = self.summary()
        lines = [
            f"  ledger: {s['path']}",
            f"  {s['live_entries']} live entries ({s['superseded']} superseded)",
            "",
            "    kind                      count",
        ]
        lines += [f"    {k:<24}  {v:>4}" for k, v in s["by_kind"].items()]
        lines += ["", "    provenance                count"]
        lines += [f"    {k:<24}  {v:>4}" for k, v in s["by_provenance"].items()]
        if s["arms_with_results"]:
            lines += ["", f"    arms with results: {', '.join(s['arms_with_results'])}"]
        return "\n".join(lines)
