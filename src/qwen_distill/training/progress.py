"""Training progress that outlives the process writing it.

The Level-2 run's numbers — validation BPB 1.317 at step 200, 1.279 at step 400 —
existed only in a Colab cell's stdout. When the runtime disconnected, the record of what
had been learned went with it, along with the checkpoints.

Progress records and checkpoints answer different questions and must have different
costs. A full checkpoint for a 94.5M model with fp32 AdamW state is roughly 1.1 GB;
writing one every step would make the experiment I/O-bound and, if it were being copied
to Drive, unusable. A progress record is a few hundred bytes. So:

    every log interval  -> progress record  (metrics, kilobytes)
    every save interval -> full checkpoint  (weights + optimizer, ~GB)

Both are written atomically, because the whole point is surviving a process that dies
mid-write.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoints import atomic_write_json


@dataclass
class ProgressWriter:
    """Append-only metrics plus an atomically-updated pointer to the latest step.

    ``metrics.jsonl`` is the full history — one record per line, append-only, so a
    partially written final line costs one record rather than the file. ``latest.json``
    answers "where did this run get to?" without parsing the history, which is what a
    recovery flow actually needs.
    """

    directory: Path
    git_commit: str | None = None
    config_sha256: str | None = None

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._metrics = self.directory / "metrics.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self._metrics

    @property
    def latest_path(self) -> Path:
        return self.directory / "progress" / "latest.json"

    def write(self, record: dict[str, Any], *, status: str = "completed_step") -> dict[str, Any]:
        """Record one step. Appended to the history and published as `latest`."""
        payload = dict(record)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        payload["status"] = status
        if self.git_commit:
            payload.setdefault("git_commit", self.git_commit)
        if self.config_sha256:
            payload.setdefault("config_sha256", self.config_sha256)

        # Append and fsync: an appended line survives a kill, a rewritten file may not.
        with open(self._metrics, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        atomic_write_json(self.latest_path, payload)
        return payload

    def read_latest(self) -> dict[str, Any] | None:
        if not self.latest_path.is_file():
            return None
        try:
            return json.loads(self.latest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def read_history(self) -> list[dict[str, Any]]:
        """Every complete record. A truncated final line is skipped, not fatal."""
        if not self._metrics.is_file():
            return []
        records: list[dict[str, Any]] = []
        for line in self._metrics.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a kill mid-write costs this record, not the history
        return records
