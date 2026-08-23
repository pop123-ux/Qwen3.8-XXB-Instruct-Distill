"""Enforce offline operation when working from a local metadata directory.

When a checkpoint path is local, no tool in this repository should contact the Hub —
not to check for a newer revision, not to resolve a tokenizer, not for anything. Two
reasons: the environments this project runs in may block egress entirely, and a silent
network fetch would undermine the claim that a report was produced purely from the
supplied files.

`transformers` and `huggingface_hub` both honour environment variables for this, so the
guard sets them for the duration of the call and restores whatever was there before.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

#: Environment variables that make the HF stack refuse network access.
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


def looks_local(model: str | Path) -> bool:
    """True when ``model`` names a directory on disk rather than a Hub repo id.

    A Hub repo id is ``owner/name`` and has no filesystem presence; anything that
    resolves to an existing directory is local.
    """
    try:
        return Path(model).is_dir()
    except (OSError, ValueError):
        return False


@contextmanager
def offline_mode(enabled: bool = True) -> Iterator[bool]:
    """Temporarily forbid Hub access.

    Yields whether the guard is active, so callers can report it. Restores the previous
    environment on exit, including unsetting variables that were not set before.
    """
    if not enabled:
        yield False
        return

    previous: dict[str, str | None] = {k: os.environ.get(k) for k in OFFLINE_ENV}
    os.environ.update(OFFLINE_ENV)
    try:
        yield True
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def offline_for(model: str | Path) -> Iterator[bool]:
    """Enable offline mode automatically when ``model`` is a local directory."""
    with offline_mode(looks_local(model)) as active:
        yield active
