"""Shared fixtures.

Tests that need `torch`/`transformers` skip cleanly when those are absent, so the
analysis-only install (``requirements/base.txt``) still has a green suite.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip  # re-exported for readability in test modules


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


HAS_TORCH = _has("torch")
HAS_TRANSFORMERS = _has("transformers")
HAS_STACK = HAS_TORCH and HAS_TRANSFORMERS

requires_stack = pytest.mark.skipif(
    not HAS_STACK, reason="requires torch and transformers (pip install -r requirements/training.txt)"
)


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """A complete miniature Qwen3.5-family checkpoint, built offline."""
    if not HAS_STACK:
        pytest.skip("requires torch and transformers")
    from qwen_distill.testing import write_tiny_checkpoint

    path = tmp_path_factory.mktemp("ckpt") / "tiny"
    write_tiny_checkpoint(path)
    return path


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_checkpoint):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tiny_checkpoint))
