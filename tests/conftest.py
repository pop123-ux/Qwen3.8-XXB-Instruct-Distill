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


# ----------------------------------------------------------------------------------
# checkpoint fixtures
# ----------------------------------------------------------------------------------
#
# One builder, because the checkpoint tests used to have three slightly different ones
# and every one of them wrote 32-byte "weights". A validator that accepts a 32-byte
# model.safetensors is exactly the validator that let a Level-2R run advertise three
# hollow checkpoints on Drive, so the fixtures have to produce files a hardened
# validator will actually accept.

#: Plausible sizes for a small synthetic checkpoint. Well above the validator's absolute
#: floors and consistent with PARAMETER_COUNT, so a fixture checkpoint passes for the
#: same reasons a real one does.
PARAMETER_COUNT = 262_912
FIXTURE_SIZES = {
    "model.safetensors": PARAMETER_COUNT * 4,
    "optimizer.pt": PARAMETER_COUNT * 8,
    "scheduler.pt": 512,
    "scaler.pt": 512,
    "rng.pt": 4096,
}


def make_checkpoint_dir(
    root,
    step: int,
    *,
    complete: bool = True,
    files=None,
    parameter_count: int | None = PARAMETER_COUNT,
    manifest: bool = True,
    metadata_contents: bool = True,
    extra_metadata: dict | None = None,
):
    """A checkpoint directory shaped exactly like the trainer's, without torch.

    ``files`` selects which artifacts exist, so a test can remove one and assert the
    validator notices. ``manifest`` controls whether ``checkpoint_manifest.json`` is
    written, so both the current format and pre-manifest checkpoints are exercised.
    """
    import json as _json
    from pathlib import Path as _Path

    from qwen_distill.training.checkpoint_validation import (
        COMPLETE_MARKER as _MARKER,
    )
    from qwen_distill.training.checkpoint_validation import (
        MANIFEST_FILENAME as _MANIFEST,
    )
    from qwen_distill.training.checkpoint_validation import (
        build_manifest as _build_manifest,
    )

    names = tuple(files) if files is not None else (
        "model.safetensors", "optimizer.pt", "training_state.json", "metadata.json",
    )
    directory = _Path(root) / f"step_{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)

    for name in names:
        if name in ("metadata.json", _MARKER, _MANIFEST):
            continue
        if name == "training_state.json":
            (directory / name).write_text(
                _json.dumps({"step": step, "epoch": 0}), encoding="utf-8"
            )
        elif name == "config.json":
            (directory / name).write_text(_json.dumps({"name": "fixture"}), encoding="utf-8")
        else:
            (directory / name).write_bytes(b"\x00" * FIXTURE_SIZES.get(name, 4096))

    if "metadata.json" in names:
        contents = sorted(names) + ([_MANIFEST] if manifest else []) + [_MARKER]
        payload = {
            "step": step,
            "complete": complete,
            "created_at": "2026-01-01T00:00:00+00:00",
            "parameter_count": parameter_count,
            # `contents` is the checkpoint's own statement of what it holds, and it
            # is what makes a later deletion detectable. A fixture that omits it is
            # simulating a checkpoint written before that existed.
            "contents": contents if metadata_contents else [],
        }
        # Merged before the manifest is built, because the manifest digests
        # metadata.json — editing it afterwards is a real corruption and the validator
        # is right to say so.
        payload.update(extra_metadata or {})
        (directory / "metadata.json").write_text(_json.dumps(payload), encoding="utf-8")

    if manifest:
        (directory / _MANIFEST).write_text(
            _json.dumps(_build_manifest(directory, step=step, parameter_count=parameter_count)),
            encoding="utf-8",
        )
    if complete:
        (directory / _MARKER).write_text(f"step {step}\n", encoding="utf-8")
    return directory
