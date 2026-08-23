"""Tests for model-loading arguments that matter at 27B scale.

Each of these guards a failure that would only show up on the expensive GPU run:

* ``dtype="auto"`` reaching `transformers` as the string. Converting it to ``None``
  makes the model load in float32 regardless of the checkpoint — ~108 GB for a 27B bf16
  checkpoint instead of ~54 GB, which fits on no single GPU.
* ``device_map`` failing fast when ``accelerate`` is missing, rather than after a long
  load.
* ``unload()`` actually freeing the model, so a paired teacher/student run does not
  hold both at once.
"""

from __future__ import annotations

import json

import pytest
from conftest import requires_stack

from qwen_distill.evaluation.runner import TransformersBackend


@pytest.fixture
def bf16_checkpoint(tmp_path):
    """A checkpoint declaring bfloat16, so 'auto' and None give different answers."""
    from qwen_distill.testing import write_tiny_checkpoint

    root = write_tiny_checkpoint(tmp_path / "ckpt")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["dtype"] = "bfloat16"
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


@requires_stack
def test_dtype_auto_respects_the_checkpoint_dtype(bf16_checkpoint):
    """The regression: 'auto' must not become None, which would load float32."""
    import torch

    backend = TransformersBackend(str(bf16_checkpoint), device="cpu", dtype="auto")
    backend.load()
    assert next(backend._model.parameters()).dtype == torch.bfloat16


@requires_stack
def test_explicit_dtype_overrides_the_checkpoint(bf16_checkpoint):
    import torch

    backend = TransformersBackend(str(bf16_checkpoint), device="cpu", dtype="float32")
    backend.load()
    assert next(backend._model.parameters()).dtype == torch.float32


@requires_stack
def test_device_map_without_accelerate_fails_fast(bf16_checkpoint):
    """Must raise before loading weights, with a message naming the fix."""
    try:
        import accelerate  # noqa: F401

        pytest.skip("accelerate is installed; the guard cannot be exercised")
    except ImportError:
        pass

    backend = TransformersBackend(str(bf16_checkpoint), device="auto", dtype="auto")
    with pytest.raises(ImportError, match="requires `accelerate`"):
        backend.load()


@requires_stack
def test_cpu_device_does_not_require_accelerate(bf16_checkpoint):
    backend = TransformersBackend(str(bf16_checkpoint), device="cpu", dtype="auto")
    backend.load()
    assert backend._model is not None


@requires_stack
def test_unload_releases_the_model(bf16_checkpoint):
    """Without this, paired_eval holds teacher and student simultaneously."""
    backend = TransformersBackend(str(bf16_checkpoint), device="cpu", dtype="auto")
    backend.load()
    assert backend._model is not None
    backend.unload()
    assert backend._model is None
    assert backend._tokenizer is None


@requires_stack
def test_backend_can_reload_after_unload(bf16_checkpoint):
    backend = TransformersBackend(str(bf16_checkpoint), device="cpu", dtype="auto")
    backend.load()
    backend.unload()
    backend.load()
    assert backend._model is not None


@requires_stack
def test_probe_reports_gpu_memory_as_unavailable_on_cpu(bf16_checkpoint):
    """A zero would be indistinguishable from a real measurement of zero."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present; this guard applies to CPU-only hosts")
    backend = TransformersBackend(
        str(bf16_checkpoint), device="cpu", dtype="float32", max_new_tokens=4
    )
    probe = backend.probe()
    assert probe.cuda_available is False
    assert probe.peak_gpu_memory_gib is None
    assert any("unavailable" in note for note in probe.notes)


@requires_stack
def test_probe_records_the_rendered_prompt_and_hash(bf16_checkpoint):
    backend = TransformersBackend(
        str(bf16_checkpoint), device="cpu", dtype="float32", max_new_tokens=4
    )
    probe = backend.probe()
    assert probe.ok
    assert probe.rendered_prompt
    assert probe.rendered_prompt_sha256
    assert len(probe.rendered_prompt_sha256) == 64
    assert probe.tokenizer_class
