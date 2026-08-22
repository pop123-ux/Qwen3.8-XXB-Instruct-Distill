"""Tests for the offline checkpoint fixture builder.

The fixture is what lets every other component be exercised without the teacher
checkpoint, so it needs to actually be loadable.
"""

from __future__ import annotations

import json

from conftest import requires_stack

from qwen_distill.architecture.params import count_parameters
from qwen_distill.testing import TINY_SPEC, write_tiny_checkpoint


@requires_stack
def test_fixture_writes_a_complete_checkpoint(tmp_path):
    root = write_tiny_checkpoint(tmp_path / "ckpt")
    for name in (
        "config.json", "generation_config.json", "chat_template.jinja",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "model.safetensors",
    ):
        assert (root / name).is_file(), name


@requires_stack
def test_fixture_is_loadable_by_transformers(tmp_path):
    from transformers import AutoModelForCausalLM

    root = write_tiny_checkpoint(tmp_path / "ckpt")
    model = AutoModelForCausalLM.from_pretrained(str(root))
    assert type(model).__name__ == "Qwen3_5ForCausalLM"
    assert sum(p.numel() for p in model.parameters()) == count_parameters(TINY_SPEC).total


@requires_stack
def test_fixture_config_declares_the_hybrid_layout(tmp_path):
    root = write_tiny_checkpoint(tmp_path / "ckpt", with_weights=False)
    config = json.loads((root / "config.json").read_text())
    layer_types = config["layer_types"]
    assert layer_types.count("full_attention") == TINY_SPEC.num_full_attention_layers
    assert layer_types[-1] == "full_attention"


@requires_stack
def test_fixture_can_include_mtp_tensors(tmp_path):
    from qwen_distill.teacher.inspect import inspect_local

    root = write_tiny_checkpoint(tmp_path / "ckpt", with_mtp=True)
    report = inspect_local(root)
    assert report.mtp_tensors
    assert all(n.startswith("mtp.") for n in report.mtp_tensors)


@requires_stack
def test_transformers_ignores_mtp_tensors_on_load(tmp_path):
    """Mirrors the real checkpoint: stock transformers discards the MTP head."""
    from transformers import AutoModelForCausalLM

    root = write_tiny_checkpoint(tmp_path / "ckpt", with_mtp=True)
    model = AutoModelForCausalLM.from_pretrained(str(root))
    assert not any("mtp" in name for name, _ in model.named_parameters())


@requires_stack
def test_config_only_fixture_skips_weights(tmp_path):
    root = write_tiny_checkpoint(tmp_path / "ckpt", with_weights=False)
    assert not (root / "model.safetensors").exists()
    assert (root / "config.json").is_file()
