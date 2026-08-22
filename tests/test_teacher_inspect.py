"""Teacher-inspection tests, driven by a synthetic checkpoint fixture.

These verify the inspector's *mechanics* (header parsing, MTP/vision partitioning,
cross-check arithmetic) without needing a 50 GB download. Verifying the real
teacher is a separate, network-dependent step: ``scripts/inspect_teacher.py``.
"""

from __future__ import annotations

import json
import struct

import pytest

from qwen_distill.architecture.spec import HybridArchSpec
from qwen_distill.teacher.inspect import (
    cross_check,
    inspect_local,
    read_safetensors_header,
)


def write_safetensors(path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    """Write a safetensors file with a valid header and zero-filled payload."""
    header: dict[str, object] = {}
    offset = 0
    dtype_bytes = {"BF16": 2, "F32": 4}
    for name, (dtype, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        size = n * dtype_bytes[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * offset)


@pytest.fixture
def fake_checkpoint(tmp_path):
    """A miniature multimodal hybrid checkpoint with MTP and vision tensors."""
    spec = HybridArchSpec(
        name="mini", hidden_size=256, num_hidden_layers=4, intermediate_size=512,
        vocab_size=1000, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=6,
        linear_key_head_dim=32, linear_value_head_dim=32, max_position_embeddings=4096,
    )
    root = tmp_path / "ckpt"
    root.mkdir()
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "torch_dtype": "bfloat16",
        "text_config": spec.to_hf_text_config(),
        "vision_config": {"model_type": "qwen3_5_vision", "depth": 2, "hidden_size": 64},
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "generation_config.json").write_text(
        json.dumps({"temperature": 0.7, "top_p": 0.8, "top_k": 20}), encoding="utf-8"
    )
    (root / "chat_template.jinja").write_text(
        "{% if reasoning_effort == 'xhigh' %}<think>{% endif %}"
        "{{ enable_thinking }}{{ preserve_thinking }}</think>", encoding="utf-8"
    )
    write_safetensors(
        root / "model-00001-of-00001.safetensors",
        {
            "model.embed_tokens.weight": ("BF16", [1000, 256]),
            "model.layers.0.mlp.gate_proj.weight": ("BF16", [512, 256]),
            "mtp.layers.0.weight": ("BF16", [256, 256]),
            "mtp.norm.weight": ("BF16", [256]),
            "model.visual.blocks.0.attn.qkv.weight": ("BF16", [192, 64]),
        },
    )
    return root, spec


def test_read_safetensors_header_roundtrip(tmp_path):
    path = tmp_path / "t.safetensors"
    write_safetensors(path, {"a": ("F32", [2, 3]), "b": ("BF16", [4])})
    header = read_safetensors_header(path)
    assert header["a"]["shape"] == [2, 3]
    assert header["b"]["dtype"] == "BF16"


def test_read_safetensors_header_rejects_truncated_file(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"\x01\x02")
    with pytest.raises(ValueError, match="too short"):
        read_safetensors_header(path)


def test_inspect_recovers_spec_from_config(fake_checkpoint):
    root, spec = fake_checkpoint
    report = inspect_local(root)
    assert report.model_type == "qwen3_5"
    assert report.spec is not None
    assert report.spec.hidden_size == spec.hidden_size
    assert report.spec.num_hidden_layers == spec.num_hidden_layers
    assert report.spec.resolved_layer_types() == spec.resolved_layer_types()


def test_inspect_detects_multimodality(fake_checkpoint):
    root, _ = fake_checkpoint
    report = inspect_local(root)
    assert report.is_multimodal
    assert report.vision_config is not None


def test_inspect_partitions_mtp_and_vision_tensors(fake_checkpoint):
    root, _ = fake_checkpoint
    report = inspect_local(root)
    assert report.mtp_tensors == ["mtp.layers.0.weight", "mtp.norm.weight"]
    assert report.vision_tensors == ["model.visual.blocks.0.attn.qkv.weight"]
    assert report.checkpoint_param_count == 1000 * 256 + 512 * 256 + 256 * 256 + 256 + 192 * 64


def test_inspect_extracts_reasoning_controls(fake_checkpoint):
    root, _ = fake_checkpoint
    report = inspect_local(root)
    for marker in ("reasoning_effort", "enable_thinking", "preserve_thinking", "xhigh"):
        assert marker in report.reasoning_controls


def test_config_only_skips_tensor_reading(fake_checkpoint):
    root, _ = fake_checkpoint
    report = inspect_local(root, config_only=True)
    assert report.tensors == {}
    assert report.checkpoint_param_count is None
    assert report.spec is not None


def test_cross_check_reports_mtp_and_vision_split(fake_checkpoint):
    root, _ = fake_checkpoint
    findings = "\n".join(cross_check(inspect_local(root)))
    assert "vision tower" in findings
    assert "MTP head" in findings
    assert "analytical text-tower parameters" in findings


def test_cross_check_flags_drift_when_config_lies(fake_checkpoint):
    """If config.json disagrees with the tensors, the cross-check must say so."""
    root, _ = fake_checkpoint
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["text_config"]["num_hidden_layers"] = 40  # inflate the analytical count
    config["text_config"]["layer_types"] = None
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    findings = "\n".join(cross_check(inspect_local(root)))
    assert "MISMATCH" in findings
    assert "ACTION" in findings


def test_missing_config_is_reported_not_raised(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    report = inspect_local(empty)
    assert report.spec is None
    assert any("config.json not found" in w for w in report.warnings)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        inspect_local(tmp_path / "nope")


def test_report_serialises_to_json(fake_checkpoint):
    root, _ = fake_checkpoint
    json.dumps(inspect_local(root).to_dict())
