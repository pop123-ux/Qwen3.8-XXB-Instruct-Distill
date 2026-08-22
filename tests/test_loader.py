"""Tests for checkpoint loader verification."""

from __future__ import annotations

import json

from conftest import requires_stack

from qwen_distill.teacher.loader import collect_versions, detect_remote_code, verify_loader


def test_collect_versions_reports_python_and_missing_packages():
    versions = collect_versions()
    assert "python" in versions
    assert all(isinstance(v, str) for v in versions.values())


def test_detect_remote_code_finds_auto_map(tmp_path):
    needed, evidence = detect_remote_code(
        str(tmp_path), {"auto_map": {"AutoModelForCausalLM": "modeling_x.XForCausalLM"}}
    )
    assert needed
    assert "auto_map" in evidence[0]


def test_detect_remote_code_finds_nested_auto_map(tmp_path):
    needed, evidence = detect_remote_code(
        str(tmp_path), {"text_config": {"auto_map": {"AutoConfig": "configuration_x.XConfig"}}}
    )
    assert needed
    assert "text_config" in evidence[0]


def test_detect_remote_code_finds_bundled_modules(tmp_path):
    (tmp_path / "modeling_custom.py").write_text("# custom", encoding="utf-8")
    needed, evidence = detect_remote_code(str(tmp_path), {})
    assert needed
    assert "modeling_custom.py" in evidence[0]


def test_detect_remote_code_clean_checkpoint(tmp_path):
    needed, evidence = detect_remote_code(str(tmp_path), {"model_type": "qwen3_5_text"})
    assert not needed
    assert evidence == []


@requires_stack
def test_verify_loader_resolves_native_class(tiny_checkpoint):
    report = verify_loader(str(tiny_checkpoint))
    assert report.errors == []
    assert report.model_type == "qwen3_5_text"
    assert report.resolved_model_class == "Qwen3_5ForCausalLM"
    assert report.resolved_model_module.startswith("transformers.")
    assert report.uses_native_transformers
    assert not report.requires_trust_remote_code
    assert report.verdict == "LOADS_NATIVELY"


@requires_stack
def test_verify_loader_reports_declared_architectures(tiny_checkpoint):
    assert verify_loader(str(tiny_checkpoint)).declared_architectures == ["Qwen3_5ForCausalLM"]


@requires_stack
def test_verify_loader_report_is_json_serialisable(tiny_checkpoint):
    json.dumps(verify_loader(str(tiny_checkpoint)).to_dict())


def test_verify_loader_records_failure_rather_than_raising(tmp_path):
    report = verify_loader(str(tmp_path / "does-not-exist"))
    assert report.errors
    assert report.verdict in ("FAILED", "UNRESOLVED")
