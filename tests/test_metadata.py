"""Tests for offline metadata ingestion and validation.

Two properties matter most here and are tested directly:

* **presence is not verification** — a field is FOUND only when its value was parsed;
* **MISSING and UNKNOWN are different** — a field the files could carry but do not is
  MISSING; a fact no metadata could ever settle (needs weights, needs a runtime run) is
  UNKNOWN. Collapsing them would overstate what has been verified.
"""

from __future__ import annotations

import json

import pytest

from qwen_distill.teacher.metadata import (
    blocking_gaps,
    load_metadata,
    summarise_counts,
    validate_metadata,
)


def field(fields, name):
    return next(f for f in fields if f.name == name)


def write(root, name, payload):
    root.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (root / name).write_text(text, encoding="utf-8")


@pytest.fixture
def minimal(tmp_path):
    """The smallest directory that supports real verification."""
    root = tmp_path / "meta"
    write(root, "config.json", {
        "model_type": "qwen3_5_text",
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 5120, "num_hidden_layers": 64, "intermediate_size": 17408,
        "vocab_size": 248320, "num_attention_heads": 24, "num_key_value_heads": 4,
        "head_dim": 256, "linear_num_key_heads": 16, "linear_num_value_heads": 48,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "tie_word_embeddings": False, "max_position_embeddings": 262144,
        "full_attention_interval": 4,
    })
    write(root, "tokenizer_config.json", {
        "tokenizer_class": "Qwen2TokenizerFast",
        "eos_token": "<|im_end|>",
        "chat_template": "{{ reasoning_effort }}<think></think>",
    })
    return root


# --- loading ----------------------------------------------------------
def test_load_reports_present_and_absent_files(minimal):
    metadata = load_metadata(minimal)
    assert metadata.file_report("config.json").present
    assert metadata.file_report("config.json").parsed
    assert not metadata.file_report("generation_config.json").present


def test_missing_optional_files_are_optional_not_missing(minimal):
    files, _ = validate_metadata(load_metadata(minimal))
    assert {f.name: f.status for f in files}["generation_config.json"] == "OPTIONAL"
    assert {f.name: f.status for f in files}["tokenizer.json"] == "OPTIONAL"


def test_missing_required_file_is_missing(tmp_path):
    root = tmp_path / "meta"
    write(root, "config.json", {"model_type": "x"})
    files, _ = validate_metadata(load_metadata(root))
    assert {f.name: f.status for f in files}["tokenizer_config.json"] == "MISSING"


def test_malformed_json_is_unknown_not_found(tmp_path):
    root = tmp_path / "meta"
    write(root, "config.json", "{ this is not json")
    metadata = load_metadata(root)
    report = metadata.file_report("config.json")
    assert report.present
    assert report.status == "UNKNOWN"
    assert "invalid JSON" in report.parse_error
    assert metadata.errors


def test_malformed_config_blocks_all_field_reads(tmp_path):
    root = tmp_path / "meta"
    write(root, "config.json", "{ broken")
    _, fields = validate_metadata(load_metadata(root))
    assert fields[0].status == "MISSING"
    assert "no field can be read" in fields[0].note


def test_non_object_json_is_rejected(tmp_path):
    root = tmp_path / "meta"
    write(root, "config.json", "[1, 2, 3]")
    assert "expected a JSON object" in load_metadata(root).file_report("config.json").parse_error


def test_missing_directory_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        load_metadata(tmp_path / "nope")


# --- field extraction --------------------------------------------------
def test_model_type_and_architectures_are_extracted(minimal):
    _, fields = validate_metadata(load_metadata(minimal))
    assert field(fields, "model_type").value == "qwen3_5_text"
    assert field(fields, "architectures").value == ["Qwen3_5ForCausalLM"]


def test_architecture_dimensions_are_read_as_values(minimal):
    """FOUND must mean the value was parsed, not that a file exists."""
    _, fields = validate_metadata(load_metadata(minimal))
    for name, expected in (
        ("hidden_size", 5120), ("num_hidden_layers", 64), ("intermediate_size", 17408),
        ("vocab_size", 248320), ("num_attention_heads", 24), ("num_key_value_heads", 4),
        ("head_dim", 256), ("linear_num_value_heads", 48), ("linear_key_head_dim", 128),
    ):
        report = field(fields, name)
        assert report.status == "FOUND", name
        assert report.value == expected, name


def test_nested_text_config_is_searched(tmp_path):
    """Multimodal checkpoints nest the language model under text_config."""
    root = tmp_path / "meta"
    write(root, "config.json", {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"hidden_size": 5120, "num_hidden_layers": 64},
        "vision_config": {"depth": 27, "hidden_size": 1152},
    })
    write(root, "tokenizer_config.json", {"tokenizer_class": "X"})
    _, fields = validate_metadata(load_metadata(root))
    assert field(fields, "hidden_size").value == 5120
    assert "text_config" in field(fields, "hidden_size").source
    assert field(fields, "vision_config").status == "FOUND"


def test_text_only_config_reports_vision_as_optional(minimal):
    _, fields = validate_metadata(load_metadata(minimal))
    assert field(fields, "vision_config").status == "OPTIONAL"


def test_layer_layout_derived_from_interval(minimal):
    report = field(validate_metadata(load_metadata(minimal))[1], "layer_layout")
    assert report.status == "FOUND"
    assert "derived from interval 4" in report.value


def test_layer_layout_explicit_when_given(tmp_path):
    root = tmp_path / "meta"
    write(root, "config.json", {
        "model_type": "qwen3_5_text", "architectures": ["Qwen3_5ForCausalLM"],
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    })
    write(root, "tokenizer_config.json", {"tokenizer_class": "X"})
    report = field(validate_metadata(load_metadata(root))[1], "layer_layout")
    assert report.status == "FOUND"
    assert "explicit (4 layers)" in report.value


def test_layer_layout_unknown_when_neither_present(tmp_path):
    root = tmp_path / "meta"
    write(root, "config.json", {"model_type": "qwen3_5_text", "architectures": []})
    write(root, "tokenizer_config.json", {"tokenizer_class": "X"})
    assert field(validate_metadata(load_metadata(root))[1], "layer_layout").status == "UNKNOWN"


def test_auto_map_is_surfaced(tmp_path):
    root = tmp_path / "meta"
    write(root, "config.json", {
        "model_type": "custom", "architectures": ["X"],
        "auto_map": {"AutoModelForCausalLM": "modeling_x.XForCausalLM"},
    })
    write(root, "tokenizer_config.json", {"tokenizer_class": "X"})
    assert field(validate_metadata(load_metadata(root))[1], "auto_map").status == "FOUND"


# --- tokenizer and template -------------------------------------------
def test_tokenizer_class_and_tokens_extracted(minimal):
    _, fields = validate_metadata(load_metadata(minimal))
    assert field(fields, "tokenizer_class").value == "Qwen2TokenizerFast"
    assert field(fields, "eos_token").value == "<|im_end|>"


def test_added_token_dicts_are_unwrapped(tmp_path):
    root = tmp_path / "meta"
    write(root, "config.json", {"model_type": "x", "architectures": []})
    write(root, "tokenizer_config.json",
          {"tokenizer_class": "X", "eos_token": {"content": "<|end|>", "lstrip": False}})
    assert field(validate_metadata(load_metadata(root))[1], "eos_token").value == "<|end|>"


def test_chat_template_found_in_tokenizer_config(minimal):
    metadata = load_metadata(minimal)
    assert metadata.chat_template_source == "tokenizer_config.json"
    assert field(validate_metadata(metadata)[1], "chat_template").status == "FOUND"


def test_standalone_jinja_template_takes_precedence(tmp_path):
    root = tmp_path / "meta"
    write(root, "config.json", {"model_type": "x", "architectures": []})
    write(root, "tokenizer_config.json", {"tokenizer_class": "X", "chat_template": "FROM_CONFIG"})
    write(root, "chat_template.jinja", "FROM_JINJA")
    metadata = load_metadata(root)
    assert metadata.chat_template == "FROM_JINJA"
    assert metadata.chat_template_source == "chat_template.jinja"


def test_missing_template_makes_reasoning_controls_unknown(tmp_path):
    root = tmp_path / "meta"
    write(root, "config.json", {"model_type": "x", "architectures": []})
    write(root, "tokenizer_config.json", {"tokenizer_class": "X"})
    _, fields = validate_metadata(load_metadata(root))
    assert field(fields, "chat_template").status == "MISSING"
    assert field(fields, "reasoning_controls").status == "UNKNOWN"


def test_reasoning_markers_extracted_from_template(minimal):
    report = field(validate_metadata(load_metadata(minimal))[1], "reasoning_controls")
    assert report.status == "FOUND"
    assert "reasoning_effort" in report.value


def test_tokenizer_vocab_size_counted_from_tokenizer_json(tmp_path):
    root = tmp_path / "meta"
    write(root, "config.json", {"model_type": "x", "architectures": []})
    write(root, "tokenizer_config.json", {"tokenizer_class": "X"})
    write(root, "tokenizer.json",
          {"model": {"vocab": {"a": 0, "b": 1, "c": 2}}, "added_tokens": [{"id": 3}]})
    assert field(validate_metadata(load_metadata(root))[1], "tokenizer_vocab_size").value == 4


# --- the honesty guarantees -------------------------------------------
def test_facts_needing_weights_are_unknown_not_found(minimal):
    """Metadata can never establish these, however complete it is."""
    _, fields = validate_metadata(load_metadata(minimal))
    for name in ("state_dict_parameter_count", "runtime_generation"):
        assert field(fields, name).status == "UNKNOWN", name


def test_medium_noop_stays_unknown_from_metadata_alone(minimal):
    report = field(validate_metadata(load_metadata(minimal))[1], "medium_reasoning_is_noop")
    assert report.status == "UNKNOWN"
    assert "runtime experiment" in report.note


def test_licence_missing_unless_file_supplied(minimal):
    report = field(validate_metadata(load_metadata(minimal))[1], "license")
    assert report.status == "MISSING"
    assert "stays UNKNOWN" in report.note


def test_licence_found_when_supplied(tmp_path):
    root = tmp_path / "meta"
    write(root, "config.json", {"model_type": "x", "architectures": []})
    write(root, "tokenizer_config.json", {"tokenizer_class": "X"})
    write(root, "LICENSE", "                    Apache License\n              Version 2.0")
    report = field(validate_metadata(load_metadata(root))[1], "license")
    assert report.status == "FOUND"
    assert "Apache-2.0" in report.value


def test_unrecognised_licence_is_not_guessed(tmp_path):
    root = tmp_path / "meta"
    write(root, "config.json", {"model_type": "x", "architectures": []})
    write(root, "tokenizer_config.json", {"tokenizer_class": "X"})
    write(root, "LICENSE", "Some bespoke terms nobody has seen before.")
    assert "unrecognised" in field(validate_metadata(load_metadata(root))[1], "license").value


def test_unknown_entries_are_not_counted_as_blocking_gaps(minimal):
    files, fields = validate_metadata(load_metadata(minimal))
    gaps = blocking_gaps(files, fields)
    assert not any("state_dict" in gap for gap in gaps)
    assert not any("runtime_generation" in gap for gap in gaps)


def test_summarise_counts_covers_every_item(minimal):
    files, fields = validate_metadata(load_metadata(minimal))
    counts = summarise_counts(files, fields)
    assert sum(counts.values()) == len(files) + len(fields)


# --- config vs the analytical model ------------------------------------
def test_supplied_config_can_drive_the_analytical_parameter_model(minimal):
    """A supplied config must flow straight into the parameter model.

    This is the join between the two halves of Phase 1: metadata says what the
    architecture *is*, and the analytical model says what it *costs*.
    """
    from qwen_distill.architecture.params import count_parameters
    from qwen_distill.architecture.spec import HybridArchSpec

    metadata = load_metadata(minimal)
    spec = HybridArchSpec.from_hf_config(metadata.config, name="from-metadata")
    assert spec.hidden_size == 5120
    assert spec.num_hidden_layers == 64
    assert spec.num_linear_attention_layers == 48
    assert spec.num_full_attention_layers == 16
    # These are the published values, so this must reproduce the published count.
    assert count_parameters(spec).total == 26_895_998_464


def test_a_config_that_differs_produces_a_different_count(tmp_path):
    """Guard against the spec silently ignoring supplied values."""
    from qwen_distill.architecture.params import count_parameters
    from qwen_distill.architecture.spec import HybridArchSpec

    root = tmp_path / "meta"
    write(root, "config.json", {
        "model_type": "qwen3_5_text", "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 4096, "num_hidden_layers": 48, "intermediate_size": 12288,
        "vocab_size": 248320, "num_attention_heads": 20, "num_key_value_heads": 4,
        "head_dim": 256, "linear_num_key_heads": 16, "linear_num_value_heads": 48,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "tie_word_embeddings": True, "max_position_embeddings": 262144,
        "full_attention_interval": 4,
    })
    write(root, "tokenizer_config.json", {"tokenizer_class": "X"})
    spec = HybridArchSpec.from_hf_config(load_metadata(root).config, name="other")
    assert spec.tie_word_embeddings is True
    assert count_parameters(spec).total != 26_895_998_464
