"""Tests pinned against the **actual supplied Qwen3.8-27B metadata**.

These are the strongest tests in the repository: they assert facts read from the real
checkpoint's own files, not from the reference implementation and not from a preset.

They skip cleanly when `vendor/qwen38-metadata/` is absent, so a contributor without
the metadata still gets a green suite — but where the metadata is present, a change to
it (or a regression in our parsing) fails here first.

Template hashes are pinned deliberately. A chat-template edit can move benchmark scores
substantially, so it is an experimental dependency and is treated like one: if a hash
changes, every measurement taken under the old template is suspect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import requires_stack

from qwen_distill.architecture.params import count_parameters, mtp_params
from qwen_distill.architecture.spec import HybridArchSpec
from qwen_distill.teacher.metadata import (
    build_verified_spec,
    hash_metadata_files,
    implementation_disagreements,
    load_metadata,
    validate_metadata,
)

METADATA_DIR = Path(__file__).resolve().parent.parent / "vendor" / "qwen38-metadata"

pytestmark = pytest.mark.skipif(
    not (METADATA_DIR / "config.json").is_file(),
    reason="vendor/qwen38-metadata/config.json not supplied (see vendor/README.md)",
)

#: Rendered-prompt hashes for the user message "What is 15 * 7?" with
#: add_generation_prompt=True. Captured from the supplied chat_template.jinja.
EXPECTED_PROMPT_SHA256 = {
    "default": "7f1de0c2b7fda736d90625a80eb094ccc0d1ee28a6b5d9004f611b62e5b11a35",
    "thinking_disabled": "8475fd3ecb780f1b067593215c95febad65f330f8ca5d6d34045f649cb6d7fd4",
    "thinking_enabled": "7f1de0c2b7fda736d90625a80eb094ccc0d1ee28a6b5d9004f611b62e5b11a35",
    "low": "51f41ace41f5cea2034d5994fc0d7a3383a22c1f71c60ef9adf3a448634ccd0e",
    "medium": "20ba983e045cdb9a66090a79cccd56941461be2610d5f334da256a388fe83abf",
    "xhigh": "7f1de0c2b7fda736d90625a80eb094ccc0d1ee28a6b5d9004f611b62e5b11a35",
}


@pytest.fixture(scope="module")
def metadata():
    return load_metadata(METADATA_DIR)


@pytest.fixture(scope="module")
def spec(metadata):
    return HybridArchSpec.from_hf_config(metadata.config, name="qwen3.8-27b")


# --- identity ----------------------------------------------------------
def test_model_type_is_qwen3_5(metadata):
    """The checkpoint reuses the qwen3_5 family rather than declaring a new type."""
    assert metadata.config["model_type"] == "qwen3_5"
    assert metadata.text_config["model_type"] == "qwen3_5_text"


def test_declared_architecture_is_multimodal(metadata):
    assert metadata.config["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert metadata.vision_config is not None


def test_no_auto_map_so_no_remote_code_needed(metadata):
    from qwen_distill.teacher.loader import detect_remote_code

    needed, evidence = detect_remote_code(str(METADATA_DIR), metadata.config)
    assert not needed, evidence


# --- architecture ------------------------------------------------------
def test_architecture_dimensions(spec):
    assert spec.hidden_size == 5120
    assert spec.num_hidden_layers == 64
    assert spec.intermediate_size == 17408
    assert spec.vocab_size == 248320
    assert spec.num_attention_heads == 24
    assert spec.num_key_value_heads == 4
    assert spec.head_dim == 256
    assert spec.linear_num_value_heads == 48
    assert spec.linear_num_key_heads == 16
    assert spec.linear_key_head_dim == 128
    assert spec.linear_value_head_dim == 128
    assert spec.max_position_embeddings == 262144
    assert spec.tie_word_embeddings is False


def test_layer_layout_is_explicit_48_linear_16_full(metadata, spec):
    """The checkpoint lists layer_types explicitly rather than relying on the interval."""
    assert "layer_types" in metadata.text_config
    assert len(metadata.text_config["layer_types"]) == 64
    assert spec.num_linear_attention_layers == 48
    assert spec.num_full_attention_layers == 16
    assert spec.resolved_layer_types()[3] == "full_attention"
    assert spec.resolved_layer_types()[-1] == "full_attention"


def test_attention_output_gate_is_declared(spec):
    assert spec.attn_output_gate is True


def test_rope_dimension_is_64(spec):
    assert spec.partial_rotary_factor == 0.25
    assert spec.rope_dim == 64


def test_no_rope_scaling_configured(metadata):
    """No YaRN by default: the 262144 context is native, not extended."""
    assert "rope_scaling" not in metadata.text_config


def test_mtp_is_declared_with_one_layer(metadata):
    assert metadata.text_config["mtp_num_hidden_layers"] == 1
    assert metadata.text_config["mtp_use_dedicated_embeddings"] is False


# --- parameter accounting ----------------------------------------------
def test_actual_config_reproduces_the_expected_parameter_count(spec):
    """The headline number, computed from the real config rather than a preset."""
    assert count_parameters(spec).total == 26_895_998_464


def test_component_breakdown_from_actual_config(spec):
    breakdown = count_parameters(spec)
    assert breakdown.embedding == 1_271_398_400
    assert breakdown.lm_head == 1_271_398_400
    assert breakdown.mlp == 17_112_760_320
    assert breakdown.full_attention == 1_677_729_792
    assert breakdown.linear_attention == 5_562_051_072
    assert breakdown.final_norm == 5_120
    assert breakdown.layer_norms == 655_360


def test_mtp_head_cost_from_declared_layer_count(metadata, spec):
    layers = metadata.text_config["mtp_num_hidden_layers"]
    assert mtp_params(spec, layers) == 424_699_392


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 4096),
        ("num_hidden_layers", 48),
        ("intermediate_size", 12288),
        ("vocab_size", 100000),
        ("num_attention_heads", 20),
        ("num_key_value_heads", 2),
        ("head_dim", 128),
        ("linear_num_value_heads", 32),
        ("linear_key_head_dim", 64),
        ("tie_word_embeddings", True),
        ("attn_output_gate", False),
    ],
)
def test_every_architecture_field_changes_the_parameter_count(spec, field, value):
    """No silent hard-coding: altering any dimension must move the estimate."""
    import dataclasses

    baseline = count_parameters(spec).total
    overrides = {field: value}
    if field == "num_hidden_layers":
        # The checkpoint lists layer_types explicitly for all 64 layers; changing the
        # depth without clearing it is a genuine inconsistency the spec rejects, so
        # re-derive the layout from the interval instead.
        overrides["layer_types"] = None
    changed = dataclasses.replace(spec, **overrides)
    assert count_parameters(changed).total != baseline, field


def test_spec_rejects_a_layer_count_inconsistent_with_layer_types(spec):
    """Changing depth while keeping a 64-entry layer_types list must not pass silently."""
    import dataclasses

    with pytest.raises(ValueError, match="layer_types has 64 entries"):
        dataclasses.replace(spec, num_hidden_layers=48)


def test_changing_layer_layout_changes_the_count(spec):
    import dataclasses

    baseline = count_parameters(spec).total
    changed = dataclasses.replace(spec, layer_types=None, full_attention_interval=2)
    assert changed.num_full_attention_layers == 32
    assert count_parameters(changed).total != baseline


# --- tokenizer, template, licence --------------------------------------
def test_tokenizer_class(metadata):
    assert metadata.tokenizer_config["tokenizer_class"] == "Qwen2Tokenizer"


def test_chat_template_comes_from_its_own_file(metadata):
    assert metadata.chat_template_source == "chat_template.jinja"
    assert metadata.chat_template


def test_licence_is_apache_2(metadata):
    _, fields = validate_metadata(metadata)
    licence = next(f for f in fields if f.name == "license")
    assert licence.status == "FOUND"
    assert "Apache-2.0" in licence.value


@requires_stack
@pytest.mark.parametrize("label,expected", sorted(EXPECTED_PROMPT_SHA256.items()))
def test_rendered_prompt_hashes_are_stable(label, expected):
    """Pin the template. A changed hash invalidates measurements taken under the old one."""
    import hashlib

    from transformers import AutoTokenizer

    from qwen_distill.utils.offline import offline_mode

    kwargs = {
        "default": {},
        "thinking_disabled": {"enable_thinking": False},
        "thinking_enabled": {"enable_thinking": True},
        "low": {"reasoning_effort": "low"},
        "medium": {"reasoning_effort": "medium"},
        "xhigh": {"reasoning_effort": "xhigh"},
    }[label]
    with offline_mode():
        tokenizer = AutoTokenizer.from_pretrained(str(METADATA_DIR), local_files_only=True)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 15 * 7?"}],
            tokenize=False, add_generation_prompt=True, **kwargs,
        )
    assert hashlib.sha256(rendered.encode()).hexdigest() == expected


@requires_stack
def test_default_reasoning_effort_is_xhigh():
    """Established directly from the template: the default branch is xhigh."""
    assert EXPECTED_PROMPT_SHA256["default"] == EXPECTED_PROMPT_SHA256["xhigh"]
    assert "default('xhigh')" in (METADATA_DIR / "chat_template.jinja").read_text(
        encoding="utf-8"
    )


@requires_stack
def test_medium_is_not_a_no_op():
    """Refutes the earlier secondary-source hypothesis.

    `medium` renders a *distinct* prompt: it omits the reasoning instruction that the
    default (xhigh) injects, so selecting it is a large prompt change, not a no-op.
    """
    assert EXPECTED_PROMPT_SHA256["medium"] != EXPECTED_PROMPT_SHA256["default"]
    assert EXPECTED_PROMPT_SHA256["medium"] != EXPECTED_PROMPT_SHA256["low"]


@requires_stack
def test_only_three_reasoning_efforts_are_accepted():
    """`high` is rejected by the template; supported values are xhigh, medium, low."""
    from transformers import AutoTokenizer

    from qwen_distill.utils.offline import offline_mode

    with offline_mode():
        tokenizer = AutoTokenizer.from_pretrained(str(METADATA_DIR), local_files_only=True)
        messages = [{"role": "user", "content": "hi"}]
        for effort in ("xhigh", "medium", "low"):
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, reasoning_effort=effort
            )
        for effort in ("high", "minimal", "none"):
            with pytest.raises(Exception, match="Unexpected reasoning effort"):
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    reasoning_effort=effort,
                )


# --- provenance --------------------------------------------------------
def test_hashes_cover_every_supplied_file(metadata):
    digests = hash_metadata_files(METADATA_DIR)
    for name in ("config.json", "tokenizer_config.json", "chat_template.jinja"):
        assert name in digests
        assert len(digests[name]) == 64


def test_implementation_disagreements_are_reported(metadata):
    """Config keys transformers 5.15.1 does not read must be surfaced, not hidden."""
    findings = " ".join(implementation_disagreements(metadata))
    assert "attn_output_gate" in findings
    assert "mamba_ssm_dtype" in findings


def test_verified_spec_is_generated_and_self_consistent(metadata):
    built = build_verified_spec(metadata, source="Qwen/Qwen3.8-27B")
    assert built["identity"]["model_type"] == "qwen3_5"
    assert built["parameters"]["total"] == 26_895_998_464
    assert built["provenance"]["file_sha256"]["config.json"]
    assert built["provenance"]["revision"] is None
    assert "not fully reproducible" in built["provenance"]["revision_note"]
    assert built["not_verified"]
    json.dumps(built)


def test_committed_verified_spec_matches_the_metadata(metadata):
    """The committed canonical spec must not drift from the files it came from."""
    committed_path = METADATA_DIR.parent.parent / "configs" / "teacher" / "qwen3_8_27b.verified.json"
    if not committed_path.is_file():
        pytest.skip("verified spec not generated yet")
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    current = hash_metadata_files(METADATA_DIR)
    assert committed["provenance"]["file_sha256"] == current, (
        "the supplied metadata changed since the verified spec was generated; "
        "regenerate it with scripts/validate_teacher_metadata.py --save-verified"
    )
    assert committed["parameters"]["total"] == count_parameters(
        HybridArchSpec.from_hf_config(metadata.config, name="x")
    ).total
