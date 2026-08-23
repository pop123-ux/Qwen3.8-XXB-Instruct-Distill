"""Regression tests for the gate-activation question (Phase 1C).

This guards a failure mode that parameter-count tests cannot catch: a model with
exactly the right shapes computing the wrong function. The tests therefore exercise
*behaviour* on inputs where the candidate activations genuinely differ, not just names.

The specific history: the checkpoint declares ``output_gate_type: "swish"`` while
``transformers`` 5.15.1 contains a hard-coded ``torch.sigmoid(gate)``. An earlier phase
recorded that as a real discrepancy. It is not — the two refer to *different gates* —
and these tests pin the reasoning so the conclusion cannot quietly rot.
"""

from __future__ import annotations

import pytest
from conftest import requires_stack

from qwen_distill.teacher.runtime_compat import (
    GATE_CONFIG_KEYS,
    SILU_EQUIVALENT_NAMES,
    TRANSFORMERS_HARDCODED_GATES,
    activations_equivalent,
    check_runtime_compatibility,
)


# --- the mathematics ---------------------------------------------------
@requires_stack
def test_swish_and_silu_are_the_same_function():
    """Swish (beta=1) is x*sigmoid(x), which is exactly SiLU."""
    import torch
    from transformers.activations import ACT2FN

    x = torch.linspace(-8, 8, 257)
    assert torch.allclose(ACT2FN["silu"](x), x * torch.sigmoid(x), atol=1e-6)
    assert torch.allclose(ACT2FN["swish"](x), ACT2FN["silu"](x), atol=0)


@requires_stack
def test_silu_and_sigmoid_differ_substantially():
    """The discriminating check: if these were close, the test above would prove nothing."""
    import torch
    from transformers.activations import ACT2FN

    x = torch.linspace(-8, 8, 257)
    difference = (ACT2FN["silu"](x) - torch.sigmoid(x)).abs().max().item()
    assert difference > 1.0, difference


@requires_stack
def test_substituting_sigmoid_for_silu_changes_a_gated_norm_output():
    """Behavioural, not nominal: the same module with the other activation differs.

    This is what makes the regression test meaningful — it demonstrates the error we
    are guarding against would actually change the model's output.
    """
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated

    torch.manual_seed(0)
    norm = Qwen3_5RMSNormGated(16).eval()
    hidden = torch.randn(2, 4, 16)
    gate = torch.randn(2, 4, 16)

    with torch.no_grad():
        as_silu = norm(hidden, gate).clone()
        norm.activation = "sigmoid"          # local to this instance; nothing global
        as_sigmoid = norm(hidden, gate).clone()

    assert not torch.allclose(as_silu, as_sigmoid, atol=1e-4)


# --- equivalence helper ------------------------------------------------
@pytest.mark.parametrize(
    "declared,runtime,expected",
    [
        ("swish", "silu", True),
        ("silu", "swish", True),
        ("silu", "silu", True),
        ("SWISH", "silu", True),      # case-insensitive
        ("sigmoid", "silu", False),   # the case that must never pass
        ("sigmoid", "swish", False),
        ("gelu", "silu", False),
    ],
)
def test_activation_equivalence(declared, runtime, expected):
    assert activations_equivalent(declared, runtime) is expected


def test_only_silu_and_swish_are_treated_as_interchangeable():
    assert sorted(SILU_EQUIVALENT_NAMES) == ["silu", "swish"]
    assert "sigmoid" not in SILU_EQUIVALENT_NAMES


# --- which key governs which gate --------------------------------------
def test_output_gate_type_governs_the_deltanet_gate_not_attention():
    """The category error that produced a false discrepancy in an earlier phase."""
    assert GATE_CONFIG_KEYS["deltanet_output_gate"] == "output_gate_type"
    assert GATE_CONFIG_KEYS["attention_output_gate"] == "attn_output_gate"


def test_recorded_runtime_activations_match_the_installed_source():
    assert TRANSFORMERS_HARDCODED_GATES["deltanet_output_gate"] == "silu"
    assert TRANSFORMERS_HARDCODED_GATES["attention_output_gate"] == "sigmoid"


@requires_stack
def test_installed_source_really_hardcodes_those_activations():
    """Read the installed module and confirm the constants above are not stale.

    If a future `transformers` changes either activation, this fails rather than
    letting the recorded values drift out of sync with reality.
    """
    import inspect

    from transformers.models.qwen3_5 import modeling_qwen3_5

    source = inspect.getsource(modeling_qwen3_5)
    # Both spellings appear across releases and mean the same thing: 5.8.0-5.12.x write
    # `F.silu(gate...)` directly, 5.13+ route it through `ACT2FN[self.activation]`.
    deltanet_is_silu = (
        'self.activation = "silu"' in source or "F.silu(gate" in source
    )
    assert deltanet_is_silu, "DeltaNet gate no longer applies silu"
    assert "attn_output = attn_output * torch.sigmoid(gate)" in source, (
        "attention output gate no longer applies sigmoid"
    )


@requires_stack
def test_installed_source_still_ignores_output_gate_type():
    """If transformers starts reading the key, our reasoning must be revisited."""
    import inspect

    from transformers.models.qwen3_5 import modeling_qwen3_5

    assert "output_gate_type" not in inspect.getsource(modeling_qwen3_5)


# --- the verdict --------------------------------------------------------
def test_swish_checkpoint_is_compatible_with_a_silu_runtime():
    report = check_runtime_compatibility(
        {"text_config": {"output_gate_type": "swish", "attn_output_gate": True}}
    )
    assert report.verdict == "VERIFIED_CORRECT"
    assert report.warnings == []


def test_sigmoid_checkpoint_would_be_flagged_as_incompatible():
    """The guard that matters: a future checkpoint declaring sigmoid must NOT pass."""
    report = check_runtime_compatibility(
        {"text_config": {"output_gate_type": "sigmoid", "attn_output_gate": True}}
    )
    assert report.verdict == "IMPLEMENTATION_MISMATCH"
    assert any("DeltaNet gate mismatch" in w for w in report.warnings)
    assert any("Do NOT use this runtime" in w for w in report.warnings)


def test_disabled_attention_gate_is_flagged():
    report = check_runtime_compatibility(
        {"text_config": {"output_gate_type": "swish", "attn_output_gate": False}}
    )
    assert report.verdict == "IMPLEMENTATION_MISMATCH"
    assert any("attn_output_gate=false" in w for w in report.warnings)


def test_default_gate_type_is_silu_when_unspecified():
    report = check_runtime_compatibility({"text_config": {}})
    assert report.verdict == "VERIFIED_CORRECT"


def test_report_is_json_serialisable():
    import json

    json.dumps(check_runtime_compatibility({"text_config": {"output_gate_type": "swish"}}).to_dict())
