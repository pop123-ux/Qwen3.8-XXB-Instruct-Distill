"""Presets must describe the experiments they claim to, and the analytical parameter
model must agree with the model `transformers` actually builds.

Both guards exist because both have already failed. The first draft of
``presets.prototype`` hard-coded a DeltaNet head dim of 64 where the real config uses 32,
and reported 4,574,308 parameters against the run record's 4,029,700 — a preset that does
not reproduce its experiment is not a record of it, it is a second, wrong description.
"""

from __future__ import annotations

import pytest
from conftest import requires_stack

from qwen_distill.architecture.params import count_parameters
from qwen_distill.architecture.presets import (
    MEASURED,
    TEACHER_RATIOS,
    architecture_fields,
    derive,
    diff,
    get_preset,
    get_spec,
    preset_names,
)
from qwen_distill.training.config import ExperimentConfig

#: Parameter counts, pinned. These are what the experiments actually built.
EXPECTED_PARAMETERS = {
    "prototype": 4_029_700,
    "level2": 94_476_448,
    "level2r": 94_476_448,
    "level3": 236_237_488,
    "teacher": 26_895_998_464,
}


@pytest.mark.parametrize("name", sorted(EXPECTED_PARAMETERS))
def test_preset_parameter_counts_are_pinned(name):
    assert get_preset(name).parameters == EXPECTED_PARAMETERS[name]


@pytest.mark.parametrize("name", ["prototype", "level2", "level2r", "level3"])
def test_preset_matches_its_config_file_field_by_field(name):
    """The drift guard. A preset is a copy of a config, and a copy that diverges is
    worse than no copy — it gives two answers to "what did that experiment run"."""
    preset = get_preset(name)
    assert preset.config, f"{name} must name the config it mirrors"
    spec_from_config = ExperimentConfig.load(preset.config).model.resolve_spec()

    differences = diff(spec_from_config, preset.spec)
    assert not differences, (
        f"preset {name!r} has drifted from {preset.config}: {differences}"
    )
    assert count_parameters(spec_from_config).total == preset.parameters


def test_every_measured_preset_points_at_a_real_config():
    from pathlib import Path

    for name in preset_names():
        preset = get_preset(name)
        if preset.kind != MEASURED:
            continue
        assert Path(preset.config).is_file(), f"{name}: {preset.config} is missing"


def test_the_student_family_keeps_the_teachers_hybrid_ratio():
    """3:1 DeltaNet to full attention, as the teacher's 48/16. A variant that abandons
    it is a different architecture family and must say so rather than inherit the name."""
    for name in ("prototype", "level2", "level2r", "level3"):
        fields = architecture_fields(get_spec(name))
        assert fields["deltanet_to_attention"] == TEACHER_RATIOS["deltanet_to_attention"]
        assert fields["full_attention_interval"] == TEACHER_RATIOS["full_attention_interval"]


def test_byte_level_students_share_one_vocabulary():
    """Bits-per-byte is only comparable within one tokenisation. The prototype is the
    exception and is documented as such — its 4096-vocab loss is not on this scale."""
    for name in ("level2", "level2r", "level3"):
        assert get_spec(name).vocab_size == 256
    assert get_spec("prototype").vocab_size == 4096
    assert "NOT byte-level" in get_preset("prototype").summary


# ----------------------------------------------------------------------------------
# derivation
# ----------------------------------------------------------------------------------


def test_derive_states_the_diff_against_its_parent():
    wider = derive("level2r", name="wider", hidden_size=1024)
    assert diff(get_spec("level2r"), wider) == {"hidden_size": (640, 1024)}


def test_derive_does_not_mutate_the_registry():
    before = get_preset("level2r").parameters
    derive("level2r", name="x", hidden_size=2048)
    assert get_preset("level2r").parameters == before
    assert get_spec("level2r").hidden_size == 640


def test_derive_rejects_an_unknown_field():
    with pytest.raises(ValueError, match="unknown architecture field"):
        derive("level2r", name="x", widht=1024)


def test_derive_refuses_an_invalid_gqa_configuration():
    """Nothing is silently adjusted. 17 attention heads has no divisor for 2 kv heads,
    and a spec that trains but is not what was asked for is worse than a raised error."""
    with pytest.raises(ValueError):
        derive("level2r", name="x", hidden_size=1088, num_attention_heads=17,
               num_key_value_heads=2)


def test_derive_recomputes_the_layout_for_a_new_depth():
    deeper = derive("level2r", name="deeper", num_hidden_layers=24)
    assert deeper.num_linear_attention_layers == 18
    assert deeper.num_full_attention_layers == 6


def test_the_registry_holds_no_unrun_architecture():
    """Future architectures are derived, not registered. A preset named `level4` would
    pre-empt the decision Level 3's result is supposed to inform."""
    for name in preset_names():
        preset = get_preset(name)
        assert preset.kind in ("measured", "reference")
        assert "future" not in name and "proposed" not in name


def test_unknown_preset_names_the_known_ones():
    with pytest.raises(KeyError, match="known:"):
        get_preset("level4")


# ----------------------------------------------------------------------------------
# analytical vs actually constructed
# ----------------------------------------------------------------------------------


@requires_stack
@pytest.mark.parametrize("name", ["prototype", "level2r", "level3"])
def test_analytical_and_constructed_parameter_counts_agree(name):
    """The formulas in `architecture.params` against the model `transformers` builds.

    On the meta device, so shapes are allocated and storage is not. If these ever
    diverge, every parameter count, memory estimate and feasibility verdict in the
    repository is wrong, and nothing downstream would notice on its own.
    """
    from qwen_distill.teacher.validate import validate_parameters

    result = validate_parameters(get_spec(name))
    if result.error and "not installed" in result.error:
        pytest.skip(result.error)
    assert result.error is None, result.error
    for component, comparison in result.comparisons.items():
        assert comparison["delta"] == 0, (
            f"{name}: {component} analytical {comparison['analytical']:,} != "
            f"constructed {comparison['measured']:,}"
        )
    assert result.details["unmatched_parameters"] == 0
    assert result.passed


@requires_stack
def test_constructed_count_matches_the_pinned_value():
    """Closes the loop: config -> spec -> real model -> the number in EXPECTED."""
    import torch
    from transformers import AutoModelForCausalLM

    from qwen_distill.teacher.validate import _build_text_config

    spec = get_spec("level3")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(_build_text_config(spec))
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS["level3"]
