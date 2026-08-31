"""The ablation matrix: a controlled design, with a falsifier on every arm."""
from __future__ import annotations

import json

import pytest

from qwen_distill.distillation.behavioral import (
    HIDDEN_DELTA,
    HIDDEN_POINTWISE,
    LOSS_TERMS,
    ROUTER_BALANCE,
)
from qwen_distill.research.ablations import ARMS, FAMILIES, arms, control, matrix, save_matrix


def test_the_matrix_is_exactly_a1_to_a4_and_b1_to_b4():
    assert sorted(ARMS) == ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
    assert len(arms("layer_matching")) == 4
    assert len(arms("context_specialisation")) == 4


def test_family_a_is_a_true_two_by_two_factorial():
    """Each cell is one combination of the two switches, and all four combinations appear.
    A ladder could not separate 'delta helps' from 'more supervision helps'."""
    cells = {
        arm: (HIDDEN_POINTWISE in ARMS[arm].loss_weights, HIDDEN_DELTA in ARMS[arm].loss_weights)
        for arm in ("A1", "A2", "A3", "A4")
    }
    assert cells == {"A1": (True, False), "A2": (False, False),
                     "A3": (False, True), "A4": (True, True)}
    assert len(set(cells.values())) == 4


def test_a2_and_a3_carry_the_same_number_of_terms():
    """A3 beating A2 must be attributable to the kind of supervision, not its quantity."""
    assert len(ARMS["A3"].loss_weights) == len(ARMS["A2"].loss_weights) + 1
    assert len(ARMS["A1"].loss_weights) == len(ARMS["A3"].loss_weights)


def test_router_balance_is_on_in_every_layer_matching_arm():
    """Without it an arm measures expert collapse rather than what it meant to measure."""
    for arm in arms("layer_matching"):
        assert arm.loss_weights[ROUTER_BALANCE] > 0


def test_every_arm_declares_a_falsifier_that_names_an_observation():
    """An arm whose claim cannot be refuted is not a hypothesis."""
    for arm in ARMS.values():
        assert len(arm.falsified_if) > 60
        assert len(arm.prediction) > 40
        assert len(arm.question) > 30


def test_the_control_arms_are_the_conventional_choices():
    assert control("layer_matching").arm == "A1"
    assert control("context_specialisation").arm == "B1"
    assert sum(1 for a in arms("layer_matching") if a.is_control) == 1


def test_every_arm_builds_a_valid_loss_configuration():
    for arm in ARMS.values():
        config = arm.loss_config()
        assert config.active
        for name in config.active:
            assert LOSS_TERMS[name].available


def test_only_a4_combines_both_hidden_terms_and_it_says_so():
    """The guard against accidentally landing in A4 must stay on for the other arms."""
    assert ARMS["A4"].combined_hidden is True
    assert ARMS["A4"].loss_config().allow_combined_hidden is True
    for arm in ("A1", "A2", "A3"):
        assert ARMS[arm].combined_hidden is False


def test_family_b_varies_data_and_not_loss():
    for arm in arms("context_specialisation"):
        assert not arm.loss_weights, f"{arm.arm} changes the loss, confounding the comparison"
        assert arm.curriculum().arm == arm.arm


def test_family_a_arms_share_one_curriculum():
    """Family A must not also vary context, or its results are unattributable."""
    used = {a.curriculum().arm for a in arms("layer_matching")}
    assert len(used) == 1


def test_matrix_lists_its_comparisons_and_admits_what_it_does_not_control():
    m = matrix()
    assert set(m["families"]) == set(FAMILIES)
    assert len(m["comparisons"]) >= 5
    for comparison in m["comparisons"]:
        assert comparison["baseline"] in ARMS and comparison["candidate"] in ARMS
    assert m["confounds_controlled"]
    assert any("variance" in note for note in m["not_controlled"]), (
        "seed variance is unmeasured and the matrix must say so"
    )


def test_matrix_serialises_and_saves(tmp_path):
    path = save_matrix(tmp_path / "matrix.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["families"]["layer_matching"]["control"] == "A1"


def test_unknown_family_is_refused():
    with pytest.raises(ValueError, match="unknown family"):
        arms("vibes")
