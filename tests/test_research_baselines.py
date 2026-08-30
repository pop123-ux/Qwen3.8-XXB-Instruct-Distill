"""The historical dense candidate is retained and is a valid control."""
from __future__ import annotations

import json

import pytest

from qwen_distill.architecture.moe_student import FROZEN_STUDENT
from qwen_distill.architecture.presets import get_spec
from qwen_distill.research.baselines import baselines, comparison, dense_h5120_l40


def test_the_old_candidate_still_exists_at_its_recorded_size():
    """It is not deleted, and it is not silently a different model than it was."""
    assert baselines()["dense_h5120_l40"].parameters == 17_763_549_760


def test_it_is_derived_from_the_teacher_not_hard_coded():
    """Deriving keeps it transfer-compatible even if the teacher preset is corrected."""
    teacher = get_spec("teacher")
    dense = dense_h5120_l40()
    assert dense.hidden_size == teacher.hidden_size
    assert dense.vocab_size == teacher.vocab_size
    assert dense.head_dim == teacher.head_dim
    assert dense.intermediate_size == teacher.intermediate_size
    assert dense.num_hidden_layers == 40 != teacher.num_hidden_layers


def test_it_shares_everything_with_the_moe_student_except_depth_and_the_ffn():
    """What makes it a control rather than merely an earlier idea."""
    dense = dense_h5120_l40()
    assert dense.hidden_size == FROZEN_STUDENT.hidden_size == 5120
    assert dense.vocab_size == FROZEN_STUDENT.vocab_size == 248_320
    assert dense.head_dim == FROZEN_STUDENT.head_dim == 256
    assert dense.num_hidden_layers != FROZEN_STUDENT.num_hidden_layers


def test_the_comparison_names_what_varies_and_what_is_held_constant():
    c = comparison()
    assert set(c["varies"]) == {"depth 40 vs 48", "dense FFN vs 8-expert top-2 MoE"}
    assert len(c["held_constant"]) >= 5
    assert c["status"].startswith("not run")


def test_the_comparison_carries_both_models_real_numbers():
    c = comparison()
    assert c["baseline"]["parameters"] == 17_763_549_760
    assert c["candidate"]["parameters"] == 13_008_505_728
    assert c["candidate"]["active_parameters_per_token"] == 9_611_119_488
    # After the expert-budget correction the sparse student is smaller than the dense
    # baseline in total as well as in active parameters, which sharpens the comparison:
    # it is no longer "more parameters, fewer active" but "fewer of both".
    assert c["candidate"]["parameters"] < c["baseline"]["parameters"]
    assert c["candidate"]["active_parameters_per_token"] < c["baseline"]["parameters"]


def test_every_baseline_says_what_it_is_a_baseline_for():
    for baseline in baselines().values():
        assert "baseline" in baseline.status
        assert len(baseline.role) > 40
        assert len(baseline.evidence) > 40


def test_it_serialises():
    json.loads(json.dumps(comparison()))
    json.loads(json.dumps({k: v.to_dict() for k, v in baselines().items()}))


def test_the_dense_baseline_is_the_one_that_fits_sixteen_gb():
    """Named in the docs as the fallback if the expert-budget decision goes the other way,
    so the claim is checked rather than asserted."""
    pytest.importorskip("transformers")
    from qwen_distill.architecture.memory import DeploymentConfig, estimate_memory
    from qwen_distill.research.memory import USABLE_GIB

    estimate = estimate_memory(dense_h5120_l40(),
                               DeploymentConfig(context_length=32_768, weight_quant="q4_k_m"))
    assert estimate.total_gib < USABLE_GIB
