"""Behavioural distillation: matching what a block computes rather than where it sits.

The tests that matter most are the ones separating the two objectives. If ``pointwise`` and
``delta`` cannot be shown to differ, the paper's central claim is untestable, so
:func:`test_delta_and_pointwise_are_different_objectives` and the span-coverage tests are
the load-bearing ones here.
"""
from __future__ import annotations

import pytest

from qwen_distill.architecture.moe_init import map_layers
from qwen_distill.architecture.moe_student import FROZEN_STUDENT, TEACHER_LAYERS
from qwen_distill.distillation.behavioral import (
    ATTENTION,
    CE,
    DELTANET_STATE,
    HIDDEN_DELTA,
    HIDDEN_POINTWISE,
    LOGIT_KD,
    LOSS_TERMS,
    MTP,
    ROUTER_BALANCE,
    CompositeLossConfig,
    ObjectiveUnavailable,
    attention_behavior_loss,
    behavioral_loss,
    describe_loss_terms,
    layer_spans,
)

torch = pytest.importorskip("torch")

MAPPING = map_layers(FROZEN_STUDENT, teacher_layers=TEACHER_LAYERS).mapping


# ---------------------------------------------------------------------------
# spans: who is responsible for the removed layers
# ---------------------------------------------------------------------------
def test_spans_tile_the_teacher_completely():
    """Every teacher layer is charged to exactly one student layer. A tiling with a hole
    would silently drop the work of the layers that were removed, which is the failure the
    whole approach exists to fix."""
    spans = layer_spans(MAPPING, TEACHER_LAYERS)
    assert len(spans) == 48
    covered = sorted(
        layer for start, end in spans.values() for layer in range(start, end)
    )
    assert covered == list(range(TEACHER_LAYERS))


def test_removed_layers_are_absorbed_by_a_neighbour():
    """16 teacher layers have no student anchor; the spans that contain them must be wider
    than one, and that width is the extra work being assigned."""
    spans = layer_spans(MAPPING, TEACHER_LAYERS)
    widths = [end - start for start, end in spans.values()]
    assert min(widths) == 1
    assert max(widths) > 1
    assert sum(w - 1 for w in widths) == TEACHER_LAYERS - 48 == 16


def test_spans_reject_a_reordering_mapping():
    with pytest.raises(ValueError, match="reorders"):
        layer_spans({0: 5, 1: 2}, 8)


def test_spans_reject_an_empty_mapping():
    with pytest.raises(ValueError, match="empty mapping"):
        layer_spans({}, 8)


# ---------------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------------
def _streams(n_student=48, n_teacher=TEACHER_LAYERS, batch=2, tokens=6, width=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        [torch.randn(batch, tokens, width, generator=g) for _ in range(n_student + 1)],
        [torch.randn(batch, tokens, width, generator=g) for _ in range(n_teacher + 1)],
    )


def test_a_student_that_matches_the_teacher_exactly_scores_zero():
    student, teacher = _streams()
    identity = {i: i for i in range(48)}
    out = behavioral_loss(student, student + teacher[49:], identity,
                          mode="pointwise", teacher_layers=TEACHER_LAYERS)
    assert float(out.total) == pytest.approx(0.0, abs=1e-6)


def test_delta_matching_scores_zero_when_the_contributions_agree():
    """Constructed so the student's per-layer contribution equals the teacher span's total.
    If this is not zero the telescoping identity is implemented wrongly."""
    g = torch.Generator().manual_seed(1)
    width, batch, tokens = 16, 1, 4
    spans = layer_spans(MAPPING, TEACHER_LAYERS)
    teacher = [torch.zeros(batch, tokens, width)]
    for _ in range(TEACHER_LAYERS):
        teacher.append(teacher[-1] + torch.randn(batch, tokens, width, generator=g))
    student = [teacher[0].clone()]
    for s in sorted(MAPPING):
        start, end = spans[s]
        student.append(student[-1] + (teacher[end] - teacher[start]))
    out = behavioral_loss(student, teacher, MAPPING, mode="delta",
                          teacher_layers=TEACHER_LAYERS)
    assert float(out.magnitude) == pytest.approx(0.0, abs=1e-6)
    assert float(out.direction) == pytest.approx(0.0, abs=1e-5)


def test_delta_and_pointwise_are_different_objectives():
    """If these agreed the paper would have nothing to compare."""
    student, teacher = _streams()
    a = behavioral_loss(student, teacher, MAPPING, mode="pointwise")
    b = behavioral_loss(student, teacher, MAPPING, mode="delta")
    assert float(a.total) != pytest.approx(float(b.total), rel=1e-3)
    assert a.mode == "pointwise" and b.mode == "delta"
    assert a.n_pairs == b.n_pairs == 48


def test_magnitude_and_direction_are_reported_separately():
    """A student doing the right thing at the wrong scale and one doing the wrong thing at
    the right scale must be distinguishable."""
    g = torch.Generator().manual_seed(2)
    base = [torch.randn(1, 4, 16, generator=g) for _ in range(3)]
    teacher = base + [torch.randn(1, 4, 16, generator=g) for _ in range(2)]
    scaled = [x * 3.0 for x in base]
    flipped = [-x for x in base]
    mapping = {0: 0, 1: 1}
    scale_only = behavioral_loss(scaled, teacher, mapping, mode="pointwise",
                                 teacher_layers=4, normalise=False)
    sign_only = behavioral_loss(flipped, teacher, mapping, mode="pointwise",
                                teacher_layers=4, normalise=False)
    assert float(scale_only.direction) < 0.01, "rescaling should not look like a direction error"
    assert float(sign_only.direction) > 1.5, "a sign flip must show up as a direction error"


def test_normalisation_stops_deep_layers_dominating():
    """Without it the per-layer term is a proxy for depth rather than for difficulty."""
    g = torch.Generator().manual_seed(3)
    student = [torch.randn(1, 4, 16, generator=g) * (i + 1) for i in range(4)]
    teacher = [torch.randn(1, 4, 16, generator=g) * (i + 1) for i in range(5)]
    mapping = {0: 0, 1: 1, 2: 2}
    raw = behavioral_loss(student, teacher, mapping, mode="pointwise",
                          teacher_layers=4, normalise=False)
    scaled = behavioral_loss(student, teacher, mapping, mode="pointwise",
                             teacher_layers=4, normalise=True)
    spread = lambda r: max(r.values()) / min(r.values())  # noqa: E731
    assert spread(raw.per_layer) > spread(scaled.per_layer)


def test_per_layer_diagnostics_name_every_student_layer():
    student, teacher = _streams()
    out = behavioral_loss(student, teacher, MAPPING, mode="delta")
    assert sorted(out.per_layer) == sorted(MAPPING)
    assert out.to_dict()["norm_ratio"] is not None


def test_the_loss_is_differentiable():
    """It has to be usable as a training term, not only as a diagnostic."""
    student = [torch.randn(1, 4, 8, requires_grad=True) for _ in range(3)]
    teacher = [torch.randn(1, 4, 8) for _ in range(5)]
    out = behavioral_loss(student, teacher, {0: 0, 1: 1}, mode="delta", teacher_layers=4)
    out.total.backward()
    assert any(t.grad is not None for t in student)


def test_a_mask_restricts_scoring_to_the_selected_positions():
    g = torch.Generator().manual_seed(4)
    student = [torch.randn(1, 6, 8, generator=g) for _ in range(3)]
    teacher = [torch.randn(1, 6, 8, generator=g) for _ in range(5)]
    mask = torch.zeros(1, 6, dtype=torch.bool)
    mask[0, :3] = True
    full = behavioral_loss(student, teacher, {0: 0, 1: 1}, mode="pointwise", teacher_layers=4)
    masked = behavioral_loss(student, teacher, {0: 0, 1: 1}, mode="pointwise",
                             teacher_layers=4, mask=mask)
    assert float(full.total) != pytest.approx(float(masked.total), rel=1e-4)


def test_a_width_mismatch_is_refused_with_a_useful_message():
    student = [torch.randn(1, 4, 8) for _ in range(3)]
    teacher = [torch.randn(1, 4, 16) for _ in range(5)]
    with pytest.raises(ValueError, match="widths must match"):
        behavioral_loss(student, teacher, {0: 0}, mode="pointwise", teacher_layers=4)


def test_a_hidden_state_tuple_of_the_wrong_length_is_refused():
    """The commonest mistake is forgetting output_hidden_states and passing logits."""
    student, teacher = _streams()
    with pytest.raises(ValueError, match="output_hidden_states"):
        behavioral_loss(student, teacher[:10], MAPPING, mode="delta",
                        teacher_layers=TEACHER_LAYERS)


# ---------------------------------------------------------------------------
# attention behaviour
# ---------------------------------------------------------------------------
def test_attention_matching_is_head_count_agnostic():
    """Student and teacher have different head counts; marginalising is what makes the
    comparison defined at all."""
    student = [torch.softmax(torch.randn(1, 24, 5, 5), -1)]
    teacher = [torch.softmax(torch.randn(1, 40, 5, 5), -1)]
    out = attention_behavior_loss(student, teacher, [(0, 0)])
    assert torch.isfinite(out.total)
    assert out.mode == "attention_kl"


def test_identical_attention_scores_zero():
    maps = [torch.softmax(torch.randn(1, 8, 5, 5), -1)]
    out = attention_behavior_loss(maps, maps, [(0, 0)])
    assert float(out.total) == pytest.approx(0.0, abs=1e-6)


def test_attention_kl_is_non_negative():
    student = [torch.softmax(torch.randn(1, 4, 6, 6), -1)]
    teacher = [torch.softmax(torch.randn(1, 4, 6, 6) * 3, -1)]
    assert float(attention_behavior_loss(student, teacher, [(0, 0)]).total) > 0


def test_mismatched_sequence_lengths_are_refused():
    student = [torch.softmax(torch.randn(1, 4, 6, 6), -1)]
    teacher = [torch.softmax(torch.randn(1, 4, 8, 8), -1)]
    with pytest.raises(ValueError, match="same sequence"):
        attention_behavior_loss(student, teacher, [(0, 0)])


def test_no_pairs_is_refused():
    with pytest.raises(ValueError, match="no attention layer pairs"):
        attention_behavior_loss([], [], [])


# ---------------------------------------------------------------------------
# the composite configuration
# ---------------------------------------------------------------------------
def test_terms_default_to_off():
    """A run's loss must be exactly what its config says, with nothing inherited."""
    assert CompositeLossConfig().active == {}
    with pytest.raises(ValueError, match="nothing to train"):
        CompositeLossConfig().validate()


def test_unavailable_terms_raise_rather_than_degrade():
    for name in (DELTANET_STATE, MTP):
        assert not LOSS_TERMS[name].available
        assert LOSS_TERMS[name].blocking_reason
        with pytest.raises(ObjectiveUnavailable, match=name):
            CompositeLossConfig(weights={LOGIT_KD: 1.0, name: 1.0}).validate()


def test_the_mtp_block_explains_why_a_result_would_be_fabricated():
    assert "fabricated" in LOSS_TERMS[MTP].blocking_reason


def test_the_deltanet_state_block_names_the_available_alternative():
    reason = LOSS_TERMS[DELTANET_STATE].blocking_reason
    assert "hidden_delta" in reason
    assert LOSS_TERMS[HIDDEN_DELTA].available


def test_combining_both_hidden_terms_needs_an_explicit_choice():
    weights = {LOGIT_KD: 1.0, HIDDEN_POINTWISE: 0.5, HIDDEN_DELTA: 0.5}
    with pytest.raises(ValueError, match="unattributable"):
        CompositeLossConfig(weights=weights).validate()
    CompositeLossConfig(weights=weights, allow_combined_hidden=True).validate()


def test_unknown_and_negative_weights_are_refused():
    with pytest.raises(ValueError, match="unknown loss terms"):
        CompositeLossConfig(weights={"free_lunch": 1.0})
    with pytest.raises(ValueError, match="reward divergence"):
        CompositeLossConfig(weights={CE: -1.0})


def test_forward_flags_are_derived_from_the_active_terms():
    config = CompositeLossConfig(weights={CE: 1.0, HIDDEN_DELTA: 0.5,
                                          ATTENTION: 0.25, ROUTER_BALANCE: 1.0})
    assert config.forward_flags() == {"output_hidden_states": True,
                                      "output_attentions": True,
                                      "output_router_logits": True}
    assert CompositeLossConfig(weights={CE: 1.0}).forward_flags() == {}


def test_config_serialises_and_describes_itself():
    config = CompositeLossConfig(weights={LOGIT_KD: 1.0, HIDDEN_DELTA: 0.5})
    data = config.to_dict()
    assert data["active"] == {HIDDEN_DELTA: 0.5, LOGIT_KD: 1.0}
    assert data["forward_flags"]["output_hidden_states"] is True
    text = describe_loss_terms()
    assert "UNAVAILABLE" in text and MTP in text
