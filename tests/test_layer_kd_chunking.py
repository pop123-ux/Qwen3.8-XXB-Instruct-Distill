"""The chunked layer objective is the same objective.

Run 003's first calibration cleared its step but not its memory gate, and the whole excess
sat in the loss: ``mse_loss`` saves both of its normalised fp32 inputs, so holding all 48
mapped pairs at 1536 positions costs roughly 4 GiB that cannot be freed until the gradient
exists. :func:`~qwen_distill.distillation.behavioral.behavioral_loss_chunked` takes that
gradient a few pairs at a time instead.

That is only admissible if it is the *same objective*. These tests are the argument that it
is: same value, same gradient, same per-layer diagnostics, for every chunk size including
ragged ones, in both matching modes, at the end-to-end level of a real training step. If
any of them fails, Run 003's layer-KD arm is no longer the arm Run 002 is compared against
and the comparison is void.

The tolerances are asserted here and the measured differences on the real 1536-token
calibration batch are recorded in docs/LAYER_KD_CHUNKING.md.
"""

from __future__ import annotations

import json

import pytest
from conftest import HAS_STACK

pytestmark = pytest.mark.skipif(not HAS_STACK, reason="requires torch and transformers")

#: The reduction is a mean over pairs, so the two forms differ only in summation order.
#: Anything larger than this is a real disagreement, not float32 associativity.
VALUE_TOLERANCE = 1e-6
#: Each pair's term reaches ``backward`` with the identical coefficient 1/n in both forms,
#: through the identical kernels, so the gradients are expected to agree far more closely
#: than the scalar does.
GRADIENT_TOLERANCE = 1e-6

STUDENT_LAYERS = 12
TEACHER_LAYERS = 16
#: Monotonic and not the identity, like the real 48 -> 64 map: teacher layers 4-7 have no
#: student anchor, which is the condition the objective exists to expose.
MAPPING = {s: s if s < 4 else s + 4 for s in range(STUDENT_LAYERS)}


def hidden_states(n_layers, *, seed, tokens=7, width=16, batch=2):
    import torch

    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(batch, tokens, width, generator=generator, dtype=torch.float32)
        for _ in range(n_layers + 1)
    ]


def leaves(states):
    return [s.clone().requires_grad_(True) for s in states]


def reference(student, teacher, **kwargs):
    """The unchunked objective, and the gradient it puts on the student's hidden states."""
    from qwen_distill.distillation.behavioral import behavioral_loss

    output = behavioral_loss(student, teacher, MAPPING, teacher_layers=TEACHER_LAYERS,
                             **kwargs)
    output.total.backward()
    return output, [s.grad for s in student]


def chunked(student, teacher, *, chunk_pairs, **kwargs):
    from qwen_distill.distillation.behavioral import behavioral_loss_chunked

    result = behavioral_loss_chunked(student, teacher, MAPPING,
                                     teacher_layers=TEACHER_LAYERS,
                                     chunk_pairs=chunk_pairs, **kwargs)
    result.backward()
    return result, [s.grad for s in student]


def value(x):
    """A scalar from either form: the reference returns graph tensors, the chunked form
    returns floats it has already detached."""
    detach = getattr(x, "detach", None)
    return float(detach() if detach else x)


def largest_difference(a, b):
    """Largest absolute gradient difference, over the tensors both forms produced."""
    worst = 0.0
    for x, y in zip(a, b, strict=True):
        if x is None and y is None:
            continue
        assert x is not None and y is not None, "one form left a hidden state ungraded"
        worst = max(worst, float((x - y).abs().max()))
    return worst


# ---------------------------------------------------------------------------
# the value and the gradient
# ---------------------------------------------------------------------------
#: 1 is the extreme, 5 and 7 divide 12 raggedly, 12 is one chunk, 32 exceeds the pair
#: count. A ragged final chunk is the case a per-chunk average would get wrong.
@pytest.mark.parametrize("chunk_pairs", [1, 2, 3, 5, 7, 12, 32])
@pytest.mark.parametrize("mode", ["pointwise", "delta"])
def test_the_chunked_objective_has_the_same_value_and_gradient(chunk_pairs, mode):
    teacher = hidden_states(TEACHER_LAYERS, seed=2)
    want, want_grads = reference(leaves(hidden_states(STUDENT_LAYERS, seed=1)),
                                 teacher, mode=mode)
    got, got_grads = chunked(leaves(hidden_states(STUDENT_LAYERS, seed=1)), teacher,
                             chunk_pairs=chunk_pairs, mode=mode)

    assert got.output.n_pairs == want.n_pairs
    assert value(got.output.total) == pytest.approx(value(want.total),
                                                    rel=VALUE_TOLERANCE)
    assert value(got.output.magnitude) == pytest.approx(value(want.magnitude),
                                                        rel=VALUE_TOLERANCE)
    assert value(got.output.direction) == pytest.approx(value(want.direction),
                                                        rel=VALUE_TOLERANCE)
    assert largest_difference(want_grads, got_grads) <= GRADIENT_TOLERANCE


@pytest.mark.parametrize("chunk_pairs", [1, 5, 12])
def test_the_per_layer_diagnostics_match(chunk_pairs):
    """``per_layer``, ``student_norm`` and ``teacher_norm`` are read off the figures and
    the run record. A chunked run whose diagnostics drifted would be unreadable against
    the calibration that preceded it."""
    teacher = hidden_states(TEACHER_LAYERS, seed=2)
    want, _ = reference(leaves(hidden_states(STUDENT_LAYERS, seed=1)), teacher,
                        mode="pointwise")
    got, _ = chunked(leaves(hidden_states(STUDENT_LAYERS, seed=1)), teacher,
                     chunk_pairs=chunk_pairs, mode="pointwise")

    assert sorted(got.output.per_layer) == sorted(want.per_layer)
    for layer, magnitude in want.per_layer.items():
        assert got.output.per_layer[layer] == pytest.approx(magnitude, rel=VALUE_TOLERANCE)
    assert got.output.student_norm == pytest.approx(want.student_norm, rel=VALUE_TOLERANCE)
    assert got.output.teacher_norm == pytest.approx(want.teacher_norm, rel=VALUE_TOLERANCE)


def test_normalisation_off_is_equivalent_too():
    """The flag is part of the objective's definition, so equivalence has to hold under
    both settings rather than only the one Run 003 uses."""
    teacher = hidden_states(TEACHER_LAYERS, seed=2)
    want, want_grads = reference(leaves(hidden_states(STUDENT_LAYERS, seed=1)), teacher,
                                 mode="pointwise", normalise=False)
    got, got_grads = chunked(leaves(hidden_states(STUDENT_LAYERS, seed=1)), teacher,
                             chunk_pairs=5, mode="pointwise", normalise=False)
    assert value(got.output.total) == pytest.approx(value(want.total), rel=VALUE_TOLERANCE)
    assert largest_difference(want_grads, got_grads) <= GRADIENT_TOLERANCE


def test_the_direction_weight_is_carried_through_unchanged():
    teacher = hidden_states(TEACHER_LAYERS, seed=2)
    want, want_grads = reference(leaves(hidden_states(STUDENT_LAYERS, seed=1)), teacher,
                                 mode="pointwise", direction_weight=0.25)
    got, got_grads = chunked(leaves(hidden_states(STUDENT_LAYERS, seed=1)), teacher,
                             chunk_pairs=5, mode="pointwise", direction_weight=0.25)
    assert value(got.output.total) == pytest.approx(value(want.total), rel=VALUE_TOLERANCE)
    assert largest_difference(want_grads, got_grads) <= GRADIENT_TOLERANCE


def test_loss_scale_scales_the_gradient_and_leaves_the_reported_value_alone():
    """Gradient accumulation divides the gradient by the number of micro-steps. The
    reported layer term must stay the objective's own value, or the two arms' logged
    losses stop being comparable."""
    teacher = hidden_states(TEACHER_LAYERS, seed=2)
    _, plain_grads = chunked(leaves(hidden_states(STUDENT_LAYERS, seed=1)), teacher,
                             chunk_pairs=5, mode="pointwise")
    scaled, scaled_grads = chunked(leaves(hidden_states(STUDENT_LAYERS, seed=1)), teacher,
                                   chunk_pairs=5, mode="pointwise", loss_scale=0.25)
    unscaled, _ = chunked(leaves(hidden_states(STUDENT_LAYERS, seed=1)), teacher,
                          chunk_pairs=5, mode="pointwise")
    assert value(scaled.output.total) == pytest.approx(value(unscaled.output.total))
    for plain, quarter in zip(plain_grads, scaled_grads, strict=True):
        if plain is None:
            continue
        assert float((plain * 0.25 - quarter).abs().max()) <= GRADIENT_TOLERANCE


# ---------------------------------------------------------------------------
# it defers the student's traversal rather than repeating it
# ---------------------------------------------------------------------------
def test_the_student_graph_is_untouched_until_backward_is_called():
    """The point of the detached stand-ins: the loss's own gradient exists before the
    student's graph is walked, and the student is then walked once, not once per chunk."""
    student = leaves(hidden_states(STUDENT_LAYERS, seed=1))
    result = chunked_no_backward = None
    from qwen_distill.distillation.behavioral import behavioral_loss_chunked

    chunked_no_backward = behavioral_loss_chunked(
        student, hidden_states(TEACHER_LAYERS, seed=2), MAPPING,
        teacher_layers=TEACHER_LAYERS, mode="pointwise", chunk_pairs=3)
    assert all(s.grad is None for s in student), \
        "the loss reached the student before backward() was called"
    assert chunked_no_backward.n_chunks == 4
    assert all(g is not None for g in chunked_no_backward.grads)

    chunked_no_backward.backward()
    assert any(s.grad is not None for s in student)
    del result


def test_hidden_states_that_need_no_gradient_are_not_forced_through_backward():
    """The equivalence harness runs both forwards under ``no_grad`` and only wants the
    loss's own gradient. ``backward()`` must be a no-op there rather than raising."""
    from qwen_distill.distillation.behavioral import behavioral_loss_chunked

    student = hidden_states(STUDENT_LAYERS, seed=1)
    result = behavioral_loss_chunked(student, hidden_states(TEACHER_LAYERS, seed=2),
                                     MAPPING, teacher_layers=TEACHER_LAYERS,
                                     mode="pointwise", chunk_pairs=4)
    assert all(g is not None for g in result.grads)
    result.backward()


def test_a_chunk_size_below_one_is_refused():
    from qwen_distill.distillation.behavioral import behavioral_loss_chunked

    with pytest.raises(ValueError, match="chunk_pairs must be at least 1"):
        behavioral_loss_chunked(hidden_states(STUDENT_LAYERS, seed=1),
                                hidden_states(TEACHER_LAYERS, seed=2), MAPPING,
                                teacher_layers=TEACHER_LAYERS, chunk_pairs=0)


# ---------------------------------------------------------------------------
# end to end, through a real training step
# ---------------------------------------------------------------------------
def train_both_ways(tmp_path, chunk_pairs):
    from test_layer_kd import history, make_config, make_teacher, run, summary

    output = tmp_path / f"chunk_{chunk_pairs}"
    config = make_config(output, layer_kd_chunk_pairs=chunk_pairs)
    assert run(config, make_teacher()) == 0
    steps = [r for r in history(output) if r.get("status") == "completed_step"]
    assert steps
    return steps, summary(output)


@pytest.mark.parametrize("chunk_pairs", [1, 3])
def test_a_chunked_run_records_the_same_losses_as_the_unchunked_run(tmp_path,
                                                                    chunk_pairs):
    """The strongest form of the claim: same seed, same batches, same optimizer, and the
    trainer's own recorded loss agrees step for step. If the two forms trained differently
    the trajectories would separate after step 1 even if step 1 matched."""
    unchunked, _ = train_both_ways(tmp_path, None)
    chunked_steps, _ = train_both_ways(tmp_path, chunk_pairs)

    assert len(chunked_steps) == len(unchunked)
    for want, got in zip(unchunked, chunked_steps, strict=True):
        assert got["step"] == want["step"]
        for field in ("loss", "layer_kd_loss", "layer_magnitude", "layer_direction",
                      "layer_norm_ratio"):
            assert got[field] == pytest.approx(want[field], rel=1e-4, abs=1e-6), field


def test_the_run_record_says_which_form_evaluated_the_objective(tmp_path):
    """Provenance: a reader must be able to tell, from the record alone, whether a run
    chunked its loss and at what width — and be told that it did not change the objective."""
    _, chunked_summary = train_both_ways(tmp_path, 3)
    _, plain_summary = train_both_ways(tmp_path, None)

    got = chunked_summary["distillation"]["layer_kd_definition"]["evaluation"]
    assert got["form"] == "chunked"
    assert got["chunk_pairs"] == 3
    assert "behavioral_loss_chunked" in got["implementation"]
    assert "not a change to the objective" in got["note"]
    assert got["equivalence"]

    plain = plain_summary["distillation"]["layer_kd_definition"]["evaluation"]
    assert plain["form"] == "unchunked"
    assert plain["chunk_pairs"] is None


def test_the_objective_definition_is_otherwise_identical(tmp_path):
    """Everything that defines *what* was supervised must be byte-identical between the
    two forms. Only the ``evaluation`` block may differ."""
    _, chunked_summary = train_both_ways(tmp_path, 3)
    _, plain_summary = train_both_ways(tmp_path, None)

    a = dict(chunked_summary["distillation"]["layer_kd_definition"])
    b = dict(plain_summary["distillation"]["layer_kd_definition"])
    a.pop("evaluation"), b.pop("evaluation")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
