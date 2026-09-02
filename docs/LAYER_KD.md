# Layer / intermediate KD — the exact objective

This is the definition Run 003 trains against. It is written down here, asserted by
`tests/test_layer_kd.py`, and copied into every run's `summary.json` under
`distillation.layer_kd_definition`, because a layer-KD number that does not say *which*
representations were matched, *how* they were paired and *how* the pairs were reduced to a
scalar cannot be interpreted, reproduced or compared against another arm.

Implementation:
[`qwen_distill.distillation.behavioral.behavioral_loss`](../src/qwen_distill/distillation/behavioral.py)
in `mode="pointwise"`. That function pre-dates Run 003; it is the module's declared **A1
control**, and Run 003 uses it unchanged. Nothing was invented for this run.

## What is compared

| | source | tensor |
| --- | --- | --- |
| teacher | `output_hidden_states=True` on the 4-bit resident teacher, same `no_grad` forward that produced the logits | `hidden_states[m(l) + 1]` — the output of teacher layer `m(l)` |
| student | `output_hidden_states=True` on the QLoRA student | `hidden_states[l + 1]` — the output of student layer `l` |

Both tuples have length `n_layers + 1`; entry `i` is the residual stream *before* layer
`i`, so entry `i + 1` is the output of layer `i`. One teacher forward serves both the
output distribution and the intermediate signal; running the teacher twice would double the
most expensive part of a KD step for no information gained.

## The mapping

`m` comes from
[`qwen_distill.architecture.moe_init.map_layers`](../src/qwen_distill/architecture/moe_init.py)
with `strategy="group"`, built at the first step from the depths the two models actually
report rather than from a constant. **Never `teacher layer i = student layer i`**: that
holds only for the first two groups and breaks immediately after.

The student's 48 layers and the teacher's 64 are both whole 4-layer hybrid groups
`[DeltaNet, DeltaNet, DeltaNet, attention]`. Twelve of the teacher's sixteen groups are
selected evenly across its depth and copied position-for-position, so a student layer always
lands on a teacher layer of its own block type — deleting every fourth layer instead would
rotate the pattern and match DeltaNet against attention. The trainer refuses a mapping that
crosses block types rather than training through it.

| student layers | teacher layers |
| --- | --- |
| 0–3 | 0–3 |
| 4–7 | 4–7 |
| 8–11 | 12–15 |
| 12–15 | 16–19 |
| 16–19 | 20–23 |
| 20–23 | 28–31 |
| 24–27 | 32–35 |
| 28–31 | 40–43 |
| 32–35 | 44–47 |
| 36–39 | 48–51 |
| 40–43 | 56–59 |
| 44–47 | 60–63 |

Teacher layers **8–11, 24–27, 36–39 and 52–55** — sixteen in all — have no student anchor
and are **not supervised**. That is what conventional layer matching does with a depth
change, and it is precisely the limitation this control exists to measure. It is not a
defect of the implementation and it is not worked around; the count is recorded in the run
record and drawn in figure F16.

## Alignment and normalisation

**Projection: none.** Teacher and student share `hidden_size = 5120`, so the comparison
needs no learned projection. `behavioral_loss` raises if the widths disagree rather than
inserting one — a projection would itself be an untested modelling choice, and a control
arm carrying an untested choice is not a control.

**Normalisation: per-token RMS scaling to unit norm** (`layer_kd_normalise`, default on).
Layers deep in a residual stream carry much larger activations than early ones, so an
unnormalised MSE is dominated by whichever mapped pairs happen to sit deepest. Normalising
makes the per-layer terms comparable, which is what lets the per-layer breakdown be read as
"which layers are struggling" rather than "which layers are deep".

## The loss

For each mapped pair `(l, m(l))`, with `h_s` and `h_t` the normalised student and teacher
hidden states over the batch's positions:

```
magnitude_l = MSE(h_s, h_t)
direction_l = 1 - mean cosine_similarity(h_s, h_t)
term_l      = magnitude_l + direction_weight * direction_l
```

and the objective is the mean of `term_l` over the 48 pairs.
`direction_weight` is `training.layer_kd_direction_weight`, default `1.0`.

**Magnitude and direction are always reported separately** and never blended into the log.
They are different failures: a student doing the right thing at half strength drifts, and a
student pushing the residual stream the wrong way cannot be fixed by rescaling. A single
number cannot tell them apart. `layer_norm_ratio` — the student's mean activation norm over
the teacher's — is logged alongside as the earliest sign of the first failure.

## Weighting

**Pure.** The layer term is the entire optimised objective, weight `1.0`.

The logit KD divergence, the cross-entropy, the teacher entropy, the top-1 agreement and
the teacher tail mass are all computed under `torch.no_grad()` and logged as diagnostics.
They contribute no gradient. This mirrors Run 002 exactly, where `kd_weight = 1.0` made
pure logit KD the whole objective and CE was likewise reported and not optimised — so the
two arms are pure single-objective controls, and the five shared diagnostics put them on
the same axes.

## Why it cannot silently become logit KD

Three separate guards, because this is the failure that would void the comparison:

1. `objectives.ObjectiveConfig.validate` refuses `layer_kd` with any signal source other
   than `online` — a stored top-k logit corpus has no hidden states and cannot serve it.
2. `train()` refuses at setup if the signal provider was built without
   `capture_hidden_states=True`, before any weight is loaded.
3. The step raises if the teacher returns no hidden states.

None of the three degrades to the logit objective. `tests/test_layer_kd.py` additionally
asserts that the loss handed to `backward()` equals the layer term and differs from the KD
divergence reported beside it.

## What this is not

It is not the project's proposed behavioural/state objective. That one is
`mode="delta"` in the same module — each student layer's *residual contribution* against
the summed contribution of the teacher span it replaced, which is defined for the sixteen
dropped layers as well. Keeping the two as separate modes of one function is deliberate:
they are the two arms of the layer-matching ablation and must be run against each other,
not merged. Run 003 is the pointwise arm; the delta arm is a later run and is not started
automatically.
