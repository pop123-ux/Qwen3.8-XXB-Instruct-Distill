# Behavioral distillation

The paper's central mechanism. Code: `src/qwen_distill/distillation/behavioral.py`.
Tests: `tests/test_behavioral.py`.

## The problem with layer matching

Depth-reduced distillation conventionally matches hidden states at mapped positions: student
layer `s` is trained so that `h_s` resembles `h_t` at some chosen teacher layer `m(s)`. The
objective has a defect that grows with the compression ratio. It asks the student to *pass
through the same points*, when what was deleted is *work* — the teacher's 64 layers each
transform the residual stream, and 16 of those transformations have no student layer to
perform them. Pointwise matching never says who took over.

## The alternative

Match the **residual contribution** instead of the position. A block's contribution is

```
delta_l = h_{l+1} - h_l
```

and contributions telescope: the total work done by a span of teacher layers `[a, b)` is
exactly `h_b - h_a`, whatever happened in between. So a student layer can be asked to
reproduce the combined contribution of every teacher layer it replaced — **including the
removed ones** — and the target costs nothing extra to compute.

| term | target | what it assumes |
|---|---|---|
| `hidden_pointwise` | `h_s[l] ~ h_t[m(l)]` | the student should visit the teacher's intermediate states |
| `hidden_delta` | `h_s[l+1] - h_s[l] ~ h_t[b] - h_t[a]` | the student should do the teacher's work, in whatever coordinates |

These are not variants of one idea. They make different predictions, and A1 vs A3 in the
ablation matrix runs them against each other.

## Spans: who is responsible for the removed layers

`layer_spans` tiles the teacher's depth so **every teacher layer is charged to exactly one
student layer**. Student layer `s` covers from its own anchor to the next student layer's
anchor; the last absorbs the remainder. Because the group mapping keeps 12 of 16 teacher
groups, four groups are skipped, and the student layers adjacent to a skipped group get a
wider span — that width *is* the extra work being assigned.

Tests check the tiling is complete (all 64 teacher layers covered, no overlap), that the
16 extra layers show up as span widths above one, and that a reordering mapping is refused
because its spans would overlap.

## Direction and magnitude are reported separately

Both are computed, and neither is blended into a single headline:

- **magnitude** (MSE) — a student doing the right thing at half strength drifts.
- **direction** (1 − cosine) — a student pushing the residual stream the wrong way cannot
  be fixed by rescaling.

A test constructs both failures in isolation: a 3x rescaled student registers direction
error < 0.01, and a sign-flipped student registers > 1.5. A single blended number would
hide which one is failing.

Hidden states are RMS-normalised per token before comparison. Layers deep in a residual
stream have much larger activations than early ones, so an unnormalised MSE is dominated by
whichever pairs sit deepest, and the per-layer diagnostic would report depth rather than
difficulty.

## Attention behaviour

Student and teacher have different attention head counts, so per-head correspondence does
not exist and inventing one would be an untested modelling choice. Averaging over heads
gives a per-query distribution over keys that both models define identically, and the KL
between those is a statement about attention behaviour that survives the head-count change.
Pairs are indices into the *attention layers only* — the student's 12 against the teacher's
16.

## What is not available, and why

**`deltanet_state` — matching the recurrent state tensor — is not implemented.** The
student's state is `(48 value heads, 128, 128)`; the teacher's has its own shape, and any
comparison needs a projection that is itself an untested modelling choice. `hidden_delta`
measures the same DeltaNet layers' behaviour at the hidden-size interface where the shapes
genuinely agree. Requesting the state term raises, and the error names the alternative.

**`mtp` is not implemented.** See [STUDENT_ARCHITECTURE.md](STUDENT_ARCHITECTURE.md#mtp).
The error says a result would be fabricated, because it would.

Neither degrades silently into an approximation.

## Composing the loss

```python
CompositeLossConfig(weights={"ce": 0.1, "logit_kd": 1.0,
                            "hidden_delta": 0.5, "router_balance": 1.0})
```

Rules the configuration enforces:

- **Terms default to off.** A run's loss is exactly what its config says; nothing is
  inherited and nothing is implied.
- **Unavailable terms raise**, with the blocking reason, rather than being dropped.
- **Negative weights are refused** — they would reward divergence.
- **Enabling both hidden terms requires `allow_combined_hidden=True`.** That combination is
  legitimate when chosen — it is the A4 cell — and unattributable when reached by accident,
  which is what merging two configs tends to produce.
- **`forward_flags()`** derives the `output_hidden_states` / `output_attentions` /
  `output_router_logits` flags from the active terms, so a term cannot be enabled without
  the forward pass that feeds it.

`router_balance` is the architecture's own load-balancing auxiliary loss at coefficient
0.001, obtained by passing `output_router_logits=True`. It is on in **every** arm of the
layer-matching ablation: without it an arm measures expert collapse rather than what it
meant to measure.

## The claim, and what refutes it

**Claim.** At a 64 -> 48 depth reduction, matching residual contributions over spans beats
matching hidden states at mapped positions, and the margin is larger at deeper layers where
more teacher work has been absorbed.

**Refuted if** A3 does not beat A1 on the aggregate benchmark score, or beats it only within
the seed-to-seed variance measured across repeated control runs. Either outcome refutes the
central claim and gets stated as such rather than reframed.

The seed variance has not been measured yet. Until it has, **no margin is significant**, and
the ablation matrix says so in its `not_controlled` field.
