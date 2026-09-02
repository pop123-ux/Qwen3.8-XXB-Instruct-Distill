# Chunked evaluation of the layer/intermediate KD objective

**The objective did not change.** Run 003 supervises the same 48 mapped layer pairs over
the same full 1536-token sequence, with the same per-token RMS normalisation, the same
per-pair magnitude and direction terms, the same `direction_weight`, and the same mean over
pairs, as [LAYER_KD.md](LAYER_KD.md) defines it. What changed is *when the gradient is
taken*.

This document is the evidence for that claim, measured rather than argued.

## Why

Run 003's first 1536-token calibration completed its step and failed its memory gate:
42.5354 GiB allocated against a 42.0 GiB gate, with 0.33 GiB of headroom left on the card.
The whole excess sat in the loss, not in the forward or the backward pass.

`F.mse_loss` saves both of its inputs for backward, and the objective normalises to fp32
first, so 48 pairs at 1536 positions hold roughly

```
48 pairs x 3 fp32 copies x (1536 x 5120 x 4 bytes) ~ 4.5 GiB
```

that cannot be released until the scalar's `backward()` runs. Gradient checkpointing does
not help: the retained tensors are loss inputs, not layer activations.

[`docs/RUN_003.md`](RUN_003.md) set out four ways forward. This is option 4 — the only one
that buys real headroom without touching the science. The memory gate was **not** raised,
the sequence length was **not** reduced, and the number of supervised pairs was **not**
reduced.

## What the chunked form does

The objective is a mean over pairs, so it splits exactly:

```
L = (1/n) * sum_s [ magnitude_s + direction_weight * direction_s ]
```

A chunk of pairs contributes `(1/n) * sum over that chunk`. Every term carries the
objective's own `1/n` — never a per-chunk average, which would overweight a ragged final
chunk — so the chunk losses sum to `L`, and each pair's term reaches `backward` with
exactly the coefficient it has in `L`. Each chunk's gradient is taken as soon as that chunk
is built, so only `layer_kd_chunk_pairs` pairs' saved inputs are ever live.

The gradient lands on **detached stand-ins** for the student's hidden states rather than
flowing straight into the student. `detach()` shares storage, so the stand-ins allocate
nothing; what is held between the chunks and the end of the loss is one gradient tensor per
supervised layer, in the hidden states' own dtype — 48 x (1536 x 5120) bf16, about 0.70
GiB. The student's own graph is then traversed **once**, by
`ChunkedBehavioralLoss.backward()`, and not once per chunk.

Both forms call the same `_pair_term`, so they cannot drift into computing different
things: the shared function is what makes the equivalence structural rather than a
coincidence that has to be re-checked whenever either path is edited.

Implementation: `qwen_distill.distillation.behavioral.behavioral_loss_chunked`, selected by
`training.layer_kd_chunk_pairs` (Run 003 uses **4**). `null` keeps every pair live at once
— the reference path, not a different objective. Every run record states which form
evaluated the objective, under `distillation.layer_kd_definition.evaluation`.

## Equivalence, asserted

`tests/test_layer_kd_chunking.py`, 27 tests. Value, gradient and per-layer diagnostics, for
chunk widths 1, 2, 3, 5, 7, 12 and 32 over 12 pairs — including the ragged ones — in both
`pointwise` and `delta` modes, with normalisation on and off, at a non-default
`direction_weight`, and end to end through the trainer's own recorded losses across a
multi-step run. Declared tolerances: `1e-6` relative on the value, `1e-6` on the gradient.

The suite also asserts the two properties the design depends on: that no gradient reaches
the student before `backward()` is called, and that the run record says which form was
used.

## Equivalence, measured on Run 003's own calibration batch

Unit tests cover the algebra on synthetic tensors. What they cannot cover is whether it
still holds at 1536 positions in bf16, through a 4-bit teacher, on the real student. So
both forms were run on the **exact batch Run 003's calibration trained on** — same corpus
(`sha256 e11ca38b…`), same tokenizer, same sequence length, same sampler, same seed 0, and
the configuration built by `kd_run.build_config` rather than restated.

Harness: `scripts/verify_layer_kd_chunking.py`.
Records: `experiments/run003_chunking_equivalence/`.

### Stage 1 — the objective and its gradient with respect to the hidden states

Both models forward under `no_grad`; the mapped hidden states become leaves; both forms are
evaluated on the *same* tensors, isolating the loss function from the student's backward.

| | reference (unchunked) | chunked, 4 pairs (12 chunks) |
| --- | ---: | ---: |
| total | 1.450157642364502 | 1.4501575492031407 |
| magnitude | 0.966771125793457 | 0.9667710885114502 |
| direction | 0.48338645696640015 | 0.48338646069169044 |
| student_norm | 2003.2148303985596 | 2003.2148303985596 |
| teacher_norm | 3823.8540534973145 | 3823.8540534973145 |

```
value    total      6.424e-08 relative     (tolerance 1e-6)
         magnitude  3.856e-08 relative
         direction  7.707e-09 relative
per-layer worst     0.0       absolute     (48 of 48 identical)
gradient worst      0.0       absolute, 0.0 relative
                              48 of 48 supervised hidden states compared
```

**The gradients are bit-identical**, at every one of the 48 supervised layers. That is the
expected result and not a lucky one: each pair's term is computed by the same code and
reaches `backward` with the same coefficient `1/48`, through the same kernels.

The scalar differs in the eighth significant figure. That is float32 summation order and
nothing else — the unchunked form reduces with `torch.stack(...).mean()` over 48 fp32
terms, the chunked form accumulates the reported value in Python floats. It is
approximately 4e-08 of the loss and it carries no gradient.

The reference's total, `1.450158`, is the `1.4501` the failed calibration logged at step 1,
which confirms the harness reproduced the calibration's batch rather than some other one.

### Stage 2 — the gradient the optimizer actually receives

Two complete forward/backward passes over the same batch, one per form, comparing every
LoRA parameter's gradient. This is the end-to-end claim: not that the loss function agrees,
but that the *run* does.

Two things had to be controlled for, and a first two-pass attempt got both wrong. Both
corrections are kept in `experiments/run003_chunking_equivalence/` rather than discarded.

**Dropout.** `lora_dropout` is 0.05 and the model trains in `train()` mode. LoRA's B is
zero-initialised, so the mask cannot affect the forward — the layer term is identical
either way — but `grad_B = grad_out @ dropout(x) A^T` depends on it entirely. The first
attempt resampled the mask between passes and disagreed by 3.5e-03 on `lora_B` gradients,
none of it about chunking. Every pass now restores the same RNG state before it starts.

**The student's own reproducibility.** With the mask fixed, the backward is *still* not
bit-reproducible: two identical unchunked passes differ by up to **9.77e-03** on a LoRA
gradient. A 13B MoE with DeltaNet kernels and gradient checkpointing accumulates through
order-dependent atomics and routing scatters. So there is no fixed reference to compare
against, and "between forms" cannot be compared with "within a form" directly — the
between set has more pairs, and the maximum of more draws is larger for no reason but the
counting. A first attempt at this criterion was biased that way and is kept as
`parameters_biased_criterion.json`.

The test therefore run is one of **exchangeability**. Four passes of each form, all eight
split evenly into two groups every possible way (35 distinct splits), the same separation
statistic computed for each, and the true form-based split ranked among them. If the form
label carries no information beyond the hardware's noise, the true split should be typical.

| statistic | within unchunked (4 runs) | within chunked (4 runs) | true split |
| --- | ---: | ---: | ---: |
| worst absolute | 9.765625e-03 | 8.7890625e-03 | 1.0742188e-02 |
| mean over the 117 varying parameters | 2.4033e-04 | 2.1149e-04 | 2.6412e-04 |

```
240 LoRA parameters, 123 identical in all 8 passes
layer term        1.450158 in all 8 passes (spread 6.424e-08 relative)
rank of the true split among 35, by max    1 of 35   p = 0.57
rank of the true split among 35, by mean  10 of 35   p = 0.29
```

The true split is **not extreme**: tenth of thirty-five by the mean statistic, and tied
with nineteen other splits at the top by the max, which one noisy parameter dominates. The
form label explains nothing the hardware does not already explain, and the between-form
mean separation (2.64e-04) sits between the two forms' own (2.40e-04 and 2.11e-04).

This is a weaker kind of evidence than stage 1's bit-identity, and it is meant to be: it
can only ever say "no separable difference was found", never "there is none". Stage 1 is
the proof that the objective and its gradient are the same; stage 2 confirms that the
shared, unmodified student backward downstream of it adds nothing attributable to the form.

Memory, measured in the same passes: **42.1117 -> 37.3305 GiB** allocated, a 4.78 GiB
saving. This harness runs neither the logit diagnostics nor the optimizer step, so these
are not the run's numbers; the calibration's are.

> The `parameters.log` transcript's split line reads "3/3" where the run was 4/4. That was
> a formatting slip in the print statement, since corrected; `repeats_per_form: 4` and
> `n_splits: 35` in `parameters.json` are the record.


## What this does not claim

- Not a capability result, not a research result. It is an implementation validation, and
  the run class it belongs to is *calibration*.
- It says nothing about whether layer KD is a good objective. It says only that the
  objective Run 003 evaluates is the one [LAYER_KD.md](LAYER_KD.md) defines.
- The memory figures in stage 1 are indicative only: both are absolute allocations with
  the teacher's weights and both sides' hidden states already resident, and the chunked
  measurement additionally carries the reference's 48 saved gradients. The memory evidence
  is the calibration run's own step profile, in [RUN_003.md](RUN_003.md).
