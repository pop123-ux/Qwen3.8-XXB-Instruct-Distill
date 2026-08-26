# Scaling study protocol — 4M → 500M on one consumer GPU

**Status: not started. This is the protocol, written before any of it runs.**

The point of writing it now is that a scaling study is easy to ruin after the fact. Once
the numbers exist, "we also changed the corpus at the 236M rung" becomes a footnote
instead of a disqualification. Fixing the rules first is the only defence.

---

## 1. What this project currently has — and why it is not a scaling law

Three training runs exist. **No two of them differ only in model size**, so the number of
usable points on a loss-vs-size curve is **zero**.

| run | params (measured) | vocabulary | corpus | result |
|---|---|---|---|---|
| `t4_prototype` | 4,029,700 | **4096** | **synthetic token sequences**, seq 256 | validation loss 2.091 |
| Level 2 | 94,476,448 | 256 (bytes) | procedural byte text, 8 MB | validation BPB 1.270 |
| Level 2R | 94,476,448 | 256 (bytes) | real public-domain English | **RUNNING** |

Why each pairing fails:

- **4M vs 94.5M** — different vocabulary (4096 vs 256) and different data (synthetic vs
  procedural). Cross-entropy over 4096 symbols and bits-per-byte over 256 are not the
  same axis. The 4M run validated the *mechanism* — forward, backward, optimizer,
  checkpoint, resume — and was never a language-modelling measurement.
- **Level 2 vs Level 2R** — identical size, different corpus. A corpus comparison. See
  [level2_vs_level2r.md](level2_vs_level2r.md).
- **4M vs Level 2R** — both of the above at once.

> **Two points do not make a scaling law, and this project does not even have two.**
> A line through two points is exact by construction and predicts nothing. It cannot
> distinguish a power law from a straight line from noise, and it has no residuals to
> inspect.

---

## 2. What a scaling law actually requires

Everything below must be **identical across rungs**. Each one that is not is a variable
the exponent silently absorbs.

| held fixed | why |
|---|---|
| the corpus — train **and** validation, pinned by SHA-256 | different data has different entropy; the corpus difference would show up as curvature |
| the tokenizer — byte-level, vocab 256 | bits-per-byte is only comparable within one tokenisation |
| the validation split rule — whole documents held out | a contiguous tail measures continuation, not generalisation |
| the validation segmentation — same sequence length and stride | byte-level loss is a mean over sequences, so segmentation moves it |
| precision | fp16 autocast and fp32 disagree to three decimals, comparable to the effect |
| optimizer family, weight decay, gradient clipping, schedule shape | |
| the **shape rule** (aspect ratios, GQA ratio, DeltaNet head ratios, full-attention interval) | otherwise a rung differs in shape as well as size, and nothing says which mattered |
| the token budget rule | see §4 |

**Varies:** model size, and the learning rate under a stated rule (§5).

**Minimum for a claim:**

- **≥4 rungs** spanning **≥1 order of magnitude**. Three can show curvature; four can be
  argued with.
- **≥1 held-out rung** not used to fit, predicted in advance, then run. A fit that
  interpolates its own inputs has demonstrated arithmetic.
- **Residuals published.** A power law that fits with a systematic residual trend is the
  wrong functional form, and only the residuals say so.

---

## 3. The ladder

Every rung is built by the **Level-2 shape rule** in
`src/qwen_distill/analysis/scaling.py`, so aspect ratios stay fixed as size varies.
Parameter counts are **measured** with `count_parameters`, never chosen — which is why
none of them is a round number.

| rung | hidden | layers | layout | measured params | FLOPs vs L2 | est. tok/s |
|---|---|---|---|---|---|---|
| R0 | 256 | 4 | 3 DeltaNet + 1 attn | **4,181,092** | 0.05× | ~43,900 |
| R1 | 512 | 8 | 6 + 2 | **30,562,028** | 0.33× | ~6,400 |
| R2 | 640 | 16 | 12 + 4 | **94,476,448** | 1.00× | **2,089.2 (MEASURED)** |
| R3 | 1024 | 16 | 12 + 4 | **236,237,488** | 2.45× | ~850 |
| R4 | 1280 | 20 | 15 + 5 | **471,678,480** | 4.87× | ~430 |

R2 is Level 2, rebuilt through the same rule and reproducing its published count exactly
(94,480,000 rounded). It is the only measured throughput in the table; the rest are FLOP-
ratio arithmetic from that one anchor and are **not predictions** — see §8.

**Note R0 is a rebuild, not the existing prototype.** The 4M run used vocab 4096 on
synthetic data and cannot join this ladder. R0 must be retrained at byte level on the
shared corpus, or the study starts at R1.

**R1 is not in the original four-rung plan and is recommended anyway.** 4M → 94.5M is a
23× gap with nothing inside it. Four points where the two smallest are 23× apart give a
fit dominated by the endpoints. R1 costs ~1.4 hours.

### Cost, at two token budgets

| rung | 32.77M tokens (Level 2's budget) | 100M tokens |
|---|---|---|
| R0 4.18M | 0.2 h | 0.6 h |
| R1 30.6M | 1.4 h | 4.3 h |
| R2 94.5M | 4.4 h | 13.3 h |
| R3 236M | 10.7 h | 32.6 h |
| R4 472M | 21.2 h | 64.7 h |
| **total** | **~38 h** | **~116 h** |

At 32.77M tokens the whole ladder is roughly one weekend of free-tier Colab, spread over
sessions. At 100M it is about three weeks. **Both assume the extrapolated rates in §8 are
right, and they have not been checked.** Budget for the ladder to cost twice this.

### Memory

Every rung fits a T4 at seq 2048 × batch 8 with gradient checkpointing and fp32 AdamW,
except R4, which needs batch 4. All five fit a 12 GB card with smaller batches, some
requiring 8-bit optimizer moments. Verify before committing hours:

```bash
python scripts/evaluate_scaling_candidates.py --throughput
```

These are estimates from an estimator calibrated at ~100M and never checked above it.

---

## 4. The token budget rule — choose one and say which

The two defensible choices measure **different things**, and mixing them produces a curve
that means neither.

**A. Fixed data** — every rung sees the same tokens.
Isolates model size cleanly and costs least. Measures *capacity at a fixed data budget*.
Larger rungs will be undertrained, so the curve **flattens at the top for a reason that
has nothing to do with capacity** — and reporting that flattening as a capacity ceiling
would be wrong.

**B. Compute-proportional** — tokens ∝ parameters (e.g. 20 tokens/param, Chinchilla-style).
Measures *compute-optimal loss*, the quantity scaling laws are usually about. Costs ∝ N²:
R4 at 20 tokens/param is 9.4 B tokens, which is roughly 250 days on a T4. **Out of reach
here.**

**Recommended: A, primary, at 100M tokens, with the limitation stated in every result.**
At 100M tokens, R2 sees ~1.06 tokens/param and R4 sees ~0.21 — every rung is far below
compute-optimal and the largest most so. Say this next to the exponent, or the number
will be read as something it is not.

**One compute-proportional control at R1** (30.6M params × 20 = 611M tokens, ~26 h) is
affordable and tells you how far the fixed-data curve sits from the compute-optimal one at
one point. That is worth more than a fourth digit on the exponent.

---

## 5. The learning-rate rule

A fixed learning rate across a 113× size range is **not** a controlled comparison: the
value tuned for 94.5M is too high for 472M and too low for 4.2M, and the curve then
measures LR mistuning at the ends.

State the rule up front. In order of preference:

1. **µP / maximal-update parametrisation** — designed for exactly this, transfers the
   optimum across widths. Requires initialisation and per-layer LR changes the current
   trainer does not implement.
2. **Width-scaled**: `lr(N) = lr_ref × sqrt(hidden_ref / hidden)`. Cheap, defensible,
   approximate.
3. **Fixed LR** — only with a documented sweep showing the endpoints are not mistuned.

**Whichever is chosen, run a 3-point LR sweep at R0 and at R4** (the cheapest and the most
expensive rungs) at ~10% of the token budget. If the optimum sits at an edge of the sweep,
the rule is wrong and the ladder is measuring it. R0's sweep costs minutes.

---

## 6. Procedure per rung

1. Verify the corpus: `python scripts/verify_corpus.py <corpus-dir>`. Record both split
   digests. **Same digests at every rung**, checked, not assumed.
2. Build the config from the shape rule. Change only `hidden_size`, `num_hidden_layers`,
   the LR under the §5 rule, and `output_dir`.
3. Confirm the fit: `python scripts/evaluate_scaling_candidates.py --devices <gb>`.
4. Train to the fixed token budget — **not** to a fixed step count. Steps are not
   comparable across rungs when batch shapes differ; tokens are.
5. `python scripts/analyze_training_run.py <run-dir> --json` — record run-wide throughput,
   sessions, plateau step, `epochs_seen`.
6. `python scripts/sanity_generate.py <run-dir> --training-text <train.txt>` — record the
   generations verbatim.
7. Work [POST_RUN_CHECKLIST.md](POST_RUN_CHECKLIST.md) before the rung counts as done.
8. Publish `RESULT.json` with establishes / does_not_establish / unknown.

**Stop the ladder if** a rung's generations are degenerate, or validation diverges from
training, or `epochs_seen` exceeds ~2 (the corpus is then the binding constraint and every
larger rung is measuring the data budget).

---

## 7. Analysis — and what may be claimed

Fit `L(N) = L_∞ + A · N^(-α)` to the **validation** bits-per-byte at each rung.

Publish, always together:

- every point, with its measured parameter count and its token budget;
- the fitted `α`, `A`, `L_∞` **with confidence intervals**;
- the **residuals**, plotted;
- the **held-out rung**: predicted before it ran, and what it actually did.

**May be claimed with 4+ rungs and a successful held-out prediction:**
"On this corpus, at this token budget, with this architecture family, validation
bits-per-byte follows `L ≈ L_∞ + A·N^(-α)` with α = _ ± _ over 4.2M–472M."

**May not be claimed, ever, from this study:**

- that α transfers to another corpus, tokenizer, or architecture family;
- that byte-level BPB scaling implies token-level scaling — different unit, different
  curve;
- that extrapolating past 472M is supported. The ladder spans 113×; the fit is evidence
  inside that range and speculation outside it;
- that a flattening at the top is a capacity ceiling, when budget A guarantees the top
  rungs are undertrained;
- that any of it says anything about **distillation**. This is from-scratch language
  modelling. Distillation is a different objective with a different curve, and a
  from-scratch scaling law does not predict it.

**A negative result is a result.** If the points do not fit a power law, publish the
points and say so. If the largest rung is no better than the one below it at this token
budget, that is a finding about the budget, and it is worth more than a forced exponent.

---

## 8. The throughput numbers above are arithmetic, not measurements

Every `tok/s` in §3 except R2's is Level 2's measured **2,089.2 tok/s** divided by the
ratio of forward FLOPs per token. That ignores memory bandwidth, kernel efficiency and
occupancy, all of which change with shape — and the Gated DeltaNet path in particular has
a Python loop whose cost does not track FLOPs.

**One measured point cannot establish how throughput scales.** Treat every derived rate as
an order of magnitude. The first rung that runs replaces its estimate with a measurement,
and the second one makes the extrapolation checkable. Until then it is arithmetic, and
`extrapolated_tokens_per_second()` labels it `UNVALIDATED EXTRAPOLATION` on every call for
that reason.

---

## 9. Prerequisites

Before rung 1:

- [ ] Level 2R has finished and been through the post-run checklist. If 94.5M cannot learn
      English structure, a scaling study over that architecture measures the rate at which
      it fails to.
- [ ] One corpus, prepared once, with both split digests recorded, large enough that the
      largest rung's token budget stays under ~2 epochs.
- [ ] The LR rule chosen and its sweep run at R0.
- [ ] Persistent checkpointing verified — R4 is ~21 h, which is many Colab sessions.
- [ ] The held-out rung nominated **in writing, before any rung runs**.
