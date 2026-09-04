# Run 003 (pointwise layer KD) vs Run 004-M (behavioral / residual-delta KD)

Concise comparison artifact derived **only** from the two measured `summary.json` /
`metrics.jsonl` files. No plots are fabricated; no benchmark numbers are invented.

## Setup parity

| field | Run 003 | Run 004-M | matched? |
|---|---|---|---|
| launch commit | `f5dd3f7…` (archived) | `05788d9d22b2aa189f305e6c5ae2711efad7c26a` | (later SHA, same trainer path) |
| teacher / revision | Qwen3.8-27B / `dbdc473…` | same | ✅ |
| teacher quantization | 4-bit NF4 | 4-bit NF4 | ✅ |
| student | `qwen38_19b_h5120_l48_moe`, 13,008,505,728 params | same checkpoint (`pilot001/transferred`) | ✅ |
| corpus sha256 | `e11ca38bb099fc89c2f74e96f5d2f1209def6a16f6a8432d4e9972acd50c100d` | **identical** | ✅ |
| corpus split | 455 seq → 433 train / 22 val, 700,000 tokens | **identical** | ✅ |
| sequence length / steps / batch / accum | 1536 / 128 / 1 / 1 | 1536 / 128 / 1 / 1 | ✅ |
| optimizer / LR / warmup / scheduler / wd | AdamW / 2e-4 / 10 / cosine / 0.0 | identical | ✅ |
| QLoRA r / α / dropout | 16 / 32 / 0.05 | 16 / 32 / 0.05 | ✅ |
| precision / seed | bf16 / 0 | bf16 / 0 | ✅ |
| KD temperature / top-k | 2.0 / 64 | 2.0 / 64 | ✅ |
| direction_weight / normalise / map / chunk_pairs | 1.0 / RMS / group / 4 | 1.0 / RMS / group / 4 | ✅ |
| GPU class | L40S | L40S | ✅ |
| **step-1 KD divergence / CE** | **7.31431 / 10.93218** | **7.31431 / 10.93218** | ✅ identical first batch |

One cosmetic non-match: `config.training.kd_weight` reads `1.0` (Run 003) vs `0.5`
(Run 004-M, kd_run default; the launcher cannot pass it). **No training effect** —
`distillation.kd_alpha` is forced to `1.0` for the `layer_kd` trainer path in both runs, so
the optimized objective and gradients are unaffected. The KD divergence is a `no_grad`
diagnostic in both.

## Intended difference (the only research variable)

| | Run 003 (pointwise) | Run 004-M (delta) |
|---|---|---|
| student target | `h_s[l+1]` | `h_s[l+1] − h_s[l]` |
| teacher target | `h_t[m(l)+1]` | `h_t[b] − h_t[a]` over assigned span `[a,b)` |
| removed teacher layers (8–11, 24–27, 36–39, 52–55) | unsupervised | absorbed into neighbouring student spans (span tiling covers all 64 exactly once) |
| supervised student layers | 48 | 48 |

## Training curves (every 16 steps)

| step | layer term R003 / R004-M | KD div R003 / R004-M | CE R003 / R004-M | top-1 agr R003 / R004-M | val R003 / R004-M |
|---:|---|---|---|---|---|
| 1 | 1.448 / 2.539 | 7.314 / 7.314 | 10.93 / 10.93 | 0.001 / 0.001 | — |
| 16 | 0.952 / 2.203 | 3.915 / 4.250 | 8.50 / 8.37 | 0.091 / 0.089 | — |
| 32 | 0.820 / 1.879 | 4.107 / 4.003 | 11.74 / 12.06 | 0.076 / 0.096 | 11.88 / 12.27 |
| 48 | 0.845 / 1.840 | 7.533 / 9.237 | 12.18 / 14.76 | 0.093 / 0.095 | — |
| 64 | 0.661 / 1.683 | 4.274 / 5.360 | 10.73 / 12.97 | 0.145 / 0.134 | 10.55 / 13.14 |
| 80 | 0.740 / 1.665 | 6.139 / 7.677 | 9.65 / 12.48 | 0.116 / 0.096 | — |
| 96 | 0.711 / 1.622 | 5.407 / 6.959 | 8.88 / 11.98 | 0.140 / 0.103 | 8.77 / 11.68 |
| 112 | 0.623 / 1.631 | 3.822 / 4.486 | 9.01 / 11.43 | 0.184 / 0.139 | — |
| 128 | 0.623 / 1.625 | 3.981 / 4.698 | 9.40 / 11.65 | 0.162 / 0.122 | 8.96 / 11.58 |

Shape: through ~step 32 the two arms track closely on KD divergence and top-1 agreement
(delta briefly ahead on agreement at step 32). From ~step 48 the pointwise arm pulls ahead
on **every** output-level metric and the gap persists to step 128.

## Final metrics

| metric | Run 003 (pointwise) | Run 004-M (delta) | favours |
|---|---:|---:|---|
| own objective term, first → final | 1.448 → **0.623** (−57%) | 2.539 → **1.625** (−36%) | — (different objectives) |
| — magnitude / direction (final) | 0.415 / 0.208 | 1.083 / 0.542 | — |
| KD divergence (teacher KL), final | **3.981** | 4.698 | pointwise (−15%) |
| KD divergence, mean over run | **5.548** | 6.445 | pointwise |
| CE diagnostic, first → final | 10.93 → **9.40** (↓) | 10.93 → **11.65** (↑) | pointwise |
| top-1 teacher agreement, final | **0.162** | 0.122 | pointwise (+33% rel) |
| top-1 agreement, mean | **0.112** | 0.100 | pointwise |
| validation loss, final | **8.960** | 11.581 | pointwise (−2.62) |
| validation loss, best | **8.771** (step 96) | 11.581 (step 128) | pointwise |
| validation trajectory | 11.88→10.55→8.77→8.96 (descends) | 12.27→13.14→11.68→11.58 (spike, ~flat) | pointwise |
| layer_norm_ratio, final | 0.617 | 0.683 (peak 0.85 @ step 48) | — (delta over-drives magnitude) |

## Systems

| | Run 003 | Run 004-M |
|---|---:|---:|
| peak allocated VRAM | 38.9455 GiB | **38.9455 GiB** (identical) |
| peak reserved VRAM | 40.7656 GiB | **40.7656 GiB** (identical) |
| VRAM guard | n/a | 45 GiB ceiling, 1193 samples, max 41.284 GiB, **0 breaches** |
| throughput | 179.0 tok/s | 197.6 tok/s (+10%, plausibly run-to-run variance) |
| runtime | 1098.3 s | 994.8 s |
| exit code | 0 | 0 |
| stability | stable | stable (validation noisier; step-64 spike to 13.14; no NaN/OOM) |

## Interpretation

**Behavioral/delta KD is trainable** — its objective falls 36 %, `layer_norm_ratio` rises
toward 1, and top-1 agreement climbs from 0 to 0.12. But under matched conditions at this
budget it is **consistently inferior to pointwise layer KD on every output-level and
held-out metric**: teacher KL (4.70 vs 3.98), CE (rose to 11.65 vs fell to 9.40), top-1
agreement (0.122 vs 0.162), validation (11.58 vs 8.96, with no genuine learning curve).
There is no compensating advantage — VRAM is identical and the throughput edge is small and
within plausible variance.

The failure mode is legible: matching residual *contributions* (`Δh`) leaves the *absolute*
residual stream unconstrained, so the student's output distribution drifts — which is
exactly what the rising CE and flat validation show, while the delta objective itself keeps
falling. Pointwise matching constrains absolute hidden states directly, which more tightly
couples to the output.

## Classification: **NEGATIVE** (128-step matched scale, single seed)

Per the campaign's evidence gate, `run004_behavioral_scale_1024` requires a matched result
that is *positive or non-inferior*. This result is inferior, so **the scale-up gate is not
cleared**. The indicated next step is the composite-objective follow-up
(behavioral + ground-truth CE + logit KD), which directly targets the observed drift — not
a pure-behavioral scale-up.

## Limitations

- Single run, single seed (0); no variance estimate.
- 128 steps / 196,608 tokens is the mechanistic regime — says nothing about capability or
  final-model quality.
- Tests **one** behavioral formulation (residual-delta, RMS-normalised,
  `direction_weight=1.0`, group span mapping). Attention-behavior matching, state-transition
  matching, alternative span assignments, and the unnormalised variant are untested.
- QLoRA rank 16 is a narrow adapter; full-parameter behavior could differ.
- Does **not** disprove RQ1 in general — it is one controlled negative data point that
  redirects the search toward composite objectives.
