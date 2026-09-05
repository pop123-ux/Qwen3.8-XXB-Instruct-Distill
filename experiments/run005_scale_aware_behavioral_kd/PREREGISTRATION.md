# RQ1 behavioral-formulation arms — PRE-REGISTERED 2026-09-05, before any training

## Analytical basis (derived from committed code + committed Run003/004-M metrics; no new runs)

`behavioral._pair_term(..., normalise=True)` RMS-normalises both tensors to unit RMS,
then computes:
    magnitude = MSE(a_hat, b_hat)
    direction = 1 - cos(a_hat, b_hat)
For unit-RMS vectors:  MSE = 2 - 2*cos = 2*(1 - cos) = 2*direction  IDENTICALLY.
Verified in committed logs: Run003 0.415307/0.207654 = 2.0000; Run004-M 1.084652/0.542333 = 2.0000.

=> The A3-norm objective is L = magnitude + 1.0*direction = 3*(1 - cos).
   It is PURE DIRECTION MATCHING and carries ZERO update-magnitude information.
   This is the specific weakness Run 004-M exposed.

## Arm B — "A3-scale" (scale-aware delta). LOCKED.

Per supervised pair p (48 pairs), per token t:
    d_s = h_s[l+1] - h_s[l]                     (student residual contribution)
    d_t = h_t[b]   - h_t[a]                     (teacher span contribution, [a,b) tiling, 64/64)
    RMS(x)   = sqrt(mean_over_hidden(x^2))
    a_hat = d_s / (RMS(d_s) + eps),  b_hat = d_t / (RMS(d_t) + eps),  eps = 1e-6

    magnitude_p = MSE(a_hat, b_hat)                                  [unchanged]
    direction_p = 1 - mean_t cos(a_hat, b_hat)                       [unchanged]
    S_p         = mean_t [ ( log(RMS(d_s)+eps) - log(RMS(d_t)+eps) )^2 ]   [NEW]

    L_B = mean_p [ magnitude_p + w_dir * direction_p + lambda_magnitude * S_p ]

    w_dir            = 1.0     (unchanged from Run003/Run004-M)
    lambda_magnitude = 1.0     (LOCKED a priori; see justification)

lambda_magnitude = 1.0 justification (principled default, NOT tuned, NOT swept):
  - Matches the existing direction_weight convention of 1.0 already frozen in the protocol.
  - S is dimensionless: a scale error of factor e gives S = 1.0, i.e. the same order as the
    observed direction term (~0.54). Committed Run004-M norm_ratio 0.694 => log(0.694)^2 ~ 0.13,
    so the new term is comparable to, and does not dominate, the direction term.
  - No alternative value will be tried this session.

Single independent variable vs Run 004-M: addition of the scale term S. Nothing else changes.
Deliberately NOT reintroducing absolute pointwise hidden-state matching: the arm stays a
pure transformation-matching objective, so the causal reading stays clean.

## Arm C — "A3-raw" (unnormalised delta). LOCKED.

Identical to Run 004-M except `layer_kd_normalise = False` (existing, already-frozen
repository flag `--layer-kd-no-normalise`). No code change whatsoever.
    L_C = mean_p [ MSE(d_s, d_t) + 1.0 * (1 - cos(d_s, d_t)) ]   on RAW deltas.
Tests whether the RMS normalisation itself caused the failure.

## Matched envelope (identical to Run003 and Run004-M, unchanged)
teacher Qwen/Qwen3.8-27B @ dbdc473dea0d6a9763042881cc33d6058d1742d2, 4bit;
student /workspace/runs/pilot001/transferred; corpus train.txt raw sha bc5972d9...;
packed 700k sha e11ca38b...; seq 1536; max_tokens 700000; steps 128; bs 1; accum 1;
lr 2e-4; AdamW; wd 0; warmup 10; cosine; bf16; grad-ckpt on; LoRA r16 a32 drop0.05;
T 2.0; top-k 64; tail bucket; group mapping; chunk_pairs 4; seed 0; log 1 / eval 32 / save 64.

## Decision rule (fixed BEFORE seeing results)
Primary endpoint: final validation loss, plus teacher/student top-1 agreement.
- If an arm materially closes the gap to Run003 (8.9599) or beats it -> scale that arm to 1024 if time allows.
- If clearly inferior -> do NOT tune lambda, do NOT sweep; report and move to the next locked arm.
Controls Run003 and Run004-M are immutable; nothing writes to their directories.
