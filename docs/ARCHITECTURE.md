# Architecture Analysis

Where the teacher's parameters, memory and compute actually go — and therefore which
levers matter. Every number here is reproducible with the scripts in this repo; none
is a benchmark measurement.

Provenance for all structural claims: [`VERIFICATION.md`](VERIFICATION.md).

## The teacher

Qwen3.8-27B is a **hybrid** model: it interleaves Gated DeltaNet (linear attention,
constant-size recurrent state) with gated full attention, at a 3:1 ratio.

```
64 layers = 16 repeats of [ DeltaNet+FFN, DeltaNet+FFN, DeltaNet+FFN, Attention+FFN ]
                            ^-- 48 linear layers --------------------^  ^-- 16 full --^
```

The full-attention layer is the **last** layer of each group, a consequence of
upstream's expansion rule `"linear_attention" if bool((i+1) % 4) else "full_attention"`.

Two details are easy to miss and both are material:

- **The attention is gated.** `q_proj` emits `n_heads * head_dim * 2`; the second half
  becomes a sigmoid gate applied to the attention output. This doubles the query
  projection's parameters.
- **The DeltaNet has its own gate.** `in_proj_z` is a full `hidden -> value_dim`
  projection on top of `in_proj_qkv`, plus scalar-per-head `in_proj_b` / `in_proj_a`.

Omitting either understates a 27B model by more than a billion parameters.

## Where the parameters are

`python scripts/estimate_vram.py --preset teacher`

| Component | Parameters | Share |
|---|---:|---:|
| MLP (SwiGLU, all 64 layers) | 17.11B | **63.6%** |
| Gated DeltaNet (48 layers) | 5.56B | 20.7% |
| Embedding | 1.27B | 4.7% |
| LM head (untied) | 1.27B | 4.7% |
| Gated attention (16 layers) | 1.68B | 6.2% |
| Norms | ~0.66M | ~0.0% |
| **Total** | **26.90B** | |

### Consequences for compression

1. **The FFN is the lever.** Nearly two thirds of the model is
   `3 x hidden x intermediate` repeated 64 times. No other single change moves the
   parameter count comparably. The teacher's FFN multiplier is 17408/5120 = **3.4x**,
   which is generous; 2.5–3.0x is common and is the first thing to test.
2. **The vocabulary is a fixed tax, and it gets worse as the model shrinks.** At
   248,320 tokens, embedding + head is 2.54B parameters — 9.4% of the teacher, but
   **over 13% of a 12B student** at the same hidden size. Tying input and output
   embeddings removes half of it outright, which is why tied variants dominate the
   search rankings.
3. **Full attention is cheap in parameters** (6.2%) but expensive in *cache*. Cutting
   attention layers buys context, not weight size.
4. **DeltaNet is not free.** At 20.7% it is over three times the parameter cost of the
   attention layers, because 48 layers each carry a `hidden -> conv_dim` projection
   with `conv_dim = 10240` (twice hidden).

## Where the memory goes

The decisive structural property: **only the 16 full-attention layers hold a cache
that grows with context.** The 48 DeltaNet layers hold a fixed
`(num_v_heads, head_k_dim, head_v_dim)` recurrent state plus a tiny conv state,
independent of sequence length.

For the teacher, batch 1, fp16 KV:

| Context | KV cache | Recurrent + conv state |
|---:|---:|---:|
| 8k | 0.50 GiB | 0.148 GiB |
| 32k | 2.00 GiB | 0.148 GiB |
| 128k | 8.00 GiB | 0.148 GiB |
| 256k | 16.00 GiB | 0.148 GiB |

A conventional 64-layer dense model with the same head configuration would need 4x
the KV cache. **This architecture has already solved most of the long-context memory
problem.**

### Which is why the binding constraint is weights, not context

`python scripts/estimate_vram.py --preset teacher --matrix`

Peak VRAM (GiB), 16 GiB card with 1 GiB reserved → 15.0 GiB usable:

| quant | 8k | 32k | 64k | 128k | 256k |
|---|---:|---:|---:|---:|---:|
| bf16 | 49.2 | 50.7 | 52.7 | 56.7 | 64.7 |
| int8 | 26.5 | 28.0 | 30.0 | 34.0 | 42.0 |
| q6_k | 22.5 | 24.0 | 26.0 | 30.0 | 38.0 |
| q5_k_m | 20.0 | 21.5 | 23.5 | 27.5 | 35.5 |
| q4_k_m | 17.7 | 19.2 | 21.2 | 25.2 | 33.2 |

**No cell fits.** At Q4_K_M the weights alone are 15.85 GiB. Even at 8k context — and
even ignoring the KV cache entirely — the teacher does not fit a 16 GB card.

This reframes the project. The goal is not "make long context affordable"; the hybrid
architecture already did that. The goal is **to remove roughly 8–11B parameters while
retaining capability**, and the FFN is where they are.

## Compute profile

Decode FLOPs per token at 32k context, teacher:

| Component | FLOPs | Share |
|---|---:|---:|
| MLP | 34.23G | 53% |
| Attention scores (context-dependent) | 12.88G | 20% |
| DeltaNet projections | 11.12G | 17% |
| Attention projections | 3.36G | 5% |
| LM head | 2.54G | 4% |
| DeltaNet state update | 0.30G | 0.5% |
| **Total** | **64.43G** | |

Only the attention-score term grows with context, and only across 16 layers.

**For single-stream decode on a consumer GPU, bandwidth binds before FLOPs.** Every
weight is read once per token, so throughput is capped near
`bandwidth x efficiency / bytes_per_token`. For the teacher at Q4_K_M:

| GPU class | Bandwidth | Ceiling |
|---|---:|---:|
| T4 | ~320 GB/s | ~14.6 tok/s |
| RTX 3060 Ti | ~448 GB/s | ~20.4 tok/s |
| RTX 5070 | ~672 GB/s | ~30.6 tok/s |

These are ceilings, not predictions, and they assume the model fits — which it does not.

## The feasible frontier

`python scripts/search_architectures.py --context <ctx>`

Ranked by non-embedding parameters (a **capacity proxy**, not measured capability),
subject to fitting 15.0 GiB usable at Q4_K_M:

| Required context | Largest feasible | Example architecture |
|---|---|---|
| 32k | ~21B total / 19.9B non-emb | hidden 5120, 64 layers, FFN 12800, interval 6, tied |
| 128k | ~16.5B total / 15.3B non-emb | hidden 5120, 40 layers, FFN 17408, interval 6, tied |
| 262k | ~13.5B total / 12.2B non-emb | hidden 5120, 32 layers, FFN 17408, interval 6, tied |

**This is the project's first substantive finding, and it contradicts the obvious
default.** The answer to "how big should the student be?" is not 7B or 10B. Under a
16 GB envelope the feasible ceiling is in the **13–21B** range, depending entirely on
how much context is demanded. Going from 32k to 262k costs roughly **7.7B parameters**
of capacity — that is the price of context, quantified.

### Two warnings about this table

- **The ranking objective is a proxy.** Non-embedding parameter count correlates with
  capacity but is not capability. Two architectures with identical parameter counts
  can differ substantially after training.
- **The ranking is biased toward `interval 6`, and that bias is suspicious.** Fewer
  full-attention layers means less KV cache, which frees budget for parameters, so the
  optimiser reaches for it. But full attention is what provides exact long-range
  recall. A 5:1 layout may well retrieve worse at 128k than a 3:1 layout with fewer
  parameters — and the capacity proxy cannot see that. **Do not adopt interval 6
  without running the long-context retrieval ablation** (`docs/EVALUATION_PLAN.md`,
  Experiment F). This is exactly the failure mode the mandate warns about: optimising
  a proxy into a real regression.

## Candidate levers, ranked by expected effect

| Lever | Parameter effect | Risk |
|---|---|---|
| Reduce FFN multiplier 3.4x → 2.5–3.0x | very large | moderate; FFN capacity is where knowledge lives |
| Tie input/output embeddings | −1.27B at hidden 5120 | low; standard in small models, slight quality cost |
| Reduce layer count | large, linear | high; depth drives reasoning |
| Reduce hidden size | large, quadratic in FFN | high; affects every component |
| Increase attention interval 4 → 6 | small parameter gain, large KV gain | **high for long-context recall** |
| Reduce DeltaNet value heads 48 → 32 | moderate | unknown; untested in this family |
| Shrink vocabulary | up to 2.54B | high; breaks tokenizer compatibility with the teacher, complicates distillation |

Vocabulary reduction deserves a note: it is tempting (9.4% of the teacher) but it
**breaks logit distillation**, because teacher and student would no longer share an
output space. Keeping the teacher's tokenizer is close to a hard requirement for the
distillation strategy in `docs/PROJECT_PLAN.md`. Tying embeddings gets half the
benefit with none of that cost.

## Open architectural questions

- Is the 3:1 DeltaNet:attention ratio optimal at 15B, or was it tuned for 27B?
- Does *where* the full-attention layers sit matter, beyond how many there are?
- Can DeltaNet value heads be reduced without hurting recall?
- Is a wider-and-shallower or narrower-and-deeper model better at fixed VRAM?
- Does MTP help a student, and does any inference engine we target support it?

None of these can be answered analytically. They are the content of
`experiments/architecture_search/`.
