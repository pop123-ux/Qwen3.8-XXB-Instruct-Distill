# Pareto evaluation and the 16 GB constraint

Code: `src/qwen_distill/research/memory.py`. Tests: `tests/test_research_memory.py`.
Report: `python scripts/student_report.py --section memory`.

## The constraint

16 GB, **end to end, fully GPU-resident**. The complete workload:

```
weights + KV cache + DeltaNet recurrent state + conv state + activations + runtime overhead
```

A quantisation table showing 11 GiB of weights and stopping there has answered a question
nobody deploys against. Two terms are frequently omitted and are the reason "it fits" turns
into an OOM: the ~0.9 GiB a PyTorch+CUDA process costs before a single model tensor is
allocated, and the fp32 logits over a 248,320-token vocabulary.

**The ceiling is not 16.0 GiB.** A 16 GB card reports **14.56 GiB** (measured on the
Level-2 T4), and a process that must coexist with a display server or another CUDA context
should not plan on the last gigabyte. Usable: **13.56 GiB**.

**CPU offload is never used to claim compliance.** An offloaded configuration has different
latency and is a different product. If one ships, it is reported separately and labelled.

## The three terms the hybrid layout changes

**KV cache — 12 layers, not 48.** `2 (K and V) x 2 kv heads x 256 head_dim x 12 layers` =
**24,576 bytes per token** in fp16. A dense 48-layer model with the same head configuration
would cost 98,304 bytes/token. At the full 262,144-token window that is 6.00 GiB instead of
24 GiB, and it is the hybrid layout's entire deployment argument.

**Recurrent state — constant.** `36 layers x 48 value heads x 128 x 128` in fp32 =
**108 MiB**, identical at 2K and at 262K. Shapes verified against a running model of this
architecture family, not assumed, and confirmed unchanged between a 32-token and a
128-token forward pass.

**Conv state — 5.6 MiB.** `(2 x key_dim + value_dim) x kernel` per DeltaNet layer, using the
runtime's own `conv_dim` formula, confirmed against the built module.

All state together is smaller than a single octave of KV cache at 8K.

## Weights are bucketed, not uniform

Routed experts are 61.57% of the model and each token touches 2 of 24, so `expert_quant` is
a separate knob from `dense_quant` (attention, DeltaNet, router, norms — the always-active
path) and `embedding_quant`. Quantising the experts harder moves far more memory per unit of
damage than a uniform setting. A uniform table would hide the one lever that matters most.

Weight counts come from the parameter audit rather than being recomputed, so the memory
table and the parameter table cannot disagree.

## The result

Q4 / Q5 / Q6, embeddings at Q6, batch 1, fully GPU-resident, against 13.56 GiB usable:

| quant | context | weights | KV | state | acts | runtime | **total** | headroom | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Q4 | 2,048 | 13.09 | 0.05 | 0.111 | 0.28 | 0.90 | **14.44** | −0.88 | DOES NOT FIT |
| Q4 | 32,768 | 13.09 | 0.75 | 0.111 | 0.28 | 0.90 | **15.14** | −1.58 | DOES NOT FIT |
| Q4 | 262,144 | 13.09 | 6.00 | 0.111 | 0.28 | 0.90 | **20.39** | −6.83 | DOES NOT FIT |
| Q5 | 2,048 | 14.91 | 0.05 | 0.111 | 0.28 | 0.90 | **16.26** | −2.70 | DOES NOT FIT |
| Q6 | 2,048 | 16.96 | 0.05 | 0.111 | 0.28 | 0.90 | **18.30** | −4.74 | DOES NOT FIT |

With embeddings also at Q4 — the genuinely cheapest all-Q4 configuration — 2,048 tokens
costs **13.93 GiB** against 13.56 usable. **Shortfall: 0.37 GiB.**

**The frozen 22.07B student does not fit a real 16 GB card at any release precision and any
context length**, including 2,048 tokens, before long context is a factor.

### The arithmetic that hides it

Planning against the nominal 16.0 GiB instead of the card's reported 14.56 GiB makes Q4
appear to reach **65,536 tokens**. That entire difference is the card's own overhead and the
gigabyte left for the rest of the system. It is the exact arithmetic that produces an
out-of-memory error on hardware that obviously had room, and a test pins both numbers side
by side.

## What does fit

46 quantisation combinations fit at some context, and **none of them uses only the release
precisions**. Fits begin one step below, at 3-bit experts:

| experts | dense | embeddings | longest context |
|---|---|---|---:|
| `q3_k_m` | `q3_k_m` | `q5_k_m` | 65,536 |
| `q3_k_m` | `q4_k_m` | `q3_k_m` | 65,536 |
| `q3_k_m` | `q4_k_m` | `q4_k_m` | 32,768 |
| `int4` | `q4_k_m` | `q3_k_m` | 16,384 |

`frontier()` produces the full list. This is the Pareto view the release decision needs: the
axes are precision kept and context reached against a fixed ceiling, and the frontier is
where neither improves without the other getting worse.

## The open decision

Two honest options. The choice belongs to whoever owns the release; the repository reports
the constraint rather than quietly re-scoping the target.

1. **Ship the 22.07B target at 3-bit experts** and report the quality cost. Reaches 64K
   context. The quality cost of 3-bit MoE experts is not yet measured here and must not be
   assumed small.
2. **Reduce the expert budget.** Experts are 61.6% of the weights and each token uses 2 of
   24, so expert count and expert width are the only levers with enough mass to close
   0.37 GiB without touching the path every token depends on. Halving the routed-expert
   count clears the ceiling at Q4 with headroom — checked arithmetically in
   `test_a_smaller_expert_budget_is_what_would_close_the_gap`.

Nothing here alters the frozen architecture. It reports what the frozen architecture costs.

## Competitor comparison

`analysis/competition.py` holds third-party figures with provenance. The measured fact worth
repeating: of the three named comparison models, **only Qwen3.5-9B actually fits 16 GB**
(7.56 GiB at 32K). Qwen3-14B needs 14.43 GiB and Gemma-3-27B 19.31 GiB.

Two rules apply to that table and are enforced by the ledger:

- **Competitor results are never fabricated.** Every third-party number carries a source; an
  entry without one is refused.
- **No claim is made before benchmarking.** Qwen3.5-9B's reported figures — MMLU-Pro 82.5,
  GPQA Diamond 81.7, IFEval 91.5, LiveCodeBench v6 65.6, LongBench v2 55.2, BFCL v4 66.1,
  TAU2-Bench 79.1 — are the target to beat. They are not evidence about our model, and this
  project holds no benchmark results of its own yet.

## The Pareto frontier the release optimises

Four axes, not one:

1. **benchmark capability** — not yet measured;
2. **VRAM** — measured analytically above;
3. **inference speed** — not yet measured; the sparse FFN and the 12-layer KV cache both
   argue for it, and neither argument is evidence;
4. **effective context capability** — defined and schematised in
   [CONTEXT_SPECIALIZATION.md](CONTEXT_SPECIALIZATION.md), not yet measured.

Only axis 2 has numbers today. Any statement ranking this model against another on axes 1,
3 or 4 would be unsupported.
