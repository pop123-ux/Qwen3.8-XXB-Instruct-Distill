# Deployment Plan

**Status: target definition and methodology. All VRAM figures below are analytical
estimates from `scripts/estimate_vram.py`. Nothing has been measured on real hardware.**

## What "fits in 16 GB" means here

Not this:

```
weight file < 16 GB  ->  "runs on a 16 GB card"
```

This:

```
weights
+ KV cache (full-attention layers only)
+ recurrent state + conv state (DeltaNet layers, constant in context)
+ activations / workspace
+ framework and CUDA runtime overhead
= peak VRAM
```

measured at a **realistic context with realistic generation**, leaving the card usable.

### Budget

| Item | Allowance |
|---|---|
| Card | 16 GiB |
| Reserved (driver, desktop compositor, other applications) | 1.0 GiB |
| **Usable** | **15.0 GiB** |

The 1 GiB reserve is deliberate. A model that fits only on a headless card with
nothing else running is not what a consumer-hardware project should ship.

## Target hardware

| GPU | VRAM | Bandwidth (approx) | Notes |
|---|---|---|---|
| RTX 3060 Ti 16GB class | 16 GB | ~448 GB/s | widely owned |
| RTX 5070 16GB class | 16 GB | ~672 GB/s | current generation |
| NVIDIA T4 | 16 GB | ~320 GB/s | free-tier cloud; no BF16, slower |

These are **not interchangeable**. The T4 in particular lacks BF16 and has roughly
half the bandwidth of the 5070, so it gets its own measured row — never an
extrapolation.

## Why bandwidth, not FLOPs, sets the speed

Single-stream decode reads essentially every weight once per token. Throughput is
therefore capped near:

```
tokens/sec ≈ bandwidth × efficiency / bytes_per_token
```

with efficiency ~0.7–0.85 for well-tuned dequantised GEMV kernels. This is a ceiling,
not a prediction — it ignores kernel launch overhead, sampling, and the cache reads
that grow with context.

Practical consequence: **quantization buys speed as directly as it buys memory.**
Halving bytes-per-weight roughly doubles the throughput ceiling.

## The teacher, for reference

`python scripts/estimate_vram.py --preset teacher --matrix`

Peak VRAM (GiB), batch 1, fp16 KV cache, against a 15.0 GiB budget:

| quant | 8k | 32k | 64k | 128k | 256k |
|---|---:|---:|---:|---:|---:|
| bf16 | 49.2 | 50.7 | 52.7 | 56.7 | 64.7 |
| int8 | 26.5 | 28.0 | 30.0 | 34.0 | 42.0 |
| q6_k | 22.5 | 24.0 | 26.0 | 30.0 | 38.0 |
| q5_k_m | 20.0 | 21.5 | 23.5 | 27.5 | 35.5 |
| q4_k_m | 17.7 | 19.2 | 21.2 | 25.2 | 33.2 |

Nothing fits. At Q4_K_M the weights alone are 15.85 GiB — the model is over budget
before a single token of context. This is the project's premise, quantified.

## Deployment matrix (to be filled by measurement)

Published only for rows actually measured. Estimated rows are labelled as such and
kept visually distinct.

| GPU | Quantization | Context | Peak VRAM | tok/s | TTFT | Status |
|---|---|---|---|---|---|---|
| RTX 3060 Ti 16GB | 4-bit | 8k | TBD | TBD | TBD | Not yet measured |
| RTX 3060 Ti 16GB | 4-bit | 32k | TBD | TBD | TBD | Not yet measured |
| RTX 3060 Ti 16GB | 4-bit | 64k | TBD | TBD | TBD | Not yet measured |
| RTX 5070 16GB | 4-bit | 8k | TBD | TBD | TBD | Not yet measured |
| RTX 5070 16GB | 4-bit | 32k | TBD | TBD | TBD | Not yet measured |
| RTX 5070 16GB | 4-bit | 64k | TBD | TBD | TBD | Not yet measured |
| T4 16GB | 4-bit | 8k | TBD | TBD | TBD | Not yet measured |

## Measurement methodology

For each row, record: peak VRAM (`torch.cuda.max_memory_allocated` **and**
`nvidia-smi` peak, since allocator reserve differs from process usage), prompt
processing speed, generation speed, time to first token, context actually achieved,
concurrency, quantization, KV/recurrent-state dtype, GPU utilisation, engine and
version, driver version, date, git commit.

Run to a **realistic generation length**, not one token. Peak VRAM commonly occurs
during prefill of a long prompt, not during decode.

## Backend support — verify before assuming

Gated DeltaNet needs a recurrent-state cache, which is not standard KV caching. For
each backend (Transformers, llama.cpp, vLLM, others):

1. Confirm the hybrid architecture is supported at all.
2. Confirm the **recurrent state** is handled correctly, not silently ignored.
3. Validate long-context retrieval against the Transformers reference path.

A backend that loads the model and produces plausible short-context output may still
be wrong at long context. That failure is quiet, so it must be tested for explicitly.

**MTP:** the teacher checkpoint ships `mtp.*` weights and `transformers` discards them
(verified — see `VERIFICATION.md`). They are used by speculative-decoding-capable
engines. Whether the student should carry an MTP head depends on whether our target
backends can use it; this needs an ablation, not an assumption.

## Quantization strategy

Architectural compression should let us ship a **good 4-bit model rather than a
desperate 2-bit one**. Sweep BF16 → FP8 → INT8 → 6/5/4-bit, measuring perplexity,
benchmarks, VRAM, throughput and stability.

Two specifics for this architecture:

- **Embeddings and LM head deserve separate treatment.** With a 248,320-token
  vocabulary they are 2.54B parameters at hidden 5120. Common practice keeps them at
  higher precision; our estimator models this explicitly (`--embedding-quant`).
- **Long context must be evaluated separately after quantization.** Quantization error
  may affect retrieval over long contexts more than short-context fluency, and a
  perplexity check will not reveal it.

## Release artifacts

- BF16 reference weights
- 4-bit quantized weights for the primary consumer target
- GGUF conversions where the architecture is supported
- Tokenizer, config, generation config
- Model card: measured benchmarks, measured hardware table, known limitations
- One-command inference examples per supported backend
- Full training and quantization recipe

A user with a 16 GB card should be able to run the model without reading any of the
research code.
