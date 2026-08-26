# Level 2, first attempt — CUDA OOM

**Status: FAILED. Kept deliberately.** This is a real experimental result, not a mistake
to be tidied away: it measures what a 94.48M hybrid model actually costs on a Tesla T4
without gradient checkpointing, under `transformers`' reference DeltaNet kernel.

| | |
|---|---|
| Date | 2026-08-24 |
| Hardware | Tesla T4, 14.56 GiB, CC 7.5, fp16 (no bf16) |
| Config | `configs/experiments/t4_level2_100m.yaml` (unchanged) |
| Model | 94.48M, 16 layers = 12 DeltaNet + 4 full attention |
| Run | seq 1024, micro-batch 8, accum 2, fp16 autocast, AdamW, checkpointing OFF |
| Outcome | OOM in the forward pass, **0 steps completed** |
| Predicted | 4.53 GiB |
| Actually demanded | ~24.8 GiB |

## The traceback pointed at the wrong thing

The exception was raised inside `torch.nn.functional.scaled_dot_product_attention`, and
that is misleading. SDPA receives `attn_mask=None` and `is_causal=True`, so it never
materialises a `batch x heads x seq x seq` matrix; all four attention layers together
retain **86 tensors, 110 MiB at batch=1**. SDPA was simply the next sizeable allocation
after the DeltaNet layers had already eaten the card.

## Root cause

The estimator did not model Gated DeltaNet activations **at all**. It treated all 16
layers as generic transformer blocks scaling with `hidden_size` and `intermediate_size`.

Measured by hooking autograd's saved tensors and attributing each to its creating module:

| scope | retained (batch=1, seq=1024) | tensors |
|---|---:|---:|
| **DeltaNet mixers** | **2155.6 MiB** | **5244** |
| MLP (DeltaNet layers) | 629.2 MiB | 96 |
| MLP (attention layers) | 209.8 MiB | 32 |
| embedding / norms / logits | 169.4 MiB | 138 |
| attention mixers | 109.7 MiB | 86 |
| **total** | **3273.6 MiB** | 5596 |

Two properties of `torch_chunk_gated_delta_rule`, the pure-PyTorch fallback that runs
when `fla` is not installed, explain it:

1. **It force-upcasts to fp32.** The function opens by casting q, k, v, beta and g to
   `torch.float32`, so 12 of 16 layers ignore fp16 autocast and retain 4-byte
   activations.

2. **A 63-iteration Python loop retains O(chunk²) clones per chunk.** Inside
   `for i in range(1, chunk_size)` it does `sub = attn[..., :i, :i].clone()` and
   multiplies by it, so autograd saves every one:

   ```
   sum(i² for i in 1..63) = 85,344 elements per (chunk, head)
   -> v_heads * 85,344 * 4 / chunk_size = 64,008 bytes per token per layer
   ```

   That one term is larger than the estimator's entire previous activation budget.

Scaling is linear in batch — measured 3273.6 MiB at batch=1 and 6186.8 MiB at batch=2,
so `360 + 2913 x batch` MiB. At batch=8: **23.1 GiB of activations**, plus 1.41 GiB of
weights, gradients and optimizer state, against a 14.56 GiB card.

## Reproducing the diagnosis

No GPU required — tensor shapes and dtypes do not depend on the device:

```bash
python scripts/probe_activations.py --config configs/experiments/t4_level2_100m.yaml --batch-size 1
python scripts/probe_activations.py --config configs/experiments/t4_level2_100m.yaml --batch-size 2
```

## What replaced it

`configs/experiments/t4_level2_100m_ckpt.yaml` — identical architecture, identical
sequence length, identical effective batch. See that file's header for each change and
its justification.
