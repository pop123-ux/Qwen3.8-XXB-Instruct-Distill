# Level 2, second attempt — trains correctly, interrupted at ~step 500

**Status: INTERRUPTED, not failed and not finished.** The model was training. The
machine went away.

| | |
|---|---|
| Date | 2026-08-24 |
| Hardware | Tesla T4, 14.56 GiB, CC 7.5, fp16 (no bf16) |
| Config | `configs/experiments/t4_level2_100m_ckpt.yaml` |
| Model | 94.48M, 16 layers = 12 DeltaNet + 4 full attention |
| Run | seq 1024, micro-batch 4, accum 4, fp16 autocast, checkpointing ON |
| Memory | **no CUDA OOM** — the revised config fits |
| Throughput | ~2100 tokens/s |
| Reached | ~step 500 of 2000 configured |
| Termination | Colab runtime disconnected; ephemeral `/content` reclaimed |

## What was measured

| step | validation loss | validation BPB |
|---:|---:|---:|
| 200 | 0.9130 | 1.317 |
| 400 | 0.8868 | 1.279 |

For scale: 8.0 BPB is a model that has learned nothing (uniform over 256 bytes). At
step 400 this model is at 1.279 and still falling.

## What this proves — and what it does not

**Proves:** the revised 94.48M configuration trains on a T4 without OOM at ~2100
tokens/s, and the architecture is learning real structure from bytes.

**Does not prove:** final model quality. The run covered roughly a quarter of the
configured 2000 steps. It is not a completed experiment and is not reported as one.
Peak VRAM was also never captured, so the 3.57 GiB estimate remains unvalidated against
a measurement.

## Why ~500 steps could not be recovered

Four separate reasons, all now fixed:

1. **Checkpoints were written in place.** A disconnect during a write left a directory
   that existed, looked plausible and could not be loaded.
2. **A checkpoint did not contain enough to resume.** It held weights, optimizer state
   and a step number — but no LR scheduler, no GradScaler, no RNG state and no data
   position. "Resuming" would have restarted the one-cycle schedule from its warmup,
   reset the loss scale and rewound the data to epoch 0.
3. **Nothing was copied off the ephemeral filesystem.** `/content` is staging, not
   storage.
4. **No pointer said which checkpoint was newest and valid.**

See [`docs/experiments/t4_level2_resumability.md`](../../../docs/experiments/t4_level2_resumability.md)
for the design that replaced this, and `RESULT.json` for the machine-readable record.

## Re-running

The next attempt starts from step 0 — these 500 steps are gone. It will not happen
again: checkpoints are now atomic, complete, and persistable to Drive.

```bash
python scripts/train_student.py --config configs/experiments/t4_level2_100m_ckpt.yaml
```
