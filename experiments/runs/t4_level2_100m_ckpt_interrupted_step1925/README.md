# Level 2, third attempt — reached step 1925 of 2000, then lost everything

**Status: INTERRUPTED at ~96% complete. Nothing recovered.** Not a failure of the model,
the memory model, or the checkpoint code. A configuration default.

| | |
|---|---|
| Hardware | Tesla T4, 14.56 GiB |
| Config | `configs/experiments/t4_level2_100m_ckpt.yaml` |
| Model | 94.48M, 16 layers = 12 DeltaNet + 4 full attention |
| Run | seq 1024, micro-batch 4, accum 4, fp16, checkpointing ON |
| Memory | **no CUDA OOM** |
| Throughput | ~2050 tokens/s |
| Reached | **step 1925 of 2000** |
| Recovered | **nothing** |

## What went wrong

The run reported, correctly and truthfully:

```
persistent copy : off (local only)
```

`training.persistent_backup` was `null`, so every checkpoint was written — atomically,
completely, verifiably — to `/content`, which Colab then reclaimed. The checkpoint
machinery did exactly what it was built to do. It was pointed at a filesystem with a
lifetime shorter than the run.

Two earlier Level-2 attempts lost ~500 steps to a disconnect and one lost the whole run
to an OOM. This one is the most expensive, because it was 96% of the way to a real
result.

## What changed because of it

1. **Persistence is now checked at startup, not at the first checkpoint.** If
   `persistent_backup` points at an unmounted Drive, the run refuses to start. Writing to
   an unmounted `/content/drive/MyDrive/...` silently creates an ordinary local
   directory — every checkpoint would report "persisted" and vanish anyway.
2. **A local-only run now says so loudly**, and `summary.json` records
   `persistence.all_checkpoints_persisted`, so "was this safe?" is answerable from the
   artifact rather than from memory.
3. **A failed copy is never reported as success** and never advances the persistent
   pointer.
4. `--status` reads persistent storage, and `--restore` brings a run back, so a fresh
   Colab session can pick up where a dead one stopped.

## Re-running

These 1925 steps are gone; the next attempt starts from step 0. See
[`docs/experiments/t4_level2_resumability.md`](../../../docs/experiments/t4_level2_resumability.md)
for the enabled-persistence commands.
