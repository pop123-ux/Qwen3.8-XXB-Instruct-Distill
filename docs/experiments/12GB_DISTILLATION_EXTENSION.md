# Extending to a 12 GB GPU

The project's stated target is **16 GB** — a Colab T4, which reports **14.56 GiB**
(measured, Level 2). This asks what changes at 12 GB: an RTX 3060 12GB, an RTX 2060 12GB,
the cards people actually own.

**Short answer: everything this project has built so far fits, and the 500M class is where
it stops.** The binding constraint is not what most people assume.

---

## 1. The budget

| | 16 GB (T4) | 12 GB |
|---|---|---|
| nominal | 16 GB | 12 GB |
| reported total | **14.56 GiB** (measured) | ~11.76 GiB (vendor spec converted — **not measured here**) |
| reserved for driver/display | 1.0 | 1.0 |
| **usable** | **13.56 GiB** | **10.76 GiB** |

Two things worth stating plainly:

- **"12 GB" is 11.76 GiB.** Planning against 12 is how a configuration that "obviously
  fits" OOMs. The gap is 0.24 GiB — about a third of the DeltaNet activation budget at
  Level 2's shape.
- **The 1 GiB reserve is not padding.** A 3060 in a desktop is usually driving a display.
  A T4 in Colab is not. The reserve is doing more work at 12 GB than at 16.

The 12 GB figure is **vendor spec, not measured.** Verify on the actual card before
committing hours:

```bash
python scripts/hardware_info.py
python scripts/benchmark_memory.py            # measure, do not assume
```

---

## 2. What fits

Largest configuration the estimator calls PLAUSIBLE, on 10.76 GiB usable:

| model | measured params | best config on 12 GB | GiB | free | limited by |
|---|---|---|---|---|---|
| Level 2 | 94,476,448 | seq 2048 × b8, ckpt, AdamW | 6.46 | 4.30 | DeltaNet activations |
| 250M class | 236,237,488 | seq 2048 × b8, ckpt, **AdamW-8bit** | 9.22 | 1.54 | DeltaNet activations |
| 250M class | 269,106,460 | seq 2048 × b8, ckpt, **AdamW-8bit** | 9.60 | 1.16 | DeltaNet activations |
| 350M class | 354,224,648 | seq 2048 × b4, ckpt, AdamW | 9.51 | 1.25 | **optimizer state** |
| 350M class | 382,746,934 | seq 2048 × b4, ckpt, **AdamW-8bit** | 8.21 | 2.55 | DeltaNet activations |
| 500M class | 471,678,480 | seq 2048 × b4, ckpt, **AdamW-8bit** | 9.51 | 1.25 | DeltaNet activations |
| 500M class | 535,727,282 | seq 2048 × b2, ckpt, **AdamW-8bit** | 7.98 | 2.78 | base weights |

At **Level 2's own configuration** (seq 1024 × batch 4, checkpointing on, fp32 AdamW), the
picture is simpler:

| model | GiB | verdict |
|---|---|---|
| Level 2 94.5M | 3.57 | PLAUSIBLE |
| 236M | 6.18 | PLAUSIBLE |
| 269M | 6.68 | PLAUSIBLE |
| 354M | 8.00 | PLAUSIBLE |
| 383M | 8.63 | PLAUSIBLE |
| 472M | 10.19 | **TIGHT** |
| 536M | 10.98 | **NOT FEASIBLE** |

**~470M is the ceiling at 12 GB with fp32 AdamW.** Above it, 8-bit moments and a smaller
batch are required, and the headroom stops being comfortable.

Reproduce with:

```bash
python scripts/evaluate_scaling_candidates.py --devices 12
```

---

## 3. The binding constraint is not the weights

At 12 GB, for these models, memory goes:

| model | weights | optimizer state | DeltaNet activations |
|---|---|---|---|
| 94.5M | 0.35 | 0.70 | 0.80 |
| 236M | 0.88 | 1.76 | 1.20 |
| 354M | 1.32 | 2.64 | 1.20 |
| 472M | 1.76 | 3.51 | 1.60 |

(GiB, at seq 1024 × batch 4 with gradient checkpointing.)

**Optimizer state is 2× the weights and usually the largest single term.** fp32 AdamW holds
a master copy plus two moments: 12 bytes per parameter against 4 for the weights. That is
where the memory is.

**DeltaNet activations are the term that scales with your knobs, not your model.** They
depend on sequence length, batch size and value-head count — not on parameter count. This
is the term whose absence caused the Level-2 OOM, and at 12 GB it is what makes seq 2048 ×
batch 8 the difference between fitting and not.

Consequence: **shrinking the model is the wrong first move at 12 GB.** In order of effect:

1. **8-bit optimizer moments** — halves the largest term. `optimizer: adamw_8bit`. Costs a
   `bitsandbytes` dependency and a small amount of optimizer precision.
2. **Halve the micro-batch, double gradient accumulation** — identical effective batch,
   halved activations. Nearly free, slightly slower.
3. **Shorter sequences** — halving seq 2048 → 1024 halves activations. Changes what the
   model can learn, so it is a research decision, not a memory knob.
4. **Gradient checkpointing** — already on everywhere here, and non-negotiable: turning it
   off multiplies retained activations by roughly 67× for this architecture. That is what
   OOMed the first Level-2 attempt.
5. **A smaller model** — last, because it changes the experiment.

---

## 4. Time, not memory, is the real 12 GB constraint

A 3060 is roughly T4-class in throughput (both are memory-bandwidth-bound here; a 3060 is
somewhat faster, and this project has measured neither).

**The anchor has changed since this table was first written.** Level 2 ran at 2,089.2
tok/s run-wide; Level 2R ran the *same architecture on the same hardware* at **1,727.9
tok/s** — 20.9% more wall-clock for an identical 32,768,000 tokens. Nothing about the model
differed. The gap is real-corpus I/O, validation on a 4.4 MB held-out set, and ten verified
Drive persists. A real training run looks like Level 2R, not Level 2, so the rows below are
anchored on **1,727.9 tok/s** with the **UNVALIDATED** FLOP-ratio extrapolation:

| model | est. tok/s | 32.8M tokens | 100M tokens |
|---|---|---|---|
| 94.5M | 1,728 (measured, Level 2R) | 5.3 h | 16.1 h |
| 236M | ~704 | 12.9 h | 39.4 h |
| 354M | ~470 | 19.4 h | 59.1 h |
| 472M | ~355 | 25.6 h | 78.2 h |

**A 12 GB card fits a 472M model and needs ~78 hours to give it 100M tokens.** That is the
extension's actual limit. Memory says yes; a weekend says no.

The 236M row is the size Level 3 will actually run at on a 16 GB T4
([level3_plan.md](level3_plan.md)), so it is the first of these estimates that will be
checked against a measurement.

Unlike Colab, a local 12 GB card has no session limit — but it is also the machine you are
using. Plan for interruption either way; the checkpoint/resume/persistence stack already
handles it (`training/checkpoints.py`, `training/persist.py`).

---

## 5. Disk, which is easy to forget

fp32 AdamW checkpoints are 12 bytes per parameter:

| model | weights | optimizer | per checkpoint | 10 checkpoints |
|---|---|---|---|---|
| 94.5M | 378 MB | 756 MB | 1.13 GB | 11.3 GB |
| 236M | 945 MB | 1.9 GB | 2.83 GB | 28.3 GB |
| 354M | 1.4 GB | 2.8 GB | 4.25 GB | 42.5 GB |
| 472M | 1.9 GB | 3.8 GB | 5.66 GB | **56.6 GB** |

A 2000-step run at `save_every: 200` leaves ten checkpoints. At 472M that is 56.6 GB —
more than a free Drive tier. Raise `save_every`, prune old checkpoints, or budget the disk.
Note 8-bit moments shrink the checkpoint too, roughly a third off the total.

---

## 6. What changes in the config

Starting from `configs/experiments/t4_level2r_100m_real_english.yaml`, for a 12 GB card at
the 250M class:

```yaml
model:
  architecture:
    hidden_size: 1024          # was 640
    num_hidden_layers: 16      # unchanged
    intermediate_size: 3456    # 3.4 x hidden, per the shape rule
    num_attention_heads: 16
    num_key_value_heads: 2
    linear_num_key_heads: 6
    linear_num_value_heads: 18
    # everything else unchanged — head_dim 64, interval 4, vocab 256, tied embeddings

training:
  optimizer: adamw_8bit        # halves the largest memory term
  batch_size: 2                # was 4
  gradient_accumulation_steps: 8   # was 4 — effective batch 16, unchanged
  gradient_checkpointing: true # NON-NEGOTIABLE
  precision: fp16              # bf16 if the card is Ampere or later (a 3060 is)

runtime:
  reserved_vram_gib: 1.0       # a desktop card is usually driving a display
```

**A 3060 is Ampere and supports bf16; a T4 is Turing and does not.** bf16 avoids the loss
scaler entirely. It is a genuine improvement — and it means a 3060 run is **not** a
controlled comparison against a T4 run. Say which was used.

---

## 7. Distillation specifically

Everything above is about **training the student**. Distillation adds a second cost the
tables do not include: getting teacher outputs.

**The teacher does not fit. Not at 12 GB, not at 16, not close.** Qwen3.8-27B is
26,895,998,464 parameters — ~13.4 GB at int4, before any activation or KV cache. Teacher
generation happens elsewhere (a rented GPU, an API, a bigger machine) and its outputs are
transported as data. That is already how `distillation/generation.py` and the manifest
system are built.

The conceptual ladder is `teacher 27B -> best student for 16 GB -> a further
distilled/compressed model for 12 GB`. The middle rung is not settled: Level 2R
established that 94.48M learns real English structure while remaining repetitive, and
Level 3 ([level3_plan.md](level3_plan.md)) tests whether 236M materially improves on it.
**Until that returns, the 12 GB architecture cannot be chosen** — it would be a
compression target for a model this project has not yet selected. Nothing here claims one.

So the 12 GB extension is entirely about the student side, and the constraint that matters
there is **storage of teacher outputs**, not VRAM. See
[DISTILLATION_DATA_REQUIREMENTS.md](DISTILLATION_DATA_REQUIREMENTS.md) — including the
finding that logit KD to a byte-level student is blocked by a vocabulary mismatch, which
is independent of how much VRAM anyone has.

Training the student on stored teacher outputs costs the same as the runs above: the
objective changes, the memory does not.

---

## 8. Status of every number here

| claim | status |
|---|---|
| T4 reports 14.56 GiB | **VERIFIED** — measured, Level 2 |
| Level 2 = 2,089.2 tok/s on T4 | **VERIFIED** — 32,768,000 tokens / 15,684.6 s (procedural corpus) |
| Level 2R = 1,727.9 tok/s on T4 | **VERIFIED** — 32,768,000 tokens / 18,963.8 s (real corpus, validation, verified persistence). This is the anchor used above. |
| Level 2 = 94,476,448 params | **VERIFIED** — `count_parameters` |
| teacher = 26,895,998,464 params | **VERIFIED** — meta-device build |
| 12 GB card reports ~11.76 GiB | **UNKNOWN** — vendor spec converted; not measured here |
| every memory figure in §2–§3 | **CORROBORATED** — from an estimator calibrated against six measured configurations at ~100M, extrapolated above that and never checked there |
| every tok/s except Level 2's | **UNKNOWN** — FLOP-ratio arithmetic from one anchor. Not a prediction. |
| checkpoint sizes | **VERIFIED** — arithmetic on measured parameter counts |
| a 3060 is T4-class in throughput | **UNKNOWN** — asserted from bandwidth reasoning, not measured |

Before spending hours on a 12 GB card, replace the UNKNOWNs with measurements:
`scripts/hardware_info.py`, then `scripts/benchmark_memory.py`, then a short run.
