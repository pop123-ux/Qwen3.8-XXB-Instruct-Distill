# Compute strategy

**Use the T4 to validate ideas cheaply. Rent larger hardware only when the experiment
genuinely requires it.**

Limited compute should not stop the project progressing — but pretending limited compute
can do work that physically requires more VRAM wastes more time than renting a GPU for
an hour.

## What actually needs a big GPU

Surprisingly little, and mostly once:

| Work | Needs | Frequency |
|---|---|---|
| Analysis, architecture search, metadata verification | CPU | continuous |
| Training-pipeline validation | T4 | continuous |
| Small distillation experiments | T4 | often |
| **Teacher baseline** | 24–48 GB | **once** |
| **Distillation data generation** | 24–48 GB | a few times |
| Student candidate comparison | 32–48 GB | a handful |
| Final student training | 48 GB+ / multi-GPU | once |

The two expensive-looking items are one-time costs whose *outputs are reusable*: the
teacher baseline is committed and compared against forever; the distillation dataset is
generated once and trained on repeatedly. That is what makes this affordable.

## Tiers, and when each is the right call

### Free — Colab T4 (16 GB)

Development, prototyping, small experiments, evaluation of small models. The project's
deployment target, so validating *there* is not a compromise — it is the point.

No bf16 (Turing). Use `precision: fp16`.

### ~$0.16–0.44/hr — 24 GB (A5000, 3090, 4090, L4)

The cheapest tier that can run the teacher at all, quantised. Enough for teacher
inference at short context and for mid-size student training.

Use it for: distillation data generation, the teacher baseline if 48 GB is unavailable,
larger student experiments.

### ~$0.69/hr — 32 GB (RTX 5090)

Buy this when 24 GB is demonstrably too tight — longer contexts, larger batches — not
speculatively.

### ~$0.33–0.35/hr — 48 GB (A6000, A40)

**Often the best value in the table.** At these snapshot prices 48 GB costs about the
same as a 24 GB RTX 4090. Comfortable for teacher baseline work and for student training
that would be cramped at 24 GB.

If you are renting once for the teacher baseline, this is the default choice.

### 80 GB+ (A100, H100)

Unquantised teacher inference and large-scale training. Justify it before booking it: at
several times the hourly cost, an experiment that fits 48 GB should run on 48 GB.

## Pricing

Indicative RunPod community pricing, **checked 2026-08-23**:

| GPU | VRAM | ~$/hr |
|---|---|---|
| RTX A5000 | 24 GB | 0.16 |
| RTX 3090 | 24 GB | 0.22 |
| RTX A6000 | 48 GB | 0.33 |
| A40 | 48 GB | 0.35 |
| RTX 4090 | 24 GB | 0.34 |
| L4 | 24 GB | 0.44 |
| RTX 5090 | 32 GB | 0.69 |

These are a **snapshot, not a promise**. Prices fluctuate, availability varies, and
community-cloud instances can be interrupted. [Check current
pricing](https://www.runpod.io/pricing) before planning around any of them, and compare
alternatives — Vast.ai, Lambda, and others occupy similar tiers.

Prices appear only in documentation, never in code, so they go stale visibly.

## Before spending anything

```bash
python scripts/hardware_info.py --simulate-vram 24 --model Qwen3.8-27B --matrix --recommend
python scripts/train_student.py --config <config> --dry-run --simulate-vram 24
```

Both run locally in seconds and tell you whether the rental would have worked. Renting
first and finding out second is the expensive order.

## A cost-control habit

Rented GPUs bill by the hour whether or not they are computing. Before starting a run:

- have the dataset already prepared and uploaded;
- have the config dry-run clean at the simulated VRAM;
- know the number of steps and the expected wall-clock;
- write checkpoints often enough that an interruption is not a total loss.

Most wasted rental time is spent debugging setup that could have been debugged on a T4.
