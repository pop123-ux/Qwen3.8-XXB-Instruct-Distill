# Hardware

What you can run, and how to find out.

## Start here

```bash
python scripts/hardware_info.py --recommend
```

Detects your accelerator, classifies it, and derives — from the project's memory model,
not from a lookup table — which models fit and which experiments are plausible. It works
on NVIDIA, on AMD/ROCm where PyTorch exposes the information, and on CPU-only machines,
where "no GPU" is reported as a normal answer rather than an error.

To preview a machine you do not have yet:

```bash
python scripts/hardware_info.py --simulate-vram 16 --simulate-name "Tesla T4" --matrix --recommend
```

Every memory figure is an **analytical estimate** until `--calibrate` measures the model
against your machine.

## Capability tiers

| Tier | VRAM | Examples | What it is for |
|---|---|---|---|
| 0 | none | CPU only | unit tests, analysis, metadata verification, architecture search |
| 1 | ≤8 GB | RX 480, GTX 1070 | small quantised inference, toy prototypes |
| 2 | 12–16 GB | **Tesla T4**, RTX 4060 Ti 16GB | the deployment target; small-student training, evaluation |
| 3 | 20–24 GB | RTX 3090, RTX 4090, A5000, L4 | quantised teacher inference, mid-size student work |
| 4 | 32 GB | RTX 5090, V100 32GB | headroom above 24 GB for longer contexts |
| 5 | 40–48 GB | A40, A6000, L40S, A100 40GB | comfortable teacher baseline; serious student training |
| 6 | 64–80 GB+ | A100 80GB, H100, MI300X | unquantised teacher inference, large-scale training |

Tiers narrow the search; the concrete fit analysis decides. A machine is classified by
its **largest single accelerator**, not the sum — without model parallelism a model must
fit one device, and calling two 8 GB cards "16 GB" would mislead exactly the people this
tool exists to help.

## The two devices you have

### Tesla T4 (16 GB, free via Colab) — Tier 2

The most useful machine available to this project, and it is a *development* machine.

Good for: small-model inference, quantised inference, architecture prototypes, small
distillation experiments, LoRA/QLoRA, evaluation, and training-pipeline debugging.

**One constraint that catches people out: the T4 is Turing (compute capability 7.5) and
has no bf16.** Use `precision: fp16`. Every config in `configs/experiments/` that targets
a T4 sets this, and the feasibility check refuses a bf16 run on a card that lacks it.

Not suitable for: bf16 Qwen3.8-27B (needs ~49 GiB), 4-bit Qwen3.8-27B (~17.7 GiB — over
even at 8k context), or full-parameter training of the eventual 13–21B student.

### AMD RX 480 (8 GB) — Tier 1

Usable for small quantised inference and toy prototypes if ROCm supports it on your
platform. ROCm's Windows support is limited and older GCN cards are not always covered;
check before planning around it. The diagnostics tool reports what ROCm actually exposes
rather than inventing NVIDIA-style capability flags for it.

## What the teacher needs

Analytical estimate, batch 1, 8k context, from `scripts/hardware_info.py --model Qwen3.8-27B`:

| Quantisation | Weights | Total @8k | Smallest tier that fits |
|---|---:|---:|---|
| bf16 | 47.3 GiB | 49.2 GiB | Tier 6 (80 GB) |
| int8 | 24.6 GiB | 26.5 GiB | Tier 4 (32 GB) |
| q6_k | 20.7 GiB | 22.5 GiB | Tier 3 (24 GB) |
| q5_k_m | 18.1 GiB | 20.0 GiB | Tier 3 (24 GB) |
| q4_k_m | 15.9 GiB | 17.7 GiB | Tier 3 (24 GB) |

**A 16 GB card cannot run the teacher at any quantisation.** That is the project's
premise, and it is why the teacher baseline needs rented hardware while the student
targets 16 GB.

## Rented GPUs

The teacher baseline is a **one-time cost**: run it once, commit the results, reuse them
for every student comparison. That makes an hour or two of rented GPU a reasonable
expense rather than an ongoing one.

Indicative community pricing from RunPod, **checked 2026-08-23** — prices fluctuate and
availability is not guaranteed, so [check current
pricing](https://www.runpod.io/pricing) before relying on any of this:

| GPU | VRAM | ~$/hr | Useful for |
|---|---|---|---|
| RTX A5000 | 24 GB | ~0.16 | quantised teacher inference; mid-size student training |
| RTX 3090 | 24 GB | ~0.22 | same |
| RTX 4090 | 24 GB | ~0.34 | same, faster |
| L4 | 24 GB | ~0.44 | same, low power |
| RTX A6000 | 48 GB | ~0.33 | comfortable teacher work — often better value than 24 GB |
| A40 | 48 GB | ~0.35 | same |
| RTX 5090 | 32 GB | ~0.69 | when 24 GB is too tight |

Prices are **not** hard-coded anywhere in the software; they live only in this document,
with a date, so they can go stale visibly rather than silently.

Note the A6000/A40 at ~$0.33–0.35: at these snapshot prices 48 GB costs about the same
as a 24 GB RTX 4090, which makes the 48 GB tier the better default for teacher work
unless 24 GB is demonstrably sufficient.

## Reading a fit report

```
q4_k_m   weights  15.85  total @8k  17.69 GiB   DOES NOT FIT
```

`total` is the **full envelope** — weights + KV cache + recurrent state + activations +
runtime overhead — not the weight file. Comparing a file size against VRAM is the
mistake this whole layer exists to prevent.

Three verdicts, deliberately:

- **FITS** — comfortable headroom.
- **TIGHT** — under 1.5 GiB spare. It will fit on an idle card and fail the moment a
  desktop compositor or a second process appears.
- **DOES NOT FIT**.

## Calibrating the estimates

```bash
python scripts/hardware_info.py --calibrate
```

Loads a small model (never the 27B teacher), measures real peak VRAM at several
contexts, and reports the measured/estimated ratio. Near 1.0 means the estimator is
trustworthy on your machine; a consistent offset means it is missing a term and should
be corrected rather than explained away. Requires CUDA — on CPU it says so rather than
reporting zeros.
