# Training on limited hardware

**The central principle: the T4 is a development machine, not the machine that trains
the final student.**

That is not a disappointment. The project's goal is a model that is cheap to *run*, not
one that is cheap to *train*. A T4 can validate every idea in the research plan at low
cost; only the final scaled run needs bigger hardware, and by then you will know the
recipe works.

## Three different training problems

Conflating these is the most common way a limited-hardware project stalls.

### A. Architecture development — cheap, do it constantly

Toy models, reduced dimensions, short sequences, synthetic data, unit tests. Runs on a
T4 in minutes, and on CPU in slightly more.

**Question answered:** does the architecture and the training code work at all?

### B. Small-scale distillation experiments — moderate, do it often

A small student, a small teacher-output dataset, QLoRA where appropriate, short
contexts. A T4 handles this.

**Question answered:** does this distillation strategy actually beat the baseline?

### C. Final student training — expensive, do it once

Potentially multi-GPU, rented, days of compute, substantial storage.

**Question answered:** how much teacher capability does the chosen recipe recover?

**A T4 cannot do C.** Anyone claiming otherwise is either training something much
smaller than this project targets, or fine-tuning an adapter and calling it a student.

## LoRA is not the final answer here

This distinction matters enough to state plainly.

LoRA and QLoRA are genuinely useful for A and B: prototyping, instruction tuning, testing
training objectives. Use them freely there.

But **applying LoRA to Qwen3.8-27B does not produce this project's deliverable.** The
goal is a *smaller architecture* — fewer parameters, less VRAM, a model that fits 16 GB.
A LoRA adapter over a 27B base is still a 27B model at inference; the adapter changes
behaviour, not size. Architecture compression requires real student parameters, trained.

So: LoRA to learn *whether an objective works*, then full training of the actual student
architecture to *build the thing*.

## The development ladder

Each level has an objective and a gate. Do not skip levels — the point is that a failure
at level 2 costs minutes rather than a rented day.

### Level 0 — CPU

**Objective:** correctness. Unit tests, analysis, architecture search, metadata
verification, dataset preparation.

**Gate:** `pytest` passes and `scripts/train_student.py --config configs/experiments/t4_prototype.yaml --dry-run` reports PLAUSIBLE.

Everything in this repository except training and generation runs here.

### Level 1 — T4 toy model — **PASSED**

**Objective:** verify the training mechanism — forward, backward, optimizer, scheduler,
checkpointing, resume, validation, logging.

**Gate:** loss falls measurably on the synthetic task. This is why the prototype trains
on a *learnable* task rather than random tokens: random data pins the loss at
`ln(vocab)` whether or not the optimizer works, so it cannot distinguish a working loop
from a broken one.

```bash
python scripts/train_student.py --config configs/experiments/t4_prototype.yaml --dry-run
python scripts/train_student.py --config configs/experiments/t4_prototype.yaml
```

**Result:** 4.03M params on a real Tesla T4, 200 steps in ~56 s. Train loss
8.2565 → 2.1008, validation 4.4904 → 2.0910. The architecture instantiates, trains and
checkpoints on real CUDA hardware.

Two false starts are worth recording, because both produced a *flat* loss that looked
like a broken training loop and was not. The first two synthetic tasks were not
learnable at toy scale, so the loss sat at `ln(vocab)` regardless. Only the third design
— a short-period induction task over a 64-token alphabet — moved. **A flat loss on an
unlearnable task is not evidence of anything**, which is the whole reason this level
exists as a separate gate.

### Level 2 — T4 small student — **configured, not yet run**

**Objective:** train a real small model on real data. Confirm throughput, memory
behaviour and checkpoint sizes match the estimates.

`configs/experiments/t4_level2_100m.yaml`: 94.48M parameters, 16 layers (12 DeltaNet +
4 full attention), hidden 640, FFN 2176. The 3:1 hybrid layout and 3.40x FFN expansion
both match the teacher, so this is a scaled-down version of the real architecture rather
than a generic transformer.

**Byte-level tokenization** (vocab 256) is the deliberate choice for this level:

- No tokenizer to download, version or licence-check — the run is fully offline.
- Any plain text file works unchanged, so the corpus is a substitution, not a rewrite.
- Nearly every parameter lands in the layers rather than in an embedding table, which
  makes a ~100M run a test of the *architecture* rather than of the embedding.
- The loss is directly interpretable as **bits per byte**: 8.0 is a model that has
  learned nothing, and good byte-level models on English land near 1.0–1.5. Unlike
  BPE perplexity it does not depend on a tokenizer's compression rate, so two runs are
  comparable.

The honest trade-off: byte sequences cover ~4x less text per token than BPE, so a
1024-byte window is roughly a 250-token window. Fine for the question this level asks.

**Gate:** a checkpoint that generates coherent output, and each *component* of the
measured VRAM within ~15% of its estimate — not the total (see below).

```bash
python scripts/train_student.py --config configs/experiments/t4_level2_100m.yaml --dry-run
python scripts/train_student.py --config configs/experiments/t4_level2_100m.yaml
python scripts/validate_checkpoint.py experiments/t4_level2_100m/final
python scripts/hardware_info.py --calibrate-run experiments/t4_level2_100m/summary.json
```

**Status: the first attempt OOMed on a real T4.** It is kept as a documented failure in
`experiments/runs/t4_level2_100m_oom_2026-08-24/`, and `configs/experiments/t4_level2_100m.yaml`
is preserved unchanged as the failed baseline. `t4_level2_100m_ckpt.yaml` supersedes it.

### What the OOM taught us

Predicted 4.53 GiB. Demanded ~24.8 GiB. Died in the forward pass, zero steps completed.

The traceback named `torch.nn.functional.scaled_dot_product_attention`, **and that was
misleading**. SDPA receives `attn_mask=None` and `is_causal=True`, so it never
materialises a `batch x heads x seq x seq` matrix; all four attention layers together
retain 110 MiB at batch=1, about 3% of the total. SDPA was simply the next sizeable
allocation after the DeltaNet layers had already consumed the card.

The real cause: **the estimator had no Gated DeltaNet term at all.** It treated all 16
layers as generic transformer blocks scaling with `hidden_size` and `intermediate_size`.
Measured by attributing every tensor autograd retains to its creating module:

| scope | retained (batch=1, seq=1024) | tensors |
|---|---:|---:|
| **DeltaNet mixers** | **2155.6 MiB** | **5244** |
| MLP (DeltaNet layers) | 629.2 MiB | 96 |
| MLP (attention layers) | 209.8 MiB | 32 |
| embedding / norms / logits | 169.4 MiB | 138 |
| attention mixers | 109.7 MiB | 86 |

Two properties of `torch_chunk_gated_delta_rule` — the pure-PyTorch fallback that runs
when `fla` is not installed — explain it:

1. **It force-upcasts to fp32.** The function opens by casting q, k, v, beta and g to
   `torch.float32`, so 12 of 16 layers ignore fp16 autocast entirely. This is also why
   fp16 bought far less speed than expected.

2. **A 63-iteration Python loop retains O(chunk²) clones.** Inside
   `for i in range(1, chunk_size)` it does `sub = attn[..., :i, :i].clone()` and
   multiplies by it, so autograd saves every one:

   ```
   sum(i² for i in 1..63) = 85,344 elements per (chunk, head)
   -> v_heads x 85,344 x 4 / chunk_size = 64,008 bytes per token per DeltaNet layer
   ```

   That single term is larger than the estimator's entire previous activation budget.

The corrected estimator models this explicitly. The loop term is derived exactly; the
remainder is fitted to six measured configurations spanning 4x in head count and 4x in
head dimension, reproducing them to within 3.4%, with a 1.10 margin so it never
underestimates. Re-run against the failed config it now reports **23.87 GiB, NOT
FEASIBLE** — which is what a dry run should have said before the GPU window was spent.

### Diagnosing this without a GPU

```bash
python scripts/probe_activations.py --config <config>            # attribute by module
python scripts/probe_activations.py --config <config> --scaling  # extrapolate by batch
```

Tensor shapes and dtypes do not depend on the device, so this runs on CPU and is a
**pre-flight check, not a post-mortem**. It is what would have caught this.

### The revised configuration

`configs/experiments/t4_level2_100m_ckpt.yaml`, estimated at **3.57 GiB**. Unchanged:
94.48M parameters, the 3:1 hybrid layout, sequence length 1024, effective batch 16.
Changed:

| Change | Why |
|---|---|
| `gradient_checkpointing: false -> true` | **The fix.** Measured 67x reduction in retained activations (3273.6 MiB -> 49.1 MiB at batch=1). Costs ~30% throughput. |
| `batch_size 8 -> 4`, `accum 2 -> 4` | **Margin, not necessity.** Batch 8 also fits at 4.54 GiB. The evidence is CPU-side tensor accounting that cannot see allocator fragmentation, and a second OOM costs another window. Effective batch is identical. |

Rejected: SDPA backend tuning (attention is 3% of the problem), shorter sequences (not
needed), and any architecture change (would replace the experiment rather than fix it).
Installing `fla` would remove the problem at source, but the run does not depend on it.

### Level 3 — T4 small distillation experiment

**Objective:** the first result that matters — does KD beat SFT?

Run the same config twice, changing only `training.objective`:

```bash
python scripts/train_student.py --config configs/experiments/distillation_small.yaml   # objective: sft
python scripts/train_student.py --config configs/experiments/distillation_small.yaml   # objective: mixed_kd
```

**Gate:** a measured difference in validation loss and downstream accuracy, in that
order. If KD does not beat SFT at small scale, scaling it up will not rescue it.

### Level 4 — rented 24 GB

**Objective:** teacher inference. Generate the distillation dataset; run the teacher
baseline; measure reasoning-token behaviour at each effort level.

**Gate:** `teacher_baseline_v1` committed, and a distillation dataset on disk.

### Level 5 — rented 32/48 GB

**Objective:** train the largest student candidate the budget allows; run the
architecture comparison that the Phase 0 search could only rank analytically.

**Gate:** a measured capability-per-VRAM curve across candidates.

### Level 6 — large distributed final training

**Objective:** the final student, using the recipe validated at levels 3–5.

**Gate:** the deployment matrix, measured on 16 GB hardware.

## The separation that makes this work

**Teacher generation and student training are separate operations.**

```
rented GPU                          any GPU, later
──────────                          ──────────────
teacher inference   ──► dataset ──► student training
(needs 24-48 GB)        (JSONL)     (fits a T4)
```

The teacher runs once and writes `data/distillation/*.jsonl`. The student trains from
that file afterwards. **A 16 GB card never has to hold a 27B teacher and a student at
the same time** — which is what makes the whole plan feasible on the hardware available.

The dataset schema is in `qwen_distill.training.data`. Every record carries not just what
the teacher said but how much it cost to say it — thinking tokens, total tokens,
reasoning setting — so reasoning-efficiency training can supervise on length without the
teacher present.

## Reasoning-efficiency training without the teacher

The teacher labels difficulty and reasoning length during generation. The student then
learns *from the labels*, offline:

```
easy prompt   → teacher answer with short reasoning   → student learns: answer briefly
hard prompt   → teacher answer with long reasoning    → student learns: reason at length
```

Start simple. Train on the shortest correct teacher response per prompt, and measure
whether easy-task tokens fall *without* hard-task accuracy falling. Build a learned
budget controller only after the simple version demonstrably works — an elaborate
mechanism that has not beaten the trivial baseline is not progress.

## Keeping a run within budget

Ordered by cost, cheapest first. The first two are free — they change memory without
changing what the experiment measures:

1. **Gradient checkpointing on.** Largest single saving.
2. **Batch size 1 + gradient accumulation.** Same effective batch, far less memory.
3. **Shorter sequences.** Changes what you measure; note it.
4. **QLoRA instead of LoRA or full.** Changes what you can conclude; note it.
5. **8-bit optimizer.** Saves ~6 bytes per trainable parameter.
6. **A bigger GPU.** Last resort, and sometimes the right answer.

`scripts/train_student.py --dry-run` prints exactly these, ordered, whenever a
configuration does not fit.

## Always dry-run first

```bash
python scripts/train_student.py --config <config> --dry-run
```

Seconds, no weights loaded, and it reports parameter count, memory breakdown and a
verdict. It is the difference between finding an OOM now and finding it forty minutes
into rented GPU time. To check a machine you have not rented yet:

```bash
python scripts/train_student.py --config <config> --dry-run --simulate-vram 24
```

The trainer refuses to start on a NOT FEASIBLE configuration unless you pass `--force`.

## Precision: what you asked for is not always what runs

`AutoModelForCausalLM.from_config` takes no dtype, so it always produces an fp32 model.
A config declaring `precision: fp16` therefore trained in **fp32** — roughly twice the
weight, gradient and optimizer memory, and none of the T4's fp16 tensor-core speed —
and nothing in the run artifacts said so. That is a whole GPU window spent on a
different experiment from the one described.

The trainer now resolves precision explicitly and records both values:

| Requested | Device | Effective | Why |
|---|---|---|---|
| `fp32` | any | `fp32` | no change requested |
| `fp16` | CUDA | `fp16` | autocast + GradScaler |
| `bf16` | Ampere+ | `bf16` | autocast, no scaler needed |
| `bf16` | Turing (T4) | `fp16` | Turing has no bf16 |
| `fp16` / `bf16` | CPU | `fp32` | autocast covers few CPU ops and is usually slower |

Mixed precision here means **autocast, not fp16 weights**. AdamW on fp16 parameters
underflows, so the master weights and gradients stay fp32 and only the matmuls are cast.
`GradScaler` then keeps fp16 gradients from flushing to zero, and gradients are unscaled
before clipping — clipping scaled gradients computes the norm against the loss scale
rather than against the gradient.

`summary.json` records `requested_precision`, `effective_precision` and a
`precision_note`, so a memory or throughput figure can be read months later against what
actually ran.

## Memory: measure the terms, never multiply the total

An earlier calibration reported a measured/estimated ratio of ~**2.85**. That number was
not usable as a correction factor, and the reason is worth stating plainly: it divided
**peak reserved** — which includes the CUDA context and every block the caching
allocator is holding — by **modelled tensors with the overhead term deliberately
zeroed**. Those are different quantities. On a small probe model the fixed CUDA context
alone rivals the whole tensor footprint, so the "factor" was mostly measuring the
context. Multiplying every future estimate by 2.85 would have made the estimator
permanently wrong in a way that looked calibrated.

What replaced it:

**1. Two predicted quantities, matching the two that are measured.**

| Quantity | Estimator field | Probe field | What it answers |
|---|---|---|---|
| Live tensors | `predicted_allocated_gib` | `peak_allocated_gib` | is our arithmetic right? |
| Process footprint | `predicted_reserved_gib` | `peak_reserved_gib` | will it fit the card? |

Only the first should ever drive a correction to the estimator. The second depends on
allocator settings and fragmentation, which the estimator does not model and should not
be blamed for — but it is the one a deployment claim has to satisfy.

**2. Every term attributed separately.** `memory_probe` snapshots seven stages of the
first training step and attributes each term by difference:

```
weights          = after_model_creation  - baseline
activations      = after_forward         - after_model_creation
gradients (net)  = after_backward        - after_forward
optimizer state  = after_optimizer_step  - after_backward
allocator reserve= peak_reserved         - peak_allocated
```

AdamW allocates its moments lazily on the first `step()`, so a snapshot taken any
earlier reads zero — and would be reported as a measurement of zero.

A total that happens to be right because two errors cancelled is still two errors, and
only a per-component comparison can see that. `--calibrate-run` does exactly this
comparison and names the term to fix:

```bash
python scripts/hardware_info.py --calibrate-run experiments/<run>/summary.json
```

It deliberately does **not** compare gradients: the probe's backward delta is net of the
activations freed during the backward pass, which is not the gross gradient size the
estimator models. Comparing them would repeat the same category error that produced 2.85.

**3. The precision scheme is modelled, not assumed.** `torch.autocast` holds fp32
weights and fp32 gradients while computing in fp16; pure-bf16 training holds bf16
weights and gradients with an fp32 master copy in the optimizer. Both come to 16 bytes
per parameter for AdamW — so a *total* cannot tell them apart, while a measurement can,
and would flag two components as wrong on a correct total.

**4. The loss path is counted properly.** `ForCausalLMLoss` does `logits =
logits.float()`, and `cross_entropy` retains a second fp32 log-softmax buffer for
backward — verified against the `transformers` source and by enumerating the tensors
autograd actually retains. Together with the `lm_head` output at the activation dtype,
the logits are held **three times over**. Modelling them once understates the term by
2.5–3x; at a 248k vocabulary that is gigabytes, and it is exactly the term that decides
whether a long-sequence run OOMs.

## Colab: keep the results, not the runtime

Colab runtimes are ephemeral — when the session recycles, anything not in git or on
Drive is gone. Checkpoints are too large for git, so the split is:

| Where | What |
|---|---|
| git | code, configs, `summary.json`, `metrics.jsonl`, small artifacts |
| Drive | checkpoints, model weights, anything measured in hundreds of MB |
| nowhere | credentials — see below |

```bash
python scripts/backup_colab_to_drive.py --dry-run   # exactly what would move
python scripts/backup_colab_to_drive.py
```

Run it **after** an experiment, not during one: synchronising every step would slow the
run and thrash Drive for no benefit.

The destination is someone's personal Drive, so the safety properties are the design
rather than a feature list:

- **Nothing is deleted by default.** `--delete-extraneous` exists, is off, and refuses
  to run without `--yes`. `--dry-run` shows precisely what it would remove.
- **Symlinks are never followed.** A stray link could otherwise pull an unrelated tree
  onto Drive.
- **Credential-shaped files are never copied** — `.env`, `*.pem`, `*.key`, `id_rsa*`,
  `.netrc`, `.git-credentials`, `*token*.json`, `.ssh/`, `.aws/` — at any depth, so a
  token dropped in the repo root does not end up in cloud storage. The patterns are
  deliberately broad, with a small exact-filename allowlist for standard model metadata
  (`tokenizer_config.json`, `tokenizer.json`, ...) that `*token*.json` would otherwise
  swallow. A backup that silently drops the teacher metadata is worse than one that
  never ran, because the loss surfaces only after the runtime is gone.
- **Unchanged files are skipped**, so re-running is cheap and idempotent.

If Drive is not mounted the script exits 2 and prints the `drive.mount` snippet rather
than silently writing to a local directory that will vanish with the runtime.

### A note on what is currently in git

`.gitignore` already excludes `*.pt`, `*.pth`, `*.ckpt` and `experiments/**/runs/`, but
`.gitignore` does not untrack files that were committed before it applied. The Level-1
prototype's three checkpoints are still tracked, at **139 MB**, for a 4.03M-parameter
toy model whose result is fully captured by the far smaller `summary.json` and
`metrics.jsonl` beside them.

That is worth deciding on deliberately rather than letting it grow. To stop tracking
them going forward while keeping the files on disk:

```bash
git rm --cached experiments/runs/t4_prototype/*/training_state.pt
```

Note that this does not shrink the repository — the blobs remain in history, and
removing them from history is a rewrite that invalidates every existing clone. The
choice to make now is whether Level-2 and later checkpoints go to Drive instead of git.
They should.
