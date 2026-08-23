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

### Level 1 — T4 toy model

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

### Level 2 — T4 small student

**Objective:** train a real small model on real data. Confirm throughput, memory
behaviour and checkpoint sizes match the estimates.

**Gate:** a checkpoint that generates coherent output, and measured VRAM within ~15% of
the estimate (`scripts/hardware_info.py --calibrate`).

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
