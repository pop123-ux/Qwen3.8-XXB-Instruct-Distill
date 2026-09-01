# Distillation roadmap

Where the project is on the path to a benchmarked 16 GB model, and what the next step is.
Direction and goals: [PROJECT_DIRECTION.md](PROJECT_DIRECTION.md).

> **Distillation is not complete.** No student has been distilled from the real teacher. No
> benchmark has been run. No external compute has been used. This document says what now
> works and what does not.

---

## 1. State of the chain

The target is now a single frozen student, `qwen38_19b_h5120_l48_moe` — 48 layers,
36 DeltaNet + 12 full attention, 8 routed experts with top-2 routing and one shared
expert. It is not a ladder and not a search space; see
[STUDENT_ARCHITECTURE.md](STUDENT_ARCHITECTURE.md). The earlier dense candidates are
retained as historical baselines, not deleted.

```
    historical from-scratch experiments (2 / 2R / 3)   — closed
            ↓
    real Qwen3.8-27B teacher                           ✅ loads, verified
            ↓
    frozen MoE student, specified + audited            ✅ 13.01B measured, structurally valid
            ↓
    initialisation: depth / FFN / KV / router          ✅ implemented + measured
            ↓
    behavioural + context objectives                   ✅ implemented, ablations defined
            ↓
    16 GB accounting                                   ✅ fits Q4/Q5/Q6, corrected to get there
            ↓
    real teacher smoke test                            ← NEXT (needs rented GPU)
            ↓
    one-step real KD pilot                             ✅ implemented, runs on fixtures
            ↓
    first benchmark                                    ⬜ registry exists, IFEval not added
            ↓
    A1-A4 and B1-B4 ablations                          ⬜ defined, none run
            ↓
    full distillation                                  ⬜
            ↓
    16 GB release                                      ⬜ architecture cleared; blocked on benchmarks
            ↓
    separate 12 GB release                             ⬜
```

### The correction that unblocked the release path

The first implementation of the frozen specification used **24 routed experts** and weighed
**22,072,134,528** parameters, of which 61.57% were routed experts. It did not fit a 16 GB
card at Q4, Q5 or Q6 at *any* context length — the best all-Q4 configuration needed
13.93 GiB at 2,048 tokens against 13.56 GiB usable.

`num_experts` 24 → 8 fixed it. One field:

| | rejected | corrected |
|---|---:|---:|
| total parameters | 22,072,134,528 | **13,008,505,728** |
| active per token | 9,615,051,648 | **9,611,119,488** |
| Q4 longest context | did not fit | **131,072** |
| Q5 longest context | did not fit | **65,536** |
| Q6 longest context | did not fit | **32,768** |

Per-token capacity is unchanged to within 0.04%: a token was only ever using two experts, so
the other sixteen cost VRAM and contributed nothing. The cost that *is* real is FFN channel
coverage, down from 100% to 39.7%. Full accounting in
[PARETO_EVALUATION.md](PARETO_EVALUATION.md); the rejected configurations are kept in
`REJECTED` in `architecture/moe_student.py`.

### The canonical path

One route from the teacher to a materialised student. Nothing else is reachable by
accident: the mock teacher is never selected implicitly, an unpinned Hub revision is
refused before the download starts, and the pilot has no architecture arguments.

```text
Qwen/Qwen3.8-27B
    |
    |  --revision <EXACT COMMIT SHA>      required for a Hub load; refused without it
    v
TeacherLoadPlan.validate()                before any bytes are fetched
    |
    v
load_verified_teacher()                   missing weights are fatal, never a warning
    |
    v
teacher tensors (streamed, one at a time)
    |
    v
FROZEN_STUDENT = qwen38_19b_h5120_l48_moe  13,008,505,728 parameters, not configurable
    |
    v
materialise_student()                     copy | KV-merge 4->2 | dense FFN -> 8 experts
    |
    v
checkpoint  ->  distillation
```

Both commands below run today against small fixtures and are the same code paths the real
teacher takes — only the weights differ:

```bash
# 0. fetch the pinned checkpoint (resumable, writes a manifest)
python scripts/download_teacher.py \
    --revision <EXACT_QWEN_COMMIT_SHA> --output /data/models/qwen3.8-27b

# 1. is the teacher operational?  (rented 24 GB card)
#    --revision is required for a Hub load and is checked before anything downloads.
python scripts/teacher_smoke_test.py \
    --local-path /data/models/qwen3.8-27b \
    --quantization 4bit \
    --revision <EXACT_QWEN_COMMIT_SHA>

# 2. teacher -> the canonical student. No architecture arguments exist.
python scripts/distill_pilot.py \
    --teacher /data/models/qwen3.8-27b \
    --revision <EXACT_QWEN_COMMIT_SHA> \
    --output runs/pilot1

# the same plan, the audit and the 16 GB verdict, loading nothing (laptop, seconds)
python scripts/distill_pilot.py --revision <EXACT_QWEN_COMMIT_SHA> --dry-run

# the mechanism regression harness — a small dense student, not a research run
python scripts/chain_selftest.py --stand-in --output runs/selftest

# every claim this repository makes about itself, checked by running it
python scripts/acceptance_gate.py
```

The pilot loads the teacher through the **verified loader**, not a bare
`from_pretrained` — `transformers` returns a freshly-initialised model rather than raising
when a checkpoint's keys do not match, and a KD loss computed against random 27B weights is
finite, falls, and means nothing.

## 2. Historical student candidates

> Superseded as *targets* by the frozen MoE student, and retained as **baselines**. The
> project's first scientific comparison is 17.76B dense L40 against 13.01B sparse-MoE L48:
> same teacher, same width, different sparsity and depth. Nothing below is deleted.

Transfer-compatible means the teacher's `head_dim` (256), GQA ratio (6 query heads per KV
head), DeltaNet ratio (3 value heads per key head), conv kernel and **vocabulary** are
inherited; `student_from_teacher()` enforces all of it by construction. Weights are
`q4_k_m`, batch 1, against 13.56 GiB usable on 16 GB and 10.76 GiB on 12 GB.

| hidden | layers | ffn | kv | dn key | params | weights | total @32K | 16 GB | 12 GB | @128K fits 16 GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 3072 | 28 | 10240 | 2 | 8 | 5.12B | 3.22 | 4.77 | ✅ | ✅ | ✅ |
| 3584 | 32 | 12160 | 2 | 8 | 7.23B | 4.48 | 6.12 | ✅ | ✅ | ✅ |
| 4096 | 40 | 13824 | 3 | 12 | 11.54B | 6.99 | 9.13 | ✅ | ✅ | ✅ |
| **4608** | **40** | **15616** | **3** | **12** | **13.98B** | 8.43 | 10.60 | ✅ | ✅ | ✅ |
| **5120** | **40** | **17408** | **4** | **16** | **17.76B** | 10.64 | 13.18 | ✅ | ❌ | ❌ |
| 5120 | 48 | 17408 | 4 | 16 | 20.81B | 12.37 | 15.18 | ❌ | ❌ | ❌ |

For scale: **Qwen3.5-9B, the model to beat, uses 7.56 GiB at 32K.** The envelope supports
far more than that, and the candidate range runs to ~17.8B.

### The finding that should shape the choice

**A depth-only student transfers as an exact copy.** `h5120 L40` keeps the teacher's width,
FFN and every head count, changing only depth. Its transfer plan is **533 tensor copies,
100% coverage, zero warnings, no width reduction anywhere**:

| student | plan | coverage | warnings |
|---|---|---|---|
| h5120 L40 (17.76B) | 533 × `copy` | 1.000 | 0 |
| h5120 L48 (20.81B) | 639 × `copy` | 1.000 | 0 |
| h4096 L40 (11.54B) | 201 slice + 270 head-subset + 40 head-select + 20 copy | 1.000 | 1 |

That matters because `slice` is explicitly a **baseline, not a method** — it assumes the
teacher's parameters are ordered by importance, which nothing guarantees. A depth-only
student removes that assumption entirely: every retained tensor is the teacher's, bit for
bit. The only open question becomes *which* layers to keep, and `group` selection already
answers that with 100% coverage across the full depth and zero block-type mismatches.

So the two targets pull apart, and should be treated as different problems:

- **16 GB**: `h5120 L40`, 17.76B, exact-copy transfer, 13.18 GiB at 32K. No long context.
- **12 GB** (and 16 GB at 128K): `h4608 L40`, 13.98B, needs width reduction — so it inherits
  the `slice`-vs-`mean_pool`-vs-`importance` question the 16 GB candidate avoids.

Neither is selected. Selection is a benchmark question and there is no benchmark yet.

---

## 3. What has to happen next, in order

**1. Run the smoke test against the real teacher.** One rented 24 GB card, minutes. It now
sweeps top-*k* and reports **tail mass, entropy, top-1 agreement and storage cost per k**,
which is the measurement the offline-vs-online decision has been waiting on. Small tail
mass at k=64 means an offline corpus loses almost nothing; large means a bigger k or
staying online.

**2. Materialise the canonical student.** `scripts/distill_pilot.py` targets
`qwen38_19b_h5120_l48_moe` and nothing else — it has no geometry flags. The KD-chain
mechanism is checked separately by `scripts/chain_selftest.py`, whose loss says nothing
about capability.

**3. A benchmark harness.** The registry exists (`evaluation/benchmark.py`: digest-pinned
suites, runs bound to what they measured, comparability checks). What is missing is a first
benchmark in it. IFEval is the cheapest: programmatic constraint checks, no model judge, no
gated dataset, no code sandbox.

**4. Measure what the expert-budget correction cost.** Resolved on the memory axis; the
quality axis is open. Two numbers to get: what dropping 60.3% of the teacher's FFN channels
costs against the 24-expert configuration on a short pilot, and whether 8 experts route as
diversely as 24 did. Either could argue for spending the remaining ~2B of budget headroom on
experts again.

**5. Run A1-A4 and B1-B4.** Defined in `research/ablations.py` with a falsifier on every
arm. A1 (pointwise layer matching) is the control for the paper's central claim; B1
(4K-only distillation) is the control for the context component. Before any margin is
called significant, the control arm has to be run more than once so seed variance is
measured — the matrix records that this is currently uncontrolled.

---

## 4. Blockers

| | blocker | blocks | resolvable here? |
|---|---|---|---|
| B1 | Teacher weights unreachable — ~54 GB, `huggingface.co` egress-blocked in this sandbox | every real-teacher run | no, needs rented compute |
| B2 | **Revision unpinned** — no commit SHA supplied or resolvable | reproducibility of any dataset | yes, on the first real run: pass `--revision` |
| B3 | Teacher needs ≥24 GB — 16.3 GiB at 4-bit vs 13.56 usable | running teacher and student on one 16 GB card | no, architectural |
| B4 | No corpus in the teacher's vocabulary — `vendor/` has no `tokenizer.json` | KD over real text at scale | yes, once the tokenizer is downloaded with the weights |
| B5 | No benchmark harness | every capability claim, and student selection | yes, and it is the critical path |
| B6 | Offline logit format unchosen | training without a resident teacher | deliberately gated on B1's tail-mass number |
| B7 | ~~Frozen student exceeds 16 GB at every release precision~~ | ~~the release itself~~ | **closed** — `num_experts` 24 → 8; fits Q4 to 131,072, Q5 to 65,536, Q6 to 32,768 |
| B10 | 262,144-token window needs an 8-bit KV cache to fit | a full-window release claim | yes, once KV-quantised retrieval accuracy is measured |
| B8 | MTP not built by the runtime | any MTP training or result | no; the field is kept as an extension point and no MTP result is claimed |
| B9 | Seed-to-seed variance unmeasured | calling any ablation margin significant | yes: run the control arm more than once |

B2 is the cheapest to close and the easiest to forget: it costs one command-line flag on the
first real run and is unrecoverable afterwards if the upstream repo moves.

---

## 5. What did not change

Level 2 / 2R / 3 remain byte-level from-scratch experiments and their records are untouched.
The byte-level tokenizer path still works and is still correct for them. No new from-scratch
level was created and none should be: the project's next result comes from the real teacher,
not from another scaling rung.
