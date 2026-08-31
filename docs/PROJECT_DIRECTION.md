# Project direction

Two goals, held at the same time, neither subordinate to the other.

**North Star 1 — the research paper.** *Beyond Layer Matching: Distilling Computational
Behavior in Hybrid Language Models*, with a second major component on context-length
specialisation during distillation. Every claim in it must be falsifiable, and every arm of
the study carries the observation that would refute it, written down before the run.

**North Star 2 — the model release.** The strongest practically deployable model under a
strict 16 GB VRAM inference constraint, where "the workload" means weights plus KV cache
plus recurrent state plus buffers plus activations plus runtime overhead — everything, at
once, on the card. No benchmark claim is made before it is measured.

The two goals share one artifact. The paper's experiments are the release's ablations, and
the release's constraint is one of the paper's variables.

---

## The student

One frozen target, `qwen38_19b_h5120_l48_moe`, distilled from `Qwen/Qwen3.8-27B`. Its
specification is in [STUDENT_ARCHITECTURE.md](STUDENT_ARCHITECTURE.md) and is canonically
defined in code at `src/qwen_distill/architecture/moe_student.py`. It is not a ladder and
not a search space: it is the target, and the research question is how to distil *into*
it, not what to distil into.

Three simultaneous reductions define the problem:

| reduction | teacher | student | where it is handled |
|---|---|---|---|
| depth | 64 layers | 48 layers | [INITIALIZATION_METHOD.md](INITIALIZATION_METHOD.md) |
| FFN | dense, 17408 wide | 8 experts x 768, top-2 | same |
| KV heads | 4 | 2 | same |

Width, vocabulary and the hybrid pattern are unchanged from the teacher. That is deliberate:
holding them fixed is what makes the three reductions attributable.

---

## What has been measured, and what it changed

Two audits were run before any training code was written, and both returned answers that
changed the plan. They are recorded here rather than in a footnote because they are the two
facts a reader most needs in order to interpret everything else.

### The student weighs 13.01B, and getting there required a correction

The first implementation of the frozen specification used 24 routed experts and came to
**22,072,134,528** parameters — 61.57% of them routed experts, and 82% of the 26.90B
teacher, which is not a distillation result. It did not fit a 16 GB card at any release
precision or any context length.

`num_experts` 24 → 8 is the entire correction:

```
exact_parameter_count     13,008,505,728      (was 22,072,134,528)
active_per_token           9,611,119,488      (was  9,615,051,648)
non_embedding             10,465,708,928
difference_from_19B       -5,991,494,272
fraction of the teacher            48.4%      (was 82%)
```

Per-token capacity is unchanged to within 0.04%, all of it the smaller router. A token was
only ever using two experts; the other sixteen cost VRAM and contributed nothing. Two
invariants make that precise, and both are counter-intuitive enough to be worth stating:

```
total  = BASE + 3.H.L.(E.W + S) + H.L.E + H.L      depends on the product E x W
active = BASE + 3.H.L.(K.W + S) + H.L.E + H.L      depends on K x W
```

So splitting a fixed expert budget between count and width is free for memory and is *not*
free for per-token capacity — which is why the correction cut the count and left the width
at 768. `parameter_model()` computes both in closed form; `audit()` builds the model and
sums its tensors; a test asserts they agree to the parameter on every component.

The name still says 19B and the model is now 6B under it. The label is not chased in either
direction: the 16 GB constraint set the size.

Reproduce: `python scripts/student_report.py --section architecture`.

### It fits 16 GB

Fully GPU-resident, quantisation overhead and runtime included, every stored expert counted:

| precision | weights | longest context |
|---|---:|---:|
| **Q4** | 7.42 GiB | **131,072** |
| **Q5** | 8.63 GiB | **65,536** |
| **Q6** | 9.99 GiB | **32,768** |

Against 13.56 GiB usable on a real 16 GB card, with 0.5 GiB held in reserve. The full
262,144-token window needs an 8-bit KV cache — fp16 KV alone is 6.00 GiB there — which is a
quality decision reported as its own row rather than folded into the headline.

**Inactive experts are not free.** 9.61B of 13.01B parameters are active per token, and all
13.01B are resident. Active parameters govern compute and latency and never reduce VRAM.
Sizing an MoE against its active count is exactly how the 22.07B architecture looked
deployable on paper.

Full accounting in [PARETO_EVALUATION.md](PARETO_EVALUATION.md);
`python scripts/student_report.py --section memory`.

### What the correction cost

Stated rather than glossed: the FFN decomposition now holds 6,912 of the teacher's 17,408
FFN channels — **39.7%, down from 100%**. The remaining 60.3% are not transferred and must
be learned. Active width per token is unchanged at 2,304, so the reconstruction bound is
unmoved; what shrinks is how much teacher FFN the router has to choose between. Whether that
costs measurable quality is an open question and the first thing a pilot should measure.

---

## What is deliberately not built

Recorded here so that nobody has to discover it from a stack trace.

**MTP is declared and not built.** The frozen specification asks for one MTP hidden layer.
`transformers` 5.15.1 exposes no `mtp_num_hidden_layers` field for `qwen3_5_moe_text` and
constructs no MTP head, and the teacher's own `mtp.*` tensors are discarded on load. The
architecture field is kept as the extension point; no MTP loss can be trained today and
any MTP result would be fabricated. `MTP_STATUS` in `architecture/moe_student.py` carries
the same statement in code, and a test pins it.

**DeltaNet state matching is not available.** The student's recurrent state and the
teacher's are different shapes, and comparing them needs a projection that is itself an
untested modelling choice. The behavioural delta term measures the same layers at the
hidden-size interface where the shapes genuinely do agree. Requesting the state term raises.

**No external compute has been used.** Everything in this repository was run in a browser
session with no GPU. The teacher has not been downloaded or executed here. Every number is
either an audit of a locally constructed model, an analytical estimate labelled as one, or
a third-party figure with a source.

---

## Ground rules

1. **Never call an estimate a measurement.** The ledger enforces this: `measured_here`,
   `reported_by_third_party` and `estimated` are a closed set, an estimate must carry its
   method, and a third-party number must carry its source.
2. **Never claim a benchmark result before running it.** Competitor figures are recorded
   with provenance and are not treated as ours to beat until ours exist.
3. **A negative result is a result.** Every ablation arm has a `falsified_if`. If the
   central claim fails, it is reported as failed.
4. **The 16 GB constraint is end-to-end and GPU-resident.** CPU offload is a different
   product and is never used to claim compliance.
5. **Nothing degrades silently.** An unavailable objective raises with the reason. A
   mismatched checkpoint is fatal, not a warning.

---

## Map

| document | question it answers |
|---|---|
| [STUDENT_ARCHITECTURE.md](STUDENT_ARCHITECTURE.md) | what exactly is being built, and what it weighs |
| [INITIALIZATION_METHOD.md](INITIALIZATION_METHOD.md) | how teacher weights become student weights |
| [BEHAVIORAL_DISTILLATION.md](BEHAVIORAL_DISTILLATION.md) | the paper's central mechanism |
| [CONTEXT_SPECIALIZATION.md](CONTEXT_SPECIALIZATION.md) | the paper's second component |
| [PARETO_EVALUATION.md](PARETO_EVALUATION.md) | the 16 GB accounting and the release frontier |
| [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md) | how results are recorded and retracted |
| [DISTILLATION_ROADMAP.md](DISTILLATION_ROADMAP.md) | what happens next, and in what order |
| [TEACHER_INTERFACE.md](TEACHER_INTERFACE.md) | the teacher, and the guards around loading it |
