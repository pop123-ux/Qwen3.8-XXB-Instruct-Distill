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
| FFN | dense, 17408 wide | 24 experts x 768, top-2 | same |
| KV heads | 4 | 2 | same |

Width, vocabulary and the hybrid pattern are unchanged from the teacher. That is deliberate:
holding them fixed is what makes the three reductions attributable.

---

## What has been measured, and what it changed

Two audits were run before any training code was written, and both returned answers that
changed the plan. They are recorded here rather than in a footnote because they are the two
facts a reader most needs in order to interpret everything else.

### The frozen target weighs 22.07B, not 19B

```
exact_parameter_count     22,072,134,528
difference_from_19B       +3,072,134,528   (+16.2%)
non_embedding             19,529,337,728
active_per_token           9,615,051,648
```

The name is the label the project inherited; 19.53B non-embedding is the likely origin of
it. The architecture is frozen, so the architecture is not adjusted to make the label true
— the difference is reported. `routed_experts` are 61.57% of the total, which is the single
most important number for every memory decision that follows.

Reproduce with `python scripts/student_report.py --section architecture`.

### It does not fit 16 GB

At Q4, Q5 or Q6, at any context length from 2,048 tokens upward, fully GPU-resident:

```
best all-Q4 configuration, 2,048 tokens    13.93 GiB
usable on a real 16 GB card                13.56 GiB
shortfall                                   0.37 GiB
```

The shortfall is small, which is why it is reported to two decimals rather than rounded to
"too big". Fits begin one precision step below the release set, at 3-bit experts, reaching
65,536 tokens.

A 16 GB card reports 14.56 GiB, and a process that must coexist with a display server
should not plan on the last gigabyte of that. Planning against the nominal 16.0 GiB instead
makes Q4 appear to reach 65,536 tokens. That gap is the entire difference between a plan
that works and an out-of-memory error on hardware that "obviously" had room.

Full accounting in [PARETO_EVALUATION.md](PARETO_EVALUATION.md);
`python scripts/student_report.py --section memory`.

**This is an open decision, not a solved problem.** Two honest options, and the choice
belongs to whoever owns the release: ship the 22.07B target at 3-bit experts and report the
quality cost, or reduce the expert budget. Experts are 61.6% of the weights and each token
touches 2 of 24, so expert count and expert width are the only levers with enough mass to
close 0.37 GiB without touching the path every token depends on. The repository does not
make that choice unilaterally, and it does not hide the constraint by quietly re-scoping
the target.

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
