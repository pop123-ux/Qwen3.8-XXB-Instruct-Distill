# Project Plan

## Objective

> Develop a Qwen3.8-derived instruct model that preserves as much of the teacher's
> benchmark capability as possible while operating within a 16 GB consumer-GPU
> deployment envelope, supporting a large context window, and reducing unnecessary
> reasoning-token expenditure and inference latency.

The parameter count is a **result**, not an input. Current analysis puts the feasible
ceiling at 13–21B depending on the required context (see
[`ARCHITECTURE.md`](ARCHITECTURE.md)); the final number comes from experiments.

### Competing objectives

Maximise: benchmark capability, instruction following, coding, mathematics, reasoning,
long-context performance, tokens/sec, capability per GB of VRAM, capability per FLOP.

Minimise: VRAM, active FLOPs, latency, unnecessary thinking tokens, unnecessary tool
calls, context consumed by reasoning, training cost, evaluation cost.

### The hard constraint

One consumer GPU with **16 GB VRAM**, where "fits" means the *complete* envelope —
weights + KV cache + recurrent state + activations + workspace + runtime overhead —
with real context and real generation, not a bare weight load. We budget **1 GiB
reserved** for driver and desktop, leaving 15.0 GiB usable. See
[`DEPLOYMENT_PLAN.md`](DEPLOYMENT_PLAN.md).

## Where the project stands

**Phase 1B: the teacher's metadata is verified. The runtime teacher is pending.**
No model has been trained. No benchmark has been run. There are no capability results.

Established in Phase 1B, **directly from the supplied checkpoint metadata**:

- `model_type: qwen3_5`, architecture `Qwen3_5ForConditionalGeneration`, loads natively
  with **no `trust_remote_code`** — so a student in this family is runnable by anyone
  with stock `transformers`. This was the open question with the largest effect on the plan.
- Every architecture dimension, and an **explicit** 64-entry layer layout (48 linear,
  16 full). The analytical model, fed the real config, yields **26,895,998,464**.
- Context 262144 is **native**: no `rope_scaling`, so it is not YaRN-extended.
- MTP declared with 1 layer, sharing the base embedding.
- Licence **Apache-2.0**.
- Reasoning controls are exactly `xhigh`, `medium`, `low`; the default is `xhigh`; and
  `medium` is **not** a no-op — it renders a distinct, shorter prompt. That refutes a
  secondary-source claim earlier phases had carried.

Established in Phase 1, empirically against the reference implementation:

- The analytical parameter model matches `transformers` **exactly** — zero delta on
  every component, checked by building the architecture on the `meta` device.
- The memory model's cache terms match a **real forward pass** byte-for-byte, and the
  DeltaNet recurrent state is measurably constant in sequence length.
- **MTP is a speculative-decoding draft model** (vLLM `_SPECULATIVE_DECODING_MODELS`),
  424.70M parameters (1.58%), discarded entirely by stock `transformers`.

Still blocked, all requiring the weights or a GPU: state-dict parameter count,
successful loading and generation, benchmark capability, reasoning-token behaviour, and
peak VRAM. See [`VERIFICATION.md`](VERIFICATION.md).

Established in Phase 0, analytically:

Established in Phase 0, analytically:

- The teacher does not fit 16 GB at any quantisation or context we modelled — at
  Q4_K_M its weights alone are 15.85 GiB.
- The binding constraint is **weights, not context**: the hybrid architecture already
  makes long context cheap (constant recurrent state; KV cache on only 16 of 64 layers).
- **63.6% of the teacher is FFN.** That is where the parameters must come from.
- The feasible ceiling under 15.0 GiB usable is ~21B at 32k context, ~16.5B at 128k,
  ~13.5B at 262k.

## Phases

### Phase 0 — Analysis infrastructure ✅

Architecture spec, exact parameter accounting, VRAM estimator, FLOP/bandwidth model,
constrained architecture search, teacher inspector, test suite.

### Phase 1 / 1B — Teacher verification ✅ metadata verified, runtime pending

Built and tested: loader verification, empirical validation of the analytical model,
the three-tier evaluation harness, the reasoning-control sweep (including a
template-level no-op detector), paired teacher/student comparison, and memory
benchmarking. All exercised end to end against a synthetic checkpoint.

Blocked on network/hardware access:

Everything in [`VERIFICATION.md`](VERIFICATION.md) marked Tier 2 or "open question"
must be closed against the real checkpoint before an architecture is chosen. The two
that can change the plan outright:

- **Does the 3.8 checkpoint load under stock `transformers`?** No `qwen3_8` module
  exists in `transformers==5.15.1`. If it needs `trust_remote_code` or an unreleased
  version, that constrains what we can train and what the community can run.
- **Does a smaller Qwen3.8-family base exist?** We found no evidence of one. If none
  exists, initialization strategy A below is unavailable.

**Deliverable:** committed inspector reports under `evaluations/baselines/`, plus a
verified spec at `configs/teacher/qwen3_8_27b.verified.json`.

### Phase 2 — Teacher baseline (`teacher_baseline_v1`)

No student result means anything without a baseline measured under *our* harness. Run
the teacher on the full evaluation suite and record everything needed to reproduce it:
benchmark version, split, sample count, prompt template, system prompt,
`reasoning_effort`, sampling parameters, context length, quantisation, engine, GPU,
software versions, date, commit hash.

Record **reasoning-efficiency metrics from the start** — thinking tokens, total
tokens, TTFT, latency, tool calls — at every `reasoning_effort` level including
disabled. The overthinking claim is central to this project and must be measured on
our own harness rather than inherited from forum reports.

Since the teacher does not fit 16 GB, baselining needs a larger GPU (rented or
borrowed). This is a one-time cost.

### Phase 3 — Initialization study

The student cannot be pretrained from scratch; that is a different and far more
expensive project. Options, to be evaluated empirically:

- **A. Smaller same-family base.** Cleanest if one exists. **Existence unconfirmed.**
- **B. Teacher surgery.** Build the student by structured reduction of the teacher —
  drop layers, narrow the FFN, prune DeltaNet heads — and transfer the weights that
  remain shape-compatible. Attractive because it inherits the teacher's tokenizer and
  representation space, which keeps logit distillation valid. Most likely primary path
  if A is unavailable.
- **C. Different-family base adapted to the hybrid architecture.** Expensive: DeltaNet
  layers would be randomly initialized.
- **D. Hybrid.** Transfer what transfers; carefully initialize the rest.

**Decision gate:** a small controlled run comparing initializations on identical data
and steps. Cheap, and it determines everything downstream.

### Phase 4 — Architecture selection

Take the analytical shortlist, train 3–5 candidates at reduced scale on identical
data, and measure. Priority comparisons:

- FFN multiplier 3.4x vs 3.0x vs 2.5x at matched total parameters
- depth vs width at matched VRAM
- attention interval 4 vs 6 — **specifically on long-context retrieval**, since this
  is where the capacity proxy is most likely to mislead (see `ARCHITECTURE.md`)
- tied vs untied embeddings
- dense vs MoE at matched *VRAM*, not matched active parameters

On MoE: sparse activation cuts compute but total expert weights still occupy VRAM,
and VRAM is our binding constraint. MoE is therefore a hypothesis to test, not a
design decision. If dense wins on capability-per-VRAM, use dense.

### Phase 5 — Distillation

Layered, in increasing order of cost:

1. **Hard-target** — teacher outputs as SFT targets. Baseline.
2. **Logit KD** — KL against the teacher's full distribution at temperature. Requires
   a **shared tokenizer**, which is why vocabulary reduction is off the table.
3. **Reasoning KD** — teacher traces on hard problems, with the explicit caveat below.
4. **Preference/ranking** — multiple candidates, ranked, for preference training.
5. **Difficulty-aware weighting** — different distillation strength by task class.

**The central tension:** reasoning KD is how the student recovers hard-task capability,
and it is also how the student inherits the teacher's overthinking. These pull in
opposite directions and must be measured against each other, not assumed to compose.

### Phase 6 — Reasoning efficiency

See [`reasoning-efficiency.md`](reasoning-efficiency.md). The objective is **adaptive
reasoning** — spend tokens in proportion to difficulty — not reasoning suppression.

The failure mode to guard against: aggressively penalising length produces a model
that answers fast, looks efficient, and fails hard problems. Every efficiency
experiment must report hard-task accuracy alongside token counts, and a drop in the
former invalidates a gain in the latter.

### Phase 7 — Long-context training

Curriculum over long documents, code and synthetic retrieval tasks. Measured as a
**curve** of score vs context length, not a single claimed number. If capability
collapses at 64k on a model advertising 128k, that gets documented.

### Phase 8 — Quantization and deployment

Architectural compression should let us ship a *good* 4-bit model rather than a
desperate 2-bit one. Sweep BF16 → FP8 → INT8 → 6/5/4-bit, measuring perplexity,
benchmarks, VRAM, throughput, and — separately — **long-context behaviour**, since
quantization may damage retrieval disproportionately.

### Phase 9 — Release

Weights, quantized variants, model card, evaluation results, hardware benchmarks,
known limitations, full training recipe. See [`DEPLOYMENT_PLAN.md`](DEPLOYMENT_PLAN.md).

## Experiment discipline

- Change **one major variable at a time**; a result from simultaneously changing
  architecture, data and objective is not a result.
- Record a hypothesis **before** the run.
- Every run logs: experiment ID, git commit, config, architecture, initialization,
  dataset version, seed, steps, LR, batch size, optimizer, schedule, GPU, VRAM,
  wall-clock, validation loss, proxy scores, reasoning-token metrics.
- Negative results are recorded, not discarded.

## Evaluation tiers

Full benchmarks after every experiment is unaffordable. Three tiers:

- **Tier 1** (every checkpoint, local): loss, perplexity, small math/coding/reasoning
  subsets, short long-context probes. Catches obvious regressions.
- **Tier 2** (promising checkpoints, single free/cheap GPU): representative subsets.
  Decides whether a checkpoint earns a full run.
- **Tier 3** (final candidates only): full suite, teacher-matched configuration.

## What would make this project fail

Named in advance, so they are recognisable when they happen:

- **Capacity is genuinely insufficient.** 15B may not hold 27B of capability at any
  distillation quality. Then the result is the measured trade-off curve — still a
  publishable finding.
- **Optimising the capacity proxy into a regression** — e.g. adopting interval 6
  because it ranks well, and shipping a model that cannot retrieve at 128k.
- **Reasoning efficiency traded against hard-task accuracy**, reported as a win by
  quoting only the token counts.
- **Benchmark contamination** from synthetic teacher data overlapping test sets.
- **Deployment claims from weight size alone**, rather than measured peak VRAM.

## Immediate next actions

1. Run `scripts/inspect_teacher.py` against the real checkpoint; commit the report.
2. Resolve the `transformers` support question (open question 1 in `VERIFICATION.md`).
3. Confirm the license and naming requirements before choosing a public model name.
4. Determine whether a smaller Qwen3.8-family base exists.
5. Establish `teacher_baseline_v1`, including reasoning-token metrics at every
   `reasoning_effort` level.

Only then choose an architecture.
