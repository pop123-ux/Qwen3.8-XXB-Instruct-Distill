# Qwen3.8 Distillation Research Program — Canonical Claude Skill

## 0. ROLE OF THIS SKILL

This skill is the canonical operating context for the Qwen3.8-27B distillation research program.

Claude must treat this file as the project's standing research protocol and institutional memory.

It exists so that a new Claude session can continue the project without reconstructing the research direction from scratch.

This skill governs:

* research questions;
* hypotheses;
* architecture invariants;
* teacher provenance;
* student provenance;
* training methodology;
* experiment design;
* GPU execution;
* compute budgeting;
* reproducibility;
* evaluation;
* plotting;
* paper development;
* future external-compute requests;
* interpretation of positive and negative results;
* Git/repository discipline.

The project owner may explicitly change this skill.

Claude must never silently override it.

---

# 1. PROJECT IDENTITY

Working research program:

**Qwen3.8-27B → qwen38_19b_h5120_l48_moe**

Primary working paper:

**"Beyond Layer Matching: Distilling Computational Behavior in Hybrid Language Models"**

Secondary research question:

**Context-Length Specialization During Distillation**

The project is an open research investigation.

It is not currently a claimed SOTA result.

It must distinguish:

* demonstrated facts;
* engineering validations;
* empirical observations;
* hypotheses;
* future proposals.

No capability claim may be made without an experiment supporting it.

---

# 2. CORE RESEARCH QUESTIONS

## RQ1 — Beyond Layer Matching

### Question

When the teacher and student have different computational topologies, is it more effective to distill the **computational behavior produced by those systems** than to force a direct layer-to-layer correspondence?

### Teacher

Qwen3.8-27B:

* 64 layers;
* hybrid DeltaNet + full attention;
* dense FFN;
* 24 query heads;
* 4 KV heads.

### Student

`qwen38_19b_h5120_l48_moe`:

* 48 layers;
* hybrid DeltaNet + full attention;
* sparse MoE;
* 24 query heads;
* 2 KV heads.

Therefore teacher and student are not isomorphic.

The research must explicitly distinguish:

1. **ordinary CE training;**
2. **logit KD;**
3. **intermediate/layer KD;**
4. **computational-behavior/state KD;**
5. **architecture-aware initialization.**

### Core hypothesis

Direct layer correspondence may be an inferior abstraction when the computational topology changes.

Behavior/state/function-level supervision may transfer information that does not have a natural one-to-one layer mapping.

This is a hypothesis.

It must not be written as a conclusion before experiments establish it.

---

# 3. SECONDARY RESEARCH QUESTION — CONTEXT SPECIALIZATION

## RQ2 — Context-Length Specialization

### Question

Does the optimal training-context distribution differ from the final deployment context, and can deliberate context specialization produce better capability/efficiency tradeoffs than:

* uniform-context training;
* maximum-context training;
* or another default mixture?

### Context regimes

The project tracks:

* 4K
* 8K
* 16K
* 32K
* 64K
* 128K
* 262K

The project must distinguish:

* native architectural context;
* training context;
* evaluation context;
* demonstrated deployment context;
* demonstrated 16 GB context.

Do not equate `max_position_embeddings=262144` with proof of 262K operational capability.

### Candidate training distributions

* uniform context mixture;
* short-context-heavy;
* medium-context-heavy;
* long-context-heavy;
* progressive context schedule.

This is a hypothesis.

---

# 4. FUTURE RESEARCH QUESTION — INITIALIZATION

## RQ3 — Architecture-Aware Initialization

Investigate whether teacher-derived initialization materially improves:

* convergence;
* teacher agreement;
* sample efficiency;
* stability;
* final capability

relative to randomly initialized student weights.

This question should be treated as a secondary ablation.

The architecture itself remains frozen.

The point is to test initialization, not redesign the student.

---

# 5. FUTURE RESEARCH QUESTION — EFFICIENCY

## RQ4 — Quality / Memory / Compute Frontier

Investigate whether the student provides a useful tradeoff between:

* quality;
* active parameters/token;
* total parameters;
* VRAM;
* throughput;
* latency;
* context length.

This is a systems question, not evidence for RQ1 by itself.

---

# 6. FROZEN TEACHER

Canonical teacher:

`Qwen/Qwen3.8-27B`

Canonical revision:

`dbdc473dea0d6a9763042881cc33d6058d1742d2`

Never silently substitute another revision.

Older revision:

`72a217afab8029b39e4af1c7273a829995a3dbaf`

must not be used as the canonical research provenance.

The adopted revision is intentional because the model weights are byte-identical to the earlier weights-upload revision while the metadata/provenance matches the project's reasoning-mode contract.

Every real experiment must record the exact teacher revision.

---

# 7. FROZEN PRIMARY STUDENT

Identifier:

`qwen38_19b_h5120_l48_moe`

Exact total parameters:

`13,008,505,728`

Exact active parameters/token:

`9,611,119,488`

## Architecture

* hidden size = 5120
* layers = 48
* DeltaNet layers = 36
* full-attention layers = 12
* layout = `[DeltaNet, DeltaNet, DeltaNet, Attention] × 12`
* vocabulary/embedding size = 248320
* maximum context = 262144
* query heads = 24
* KV heads = 2
* head dimension = 256
* partial rotary factor = 0.25
* RoPE theta = 10000000
* DeltaNet key heads = 16
* DeltaNet value heads = 48
* DeltaNet head dimension = 128
* convolution kernel = 4
* recurrent state = fp32
* routed experts = 8
* expert width = 768
* routing = top-2
* shared expert = 1
* router auxiliary loss = 0.001
* embeddings = untied
* normalization = RMSNorm
* MTP = declared but not implemented by current runtime

These are frozen research invariants.

If a contradiction is found:

1. stop the affected action;
2. document the contradiction;
3. report it;
4. do not silently redesign the architecture.

---

# 8. WHY THIS STUDENT EXISTS

The student is designed as a computationally different but substantially capable descendant of the teacher.

Transformation:

```text
Qwen3.8-27B
64 layers
hybrid DeltaNet + attention
dense FFN
24 Q / 4 KV
        ↓
teacher-derived transformation
        ↓
13.0085B student
48 layers
36 DeltaNet + 12 attention
sparse MoE
24 Q / 2 KV
```

The project is specifically interested in whether useful teacher behavior can survive this nontrivial topology transformation.

Do not simplify the research story into:

"smaller Qwen model."

The research story is:

**distilling useful computation across a changed computational topology.**

---

# 9. HARD DEPLOYMENT TARGET

The student is intended to support:

**16 GB end-to-end GPU-resident inference.**

This includes:

* weights;
* quantization overhead;
* KV cache;
* recurrent state;
* runtime/workspace.

A training-memory measurement must never be called a deployment result.

The project must separately measure:

* training memory;
* inference memory;
* long-context memory;
* quantization effects.

---

# 10. CURRENT RESEARCH STATUS

The project has passed the following engineering milestones:

## Teacher integration

* real Qwen3.8-27B checkpoint available locally;
* pinned revision verified;
* pretrained weight loading verified;
* tokenizer verified;
* reasoning-mode handling verified;
* real online teacher logits available.

There remains a known quantized prefix-consistency numerical issue:

* argmax agreement is effectively identical;
* small relative logit differences remain under 4-bit execution.

This must remain documented.

Do not claim exact numerical equivalence under 4-bit.

## Student materialization

The canonical student has been materialized.

Verified:

* exact 13,008,505,728 base parameters;
* 48 layers;
* 36 DeltaNet;
* 12 attention;
* 8 routed experts;
* top-2;
* shared expert;
* no missing transferred tensors;
* complete transfer;
* documented coverage.

## Training infrastructure

QLoRA is now genuinely implemented.

Current canonical mechanism:

* frozen quantized student base;
* LoRA adapters;
* real teacher;
* real teacher logits;
* actual forward;
* actual KD loss;
* actual backward;
* actual optimizer step;
* persistent checkpoints;
* experiment ledger.

---

# 11. RUN HISTORY

## Run 001 — Canonical KD Mechanism Validation

### Purpose

Prove that the real teacher can drive the real canonical student through the actual KD stack on the A40.

### Configuration

* teacher = Qwen3.8-27B;
* teacher revision = dbdc473...;
* teacher = 4-bit;
* student = canonical 13B;
* QLoRA;
* LoRA r=16;
* alpha=32;
* sequence length = 1024;
* batch = 1;
* mixed KD;
* KD weight = 0.5;
* CE weight = 0.5.

### Result

1-step smoke + 50-step pilot completed.

Pilot:

* 51,200 tokens;
* ~598.8 s;
* validation loss decreased;
* teacher/student agreement increased;
* adapters received gradients;
* checkpoints validated.

### Memory

Approximately:

* 37.79 GiB peak allocated;
* 38.30 GiB peak reserved;
* ~40.6 GB observed via `nvidia-smi` during pilot activity.

### What Run 001 demonstrates

Run 001 demonstrates:

> The canonical 13B student can be trained through the real teacher-in-the-loop KD mechanism using QLoRA on the A40.

### What it does NOT demonstrate

Run 001 does NOT demonstrate:

* capability improvement;
* superiority of logit KD;
* superiority of layer KD;
* superiority of behavioral KD;
* SOTA performance;
* long-context capability;
* context specialization.

It is an engineering/mechanism validation experiment.

---

# 12. RUN 002 STATUS

The first Run 002 attempt consisted of a **2048-token calibration**.

### Configuration

* pure logit KD;
* QLoRA;
* 2048 sequence length;
* batch 1.

### Measured result

* peak allocated = 43.0137 GiB;
* peak reserved = 43.5410 GiB;
* A40 usable = ~44.43 GiB.

The project's 42 GiB safety gate therefore failed.

### Consequence

The 128-step 2048-token Run 002 was **not launched**.

This calibration is not the Run 002 research result.

It is a memory calibration.

### Current approved Run 002 concept

Run 002 should be:

**Pure Logit KD Control**

with:

* sequence length = 1536;
* batch = 1;
* 128 optimizer steps;
* QLoRA;
* r=16;
* alpha=32;
* pure KD;
* temperature = 2.0;
* top-k = 64;
* seed = 0.

Expected target tokens:

`196,608`

The sequence length must be recalibrated first.

---

# 13. RESEARCH EXPERIMENT LADDER

The core scientific ladder is:

```text
Run 001
Engineering mechanism validation
        ↓
Run 002
Pure Logit KD
        ↓
Run 003
Layer / Intermediate KD
        ↓
Run 004
Computational-Behavior / State KD
        ↓
Controlled analysis
        ↓
Context specialization
        ↓
16 GB deployment evaluation
        ↓
Paper
```

The exact number of runs is not sacred.

Scientific comparability is more important than run count.

---

# 14. CORE CONTROLLED EXPERIMENTS

## Baseline A — CE-only

Question:

What can the canonical student learn without teacher distillation?

## Baseline B — Logit KD

Question:

What does conventional teacher-logit supervision provide?

## Baseline C — Layer KD

Question:

Does direct intermediate/layer matching provide additional benefit?

## Proposed — Behavioral/State KD

Question:

Does supervising computational behavior/state outperform direct layer correspondence?

Whenever possible, these must share:

* same student;
* same teacher;
* same tokenizer;
* same dataset;
* same token budget;
* same seed policy;
* same optimizer;
* same PEFT setup;
* same precision;
* same evaluation procedure.

The intended experimental difference should be the distillation objective itself.

---

# 15. BEHAVIORAL / STATE KD DESIGN PRINCIPLE

The proposed method should not simply create a renamed layer loss.

It should measure or supervise quantities that are meaningful for the computation being performed.

Possible signals include:

* hidden states;
* DeltaNet recurrent states;
* attention outputs or attention-derived behavior;
* FFN/MoE reconstruction behavior;
* logits;
* function/output agreement;
* teacher/student state transitions.

For non-isomorphic components, prefer functionally meaningful correspondences over arbitrary layer indices.

Any final formulation must be specified mathematically in the research documentation before being presented as the main method.

---

# 16. CONTEXT EXPERIMENT PROGRAM

After the core KD comparison, investigate:

### Training distributions

* uniform;
* short-heavy;
* medium-heavy;
* long-heavy;
* progressive.

### Evaluation lengths

* 4K;
* 8K;
* 16K;
* 32K;
* 64K;
* 128K;
* 262K.

The primary question is:

> Does the training-context distribution change the capability curve across evaluation context lengths?

Do not cherry-pick only the context where a method wins.

Report the entire curve when data exists.

---

# 17. CONTEXT × MEMORY EXPERIMENT

Eventually characterize:

```text
context length
        ×
quantization
        ×
peak VRAM
        ×
throughput
```

This produces an important systems result for the 16 GB target.

Do not call a context feasible based solely on theoretical memory estimates.

Measure it.

---

# 18. MOE RESEARCH

Measure:

* expert utilization;
* expert token counts;
* routing entropy;
* dead experts;
* load imbalance;
* utilization over training;
* utilization by context;
* specialization where identifiable.

Do not call an expert "specialized" unless measured behavior supports the statement.

Do not assume sparsity automatically produces useful specialization.

---

# 19. EVALUATION HIERARCHY

Evaluation must proceed from cheapest to most scientifically informative.

## Level 1 — Mechanism metrics

* training loss;
* validation loss;
* KD loss;
* CE loss;
* teacher entropy;
* top-1 teacher agreement;
* tail mass;
* state similarity.

## Level 2 — Language-model quality

Use reproducible held-out evaluation.

## Level 3 — Capability benchmarks

Only once the model has received enough training to make capability measurement meaningful.

## Level 4 — Deployment

Measure:

* memory;
* throughput;
* latency;
* context handling;
* quantization stability.

Do not use infrastructure success as a substitute for capability evaluation.

---

# 20. STATISTICAL / SCIENTIFIC DISCIPLINE

Where budget permits:

* use multiple seeds for key claims;
* report variance;
* use consistent evaluation sets;
* preserve raw metrics;
* avoid post-hoc metric selection.

When multiple seeds are impossible because of compute:

* state the limitation;
* avoid overstating certainty;
* distinguish exploratory findings from confirmed findings.

---

# 21. PAPER STRUCTURE

The eventual paper should contain approximately:

## 1. Introduction

* problem;
* topology mismatch;
* limitations of direct layer matching;
* proposed behavioral/state perspective;
* 16 GB motivation.

## 2. Related Work

Cover:

* knowledge distillation;
* intermediate/layer distillation;
* cross-architecture distillation;
* state/function supervision;
* hybrid recurrent/attention architectures;
* MoE distillation;
* long-context training;
* context specialization.

Never claim novelty merely because the project uses a different model combination.

Novelty must be argued at the method/problem level.

## 3. Teacher and Student

Document exact architectures.

## 4. Architecture Transformation

Explain:

* depth reduction;
* DeltaNet/attention layout;
* KV reduction;
* FFN→MoE transformation;
* teacher-derived initialization.

## 5. Distillation Methods

Define:

* CE;
* logit KD;
* layer KD;
* behavioral/state KD.

## 6. Context Specialization

Define training distributions and evaluation grid.

## 7. Experimental Setup

Include:

* compute;
* tokens;
* seeds;
* optimizer;
* PEFT;
* precision;
* teacher revision;
* dataset;
* context length.

## 8. Results

Only real measurements.

## 9. Ablations

Test which components matter.

## 10. Deployment

16 GB constraints and measured behavior.

## 11. Limitations

Be explicit.

## 12. Conclusion

Only claims supported by data.

---

# 22. PAPER CLAIM STANDARD

Every important statement must fall into one category:

### Demonstrated

Supported by a reproducible experiment.

### Observed

Measured but not necessarily causally established.

### Hypothesized

Proposed before evidence.

### Planned

Not yet executed.

Claude must label these internally and avoid collapsing them.

---

# 23. PAPER FIGURE PROGRAM

Eventually produce:

### Figure 1 — Model Compression

Teacher vs student:

* total parameters;
* active parameters/token;
* depth;
* expert sparsity.

### Figure 2 — 16 GB Pareto Frontier

x:

peak VRAM.

y:

quality metric.

Optional:

tokens/sec or context.

Highlight:

* canonical student;
* constraint boundary;
* relevant competitors.

### Figure 3 — Context Specialization

Training context distribution versus evaluation performance across context lengths.

### Figure 4 — Distillation Recovery

Training tokens/steps versus:

* teacher KL;
* validation loss;
* benchmark score.

Compare:

* CE;
* logit KD;
* layer KD;
* behavioral KD.

### Figure 5 — Teacher/Student Computational Behavior

Where data exists:

* hidden-state similarity;
* DeltaNet-state similarity;
* attention similarity;
* FFN/MoE reconstruction error;
* logits similarity.

### Figure 6 — MoE Routing

* utilization;
* entropy;
* tokens/expert;
* dead experts;
* imbalance.

### Figure 7 — Context × Memory

context length versus peak VRAM with quantization curves.

---

# 24. PLOT TRACEABILITY

Every plotted point must be traceable to:

* experiment ID;
* source artifact;
* source metric;
* configuration.

Plotting scripts must read structured artifacts.

Never hard-code experimental results.

Plots must never fabricate missing points.

If no result exists:

* leave it absent;
* or fail with a clear missing-data message.

---

# 25. REPRODUCIBILITY STANDARD

Every canonical run must preserve:

* exact Git SHA;
* teacher revision;
* tokenizer hashes;
* dataset hashes;
* architecture hash;
* experiment configuration;
* environment;
* hardware;
* random seed;
* checkpoints;
* metrics;
* logs;
* checksums;
* termination state.

A researcher should be able to answer:

> What code, model, data, environment and configuration produced this number?

without guesswork.

---

# 26. LOCAL COMPUTE STRATEGY

Current local environment:

* RunPod A40 48 GB;
* project GPU budget = $50 maximum.

Use local GPU time primarily for:

* pipeline validation;
* controlled pilots;
* mechanism tests;
* selected comparative experiments;
* deployment measurements.

Do not attempt full-parameter 13B AdamW training on the A40.

---

# 27. FUTURE FULL-DISTILLATION STRATEGY

The local $50 phase is a **proof-of-method phase**, not necessarily the final training phase.

If the controlled experiments produce promising evidence, prepare for a second research phase:

## Phase II — Full Distillation

Goal:

Perform a substantially larger distillation campaign using donated, sponsored, academic, or lab compute.

Potential targets:

* full-parameter training;
* much larger token budget;
* more seeds;
* richer datasets;
* longer training;
* full context curriculum;
* extensive ablations.

The local evidence should make a future compute request concrete rather than speculative.

---

# 28. EXTERNAL COMPUTE / LAB PROPOSAL STRATEGY

If the project produces promising early evidence, Claude should help prepare a professional compute request.

The request should NOT say:

"Give me GPUs to see whether my idea works."

Instead:

"Here is the hypothesis, here is the frozen architecture, here is the local validation, here is the measured evidence, here is the missing experiment, here is the estimated compute requirement, and here is exactly what the donated compute would establish."

A strong future compute proposal should contain:

### A. Research hypothesis

Clearly stated and falsifiable.

### B. Evidence already obtained

Include:

* teacher verification;
* student materialization;
* canonical KD feasibility;
* measured memory;
* preliminary KD behavior.

### C. Missing evidence

Explicitly state that large-scale capability comparison has not yet been demonstrated.

### D. Proposed experiment

Provide:

* teacher;
* student;
* token budget;
* precision;
* optimizer;
* context distribution;
* baselines;
* evaluation;
* seeds.

### E. Compute estimate

Break down by:

* training hours;
* GPU type;
* GPU count;
* storage;
* checkpoint frequency;
* evaluation compute.

### F. Scientific deliverables

Examples:

* controlled KD comparison;
* state-behavior analysis;
* context specialization curves;
* deployment measurements;
* reproducible artifacts;
* paper/preprint.

### G. Open release plan

Where appropriate:

* code;
* configs;
* experiment records;
* plots;
* checkpoints;
* final model;
* reproducibility package.

Never exaggerate expected impact to obtain compute.

---

# 29. COMPUTE REQUEST PHILOSOPHY

The strongest request will come after the project can say:

> "We have already demonstrated the entire pipeline and obtained preliminary controlled evidence. The remaining uncertainty is a specific scientific question requiring larger compute."

That is much stronger than requesting hardware before any empirical validation.

Therefore:

**Local phase = establish credibility.**

**External-compute phase = establish statistical/scientific strength.**

---

# 30. FUTURE FULL-DISTILLATION EXPERIMENT PLAN

When sufficient compute becomes available, the preferred sequence is:

## Stage I

Teacher reproduction and final provenance lock.

## Stage II

Final student materialization and initialization audit.

## Stage III

Small real KD sanity check.

## Stage IV

Matched baselines:

* CE;
* logit KD;
* layer KD;
* behavioral KD.

## Stage V

Scaling study:

* increasing token budget;
* increasing context;
* convergence behavior.

## Stage VI

Context specialization.

## Stage VII

Ablations.

## Stage VIII

Deployment evaluation.

## Stage IX

Statistical confirmation with additional seeds where feasible.

## Stage X

Final paper and public release.

---

# 31. DO NOT OPTIMIZE FOR PAPER APPEARANCE

Do not:

* make plots look stronger than the data;
* hide failed experiments;
* remove negative results;
* cherry-pick checkpoints;
* omit expensive controls because they weaken the story;
* retroactively modify experiment definitions.

The research record must remain faithful to what happened.

---

# 32. EXPERIMENT COST ACCOUNTING

For each GPU run record:

* GPU type;
* GPU hours;
* cost;
* tokens processed;
* tokens/sec;
* peak VRAM;
* useful research output.

Also track:

`scientific information gained / dollar`

Prefer experiments with high expected information value.

---

# 33. RUN VALIDITY CLASSES

Each run must have one class:

### Calibration

Answers whether a configuration fits or behaves safely.

### Mechanism validation

Answers whether the pipeline functions.

### Controlled research experiment

Tests a defined hypothesis against a control.

### Ablation

Tests causal importance of a component.

### Capability evaluation

Measures model quality.

### Systems evaluation

Measures deployment behavior.

Never mix these categories.

---

# 34. WHAT COUNTS AS SUCCESS

## RQ1 success

Evidence that behavioral/state KD improves a predefined metric relative to layer KD and/or conventional KD under controlled conditions.

Possible metrics:

* teacher agreement;
* state similarity;
* held-out loss;
* benchmark capability;
* convergence efficiency.

The metric must be declared before interpreting the result.

## RQ1 negative result

Behavioral/state KD does not outperform layer KD.

This remains a scientifically useful result.

## RQ2 success

A context-specialized curriculum produces a reproducible improvement in the capability/context tradeoff relative to the control.

## RQ2 negative result

No meaningful advantage is found.

Also scientifically useful.

---

# 35. LIMITATIONS THAT MUST REMAIN VISIBLE

Current known limitations include:

* local compute is insufficient for broad full-parameter 13B training;
* QLoRA/PEFT means early experiments are not equivalent to full-parameter optimization;
* short pilots are not capability evidence;
* 4-bit teacher introduces measurable numerical differences;
* limited GPU budget restricts seeds/sweeps;
* large-context capability is not demonstrated merely by declaring 262K;
* benchmark breadth may initially be limited.

Do not hide these limitations in paper preparation.

---

# 36. REPOSITORY STRUCTURE

Keep one canonical path for each function:

```text
src/
    qwen_distill/
        architecture/
        distillation/
        training/

scripts/
    teacher_smoke_test.py
    distill_pilot.py
    kd_run.py
    run_record.py
    ...

docs/
    RESEARCH.md
    REAL_TEACHER_RUN.md
    DISTILLATION_ROADMAP.md
    RUN_001.md
    ...

experiments/
    ledger.jsonl
    kd_run_001/
    ...

plots/
    README.md
    common.py
    outputs/
        paper/
        readme/
```

Avoid duplicate execution paths.

---

# 37. REPOSITORY CLEANLINESS

Never commit:

* teacher weights;
* student checkpoints;
* optimizer state;
* datasets;
* credentials;
* tokens;
* caches;
* huge generated artifacts.

Commit:

* code;
* configs;
* run metadata;
* concise metrics;
* reproducibility manifests;
* research documentation;
* plotting code;
* paper figures when legitimately derived from real artifacts.

---

# 38. GIT RULES DURING EXPERIMENTS

Before launch:

```bash
git status --short
git rev-parse HEAD
git branch --show-current
```

Record the launch SHA.

Do not alter the experimental code path while a run is active unless the process is isolated and the exact pre-run SHA is already recorded.

After completion:

1. validate artifacts;
2. inspect diff;
3. run relevant tests;
4. run lint;
5. run `git diff --check`;
6. commit deliberate changes;
7. push only when authorized;
8. verify remote SHA.

---

# 39. ACTIVE RUN RULE

During training:

Do not:

* launch another training process on the same GPU;
* run CUDA-heavy tests unnecessarily;
* launch broad model-loading test suites;
* alter configuration;
* silently change sequence length;
* automatically retry failed experiments;
* automatically start the next run.

A temporary GPU utilization drop is not by itself evidence of failure.

Use:

* run logs;
* metrics;
* process state;
* checkpoint timestamps;
* GPU state

to diagnose the run.

---

# 40. CLAUDE BEHAVIOR RULES

Claude must prioritize:

1. scientific validity;
2. reproducibility;
3. budget efficiency;
4. safety;
5. execution speed.

Claude should avoid wasting paid GPU time on:

* repeated audits;
* rereading unchanged files;
* redundant subagents;
* full test suites that load models;
* cosmetic refactors;
* unnecessary dependency work.

When a real blocker is found:

* identify the minimum fix;
* implement only that fix;
* test it;
* continue.

Do not use a blocker as an excuse for broad redesign.

---

# 41. WHEN TO PAUSE AND ASK THE PROJECT OWNER

Claude should stop rather than improvise when:

* architecture would need changing;
* teacher revision would need changing;
* research question would change;
* experimental control would be invalidated;
* budget would materially increase;
* a new training methodology would alter interpretation;
* a proposed result would require unsupported claims.

---

# 42. WHEN NOT TO ASK

Claude should not waste time asking for approval for:

* ordinary file inspection;
* safe artifact validation;
* standard checks;
* known calibration steps;
* predefined run execution;
* routine metric recording;
* routine checkpoint validation.

Use the existing protocol.

---

# 43. PAPER / PREPRINT READINESS CHECKLIST

Before claiming a paper is ready:

* [ ] RQ1 clearly defined
* [ ] RQ2 clearly defined
* [ ] baselines defined
* [ ] proposed method mathematically specified
* [ ] architecture documented
* [ ] teacher provenance pinned
* [ ] student provenance pinned
* [ ] dataset provenance recorded
* [ ] controlled experiments completed
* [ ] key ablations completed
* [ ] capability evaluation completed
* [ ] deployment evaluation completed
* [ ] figures traceable
* [ ] negative results retained
* [ ] limitations documented
* [ ] reproducibility package complete
* [ ] no fabricated claims
* [ ] no unsupported SOTA language

---

# 44. FUTURE MODEL RELEASE READINESS

A public model release should include:

* model configuration;
* tokenizer information;
* exact base/student provenance;
* quantization information;
* usage documentation;
* license information;
* reproducibility information;
* known limitations;
* benchmark results;
* memory results;
* context results.

Never release a model as a proven research success before its evaluation is complete.

---

# 45. PROJECT END STATES

The project may legitimately end in several ways:

## Outcome A — Strong positive result

Behavioral/state KD materially outperforms controls.

Proceed to:

* larger confirmation runs;
* external compute;
* publication;
* public release.

## Outcome B — Mixed result

Behavioral KD improves some metrics but not others.

Analyze where and why.

This may yield the most interesting paper.

## Outcome C — Null result

Behavioral KD does not beat conventional methods.

Publishable as a careful negative result if the experimental design is strong.

## Outcome D — Systems result

The architecture is useful under a 16 GB constraint even if the central research hypothesis is not established.

Separate the systems result from the scientific claim.

## Outcome E — Methodological blocker

Training or evaluation limitations prevent a strong conclusion.

Document the limitation rather than forcing a conclusion.

---

# 46. THE STRATEGIC ROADMAP

The whole project should be understood as:

```text
PHASE 0
Architecture + provenance
        ↓
PHASE 1
Teacher verification
        ↓
PHASE 2
Student materialization
        ↓
PHASE 3
KD mechanism validation
        ↓
PHASE 4
Controlled KD comparison
        ↓
PHASE 5
Context specialization
        ↓
PHASE 6
Behavior/state analysis
        ↓
PHASE 7
16 GB deployment characterization
        ↓
PHASE 8
Statistical confirmation
        ↓
PHASE 9
Paper / preprint
        ↓
PHASE 10
External compute proposal
        ↓
PHASE 11
Full-scale distillation
        ↓
PHASE 12
Final model + open release
```

---

# 47. CURRENT PHASE

The project is in:

**Phase 4 — controlled KD comparison**

Completed:

* Run 001 — mechanism validation.
* Run 002 calibration at 2048 — failed the safety gate; sequence reduced to 1536.
* Run 002 calibration at 1536 — passed.
* **Run 002 — pure logit KD, 128 steps, 196,608 tokens. COMPLETE.** Peak allocated
  40.58 GiB. Record `experiments/run002_logit_kd/`; the trained adapter is preserved in
  the repository through Git LFS.
* Run 003 calibration at 1536, unchunked — **failed** the gate at 42.5354 GiB. The
  scientific configuration was not changed in response.
* Chunked layer-KD evaluation implemented and proved equivalent: student hidden-state
  gradients bit-identical at all 48 supervised pairs, objective differing by 6.424e-08
  relative from float32 summation order alone. Record
  `experiments/run003_chunking_equivalence/`.
* Run 003 calibration at 1536, chunked over 4 mapped pairs — **passed** at 38.0289 GiB
  allocated / 39.8906 GiB reserved against a 42.0 GiB gate and 44.4316 GiB usable.
  Margin 3.9711 GiB. Record `experiments/run003_calibration_1536_chunked/`.

**The next intended action is the 128-step Run 003.**

It is prepared and needs no design work. Take
`experiments/run003_calibration_1536_chunked/command.txt` verbatim and change exactly one
thing: `--steps 1` becomes `--steps 128`. Nothing else may move — not the sequence length,
not the teacher, not the revision, not the student, not the seed, not the optimizer, not
the QLoRA geometry, not the batch, not the 48 supervised pairs, not the objective. Run 002
and Run 003 are a controlled pair and `scripts/record_run003_kd.py` checks that field by
field when it writes the ledger entry.

Before spending any budget, verify the re-materialised student: exactly
**13,008,505,728** parameters, `missing == []`, coverage **0.999830**. If any of those
three fails, stop and do not train.

**38.0289 GiB is a calibration result, not a training result.** The 128-step Run 003 has
not been run. Do not confuse calibration with research results, and do not let the passed
gate be reported as an outcome of Run 003.

Run 004 (computational-behaviour / state KD) must not be started until Run 003 is complete
and recorded.

---

# 48. IMMEDIATE EXPERIMENT PRIORITY

The immediate sequence is:

```text
Run 002
Pure Logit KD
        ↓
Run 003
Layer KD
        ↓
Run 004
Behavioral / State KD
        ↓
Compare under matched conditions
        ↓
Decide what context experiments are worth the remaining budget
```

Do not spend substantial budget on context experiments before establishing the core KD comparison.

---

# 49. LOCAL $50 PHASE OBJECTIVE

The $50 local compute allocation is not intended to produce the final fully optimized model.

Its purpose is to answer:

1. Is the training mechanism viable?
2. Is conventional KD measurably useful?
3. Does layer KD add value?
4. Is behavioral/state KD promising enough to justify larger-scale training?
5. Is the architecture sufficiently interesting to justify an external compute request?
6. What specific evidence should a lab/team receive in a compute proposal?

The local phase should maximize **evidence density per dollar**.

---

# 50. EXTERNAL-COMPUTE GATE

Do not approach a lab/team merely because the model is large.

A compute request becomes justified when the project can present:

* reproducible pipeline;
* frozen architecture;
* pinned teacher;
* successful local KD;
* controlled baseline;
* preliminary evidence for the proposed method;
* precise unanswered question;
* estimated compute requirement;
* expected deliverables.

The request should be framed as:

**"Here is a validated research experiment that now requires scale."**

not:

**"I have an idea and need GPUs."**

---

# 51. FINAL PRINCIPLE

This project is fundamentally an empirical research program.

The objective is not:

"make the biggest model possible."

The objective is:

> Determine whether computational behavior can be distilled across a changed hybrid topology, determine how context specialization interacts with that process, and quantify the resulting quality/efficiency tradeoff under realistic deployment constraints.

The project must remain falsifiable.

A negative result is acceptable.

A failed experiment is acceptable.

A smaller result than expected is acceptable.

An unsupported claim is not.

Every GPU dollar, every run, every plot, and every paragraph of the paper should move toward a reproducible answer to the research questions.

