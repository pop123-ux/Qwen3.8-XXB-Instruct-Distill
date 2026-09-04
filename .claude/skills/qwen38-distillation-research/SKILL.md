---
name: qwen38-distillation-research
description: Canonical operating rules and scientific state for the Qwen3.8 heterogeneous-topology distillation research program.
---

# Qwen3.8 Distillation Research Lab — Canonical Skill

This skill is the operational memory for `pop123-ux/Qwen3.8-XXB-Instruct-Distill`.

The project is an empirical research program, not a model-building demo. Optimize decisions in this order:

1. scientific validity;
2. reproducibility;
3. information gained per GPU dollar;
4. preservation of evidence;
5. execution speed.

Never manufacture a result, silently change a protocol, redesign the frozen architecture, or turn an engineering success into a capability claim.

---

## 1. Research questions

### RQ1 — Beyond Layer Matching

When teacher and student computational topologies differ substantially, can topology-aware or dynamics-aware distillation transfer useful computation more effectively than conventional pointwise intermediate-state matching?

The student is not a depth-truncated clone. It changes depth, KV geometry and dense-FFN topology, so correspondence itself is part of the research problem.

### RQ2 — Context-Length Specialization

Under matched total training tokens and evaluation, does a deliberate context-length curriculum change the quality/context/efficiency tradeoff relative to a non-specialized control?

RQ2 is **not active yet**. Finish the decisive RQ1 objective comparison first.

---

## 2. Canonical teacher

Teacher:

- model: `Qwen/Qwen3.8-27B`
- revision: `dbdc473dea0d6a9763042881cc33d6058d1742d2`
- runtime class observed: `Qwen3_5ForCausalLM`
- hidden size: 5120
- vocabulary embedding rows: 248320
- 64 layers
- hybrid pattern: 16 × (3 Gated DeltaNet + 1 full-attention layer)
- attention: 24 query heads / 4 KV heads, head dim 256
- DeltaNet: 48 value heads / 16 key heads
- FFN width: 17408
- native context: 262144

The canonical online-teacher experiments use 4-bit teacher loading because the full teacher and trainable student must coexist on one 48 GB-class GPU.

4-bit prefix checks established essentially identical argmax behavior but small numerical differences. Never describe 4-bit execution as numerically exact.

Do not change the teacher revision for a controlled family.

---

## 3. Canonical student — frozen architecture

ID: `qwen38_19b_h5120_l48_moe`

Frozen base parameters: **13,008,505,728**

Approximate active parameters/token: **9,611,119,488**

Architecture:

- hidden size 5120
- 48 layers
- 36 DeltaNet + 12 attention
- layout `[DeltaNet, DeltaNet, DeltaNet, Attention] × 12`
- 24 query heads / 2 KV heads
- attention head dim 256
- partial rotary factor 0.25
- RoPE theta 10,000,000
- DeltaNet: 16 key heads / 48 value heads, head dim 128, conv kernel 4
- recurrent state kept fp32 where the implementation requires it
- sparse MoE: 8 routed experts, top-2
- 1 shared expert
- expert width 768
- router auxiliary loss 0.001
- untied embeddings
- RMSNorm
- MTP declared in architecture metadata but not constructed by the current Transformers runtime

The architecture is frozen. If an experiment seems to require changing it, stop and ask the project owner instead of silently redesigning the model.

Canonical transferred checkpoint used by the GPU workflow has historically lived at:

`/workspace/runs/pilot001/transferred`

It was materialized completely with no missing tensors and high transfer coverage. A future controlled run must fingerprint the actual checkpoint it sees; a historical path is not an identity proof.

---

## 4. Training mechanism

Current canonical constrained-compute mechanism:

- QLoRA
- NF4 frozen student base
- LoRA rank 16
- LoRA alpha 32
- LoRA dropout 0.05
- adapters on DeltaNet qkv/out and full-attention q/k/v/o paths
- sparse expert matrices excluded from adapters because routed expert gradients were too sparse for the intended short mechanistic comparisons
- real online teacher
- exact teacher revision
- top-k teacher distribution with exact tail bucket
- gradient checkpointing
- BF16 compute

`prepare_model_for_kbit_training` is intentionally not part of the canonical path because it caused unwanted embedding/lm-head upcasting. The implementation uses input-gradient enabling needed by the PEFT path instead.

---

## 5. Historical experiments — immutable evidence

Historical outputs and `CLAUDE_CLI_RUNS/` transcripts must never be rewritten to make later infrastructure look older than it is.

### Run 001 — mechanism validation

Real teacher + canonical student + QLoRA completed a 1-step smoke and 50-step pilot at sequence length 1024. This proved the training chain works. It is not a capability result.

### Run 002 — pure logit KD

A 2048-token calibration exceeded the then-used safety gate. The matched control was therefore run at sequence length 1536.

Completed 128 steps / 196,608 training tokens on an A40. It is a conventional output-KD control. A40 versus later L40S runs is a documented hardware confound for systems comparison.

### Run 003 — pointwise layer/intermediate KD

The first unchunked 1536-token calibration exceeded its memory gate because all mapped hidden-state loss inputs were retained.

A chunked implementation was proved mathematically/effectively equivalent: same 48 pairs and same objective, with hidden-state gradients matching and only tiny scalar summation-order differences. The chunked 128-step run completed on L40S.

Run 003 is the historical pointwise-state control.

### Run 004 exploratory

An early behavioral-delta run accidentally consumed the full Gutenberg source because the intended token cap was not applied. Preserve it as exploratory/shakeout evidence only. Do not use it as the matched RQ1 comparison.

### Run 004-M — matched topology-span delta

Matched Run 003 scientific conditions:

- sequence length 1536
- `max_tokens=700000`
- 128 steps
- batch 1, accumulation 1
- AdamW
- LR 2e-4
- warmup 10, cosine schedule
- BF16
- QLoRA r16 / alpha32 / dropout0.05
- seed 0
- teacher T=2, top-k=64, tail bucket
- same corpus/tokenizer/packing stream
- same L40S class

Behavioral target:

`student: h_s[l+1] - h_s[l]`

against the complete teacher span assigned to that student transition:

`teacher: h_t[b] - h_t[a]`

Teacher spans tile all 64 teacher layers exactly once across the 48 student transitions. Removed teacher layers are absorbed into spans rather than silently dropped.

Result: the span-delta objective was trainable and decreased substantially, but the pure span-delta arm underperformed pointwise Run 003 on output/validation diagnostics at the matched short budget.

Correct classification: **negative for this exact pure-span formulation under the matched 128-step regime**.

Do not write “behavioral KD fails.”

Current mechanistic hypothesis: a transition objective can learn approximately the right movement while the absolute residual state is already in the wrong region. This is a hypothesis until a functional transition audit tests absolute-state error, logit-change behavior and influence diagnostics.

---

## 6. Current phase

The project is now in **RQ1 controlled objective-family laboratory work**.

Active control files:

- `research/plans/RQ1_OBJECTIVE_LAB_V1.json`
- `research/protocols/RQ1_OBJECTIVES_V2.json`
- `experiments/research_campaign.json`
- `docs/REPRODUCIBILITY.md`
- `scripts/lab_preflight.py`
- `scripts/research_guard.py`
- `scripts/capture_research_environment.py`

`RQ1_V1.json` is retained as an earlier reproducibility draft. New controlled runs use the V2 family once their arm is CPU-ready.

The only permitted scientific independent variable inside V2 is the registered **arm**. Any change to LR, schedule, optimizer, LoRA settings, seed, sequence length, data stream, token budget, teacher, student initialization or objective coefficient requires a new protocol version.

---

## 7. RQ1 registered objective program

Read `research/plans/RQ1_OBJECTIVE_LAB_V1.json` before acting. Its statuses are execution gates.

### Arm A — pointwise state KD

Conventional mapped intermediate hidden-state magnitude + direction matching.

Historical anchor: Run 003.

### Arm B — published FDD comparator

This is the closest registered published trajectory/dynamics comparator and must be represented faithfully enough to deserve the label.

Gong et al., ACL 2025, “Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation”:

- select corresponding intermediate layers;
- decode each model's selected intermediate features through its own pretrained LM head into the shared vocabulary/prediction space;
- align the prediction-space trajectory;
- align finite-difference/first-order prediction dynamics between adjacent selected layers;
- combine these with conventional output KD.

The paper reports `alpha=1`, `beta=1` in its experiments and uniformly sampled intermediate layers. Any project adaptation must be documented explicitly.

**A raw residual hidden-state adjacent delta is not FDD. Never label it FDD.**

### Arm C — topology-aware span delta

Pure aggregate teacher-span residual contribution matched to one student transition. Historical anchor: Run 004-M.

### Arm D — pointwise + adjacent residual-delta ablation

Internal abstraction control: absolute state plus local adjacent teacher residual transitions. This isolates local transition supervision from topology-compressed span supervision. It is not prior-art FDD.

### Arm E — pointwise + topology-span delta

Critical test of the current mechanistic hypothesis. It asks whether the topology-aware span term becomes useful once absolute-state drift is constrained.

### Arm F — output + state + span composite

Do not execute until all coefficients are preregistered in a protocol revision. Composite weights are scientific hyperparameters and must never be chosen after inspecting A-E results.

An arm whose plan/protocol status is not `existing_cpu_tested`, `cpu_tested`, or `ready` is **not runnable**. Do not use GPU time to finish its implementation.

---

## 8. Prior-art discipline

Closest trajectory/dynamics work includes Gong 2025 FDD and Chi 2026 MTA. The campaign also records heterogeneous-architecture KD, depth/intermediate distillation, MoE distillation and long-context work.

Do not claim the proposed topology-span idea is the first method of its kind merely because the exact teacher/student combination differs.

Novelty requires:

1. a faithful closest-prior-art comparator;
2. a clear definition of what differs;
3. controlled evidence;
4. replication of any central result;
5. a literature review current at paper time.

---

## 9. Reproducibility-first lab policy

### Free/CPU work happens before GPU rental

Do not spend RunPod GPU time on:

- repository browsing;
- checking whether files exist;
- writing documentation/protocols;
- static validation;
- CPU-only unit tests;
- literature review;
- plotting existing results;
- cosmetic refactors;
- container builds.

Before a GPU session, the repository CI and local/free preflight should already establish the control plane.

Canonical free gate:

`python scripts/lab_preflight.py --json`

### GPU work is reserved for GPU-essential actions

Once a pod exists, allowed setup work is deliberately short:

- record the exact GPU/runtime identity;
- verify mounted/local teacher, student and corpus paths/fingerprints;
- calibrate VRAM only if a new objective materially changes the memory path;
- train controlled arms;
- run GPU-dependent evaluation/deployment measurements.

Do not start a broad repository audit on the paid pod.

---

## 10. New RQ1 environment

Historical runs did not record every transitive package version. Do not invent them.

New V2 experiments therefore use a fresh reproducible environment rather than pretending to recreate the unknown historical one exactly.

Key files:

- `environment/Dockerfile.research`
- `requirements/research-rq1-direct.txt`
- `.github/workflows/research-image.yml`
- `.github/workflows/research-lab-ci.yml`

The image is built off the paid GPU and published immutably. It embeds `/opt/research-pip-freeze.txt`. Every GPU session records the actual image digest through:

`RESEARCH_CONTAINER_DIGEST`

and captures the complete runtime environment with:

`python scripts/capture_research_environment.py --output <path>`

### Allocator environment variable

New launch instructions use PyTorch's canonical:

`PYTORCH_ALLOC_CONF=expandable_segments:True`

`PYTORCH_CUDA_ALLOC_CONF` is only a backward-compatible legacy alias. The guard/capture code accepts either and refuses conflicting values.

---

## 11. Scientific quality vs throughput comparability

Do not confuse a systems confound with a scientific one.

Fatal quality locks include:

- teacher/student/data identity;
- all registered training hyperparameters;
- critical Python/PyTorch/Transformers/CUDA versions;
- L40S GPU family/count/compute capability for this family;
- registered allocator behavior.

Systems comparability additionally requires the registered host driver and image digest.

A different host driver or container digest means direct throughput comparison must be disabled/qualified. It does not automatically mean the loss comparison is scientifically meaningless if every scientific quality lock passes.

`research_guard.py` writes both `quality_comparable` and `throughput_comparable` rather than collapsing these concepts.

---

## 12. Hardware and memory

Current RQ1 GPU class: **single NVIDIA L40S 48 GB nominal**.

Observed usable PyTorch device memory in prior RunPod L40S session: about 44.39 GiB.

Hard training ceiling for this project: **45 GiB**. Never raise the ceiling to make an objective pass.

Run 004-M historical peak allocated was about 38.95 GiB and peak reserved about 40.77 GiB, with an external guard maximum around 41.28 GiB.

Treat these as historical measurements, not guarantees for a new objective path.

Do not switch to A40 merely to save money inside a controlled L40S family without creating/documenting the resulting protocol/system distinction.

---

## 13. Budget policy

The project owner reported **$6.54 RunPod balance on 2026-09-04**. This is a session budget fact, not a permanent project invariant.

Maximize scientific information per dollar.

A matched 128-step Run 004-M took ~994.8 seconds on L40S, so short arms are affordable; long-budget and long-context experiments are not to be launched speculatively.

Do not automatically run every seed or every context. Execute the minimum matrix that resolves the current causal question, then replicate the decisive pair(s).

Preserve enough balance to stop cleanly and archive results.

---

## 14. Controlled recipe for the current short RQ1 family

Unless a new protocol explicitly replaces it, V2 freezes:

- sequence length 1536
- corpus cap 700000 tokens
- 128 optimizer steps
- batch 1
- accumulation 1
- QLoRA
- AdamW
- LR 2e-4
- weight decay 0
- warmup 10
- cosine schedule
- BF16
- gradient checkpointing on
- LoRA r16 / alpha32 / dropout0.05
- seed 0 for the first matrix
- KD temperature 2
- top-k 64
- exact tail bucket
- intermediate direction weight 1
- per-token RMS normalization on
- type-preserving group mapping
- chunk size 4 where the registered intermediate objective uses the validated chunk path
- eval every 32
- checkpoint every 64
- metrics every step

Never rely on a CLI default to reconstruct these values. The resolved config must contain them explicitly.

---

## 15. Data identity

The matched historical stream is registered by hashes in the active protocol, including:

- corpus SHA256 `e11ca38bb099fc89c2f74e96f5d2f1209def6a16f6a8432d4e9972acd50c100d`
- packing version 1.0
- blank-line document separation
- tokenizer file hashes
- vocab facts and EOS identity

A familiar path such as `/workspace/corpora/gutenberg/train.txt` is not proof that the bytes match. Verify fingerprints.

The validation split and deterministic sampler stream must not drift between arms.

---

## 16. Required run artifacts

Every controlled run must answer: **what code, model, data, environment and configuration produced this number?**

Retain:

- Git SHA and clean/dirty state;
- protocol ID + fingerprint;
- fully resolved configuration + fingerprint;
- arm and seed;
- exact command;
- teacher revision;
- student checkpoint fingerprint;
- corpus/tokenizer/packing fingerprints;
- runtime environment capture;
- GPU/driver/container identity;
- per-step metrics;
- summary;
- memory profile/guard result;
- checkpoints required by protocol;
- terminal state;
- cost, elapsed time, tokens and throughput;
- artifact/archive checksums.

A process exit code of zero by itself is not a complete experiment.

---

## 17. Claim standard

Internally classify every statement as:

- **demonstrated** — supported by controlled reproducible evidence;
- **observed** — measured, but not necessarily causal;
- **hypothesized** — explanatory proposal still awaiting a diagnostic/ablation;
- **planned** — not executed.

Never collapse these categories in documentation or paper prose.

A 128-step PEFT run is a mechanism/objective comparison. It cannot prove the final 13B student's competitive benchmark capability.

---

## 18. Evidence gates

The campaign file is authoritative for paper-level gates. Important remaining gates include:

- faithful closest-prior-art comparator;
- adjacent-vs-span abstraction ablation;
- objective-family comparison;
- functional transition-influence audit;
- longer-budget survival only after a method earns it;
- context-specialization ablation;
- capability benchmarks;
- measured 16 GiB deployment accounting;
- additional-seed replication of a central result.

Negative/null results remain visible.

---

## 19. Functional audit after a decisive RQ1 pair

When a decisive pair exists, examine the registered diagnostics such as:

- residual cosine;
- relative residual error;
- logit-delta cosine;
- perturbation KL;
- influence rank correlation;
- residual scale sensitivity.

This is how the project tests the “right movement from the wrong state” explanation instead of merely repeating it.

---

## 20. Longer-budget gate

Do not jump directly to 4096/16384-step runs.

A longer run is justified only when:

1. a short objective is CPU-verified and GPU-stable;
2. its matched comparison is positive or scientifically decisive enough to warrant scale;
3. the key result has an appropriate replication plan;
4. budget remains;
5. the full environment/container is pinned;
6. the protocol for the longer budget is created before launch.

A longer token budget is a new protocol, not a casual CLI override.

---

## 21. RQ2 and long-context work

Context targets eventually include 4K through 262K and curricula such as uniform, short-heavy, medium-heavy, long-heavy and progressive.

RQ2 requires:

- a real context scheduler/mixture implementation;
- same total training tokens across arms;
- a predefined evaluation grid;
- measured memory and throughput;
- no cherry-picking only the context where one curriculum wins.

Do not spend current RQ1 budget on RQ2 merely because the architecture declares a 262K maximum.

---

## 22. Deployment target

The deployment target is **16 GiB end-to-end GPU-resident inference**, including:

- weights;
- quantization metadata/overhead;
- KV cache;
- recurrent state;
- runtime/workspace.

Training memory is not deployment memory.

Measure inference memory, latency, throughput, quantization stability and context behavior separately before making a 16 GB feasibility claim.

---

## 23. MoE measurements

For capability-scale work, measure rather than assume:

- routed expert utilization;
- tokens/expert;
- routing entropy;
- dead experts;
- load imbalance;
- changes with training/context.

Do not call experts specialized unless behavior supports the claim.

---

## 24. Repository hygiene

Never commit:

- teacher weights;
- canonical student weight files unless explicitly intended and properly LFS-managed;
- optimizer state from large runs;
- datasets;
- credentials/tokens;
- caches;
- accidental shell files.

Do commit:

- source code;
- protocols/plans;
- tests;
- concise run metadata;
- reproducibility manifests;
- research documentation;
- traceable plots/figures.

Preserve `CLAUDE_CLI_RUNS/` unless the owner explicitly requests otherwise.

---

## 25. Git/run discipline

Before a controlled launch record:

`git status --short`

`git rev-parse HEAD`

`git branch --show-current`

Run only from a known commit. Do not mutate the active experiment code path during training.

After a run, validate artifacts before interpreting the result. Never automatically retry an OOM with changed hyperparameters and then call the retry the same arm.

---

## 26. Claude behavior on a paid GPU

Once the owner starts an L40S, do not repeat the repository work that should already be done.

Preferred sequence:

1. load this skill;
2. confirm the expected launch SHA and active V2 plan/protocol;
3. run the short lab preflight once;
4. capture runtime environment and immutable image digest;
5. verify GPU-essential mounted teacher/student/corpus fingerprints;
6. run the research guard with a fully resolved config;
7. execute only an arm marked CPU-ready;
8. monitor the active run, memory ceiling and artifacts;
9. stop between scientific arms when analysis/gating requires it.

Do not spawn redundant subagents to rediscover the repository on paid time.

---

## 27. When to ask the owner

Stop and ask before:

- changing the frozen architecture;
- changing teacher revision;
- changing a research question;
- introducing a new scientific coefficient/hyperparameter not preregistered;
- materially increasing budget;
- relaxing the 45 GiB ceiling;
- replacing a required prior-art comparator with a cheaper but semantically different proxy;
- making a claim not supported by existing evidence.

Do not ask for permission for ordinary validation already required by the protocol.

---

## 28. End-state principle

This project may end with a strong positive, mixed, null, systems-only, or methodological result. All are legitimate if the evidence is clean.

The unacceptable outcome is an apparently strong story built from uncontrolled or unreproducible experiments.

Every GPU dollar and every repository change should move the project toward a falsifiable answer to RQ1/RQ2 and a reproducible record that another researcher can audit.
