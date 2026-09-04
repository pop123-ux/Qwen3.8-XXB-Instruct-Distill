# Novelty hardening plan

## Goal

The project should aim for a **defensible empirical contribution**, not a blanket claim that the underlying idea of trajectory or delta distillation is new.

The closest prior work already establishes feature-dynamics distillation (FDD) and trajectory alignment (MTA). Therefore the paper should earn novelty through a sharper question, explicit prior-art baselines, controlled experiments, and evidence that isolates the proposed operation.

## Proposed contribution to test

Working name:

**Computational Span Distillation (CSD)**

Definition:

> Instead of requiring a student block to match one teacher hidden state or one adjacent teacher transition, assign a student transition to a teacher computational span and match the residual contribution accumulated across that span.

For student transition `l -> l+1` and teacher span `[a,b)`:

`Δh_student(l) = h_student(l+1) - h_student(l)`

`Δh_teacher(a,b) = h_teacher(b) - h_teacher(a)`

The important distinction is therefore **span aggregation under depth/topology mismatch**, not the use of finite differences itself.

## Prior-art baselines the experiment matrix should distinguish

### P0 — CE only

No teacher internal supervision. Measures the floor.

### P1 — Logit KD

Conventional output-distribution supervision.

### P2 — Pointwise intermediate KD

The project's existing layer-control formulation.

### P3 — FDD-compatible adjacent dynamics

A faithful implementation of the closest feature-dynamics idea available to the project. Where the published method uses prediction-space dynamics, reproduce that representation rather than renaming the project's hidden-delta loss as FDD.

### P4 — CSD / span behavioral KD

The proposed teacher-span → student-transition objective.

### P5 — Composite

CE + logit KD + pointwise KD + CSD, used to test whether the proposed objective adds information rather than merely replacing another useful loss.

## The key ablation

The paper's cleanest causal ablation is:

**Adjacent dynamics vs span dynamics**

Hold teacher, student, token budget, optimizer, initialization, context, LoRA, seed, and evaluation fixed.

Change only whether the behavior target is:

- adjacent teacher transition; or
- aggregated teacher-span transition.

If span supervision is better, the result is much more informative than simply showing that some delta loss can train.

## Functional evidence, not only loss curves

A loss falling is not enough to establish “computational behavior transfer.” The project should measure whether trained students reproduce the teacher's **functional influence profile**.

Add an evaluation pass that records, for teacher spans and student transitions:

- residual-vector cosine similarity;
- relative residual magnitude error;
- logit-change cosine similarity;
- KL change after perturbing/ablating the transition;
- rank correlation of transition influence across depth;
- sensitivity under small residual scaling perturbations.

These measurements ask whether the student and teacher transitions have similar effects on downstream predictions, rather than merely similar coordinates.

## Topology-aware analysis

The paper should report the teacher/student computational maps explicitly:

- teacher attention blocks → student attention transitions;
- teacher DeltaNet blocks → student DeltaNet transitions;
- teacher dense FFN regions → student sparse-MoE regions;
- teacher depth removed/absorbed by each student span;
- 4 KV heads → 2 KV heads.

A useful analysis is to compare multiple span maps:

1. uniform depth mapping;
2. block-type-preserving mapping;
3. teacher block-influence-weighted mapping.

The third should be treated as a post-hoc ablation unless pre-registered before training.

## Strong evidence standard

The central claim should require all of the following before being presented as supported:

1. matched-budget comparison against pointwise KD;
2. explicit comparison to an adjacent-dynamics/FDD-compatible baseline;
3. improvement on output-level teacher imitation metrics, not only the proposed loss;
4. improvement or parity on held-out capability metrics;
5. evidence that the effect survives a materially longer budget;
6. at least two random seeds for the final central comparison, if compute permits;
7. functional-influence analysis showing a change consistent with the claimed mechanism;
8. all hyperparameters and mappings frozen before the decisive comparison.

A result that only improves the behavioral loss but not teacher agreement or capability should be reported as an optimization result, not evidence of superior knowledge transfer.

## Falsification tests

The hypothesis is weakened or falsified if:

- span-KD is consistently worse than pointwise KD under matched conditions;
- an FDD-compatible adjacent-delta baseline is equally good and materially simpler;
- CSD only improves its own training loss without affecting teacher agreement or capabilities;
- gains disappear after controlling for loss magnitude or training compute;
- gains are confined to one seed;
- gains disappear on a second data distribution;
- functional influence profiles remain teacher/student-incongruent.

## Recommended run ladder

### Stage H0 — matched controls

Run P1/P2/P4 under the current 1536-token, 128-step matched protocol.

### Stage H1 — published-method comparator

Add P3, using the closest faithfully reproducible FDD formulation.

### Stage H2 — transition abstraction ablation

Compare adjacent dynamics vs teacher-span dynamics.

### Stage H3 — longer scaling

Scale the best-supported method and the strongest comparator to 1,024 and 4,096 steps. Consider 16,384 only if the earlier stages are healthy and within budget.

### Stage H4 — capability validation

Evaluate all surviving methods on the same held-out suite.

### Stage H5 — seed replication

Repeat the decisive central comparison with at least one additional seed.

### Stage H6 — functional-behavior audit

Run transition influence / perturbation analysis on matched checkpoints.

## What would constitute a strong paper result

A strong result would look like:

> A teacher-span transition objective consistently outperforms pointwise and adjacent-dynamics distillation under matched training budgets for a materially non-isomorphic teacher/student pair, with the advantage persisting at longer budgets, translating to held-out capability gains, and being visible in functional transition-influence measurements.

That claim would be substantially stronger and more defensible than “we invented behavioral KD.”

## What would not constitute novelty

The following are already represented by prior work and should not be claimed as new by themselves:

- matching hidden states;
- matching intermediate layers;
- matching layer deltas;
- viewing Transformer depth through an ODE/dynamics lens;
- heterogeneous-architecture KD in general;
- MoE distillation in general;
- long-context distillation in general.

## Literature anchor

The paper must explicitly discuss at minimum:

- Gong et al., *Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation*, ACL 2025.
- Chi et al., *MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation*, ACL 2026.
- Hao et al., *One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation*, 2023.
- Liu et al., *Cross-Architecture Knowledge Distillation*, 2022.
- Sun et al., *Patient Knowledge Distillation for BERT Model Compression*, 2019.
- Jiao et al., *TinyBERT: Distilling BERT for Natural Language Understanding*, 2020.
- NVIDIA, *Compact Language Models via Pruning and Knowledge Distillation (MINITRON)*, 2024.
- Men et al., *ShortGPT: Layers in Large Language Models are More Redundant Than You Expect*, 2024.
- Yang et al., *LaCo: Large Language Model Pruning via Layer Collapse*, 2024.
- Gu et al., *MiniLLM: Knowledge Distillation of Large Language Models*, 2023.
- Kim et al., *Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models*, 2025.
- Huber et al., *Short Data, Long Context: Distilling Positional Knowledge in Transformers*, 2026.

See [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) for links and scope notes.
