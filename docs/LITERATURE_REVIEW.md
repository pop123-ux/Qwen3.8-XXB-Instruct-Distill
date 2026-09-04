# Literature review: topology-aware and trajectory-based distillation

This project does **not** claim that feature-delta or trajectory distillation is new. Those ideas are established. The research question here is narrower: whether **computational-span-aligned transition supervision** is a useful abstraction when teacher and student have materially different computational topologies.

This document is intentionally part of the experiment record. Published methods below are treated as prior art and, where feasible, as experimental baselines rather than rhetorical background.

## Closest prior work

### 1. Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation

Gong et al., ACL 2025.

- ACL Anthology: https://aclanthology.org/2025.acl-long.1125/
- DOI: https://doi.org/10.18653/v1/2025.acl-long.1125

This is the closest conceptual precedent. FDD treats Transformer depth as a discrete dynamical system and augments KD with both trajectory alignment and first-order feature-delta alignment. The published method therefore establishes that matching **how representations change across depth** is a viable KD abstraction.

**Consequence for this project:** we must not describe “delta KD” or “trajectory KD” itself as the novel contribution.

Our potentially distinct question is whether a student transition can be supervised against the **aggregate residual contribution of a teacher span** when the student has fewer or structurally different blocks.

### 2. MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation

Chi et al., ACL 2026.

- arXiv: https://arxiv.org/abs/2605.01374
- ACL Anthology: https://aclanthology.org/2026.acl-long.1507/
- DOI: https://doi.org/10.18653/v1/2026.acl-long.1507

MTA extends trajectory-oriented distillation using layer-adaptive semantic granularity, dynamic structural alignment, and hidden-representation alignment. It reinforces the observation that fixed pointwise layer correspondence is not the only useful way to represent teacher information.

**Consequence for this project:** a serious paper must position span-aligned transition KD against the broader trajectory-alignment family, not only against vanilla layer matching.

## Heterogeneous-architecture distillation

### 3. One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation

Hao et al., 2023.

- arXiv: https://arxiv.org/abs/2310.19444

This work studies KD when teacher and student architectures are heterogeneous, finding that direct feature matching can be problematic. It projects intermediate features into an aligned latent space and uses adaptive target enhancement.

**Relevance:** it establishes that architecture mismatch is a real KD problem and that architecture-specific intermediate states should not automatically be assumed equivalent.

### 4. Cross-Architecture Knowledge Distillation

Liu et al., 2022.

- arXiv: https://arxiv.org/abs/2207.05273

This work explicitly targets Transformer-to-CNN cross-architecture transfer using projection mechanisms rather than naive feature equality.

**Relevance:** further evidence that cross-architecture KD requires an abstraction that survives architectural differences.

### 5. DisWOT: Student Architecture Search for Distillation WithOut Training

Dong, Li & Wei, 2023.

- arXiv: https://arxiv.org/abs/2303.15678

Studies how architecture differences affect distillation performance and uses feature-semantic/relation similarity to select student architectures.

**Relevance:** supports treating teacher/student topology as an experimental variable rather than a nuisance.

## Intermediate representation and depth reduction

### 6. Patient Knowledge Distillation for BERT Model Compression

Sun et al., 2019.

- arXiv: https://arxiv.org/abs/1908.09355

Introduces PKD-Last and PKD-Skip, explicitly supervising multiple teacher intermediate layers rather than only the final output.

### 7. TinyBERT: Distilling BERT for Natural Language Understanding

Jiao et al., 2020.

- arXiv: https://arxiv.org/abs/1909.10351

A foundational Transformer distillation framework that matches embedding outputs, attention matrices, and hidden states across mapped layers.

### 8. MINITRON: Compact Language Models via Pruning and Knowledge Distillation

NVIDIA, 2024.

- arXiv: https://arxiv.org/abs/2407.14679

Particularly relevant to this project because it studies depth/width-pruned LLMs and combines logit, embedding and intermediate-state distillation. Their analysis also explicitly maps teacher and student blocks despite different depths.

**Relevance:** this is a strong baseline family for depth-reduced LLMs. Our work should make clear how span-transition supervision differs from mapping hidden states at selected teacher/student blocks.

### 9. ShortGPT: Layers in Large Language Models are More Redundant Than You Expect

Men et al., 2024.

- arXiv: https://arxiv.org/abs/2403.03853

Shows that some LLM layers can be removed with comparatively small degradation and introduces block influence as a depth-redundancy measure.

### 10. LaCo: Large Language Model Pruning via Layer Collapse

Yang, Cao & Zhao, 2024.

- arXiv: https://arxiv.org/abs/2402.11187

Collapses later layers into earlier layers to reduce depth while preserving much of the original structure.

### 11. Iterative Layer-wise Distillation for Efficient Compression of Large Language Models

Kovalev & Tikhomirov, 2025.

- arXiv: https://arxiv.org/abs/2511.05085

Combines iterative layer-importance analysis with KL/MSE retraining for depth reduction.

**Relevance:** reinforces the need to distinguish our loss abstraction from ordinary layer-reduction/retraining recipes.

## LLM KD objectives

### 12. MiniLLM: Knowledge Distillation of Large Language Models

Gu et al., 2023.

- arXiv: https://arxiv.org/abs/2306.08543

Establishes reverse-KL distillation as a strong objective for generative LMs.

**Relevance:** the objective-family controls in this project should separate the question of **what distributional divergence** is used from the question of **what internal computational signal** is supervised.

### 13. Sequence-Level Knowledge Distillation

Kim & Rush, 2016.

- arXiv: https://arxiv.org/abs/1606.07947

Foundational sequence-level KD showing the value of teacher-generated targets beyond direct token-level matching.

## MoE-specific distillation

### 14. Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

Kim, Chu & Yang, 2025.

- arXiv: https://arxiv.org/abs/2502.12947

Shows that inactive experts can still contain useful knowledge and proposes MoE-specific distillation mechanisms, including Knowledge Augmentation and Student-Aware Router.

**Relevance:** our dense-teacher → sparse-MoE student is not merely a depth reduction. The expert decomposition itself is a transfer problem and should be measured separately from trajectory alignment.

## Long-context distillation

### 15. Short Data, Long Context: Distilling Positional Knowledge in Transformers

Huber et al., 2026.

- arXiv: https://arxiv.org/abs/2604.06070

Shows that logit-based KD can transfer positional/long-context information to students trained on packed short-context data and analyzes where positional information enters the distillation signal.

**Consequence:** context-length specialization is also not a blank-slate idea. The contribution must be the controlled comparison of **context exposure distributions** under equal token budgets and the interaction with the topology-aware KD objective.

### 16. Combining On-Policy Optimization and Distillation for Long-Context Reasoning in Large Language Models

Ramos, Alves & Martins, 2026.

- arXiv: https://arxiv.org/abs/2605.12227

Studies long-context reasoning with dense teacher guidance plus outcome-based optimization.

**Relevance:** a later extension path, not part of the first controlled experiment.

## What is and is not potentially novel here

### Established

- output/logit KD;
- hidden-state/intermediate-state KD;
- multi-layer mapping;
- feature trajectory alignment;
- first-order feature-delta alignment;
- heterogeneous-architecture KD;
- depth reduction followed by KD;
- MoE-aware KD;
- long-context KD.

### Potentially distinct

This project studies a more specific construction:

> **For a structurally non-isomorphic teacher/student pair, map each student transition to a teacher computational span and supervise the student residual contribution against the aggregate teacher span contribution, then test whether that abstraction improves transfer relative to pointwise and adjacent-delta baselines.**

The novelty claim therefore lives in the **combination and controlled empirical test of span-aligned computational transitions under topology mismatch**, not in the existence of feature dynamics itself.

## Required citation discipline

Any paper claim in the README, research protocol, or eventual paper must cite the relevant prior work above. In particular, the final paper must cite FDD and MTA in the first section that introduces trajectory/feature-delta supervision, and must cite heterogeneous-KD work when motivating the topology mismatch.

No statement such as “first,” “novel,” “unprecedented,” or “no previous work” is permitted unless a future literature audit supports it with an explicit scope and date.
