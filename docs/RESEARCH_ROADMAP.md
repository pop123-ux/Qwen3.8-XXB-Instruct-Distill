# Research Roadmap — Qwen3.8-XXB-Instruct-Distill

## Purpose

These files make the repository the source of truth for research execution. Future sessions should read `research/ROADMAP.yaml` and `research/STATE.yaml`, preserve the frozen experimental envelope, repair technical blockers without changing the scientific question, execute the next admissible experiment, and update state.

**Negative scientific results advance the research tree. Technical failures are repaired and the same intended experiment resumes.**

The goal is not to tune until behavioral KD wins. The goal is to accumulate clean evidence toward the project’s research questions.

## RQ1

**When teacher and student topologies differ materially, does behavioral/residual-delta knowledge distillation provide a better supervision signal than conventional pointwise hidden-state matching?**

The working hypothesis is that absolute hidden-state correspondence may become less meaningful as teacher/student topology diverges, while transformation-level correspondence may remain meaningful.

### Run003 — pointwise control

Run003 is the immutable matched control.

| Metric | Run003 |
|---|---:|
| Final validation loss | 8.9599 |
| Best validation loss | 8.7705 |
| Final top-1 agreement | 0.16222 |
| Final logit KD loss | 3.9808 |
| Final CE loss | 9.4026 |
| Runtime | 1098.3 s |
| Throughput | 179.0 tok/s |
| Peak allocated / reserved | 38.945 / 40.766 GiB |

### Arm A / Run004-M — normalized residual-delta KD

Arm A completed with integrity PASS and clearly lost to Run003 on the important shared metrics.

| Metric | Run003 | Run004-M |
|---|---:|---:|
| Final validation loss | **8.9599** | 11.8886 |
| Best validation loss | **8.7705** | 11.8886 |
| Final top-1 agreement | **0.16222** | 0.12117 |
| Final logit KD loss | **3.9808** | 4.9099 |
| Final CE loss | **9.4026** | 11.9913 |

This is a clean **NO-GO for the normalized-delta formulation**. It does not by itself disprove every form of transformation-level supervision.

### Arm B — scale-aware behavioral KD

Arm B is already authorized and may currently be running. It must **not** be stopped, restarted, reconfigured, or redesigned because these roadmap files were added.

Its purpose is to test whether preserving update magnitude as well as direction improves transformation matching. The exact already-locked loss equation and coefficients must be copied from the active run’s local config/artifacts before interpretation. They must not be chosen after seeing the result.

If Arm B becomes competitive or superior, preserve it and use remaining compute for a complete confirmation or a 1024-step persistence run of that exact formulation.

If Arm B is clearly inferior, do not sweep coefficients. Advance to Arm C.

### Arm C — raw residual matching

Arm C is a single matched 128-step raw/unnormalized residual-matching arm. It tests whether normalization itself caused the failure of Arm A. Its exact equation must be locked before the outcome is observed.

If Arm C also clearly loses, the project should stop inventing loss variants in the same session. Multiple principled transformation-matching formulations losing to Run003 would be meaningful evidence favoring pointwise KD for this teacher/student topology under the tested protocol.

## Evidence needed for a strong RQ1 claim

Evidence becomes progressively stronger when:

1. a behavioral arm beats or credibly matches the pointwise control under the matched 128-step protocol;
2. the same exact formulation persists at a longer horizon;
3. the effect replicates;
4. the effect appears in another heterogeneous teacher/student topology.

A single teacher/student pair cannot establish universal superiority.

## Frozen experimental envelope

Important frozen identities include:

- teacher revision: `dbdc473dea0d6a9763042881cc33d6058d1742d2`
- train SHA256: `bc5972d9a52580ff14ab1b3b1753f9cd68c726c63cc625a7ed3913ec3c5dc5c5`
- packed 700k SHA256: `e11ca38bb099fc89c2f74e96f5d2f1209def6a16f6a8432d4e9972acd50c100d`
- sequence length: 1536
- max corpus tokens: 700000
- matched horizon: 128 steps
- batch size: 1
- gradient accumulation: 1
- learning rate: 2e-4
- AdamW, bf16, warmup 10, cosine scheduler
- LoRA rank/alpha/dropout: 16 / 32 / 0.05
- seed: 0
- KD temperature: 2.0
- KD top-k: 64
- chunk pairs: 4
- VRAM guard: 44.0 GiB

Historical deterministic student materialization uses group layer strategy, mean KV merge, contiguous FFN partitioning, gate compensation enabled, seed 0, and the exact teacher revision above.

## Blocker handling

Missing corpus/checkpoint, OOM, NaN, wrong revision, wrong dataset, corruption, or an implementation bug is a **technical failure**. Make the minimum repair and resume the intended locked experiment.

Bad validation loss or poor teacher agreement is a **scientific result**. Preserve it and follow the roadmap branch. Do not retune until the desired result appears.

When compute is limited, prefer a complete matched experiment over an incomplete long run.

## RQ2

The exact prior RQ2 definition is not available in the planning context used to generate these files, so it is deliberately **not invented**.

Before RQ2 execution, a local session must recover the exact existing RQ2 question, hypothesis, controls, arms, protocols, dependencies, and relationship to RQ1 from the repository itself and copy them faithfully into `research/ROADMAP.yaml` and `research/STATE.yaml`.

This recovery is CPU-side bookkeeping and must not interrupt Arm B or another active experiment.

## Future-session contract

When told to **continue the research roadmap**, the agent should:

1. read `research/ROADMAP.yaml`;
2. read `research/STATE.yaml`;
3. validate relevant completed artifacts;
4. identify the next admissible experiment;
5. repair technical blockers while preserving scientific invariants;
6. run the experiment;
7. preserve its artifacts;
8. update `STATE.yaml`;
9. follow the result branch;
10. continue until a genuine research gate, resource boundary, or explicit user intervention requires stopping.
