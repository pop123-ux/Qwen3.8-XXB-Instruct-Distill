# Research Reproducibility Contract

Completed experiment outputs are immutable. The reproducibility system is additive: it adds controls for future runs without rewriting historical summaries, ledgers, commands, checkpoints, or plots.

## Active protocol family

Every controlled experiment belongs to a versioned protocol under `research/protocols/`. A protocol freezes the scientific conditions that must remain constant across an ablation. Its explicitly declared independent variable is the only scientific field allowed to vary.

The active new-run family is `research/protocols/RQ1_OBJECTIVES_V2.json`, paired with `research/plans/RQ1_OBJECTIVE_LAB_V1.json`. `RQ1_V1.json` is retained as the earlier reproducibility draft and is not rewritten retroactively.

A changed learning rate, optimizer, scheduler, batch size, accumulation, LoRA setting, tokenizer, corpus, packing rule, teacher revision, student checkpoint, loss coefficient, sequence length, or token budget requires a new scientific protocol version. Hidden CLI-default inheritance is forbidden.

## CPU first: paid GPU time is not repository-validation time

Before a GPU is rented, run:

`python scripts/lab_preflight.py --json`

The preflight is standard-library-only. It validates the lab plan/protocol relationship, teacher/student identities, matched recipe, objective registry, FDD semantics, container declaration, required files and execution gates without importing PyTorch or loading a model.

The following work belongs off the paid GPU whenever possible:

- repository inspection and file-existence checks;
- protocol/design writing;
- literature review;
- JSON/schema/static validation;
- CPU-only objective/unit tests;
- syntax/lint checks;
- plotting from existing artifacts;
- container building and publishing.

GitHub Actions provides the CPU control-plane checks in `.github/workflows/research-lab-ci.yml` and builds/publishes the research image in `.github/workflows/research-image.yml`.

The GPU session is reserved for things that genuinely require its runtime: GPU identity capture, local teacher/student/data-path validation, memory calibration when a new objective changes memory behavior, controlled training, and GPU-dependent evaluation/deployment measurements.

## Environment identity

`environment/research-baseline.json` records the observed environment of Run 004-M. Historical transitive package versions were not fully captured, so the repository must not pretend that environment can be reconstructed byte-for-byte.

New RQ1 runs therefore open a fresh reproducible environment:

- Python 3.12.3;
- PyTorch 2.8.0+cu128;
- Transformers 5.15.1;
- direct package pins in `requirements/research-rq1-direct.txt`;
- immutable image built from `environment/Dockerfile.research`;
- package snapshot embedded at `/opt/research-pip-freeze.txt`;
- immutable registry image digest recorded through `RESEARCH_CONTAINER_DIGEST`;
- full runtime capture written by `scripts/capture_research_environment.py`.

This is intentionally a **new experimental environment**. Historical Run 003/004-M remain anchors, but a clean A-F comparison should be performed within the new family rather than pretending an unrecorded historical dependency set is identical.

### PyTorch allocator variable

New launch instructions use:

`PYTORCH_ALLOC_CONF=expandable_segments:True`

`PYTORCH_CUDA_ALLOC_CONF` is accepted as the legacy compatibility alias by the capture/guard scripts. If both are set to different values, the guard refuses the run.

## Scientific validity versus systems comparability

`research_guard.py` separates two questions.

**Scientific/quality locks are fatal on mismatch:** teacher/student/data recipe, critical Python/PyTorch/Transformers/CUDA versions, L40S GPU family/count/compute capability, allocator and all registered training hyperparameters.

**Systems locks affect throughput comparability:** exact host NVIDIA driver and immutable container image digest. A driver mismatch does not by itself turn a matched quality experiment into a different loss function, but a run with a different driver/container must be marked `throughput_comparable=false` and must not be used for direct systems-speed claims.

This distinction prevents two opposite errors: wasting GPU money to chase an irrelevant host-driver match for a quality ablation, and silently presenting throughput from different systems as if it were controlled.

## Scientific controls

For matched objective comparisons, standardize:

- exact teacher revision and quantization;
- exact student architecture and starting checkpoint identity;
- exact corpus SHA, tokenizer hashes, document separator and packing version;
- exact token budget, sequence length, batch size and gradient accumulation;
- exact optimizer, learning rate, warmup, scheduler and weight decay;
- exact precision, gradient checkpointing and allocator configuration;
- exact LoRA rank, alpha and dropout;
- exact KD temperature, top-k, tail handling and intermediate-loss settings;
- exact objective coefficients;
- exact evaluation/checkpoint/logging cadence;
- exact seed policy.

Quality comparisons are based on matched **tokens seen and scientific conditions**, not wall-clock time. Throughput is a separate systems result.

## Objective implementation gate

An objective being named in a roadmap does not make it runnable. `research/plans/RQ1_OBJECTIVE_LAB_V1.json` and `RQ1_OBJECTIVES_V2.json` assign every arm an implementation status. `research_guard.py` refuses any arm that has not reached a CPU-tested state.

In particular, the published FDD comparator is not a raw hidden-residual delta. The registered FDD arm must operate in LM-head prediction space and contain output KD, trajectory KL and finite-difference/derivative alignment. The adjacent residual-delta experiment is a separate internal ablation and must never be labelled FDD.

Composite loss weights are scientific hyperparameters. Arm F stays blocked until its weights are preregistered; they must not be selected after reading A-E outcomes.

## Required artifacts for every controlled run

A run is not complete merely because the training process exits zero. Preserve at minimum:

- exact Git SHA and clean/dirty state;
- protocol and protocol fingerprint;
- fully resolved configuration and fingerprint;
- independent arm/seed;
- teacher revision and student checkpoint identity;
- corpus/tokenizer/packing fingerprints;
- full runtime/environment capture;
- hardware and memory record;
- exact command;
- per-step metrics and summary;
- checkpoints required by the protocol;
- terminal/termination state;
- cost/runtime/tokens/throughput accounting;
- checksums or archive index for retained artifacts.

## Historical artifacts

Run 003, Run 004-M and earlier results remain historical records. New infrastructure must not overwrite their `summary.json`, `metrics.jsonl`, command files, archived plots, transcripts, or ledger entries. Negative and null results remain part of the research record.

## Plots and claims

Every plotted point must resolve to an experiment ID and source artifact. A comparative figure must not visually imply that incompatible protocol families form a controlled ablation.

Use the claim classes **demonstrated**, **observed**, **hypothesized**, and **planned**. A 128-step mechanism comparison is not a capability result, and no novelty/SOTA claim is permitted without the registered prior-art controls and measured capability/deployment evidence.
