# Research Reproducibility Contract

Completed experiment outputs are immutable. The reproducibility system is additive: it adds controls for future runs without rewriting historical summaries, ledgers, commands, checkpoints, or plots.

## Protocol families

Every controlled experiment belongs to a versioned protocol family under `research/protocols/`. A family freezes the scientific conditions that must remain constant across an ablation. The explicitly declared independent variable is the only field allowed to vary inside that family.

A changed learning rate, optimizer, scheduler, batch size, accumulation, LoRA setting, tokenizer, corpus, packing rule, teacher revision, student checkpoint, software stack, container, CUDA stack, or hardware class requires a new protocol family/version. It must never be presented as a continuation of the previous matched experiment.

`research/protocols/RQ1_V1.yaml` is the frozen baseline family for the next objective-comparison work. It is based on the conditions of Run 003 and matched Run 004-M.

## Environment identity

`environment/research-baseline.json` records the observed critical environment used by Run 004-M:

- Python 3.12.3
- PyTorch 2.8.0+cu128
- Transformers 5.15.1
- CUDA runtime 12.8
- NVIDIA L40S, compute capability 8.9
- NVIDIA driver 580.159.04

`environment/Dockerfile.research` is the reproducible container baseline. Host GPU driver identity remains external to the container and is recorded separately.

Before a final long research run, capture a full package/environment fingerprint with:

`python scripts/capture_research_environment.py --output <path>`

The full package set, CUDA/cuDNN, GPU identity, driver, allocator setting, platform, and container identity are then retained as run provenance.

## Scientific controls

For matched objective comparisons, standardize:

- exact teacher revision and quantization;
- exact student architecture and starting checkpoint;
- exact corpus SHA, tokenizer hashes, document separator and packing version;
- exact token budget, sequence length, batch size and gradient accumulation;
- exact optimizer, learning rate, warmup, scheduler and weight decay;
- exact precision, gradient-checkpointing and allocator configuration;
- exact LoRA rank, alpha and dropout;
- exact KD temperature, top-k, tail handling and intermediate-loss settings;
- exact evaluation/checkpoint/logging cadence;
- exact seed policy;
- exact container and critical software versions;
- exact GPU model and driver for throughput comparisons.

Quality comparisons are based primarily on matched **tokens seen and scientific conditions**, not wall-clock time. Throughput is a systems measurement. A throughput comparison is marked non-comparable when GPU, driver, CUDA stack or container identity differs.

## Defaults are not protocol values

CLI defaults are not part of the research contract. A controlled run must record the resolved configuration explicitly. A researcher must not rely on whichever default happens to be present in the current trainer version.

## Historical artifacts

Run 003, Run 004 and earlier results remain historical records. New reproducibility infrastructure must not overwrite their `summary.json`, `metrics.jsonl`, command files, archived plots, or ledger entries. New code can improve future enforcement without retroactively changing what a completed run actually did.

## Plots

Comparative research figures must be backed by experiment IDs and a compatible protocol family. A figure must not visually imply that incomparable runs form a controlled ablation.
