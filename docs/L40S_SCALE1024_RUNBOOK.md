# L40S Runbook — RQ1 1024-step behavioral scale test

This branch exists to minimize paid GPU idle time. Do not redesign the experiment on the pod.

## Why this is the next run

Run 003 completed the 128-step pointwise layer-KD control on an L40S. Run 004 then implemented residual-contribution behavioral KD. The initial Run 004 was exploratory because its corpus envelope did not match Run 003. Run 004-M is the matched 128-step behavioral replication. The preregistered next evidence gate is therefore G4: determine whether the behavioral objective survives a longer 1,024-step training horizon.

The frozen protocol is `research/protocols/RQ1_V2_SCALE1024.yaml`.

## Before starting the paid pod

Have these persistent paths ready or restore them immediately:

- `/workspace/models/qwen3.8-27b-dbdc473`
- `/workspace/runs/pilot001/transferred`
- `/workspace/corpora/gutenberg/train.txt`

Two hashes describe the same frozen data pipeline at different stages and must not be confused:

- raw `/workspace/corpora/gutenberg/train.txt`: `bc5972d9a52580ff14ab1b3b1753f9cd68c726c63cc625a7ed3913ec3c5dc5c5`
- deterministic packed stream after teacher tokenization + EOS packing + the 700,000-token cap: `e11ca38bb099fc89c2f74e96f5d2f1209def6a16f6a8432d4e9972acd50c100d`

The launcher verifies the raw hash before loading weights and verifies the packed-stream hash from `summary.json` after training. This prevents a source-file mismatch from consuming GPU time while also proving that the matched token stream was reproduced.

Use the same L40S software baseline used by the matched experiment: Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 5.15.1.

## On the pod

Checkout `prep/l40s-rq1-scale1024`, pull the latest branch state, leave the working tree clean, and run exactly:

```bash
bash scripts/launch_l40s_scale1024.sh
```

The launcher refuses the wrong branch, a dirty working tree, wrong GPU count/model, missing teacher/student/corpus/configs, wrong raw corpus hash, wrong critical Python/PyTorch/Transformers versions, or a non-empty target run directory. It records provenance before training.

The historical command specified a 45-GiB external guard, but the archived L40S exposed only about 44.39 GiB physical capacity, making that `nvidia-smi` threshold ineffective. The paid-run launcher therefore uses a 44.0-GiB operational safety guard. This does not alter the model, data, optimizer, loss, precision, seed, or schedule; Run 003's recorded training peak was about 38.95 GiB allocated / 40.77 GiB reserved.

Do not use the L40S session for package research, architecture edits, plotting, documentation, benchmark selection, or hyperparameter tuning.

## What changes from Run 004-M

The intended optimization change is the training horizon: 128 -> 1,024 steps. Architecture, teacher revision, student initialization, sequence length, batch size, QLoRA geometry, seed, objective definition, temperature, top-k, corpus identity, and 700k-token corpus cap remain frozen.

The instrumentation cadence follows the historically preregistered 1,024-step command: log every 16 steps, evaluate every 128, save every 512. These intervals do not change the optimization objective.

The longer run will revisit the finite 700k-token packed stream because 1,024 x 1,536 = 1,572,864 processed positions, which exceeds the unique packed-token count. This is deliberate for continuity with the matched envelope and must be reported when interpreting the scale result; it is not evidence from 1.57M unique corpus tokens.

## After the run

Before bundling evidence, the launcher requires `summary.json` and the behavioral manifest and verifies: completed outcome, 1,024 steps, 700,000 packed tokens, 1,536 sequence length, exact packed-stream hash, behavioral/delta objective identity, exact teacher revision, and the 700k cap.

It then writes `evidence_bundle.tgz` inside the run directory without checkpoints, optimizer state, model weights, or recursively including the archive itself. Preserve that bundle before terminating the pod.

The decision after this run is binary:

- if training is stable and output-level teacher alignment / validation behavior preserve the Run 004-M signal, advance to the separately locked 4,096-step protocol;
- otherwise stop scale-up and analyze the failure before spending more GPU time.

Do not infer success from behavioral training loss alone.
