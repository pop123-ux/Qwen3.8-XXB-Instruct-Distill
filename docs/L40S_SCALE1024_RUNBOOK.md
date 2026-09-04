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

The corpus must hash to:

`e11ca38bb099fc89c2f74e96f5d2f1209def6a16f6a8432d4e9972acd50c100d`

Use the same L40S software baseline used by the matched experiment: Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 5.15.1.

## On the pod

Checkout this branch and run exactly:

```bash
bash scripts/launch_l40s_scale1024.sh
```

The launcher refuses the wrong GPU, missing teacher/student/corpus, wrong corpus hash, wrong critical Python/PyTorch/Transformers versions, or missing Run 004/VRAM-guard code. It records provenance before training and hard-stops the child process if used GPU memory exceeds 45 GiB.

Do not use the L40S session for package research, architecture edits, plotting, documentation, benchmark selection, or hyperparameter tuning.

## What changes from Run 004-M

The intended scientific change is the training horizon: 128 -> 1,024 steps. Architecture, teacher revision, student initialization, sequence length, batch size, QLoRA geometry, seed, objective definition, temperature, top-k, corpus identity, and 700k-token corpus cap remain frozen.

The longer run will revisit the finite 700k-token packed stream because 1,024 x 1,536 exceeds the unique packed-token count. This is deliberate for continuity with the matched envelope and must be reported when interpreting the scale result; it is not evidence from 1.57M unique corpus tokens.

## After the run

The launcher writes `evidence_bundle.tgz` inside the run directory without checkpoints, optimizer state, or model weights. Preserve that bundle before terminating the pod.

The decision after this run is binary:

- if training is stable and output-level teacher alignment / validation behavior preserve the Run 004-M signal, advance to the separately locked 4,096-step protocol;
- otherwise stop scale-up and analyze the failure before spending more GPU time.

Do not infer success from behavioral training loss alone.
