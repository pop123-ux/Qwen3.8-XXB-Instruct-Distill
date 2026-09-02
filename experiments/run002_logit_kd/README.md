# Run `run002_logit_kd` — knowledge distillation

Run ID: `run002_logit_kd`
Created: 2026-09-02T19:30:43.304697+00:00
Status: see `manifest.json` -> `status`

This directory, not any terminal scrollback or chat transcript, is the record of this run.

## What is here

| Path | What it is |
| --- | --- |
| `manifest.json` | The complete statement of what this run is: commit, teacher revision, tokenizer, corpus, hardware, every hyperparameter. Written **before** the run started. |
| `config.json` | The resolved experiment config, exactly as the trainer read it. |
| `command.txt` | The exact command that launched the run. |
| `environment.txt` | Python, PyTorch, CUDA, Transformers, full `pip freeze`. |
| `hardware.txt` | `nvidia-smi`, GPU model and VRAM, driver, RAM, filesystems. |
| `git.txt` | Repository, branch, commit SHA, clean/dirty state, modified paths. |
| `teacher_provenance.json` | Teacher model id, **exact upstream revision SHA**, metadata checksums. |
| `tokenizer_provenance.json` | Tokenizer identity, vocabulary size, file checksums. |
| `dataset_provenance.json` | Corpus identity, byte counts, `train`/`validation` SHA-256. |
| `metrics.jsonl` | Append-only, fsynced per record: loss, KD components, LR, step, throughput, elapsed, validation. |
| `training.log` | Full stdout+stderr of the run, tee'd as it is produced. |
| `progress/latest.json` | Atomically updated "where did this run get to". |
| `checkpoints/` | Full resumable checkpoints (weights, optimizer, scheduler, scaler, RNG, data position). |
| `artifacts/` | Plots, evaluation outputs, anything derived. |
| `final/` | The end-state student, once the run completes. |
| `CHECKSUMS.txt` | SHA-256, size and source path for every important artefact. |
| `termination.json` | How the run ended, written whether it succeeded or failed. |

## Archival

See the repository's `experiments/run002_logit_kd/` directory.

## If this run failed

Do not delete it, do not overwrite it, and do not restart it from step 0. A failed run
is a measurement. Preserve the partial `metrics.jsonl`, the `training.log`, the last
complete checkpoint and `termination.json`, and resume with
`runtime.resume_from` pointing at `checkpoints/latest.json`.

## Reproducing

    git clone git@github.com:pop123-ux/Qwen3.8-XXB-Instruct-Distill.git
    git checkout f5dd3f75bf4e069c75442396bcf43f2f52d0471d
    # teacher: Qwen/Qwen3.8-27B @ dbdc473dea0d6a9763042881cc33d6058d1742d2
    PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/kd_run.py --teacher /workspace/models/qwen3.8-27b-dbdc473 --revision dbdc473dea0d6a9763042881cc33d6058d1742d2 --quantization 4bit --student canonical --pretrained /workspace/runs/pilot001/transferred --text-path /workspace/corpora/gutenberg/train.txt --sequence-length 1536 --max-tokens 700000 --steps 128 --batch-size 1 --gradient-accumulation-steps 1 --objective logit_kd --kd-weight 1.0 --kd-temperature 2.0 --kd-top-k 64 --strategy qlora --optimizer adamw --lora-rank 16 --lora-alpha 32 --precision bf16 --seed 0 --log-every 1 --eval-every 32 --save-every 64 --name run002_logit_kd --output /workspace/runs/run002_logit_kd
