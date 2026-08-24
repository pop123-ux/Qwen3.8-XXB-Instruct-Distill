# Making Level 2 survive a Colab disconnect

The Level-2 run reached ~step 500 on a Tesla T4. No OOM, ~2100 tokens/s, validation
bits-per-byte down from 1.317 at step 200 to 1.279 at step 400. Then the Colab runtime
disconnected, `/content` was reclaimed, and all of it was gone.

Nothing about the model was wrong. Everything about the persistence was.

## The invariant

> A crash may lose the step currently executing. It must never invalidate the last
> checkpoint that completed.

Note what this does *not* promise. A hard kill — SIGKILL, a Colab runtime teardown, a
preempted VM — cannot be caught by any handler, so no amount of graceful-shutdown code
saves the step in flight. The guarantee is about everything before it.

## Why the old checkpoints could not have been resumed anyway

Even if the files had survived, "resume" would have produced a different run wearing the
old run's step number. A checkpoint held weights, optimizer state and a step counter.
An audit of what a resume actually needs:

| state | was it saved? |
|---|---|
| model weights | yes |
| optimizer state | yes |
| global step, best validation loss, history | yes |
| config | yes |
| **LR scheduler state** | **no** — the one-cycle schedule would restart from warmup |
| **AMP GradScaler state** | **no** — the loss scale would reset |
| **RNG state** (Python, torch, CUDA) | **no** — different dropout, different shuffle |
| **data position** | **no** — the stream would rewind to epoch 0 and re-train seen data |
| tokens seen, epoch, elapsed time | no |
| git commit, precision, hardware | only at run level, not per checkpoint |

Eleven of seventeen items missing. The scheduler and scaler omissions are silent
correctness bugs: the loss curve bends and nothing reports why.

## Two artifacts, two costs

These answer different questions and must not share an interval.

| | progress record | full checkpoint |
|---|---|---|
| contains | metrics, step, tokens, timestamps | weights, optimizer, scheduler, scaler, RNG, data position |
| size | ~400 bytes | ~1.1 GB at 94.5M with fp32 AdamW state |
| interval | `log_every` (25 steps) | `save_every` (100–500 steps) |
| written to | `metrics.jsonl` + `progress/latest.json` | `checkpoints/step_NNNNNN/` |

Writing a full checkpoint every step would make the experiment I/O-bound and, if it were
also being copied to Drive, unusable. Writing progress every step costs nothing. So they
are separate, and both are configurable:

```yaml
training:
  save_every: 500        # full checkpoint
  log_every: 25          # progress record (progress_every defaults to this)
  progress_every: 25     # optional, if you want them to differ
  persistent_backup: null   # or a Drive path; off by default
```

## Layout

```
experiments/runs/t4_level2_100m_ckpt/
    checkpoints/
        latest.json                  <- pointer, updated only after a verified write
        step_000500/
            model.safetensors        <- weights (no pickle, memory-mappable)
            optimizer.pt
            scheduler.pt
            scaler.pt
            rng.pt                   <- python + torch + numpy + cuda
            training_state.json      <- step, tokens, data position, history
            config.json
            metadata.json            <- what this checkpoint is, standalone
            COMPLETE                 <- written last, after everything is fsynced
        .step_000600.incomplete/     <- a killed write; ignored, removed at startup
    progress/
        latest.json
    metrics.jsonl
    summary.json
```

Step numbers are zero-padded so lexical order matches numeric order — `step_000500`
sorts before `step_002000` in any file listing.

## Atomicity

Every checkpoint is written to `.step_NNNNNN.incomplete/`, fsynced, checked for its
required files, then `os.replace`d into place — atomic on POSIX — and only then does
`latest.json` change. A reader that trusts `latest.json` and the `COMPLETE` marker can
never be handed a partial write.

`is_complete()` requires **both** the marker and every required file, plus
`"complete": true` in the manifest. The marker alone would trust a write interrupted
just after it; the files alone would trust a directory mid-write.

So if step 600's write dies halfway:

- `step_000500/` is untouched and still loadable
- `latest.json` still says 500
- `.step_000600.incomplete/` is ignored by discovery and deleted on next startup
- `--resume latest` resumes from 500

## One checkpoint format, agreed on by every tool

There is exactly one canonical layout, written by the trainer and read by the validator
and by resume. There is **no `final/` directory** — an earlier implementation wrote one
in a bespoke layout while the other tools expected `checkpoints/step_NNNNNN/`, and that
disagreement produced a real T4 session where a successful 20-step run left nothing the
validator or `--resume` could find. Two checkpoint systems is the bug; a compatibility
alias would have kept it.

A note on how that failure presented, because it is a trap worth naming: `--resume` did
not exist as an option at the time, only `--resume-from`. **argparse silently accepts an
unambiguous prefix**, so `--resume latest` bound to `--resume-from` and `latest` became a
literal relative path — the loader then looked for `latest/training_state.pt`. A missing
option became a wrong value instead of an error. `--resume` is now a real option, typed
as a string rather than a `Path`, so `latest` stays a token for the resolver to interpret.

## Resume

```bash
python scripts/train_student.py --config configs/experiments/t4_level2_100m_ckpt.yaml --resume latest
```

`latest` resolves through `latest.json`, verifies the checkpoint, and falls back to the
newest directory that verifies on its own terms if the pointer is stale or missing — a
Drive copy can arrive out of order. You can also pass a step number (`--resume 400`) or
a path.

**If nothing valid matches, it exits 2 rather than starting from step 0.** Silently
restarting is how you lose 500 steps twice.

### Extending a run

`max_steps` is the intended **total** training length, so raising it continues the run:

```bash
python scripts/train_student.py --config <config> --max-steps 20            # 0 -> 20
python scripts/train_student.py --config <config> --resume latest --max-steps 40   # 20 -> 40
```

OneCycleLR's learning rate at step *t* is a function of `total_steps`, so its saved state
cannot be replayed against a longer horizon. The schedule is **rebuilt for the new total
and fast-forwarded** to the restored step — verified to be exactly a fresh 40-step curve
advanced 20 steps. That genuinely changes the LR curve for the remaining steps, so it is
reported, never silent:

```
    training target extended: 20 -> 40 steps
    the one-cycle LR schedule is rebuilt for the new total and fast-forwarded;
    remaining steps follow the new curve, not the original one
```

Nothing else resets: model, optimizer, scaler, RNG, data position and token count all
carry over.

### What is still refused

Flexibility about `max_steps` is not permission to resume onto anything. Differences that
would make the weights or the saved data position meaningless remain fatal:

| change | why it is fatal |
|---|---|
| `model.architecture`, `model.pretrained`, `model.spec_path` | the checkpoint's tensors have different shapes |
| `data.max_sequence_length` | the corpus is re-chunked, so the saved data position addresses sequences that no longer exist |
| `training.batch_size` | the saved position is a batch index, which does not carry across |
| `training.strategy` | a different strategy holds different parameters |

Changes that alter what is being optimised — learning rate, optimizer, precision,
accumulation, corpus — are **allowed and itemised** in the resume banner rather than
being either blocked or hidden.

### Is a resumed run the same run?

Tested directly. Run A trains 0→8 continuously; run B trains to 4, checkpoints, reloads
and continues to 8. On CPU the final weights are **bit-identical across all tensors**,
and the scheduler's `last_epoch` and `_last_lr` match exactly.

This is **not** guaranteed on GPU. cuDNN algorithm selection and atomic reductions make
some kernels non-deterministic across processes, so a GPU resume should be expected to
match closely, not exactly. We do not claim bit-identical GPU training, because we have
not measured it.

## Persistence

`/content` is staging, not storage. A completed checkpoint is only safe once it is
somewhere else.

```bash
# after a run, or between runs
python scripts/backup_colab_to_drive.py --source experiments/runs/t4_level2_100m_ckpt \
    --destination /content/drive/MyDrive/qwen-distill/t4_level2_100m_ckpt --checkpoints-only

# see exactly what would move, change nothing
python scripts/backup_colab_to_drive.py --source ... --destination ... --dry-run
```

`--checkpoints-only` copies checkpoints plus the kilobyte-sized files that make a
recovered checkpoint interpretable (`metrics.jsonl`, `summary.json`, `progress/`), and
skips code — code belongs in git.

The backup **refuses to copy an incomplete checkpoint**. Publishing a half-written
checkpoint to Drive is worse than not copying it at all: it becomes the newest thing
there, and recovery reaches for the newest. All the existing safety properties still
hold — never follows symlinks, never copies credential-shaped files, never deletes
without both `--delete-extraneous` and `--yes`.

Automatic per-checkpoint backup is available but **off by default**, because the trainer
must work with no Drive, no network and no mounted volume:

```yaml
training:
  persistent_backup: /content/drive/MyDrive/qwen-distill/t4_level2_100m_ckpt/checkpoints
```

A backup failure is reported and does not stop training. Losing the copy is recoverable;
losing the run is not.

## Validating a checkpoint

```bash
# an explicit checkpoint
python scripts/validate_checkpoint.py experiments/runs/t4_level2_100m_ckpt/checkpoints/step_000020

# or the run directory, which resolves to its newest verified checkpoint
python scripts/validate_checkpoint.py experiments/runs/t4_level2_100m_ckpt
```

A directory that is itself a checkpoint is always used exactly as given, so the interface
stays unambiguous. Checkpoints written before this format are still readable, so existing
Level-1 artifacts do not become unverifiable.

## Where did my run get to?

```bash
python scripts/train_student.py --config configs/experiments/t4_level2_100m_ckpt.yaml --status
```

Answers from files on disk alone, so it works in a fresh Colab session that has never
seen the run — which is exactly when the question gets asked. Reports the latest
progress record, every complete checkpoint, and whether the run is resumable.

## Full Colab workflow

### First session

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
cd /content
git clone https://github.com/pop123-ux/Qwen3.8-XXB-Instruct-Distill
cd Qwen3.8-XXB-Instruct-Distill
pip install -e ".[dev]" && pip install -r requirements/training.txt

python scripts/train_student.py --config configs/experiments/t4_level2_100m_ckpt.yaml --dry-run
python scripts/train_student.py --config configs/experiments/t4_level2_100m_ckpt.yaml

# after the run, or any time you want the work safe
python scripts/backup_colab_to_drive.py \
    --source experiments/runs/t4_level2_100m_ckpt \
    --destination /content/drive/MyDrive/qwen-distill/t4_level2_100m_ckpt \
    --checkpoints-only
```

### After a disconnect

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
cd /content
git clone https://github.com/pop123-ux/Qwen3.8-XXB-Instruct-Distill
cd Qwen3.8-XXB-Instruct-Distill
pip install -e ".[dev]" && pip install -r requirements/training.txt

# bring the experiment back from Drive
mkdir -p experiments/runs
cp -r /content/drive/MyDrive/qwen-distill/t4_level2_100m_ckpt experiments/runs/

# confirm what survived before spending GPU time on it
python scripts/train_student.py --config configs/experiments/t4_level2_100m_ckpt.yaml --status

# continue from where it stopped
python scripts/train_student.py --config configs/experiments/t4_level2_100m_ckpt.yaml --resume latest
```

## Storage cost

At 94.48M parameters in fp32 with AdamW state, one checkpoint is roughly:

| file | size |
|---|---|
| `model.safetensors` | ~360 MB |
| `optimizer.pt` (two AdamW moments) | ~720 MB |
| everything else | < 1 MB |
| **total** | **~1.1 GB** |

At `save_every: 500` over 2000 steps that is four checkpoints, ~4.5 GB. Comfortable on a
15 GB free Drive; prune older ones if you lower the interval. Progress records over the
same run total a few hundred kilobytes.

**Do not commit checkpoints to git.** `.gitignore` already excludes `*.pt`, `*.ckpt` and
`experiments/**/runs/`; `*.safetensors` is excluded too. Git holds code, configs, tests,
docs and compact experiment metadata. Drive holds weights.
