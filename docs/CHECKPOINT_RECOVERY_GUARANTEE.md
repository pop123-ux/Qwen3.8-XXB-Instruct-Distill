# The checkpoint recovery guarantee

> **A checkpoint is not considered persisted until the destination copy has been
> independently verified as loadable and complete.**

> **Accidental deletion of a previously persisted checkpoint is detected at the next
> status, restore or resume operation, and does not cause silent corruption or false
> resumability.**

Those two sentences are the contract. The rest of this document says what enforces them,
what they cost, and exactly what they do not cover.

---

## 1. What went wrong

A Level-2R run reached approximately step 800. Persistent storage held:

```
step_000200/   COMPLETE  config.json  metadata.json  rng.pt
step_000400/   scaler.pt  scheduler.pt  training_state.json
step_000600/
```

Every small file. **No `model.safetensors`. No `optimizer.pt`.** And `latest.json`
recording `"complete": true`.

Whether those files were deleted by hand, dropped by Drive, lost to an interrupted copy,
or never written is **not recoverable after the fact, and the system must not depend on
knowing**. What is recoverable is why nothing noticed.

### The root cause

`is_complete()` asked whether the required *filenames* were present:

```python
if not all((directory / name).is_file() for name in REQUIRED_FILES):
    return False
```

`Path.is_file()` is `True` for a zero-byte file, for a file truncated to 50 bytes, and for
360 MB of zeros. Four consequences followed from that one line:

1. **Nothing recorded what the files should be.** No size, no digest — so nothing
   downstream *could* tell a whole `model.safetensors` from a hollow one. Only total
   absence was detectable, and only by luck.
2. **The destination was checked with the same weak test.** `persist_checkpoint` copied
   with `shutil.copytree`, asked `is_complete(final)`, and printed `persisted -> …`. A
   copy that arrived truncated passed.
3. **Nothing was fsynced at the destination.** `shutil.copytree` never fsyncs, and on a
   Drive FUSE mount `close()` can return long before the bytes are durable. A copy could
   verify from cache and be short on disk minutes later.
4. **`latest.json` was believed.** It stores `"complete": true` as a literal field written
   at save time. That is a claim about a moment in the past, and it outlived the files it
   described.

Every caller — trainer, backup, restore, status report, run analysis — shared that one
definition, so hardening any of them individually would have left the others wrong. That
is the failure this phase fixes, and it is bigger than any one deletion.

---

## 2. The validity contract

A checkpoint is **VALID** only if all of the following hold, checked *now*:

| # | requirement |
|---|---|
| 1 | the `COMPLETE` marker is present |
| 2 | every **core** artifact is present: `model.safetensors`, `training_state.json`, `metadata.json` |
| 3 | every **resume** artifact is present: `optimizer.pt` |
| 4 | every artifact the checkpoint's **own metadata says it contains** is present |
| 5 | no required file is zero-length |
| 6 | no file is below its **format floor** (a safetensors file is not 50 bytes) |
| 7 | no file is outside the **band its recorded parameter count allows** |
| 8 | every size matches `checkpoint_manifest.json` *(manifest level)* |
| 9 | every SHA-256 matches `checkpoint_manifest.json` *(manifest level)* |
| 10 | the weights deserialize and the tensor count matches the architecture *(load level)* |
| 11 | the optimizer state deserializes and is an optimizer state dict *(load level)* |
| 12 | `metadata.json` records `complete: true` |

**A directory containing `COMPLETE` but missing `model.safetensors` or `optimizer.pt` is
INVALID.** Requirement 4 is the one that makes this work on the three checkpoints already
on Drive: they predate manifests, but their `metadata.contents` lists the files they were
written with, so a file named there and now absent is a deletion — detectable with no new
information.

### One validator, one definition

`src/qwen_distill/training/checkpoint_validation.py` is the only place that decides.
Everything else calls it:

| caller | what it uses |
|---|---|
| `checkpoints.is_complete()` | a thin boolean over the validator |
| `checkpoints.list_checkpoints()` | verified checkpoints only |
| `checkpoints.resolve_checkpoint("latest")` | `resolve_latest()` |
| `checkpoints.update_latest_pointer()` | refuses, with a reason |
| `checkpoints.load_checkpoint()` | refuses, with a reason |
| `persist.persist_checkpoint()` | source **and** destination |
| `persist.restore_run()` | source, and the restored copy |
| `persist.persistent_status()` / `preflight()` | every checkpoint, live |
| `analysis.build_checkpoint_timeline()` | the same verdict |
| `scripts/validate_checkpoint.py` | all levels |

A failure returns structured detail, not a bare `False`: `missing_files`,
`zero_length_files`, `implausible_sizes`, `size_mismatches`, `checksum_mismatches`,
`load_failure`, `architecture_mismatch`, `resume_failure`.

### Verification levels

| level | checks | cost |
|---|---|---|
| `structure` | 1–7, 12 | no large reads; milliseconds |
| `manifest` | + 8, 9 | one full read; ~3 s per 1.13 GB checkpoint locally |
| `load` | + 10, 11 | full deserialization |

`structure` is the default for scanning a directory of checkpoints, because a status
report that costs a full read of every checkpoint is a status report nobody runs.
`manifest` is the default for **persistence**, and `load` is the default for
`validate_checkpoint.py`.

### Serialization formats, stated explicitly

| artifact | format | required |
|---|---|---|
| `model.safetensors` | safetensors | always |
| `optimizer.pt` | torch pickle | to resume |
| `scheduler.pt`, `scaler.pt`, `rng.pt` | torch pickle | when written |
| `training_state.json`, `metadata.json`, `config.json`, `checkpoint_manifest.json` | JSON | always / when written |
| `COMPLETE` | text | always, written last |

### Size plausibility

Two independent floors, because they catch different things. Exact sizes are **not**
hard-coded — dtype, tied embeddings and optimizer choice all move the true figure.

- **Format floor**: 64 bytes for `model.safetensors` and `optimizer.pt`. Below this the
  file is not small, it is destroyed.
- **Parameter band**: `model.safetensors` within `[0.5, 6.0]` bytes/parameter,
  `optimizer.pt` within `[0.3, 24.0]`, plus 1 MB of format overhead on the ceiling so a
  tiny test model is not falsely rejected.

For the Level-2R 94,476,448-parameter model that puts `model.safetensors` in
**[47 MB, 568 MB]**. A 50-byte file, a 0-byte file, and a 4 KB file are all decisively
wrong.

---

## 3. The persistence sequence

Every step is a gate. `persisted ->` is printed by exactly one code path, after step 8.

```
 1. validate the SOURCE                     -> refuse to copy a damaged checkpoint
 2. take the source's real sizes + SHA-256   (from its manifest, or computed now)
 3. copy every file EXCEPT COMPLETE into
    <dest>/checkpoints/.step_NNNNNN.incomplete/
    fsyncing each file, then the directory
 4. RE-READ the destination: size and
    SHA-256 of every file vs the source     -> any mismatch stops here
 5. write COMPLETE at the destination        (only now — a failed staging directory
                                              can never look finished)
 6. run the full validator on the staging copy
 7. promote atomically (os.replace), then
    re-validate the promoted directory
 8. update <dest>/checkpoints/latest.json
```

On failure at any step:

- the previously persisted checkpoint is **untouched**;
- `latest.json` is **not updated**;
- the staging directory is **left in place**, named `.incomplete`, skipped by every
  discovery path — it is the only evidence of what the copy managed to write;
- `PersistResult.verified` is `False` and the report names the failing files;
- **training continues**, because the local checkpoint is intact and losing a copy is
  recoverable, but the failure is printed to stderr and recorded in `summary.json`.

### What the operator sees

Success:

```
    checkpoint: step_000600
    persistent verification: PASS (manifest)
      model.safetensors: PASS (377,905,792 bytes)
      optimizer.pt: PASS (755,811,584 bytes)
      checksum verification: PASS
    persisted -> /content/drive/MyDrive/.../checkpoints/step_000600
```

Failure:

```
    checkpoint: step_000600
    persistent verification: FAILED
      2 file(s) did not arrive intact at .../step_000600

    missing:
      model.safetensors
      optimizer.pt

    latest pointer NOT updated.
    incomplete copy left at .../.step_000600.incomplete
```

---

## 4. Detecting a deletion that happens later

Verification at write time cannot protect a file from being deleted next week. What the
system guarantees is that the deletion is **found at the next operation that matters**,
and never papered over.

```
step_000600 persisted and verified
        |
        v
model.safetensors deleted  (by hand, by sync, by anything)
        |
        v
status / restore / resume / preflight
        |
        v
step_000600 reported INVALID — missing model.safetensors
        |
        v
latest pointer falls back to the newest checkpoint that verifies NOW
        |
        v
"falling back to step_000400 / resumable at step 400"
```

`latest.json` no longer means *"this directory used to be complete"*. It is a hint whose
target is re-validated on every read, and the substitution is always reported:

```
latest checkpoint step_000600 is invalid:
  missing model.safetensors — the checkpoint's own metadata says these were written

falling back to step_000400

resumable at step 400
```

Two things the system deliberately does **not** do:

- **it never recreates a missing file**, silently or otherwise;
- **it never reports a damaged checkpoint as resumable**, even when it is the newest and
  even when the pointer names it.

A fresh session counts *verified resumable* checkpoints, never directories:

```
persistent checkpoints found: 4 (3 verified resumable)

  step_000200  VALID
  step_000400  VALID
  step_000600  INVALID — missing model.safetensors
  step_000800  VALID

newest valid checkpoint: step_000800
```

---

## 5. Commands

**Validate persistent storage** (the copy that matters after a runtime dies):

```bash
python scripts/validate_checkpoint.py /content/drive/MyDrive/<run> --persistent
python scripts/validate_checkpoint.py /content/drive/MyDrive/<run> --persistent --level manifest
```

**Validate one checkpoint, including a real load:**

```bash
python scripts/validate_checkpoint.py <run>/checkpoints/step_000400 --level load
python scripts/validate_checkpoint.py <run>/checkpoints/step_000400 --behaviour
```

**Check status without training:**

```bash
python scripts/train_student.py --config <config> --persistent-dir <drive> --status
```

**Prove the guarantee on this machine and this filesystem:**

```bash
python scripts/test_persistence.py
python scripts/test_persistence.py --destination /content/drive/MyDrive/persist-selftest
```

**Start a clean Level-2R run** (the config is unchanged; the destination is new):

```bash
python scripts/train_student.py \
    --config configs/experiments/t4_level2r_100m_real_english.yaml \
    --persistent-dir /content/drive/MyDrive/t4_level2r_100m_real_english_v2
```

**Resume after Colab expires**, in a fresh session:

```bash
python scripts/train_student.py \
    --config configs/experiments/t4_level2r_100m_real_english.yaml \
    --persistent-dir /content/drive/MyDrive/t4_level2r_100m_real_english_v2 \
    --restore --resume latest
```

`--restore` copies back only checkpoints that verify, names any it skipped and why, and
rebuilds the local pointer from what verifiably arrived. `--resume latest` re-validates
before resuming and falls back with a stated reason.

**Do not reuse the previous persistent directory.** The three checkpoints in it are
invalid, and the new run starts from step 0 into `..._v2`.

---

## 6. Cost

| operation | cost |
|---|---|
| `structure` validation of one checkpoint | milliseconds; no large reads |
| `manifest` write at save time | one pass over the checkpoint (~3 s per 1.13 GB locally) |
| `manifest` verification at the destination | one pass over the copy |
| `load` verification | full deserialization |

At Level-2R's `save_every: 200` — roughly 26 minutes of training per checkpoint — digesting
1.13 GB locally is about **0.2% overhead**. On a Drive mount the read is slower and the
figure is correspondingly higher; it is still small against the alternative, which is
discovering at step 800 that nothing since step 0 is recoverable.

`fsync` is requested on every destination file and directory. On a FUSE filesystem that is
the strongest request available and **not a guarantee** — which is exactly why the
destination is re-read and digested afterwards rather than trusted.

---

## 7. What this does NOT guarantee

Stated plainly, because a guarantee whose limits are unstated is a claim.

- **It cannot stop a deletion.** Nothing in a training script can. It guarantees the
  deletion is detected and never masquerades as a valid checkpoint.
- **It cannot make an unreliable filesystem reliable.** If Drive accepts bytes, reports
  success, and discards them later, verification at write time will pass. Detection then
  happens at the next status, restore or resume — which is the guarantee, and it is why
  more than one checkpoint is kept.
- **`manifest` level proves byte-identity, not correctness.** A checkpoint that is a
  perfect copy of a wrong checkpoint passes. `--level load` and `--behaviour` address
  that; `scripts/validate_checkpoint.py --behaviour` is what proves the weights reload
  into the expected architecture and generate reproducibly.
- **It says nothing about model quality.** A perfectly persisted checkpoint can hold a
  model that generates `"and and and"`. See
  [experiments/POST_RUN_CHECKLIST.md](experiments/POST_RUN_CHECKLIST.md).
- **The size bands are plausibility checks, not predictions.** They are deliberately loose.
  A file inside the band can still be wrong; that is what the digest is for.

---

## 8. Verification status

| claim | status |
|---|---|
| the observed directories held metadata and no weights | **CORROBORATED** — reported by the user; not observed by this analysis |
| `Path.is_file()` returns True for a zero-byte file | **VERIFIED** — language semantics, and tested in `test_C_model_safetensors_zero_bytes` |
| `shutil.copytree` performs no fsync | **VERIFIED** — CPython source |
| the old check accepted a 32-byte `model.safetensors` | **VERIFIED** — three test fixtures did exactly that until this phase |
| deletion is detected on pre-manifest checkpoints via `metadata.contents` | **VERIFIED** — `test_a_pre_manifest_checkpoint_still_detects_deletion` |
| a truncated destination copy does not advance the pointer | **VERIFIED** — `test_J_...`, and stage 6 of the self-test |
| restore falls back to the newest valid checkpoint | **VERIFIED** — `test_N_...`, and stage 5 of the self-test |
| ~0.2% overhead at Level-2R's save interval | **CORROBORATED** — measured locally at 393 MB/s; Drive is slower and unmeasured |
| fsync makes bytes durable on a Drive FUSE mount | **UNKNOWN** — requested, not guaranteed; the re-read exists because of this |
