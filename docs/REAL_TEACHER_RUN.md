# Running the first real Qwen3.8-27B experiment

Everything needed to take this repository to an external GPU and start. Two commands.

The 27B teacher does **not** fit a 16 GB card. Estimated ~16.3 GiB at 4-bit against 13.56
GiB usable, so **24 GB is the practical floor**. That constrains where the *teacher* runs
and says nothing about the 16 GB student target — they are separate budgets.

---



## The pin: `72a217a`, not yet resolved to 40 characters

The teacher's checkpoint-upload commit is known in abbreviated form:

```
Qwen/Qwen3.8-27B @ 72a217a          (abbreviated — NOT usable as the research pin)
```

`TeacherLoadPlan.validate()` accepts a 7-to-40 character commit id, so `72a217a` will pass
the gate and load. **Do not use it as the recorded pin.** An abbreviation is not immutable:
it is a prefix, and a prefix can become ambiguous as the repository grows. Final provenance
must carry the full 40-character SHA.

Resolving it needs one Hugging Face metadata request, which the authoring sandbox cannot
make — `huggingface.co` egress is denied there by organisation policy. On the GPU machine:

```bash
python - <<'EOF'
from huggingface_hub import HfApi
print(HfApi().model_info("Qwen/Qwen3.8-27B", revision="72a217a").sha)
EOF
```

That is one request and it returns the full SHA. Use that value everywhere afterwards, and
record it in the ledger alongside the run. The abbreviation is recorded here only so the
next session does not have to rediscover which commit is meant.

## The one-command start

```bash
# 1. fetch the pinned checkpoint. Resumable; writes teacher_download_manifest.json.
python scripts/download_teacher.py \
    --revision <EXACT_QWEN3.8-27B_COMMIT_SHA> \
    --output /data/models/qwen3.8-27b

# 2. verify the weights actually load. THIS is the authoritative check.
python scripts/teacher_smoke_test.py \
    --local-path /data/models/qwen3.8-27b \
    --revision <EXACT_QWEN3.8-27B_COMMIT_SHA> \
    --quantization 4bit \
    --json runs/teacher_smoke.json
```

Three things are worth keeping straight, because they are easy to conflate:

- **the revision SHA is the research pin.** A repo id serves different weights over time, so
  every artifact records the SHA and an unpinned Hub load is refused before it downloads;
- **the manifest records the download.** It says which files arrived and how big they were;
- **the smoke test performs the pretrained-weight verification.** The downloader proves
  files exist. It cannot prove they load as the intended teacher — `transformers` returns a
  freshly-initialised model rather than raising when keys do not match, so the missing-weight
  gate in `load_verified_teacher()` is the only thing standing between you and a 27B teacher
  full of random numbers generating fluent nonsense.

A fresh GPU machine should need nothing beyond those two commands before the pilot.

## A. Setup

```bash
git clone https://github.com/pop123-ux/Qwen3.8-XXB-Instruct-Distill.git
cd Qwen3.8-XXB-Instruct-Distill
pip install -r requirements/base.txt
pip install accelerate bitsandbytes        # device_map and 4-bit; not in base
export PYTHONPATH=src
python -m pytest tests/ -q                 # must be green before trusting anything
```

## B. Pin the teacher

Open <https://huggingface.co/Qwen/Qwen3.8-27B>, go to **Files and versions**, and copy the
full commit SHA of the revision you intend to use.

Pass it as `--revision`. **The smoke test refuses a hub load without one** — the same repo
id serves different weights over time, so an unpinned run cannot be reproduced and its
numbers could not be attributed to a specific checkpoint later. Do not invent a SHA.

Set `HF_TOKEN` if the repo is gated.

## C. Smoke test

```bash
python scripts/teacher_smoke_test.py \
    --quantization 4bit \
    --revision <EXACT_QWEN_COMMIT_SHA> \
    --json runs/teacher_smoke.json
```

Ten checks: load (with the missing-weight gate armed), tokenizer, chat template across all
four reasoning modes, tokenize, generate, logits, TeacherSignal, dimensions, alignment,
provenance. Roughly 10–30 minutes, mostly download.

On >40 GB drop `--quantization 4bit` and add `--dtype bfloat16`. To bound placement:
`--max-memory '{"0":"22GiB","cpu":"64GiB"}' --offload-folder /tmp/offload`.

## D. Reading the result

**`ALL TEN CHECKS PASSED`** means the teacher is operational and the KD path is unblocked.

The numbers that matter are in the check-9 sweep:

| | meaning |
|---|---|
| `tail mass` at k=64 | teacher probability outside the stored shortlist. Small → an offline logit corpus loses little. Large → raise *k* or stay online. |
| `entropy` | near zero means a near-deterministic teacher, so KD carries little more than its argmax |
| `top1 agree` | sanity only; 1.000 against its own signal is expected |
| `MEASURED peak GPU memory` | the **only** real memory number produced. Every other figure is an analytical estimate, labelled as such. |

Failures:

| message | meaning |
|---|---|
| `which means those weights are RANDOM` | the checkpoint did not load. **Never proceed** — transformers does not raise here, and the resulting KD loss would be finite, falling and meaningless. Re-download; check the revision. |
| `not the architecture this project verified` | a different checkpoint under the expected name. Stop. |
| `reasoning modes render identical prompts` | the controls do nothing; any reasoning-cost comparison would measure noise |
| `could not load the tokenizer` | vendored metadata has no `tokenizer.json`; it must come with the weights |
| CUDA OOM | use `--quantization 4bit`, or `--max-memory` / `--offload-folder` |

## E. First transfer: teacher -> the canonical student

Only after the smoke test passes. Assumes the weights are already on disk from step C.

```bash
python scripts/distill_pilot.py \
    --teacher <PATH_TO_DOWNLOADED_QWEN3.8-27B> \
    --revision <EXACT_QWEN_COMMIT_SHA> \
    --output runs/pilot1
```

The pilot has **no architecture arguments**. The student is always the canonical frozen
target `qwen38_19b_h5120_l48_moe` — 13,008,505,728 parameters, 48 layers, 8 routed experts
of width 768 with top-2 routing plus one shared expert. A pilot that could quietly run a
different geometry would produce a result nobody could attribute.

Chain: pinned revision → verified load → 64→48 group-aligned layer mapping → materialise
(copy, KV-merge 4→2, decompose the dense FFN into experts) → coverage report → checkpoint.

Add `--dry-run` to see the plan, the parameter audit and the 16 GB verdict without loading
or writing anything. It runs in seconds on a laptop and is worth doing before the GPU is
rented.

What to expect: `complete: True` with no missing tensors, and coverage just under 100% —
the router and the shared-expert gate have no teacher counterpart and are initialised
rather than transferred, so a report claiming 100% would be wrong.

**Mechanism check, not this.** `scripts/chain_selftest.py` is the developer harness that
drives a small dense student through transfer → KD → checkpoint. It keeps geometry flags
because varying geometry is its job. It is not a research run and its loss means nothing
about capability.


## E2. Tokenising a corpus for the canonical student

The canonical student has a **248,320-entry embedding**, inherited from the teacher. The
byte-level corpus path (`data.text_corpus`) emits ids in 0-255 and would only ever index
the first 256 rows of it, so it **cannot** train this student. That path is not going
away — every historical experiment through Level 2R used it, and it stays the byte-level
baseline — but it is legacy for the canonical target.

Tokenizer-backed training uses **the teacher's own tokenizer**, taken from the pinned
checkpoint downloaded in step C:

```bash
/data/models/qwen3.8-27b/        # from step C; tokenizer.json lives here
```

Set the data block to:

```yaml
data:
  tokenized_text: true
  text_path: corpora/train.txt          # plain UTF-8
  tokenizer_path: /data/models/qwen3.8-27b
  document_separator: blank_line        # or "line", or "file"
  max_sequence_length: 4096
  expected_vocab_size: 248320           # the student's vocab_size; a mismatch refuses
```

Points worth being explicit about:

- **The tokenizer path loads tokenizer files only.** `AutoTokenizer.from_pretrained(...,
  local_files_only=True)` reads a few megabytes beside the 54 GB of weights and never
  opens them. Corpus preparation therefore needs no GPU and no network, and can be done
  before the machine is rented.
- **`expected_vocab_size` is a refusal, not a repair.** If the tokenizer disagrees with the
  student, the run stops. Nothing resizes the embedding: that would leave rows the run
  never trains and a checkpoint whose vocabulary cannot be accounted for.
- **248,320 is never hardcoded as the tokenizer's answer.** The vocabulary is read off the
  loaded tokenizer with `len(tokenizer)` and recorded; `expected_vocab_size` is the
  student's number, checked against it.
- **Packing is `document + EOS + document + EOS + ...`**, chunked at exactly
  `max_sequence_length`. The trailing partial chunk is dropped and counted rather than
  padded, because the trainer feeds one rectangular tensor as both input and labels with
  no mask and no `-100` — padding would be trained on as if it were text.
- **`vendor/qwen38-metadata` cannot serve as `tokenizer_path`.** It carries
  `tokenizer_config.json` but no `tokenizer.json`, so it has no vocabulary to encode with.
  Use the downloaded checkpoint.
- **The run summary records the tokenizer** — class, vocabulary size, EOS id, source path,
  per-file SHA-256, and the teacher model and revision when the config supplies them —
  inside the existing `corpus` block of `summary.json`.

A smoke run needs no large corpus: `max_documents` and `max_tokens` bound the stream, and
a few kilobytes of text is enough to prove the path end to end.

**What CPU work has and has not established.** The data path, its packing, its vocabulary
refusals and its trainer integration are tested offline against the repository's own tiny
tokenizer fixture (`tests/test_tokenized_data.py`). None of that touches the real Qwen
tokenizer, the real teacher, or a GPU. The full teacher SHA is resolved here, on the GPU
machine, in step B — it was not available to the CPU work.


## F. Preserve these

- `runs/teacher_smoke.json` — provenance, tail-mass sweep, measured peak memory
- `runs/kd_pilot/pilot_record.json` and `summary.json` — transfer coverage, KD diagnostics
- the exact `--revision` you used

Small JSON. Bring them back to the repository.

## G. Never commit

Model weights, student checkpoints, optimizer state, generated teacher datasets, corpora,
`HF_TOKEN` or any credential, or the downloaded teacher directory. `.gitignore` covers the
usual paths; the JSON above is the only thing worth keeping.
