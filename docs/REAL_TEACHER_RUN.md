# Running the first real Qwen3.8-27B experiment

Everything needed to take this repository to an external GPU and start. Two commands.

The 27B teacher does **not** fit a 16 GB card. Estimated ~16.3 GiB at 4-bit against 13.56
GiB usable, so **24 GB is the practical floor**. That constrains where the *teacher* runs
and says nothing about the 16 GB student target — they are separate budgets.

---

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


## F. Preserve these

- `runs/teacher_smoke.json` — provenance, tail-mass sweep, measured peak memory
- `runs/kd_pilot/pilot_record.json` and `summary.json` — transfer coverage, KD diagnostics
- the exact `--revision` you used

Small JSON. Bring them back to the repository.

## G. Never commit

Model weights, student checkpoints, optimizer state, generated teacher datasets, corpora,
`HF_TOKEN` or any credential, or the downloaded teacher directory. `.gitignore` covers the
usual paths; the JSON above is the only thing worth keeping.
