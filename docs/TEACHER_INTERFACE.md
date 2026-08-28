# The real teacher: Qwen3.8-27B

Phase 4A made the teacher operational. This document is what you need to load it, what it
costs, and what it refuses to do.

---

## 1. Exact identity

| | |
|---|---|
| **Model ID** | `Qwen/Qwen3.8-27B` |
| **Revision** | **UNPINNED** — see §7 |
| Declared architecture | `Qwen3_5ForConditionalGeneration` (multimodal) |
| Class loaded here | `Qwen3_5ForCausalLM` (text tower only) |
| `model_type` | `qwen3_5`, text tower `qwen3_5_text` |
| Parameters | 26,895,998,464 |
| Layout | 64 layers, period-4 hybrid: 48 Gated DeltaNet + 16 gated attention |
| Hidden / FFN | 5120 / 17408 |
| Attention | 24 query / 4 KV heads, `head_dim` 256, `partial_rotary_factor` 0.25 |
| DeltaNet | 48 value / 16 key heads, head dim 128, conv kernel 4 |
| Vocabulary | 248,320, `tie_word_embeddings: false` |
| Context | 262,144 |

`EXPECTED_ARCHITECTURE` in `distillation/real_teacher.py` pins the first eight of these and
is checked against the loaded config on every load. A mismatch raises: every transfer plan,
memory estimate and parameter count in this repository was derived from this architecture,
so loading a different one would leave all of them silently wrong.

**The checkpoint is multimodal; we load only its text tower.** `Qwen3_5ForCausalLM`
declares `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`, so the
vision tower and the MTP head are discarded by design. The count of discarded tensors is
reported rather than hidden, so a checkpoint that starts shipping something else is noticed.

---

## 2. Tokenizer

| | |
|---|---|
| Class | `Qwen2Tokenizer` |
| Vocabulary | 248,320 |
| `model_max_length` | 262,144 |
| BOS | **none** — `bos_token: null`, `add_bos_token: false` |
| EOS | `<|im_end|>` (248046) |
| PAD | `<|endoftext|>` (248044) |
| `<think>` / `</think>` | 248068 / 248069, **single tokens each** |

Two things worth knowing:

**`add_bos_token` is verified by measurement, not read from the config.** `describe_tokenizer`
encodes a probe string with and without special tokens and compares. The flag and the
behaviour can disagree, and the alignment of every teacher signal depends on the behaviour.

**`config.json` and `tokenizer_config.json` disagree about EOS.** The config declares
`eos_token_id: 248044` (`<|endoftext|>`); the tokenizer declares `<|im_end|>` (248046).
`generation_config.json` settles it by listing both: `eos_token_id: [248046, 248044]`.
Generation uses the tokenizer's, which is the chat-template-appropriate one.

`</think>` being a single token is what makes the reasoning split **exact**. Token counts
are produced by splitting the generated ids at that token — never by counting whitespace,
and never by re-tokenising decoded strings, which is lossy. If a tokenizer ever lacks the
token, the split is reported as unavailable rather than approximated silently.

---

## 3. Reasoning modes

Read off `vendor/qwen38-metadata/chat_template.jinja`, and verified against it in the tests:

| mode | template kwargs | rendered length (probe) |
|---|---|---|
| `thinking_disabled` | `enable_thinking=False` | 114 |
| `low` | `reasoning_effort="low"` | 269 |
| `medium` | `reasoning_effort="medium"` | **103 — the shortest** |
| `xhigh` | `reasoning_effort="xhigh"` | 340 |
| *(default)* | none | identical to `xhigh` |

`high` is **not** accepted and raises inside the template. `medium` is not a no-op: it
injects no reasoning instruction while the default injects the long `xhigh` one, so it
renders the shortest prompt of the four.

**The teacher refuses to fall back.** `evaluation.runner` catches a `TypeError` from the
template and re-renders without the controls, which is right for a survey backend measuring
whether a control does anything. `render_prompt` raises `TemplateRejectedMode` instead: a
record labelled with a mode the prompt never carried would make the reasoning-cost
comparison measure nothing.

`mode_changes_the_prompt()` renders every mode and reports collisions, so "the controls do
nothing" is discovered before a generation run rather than inferred from flat token counts
afterwards.

---

## 4. The gate that matters

**`transformers` does not raise when a checkpoint's keys do not match the model class.** It
returns a freshly-initialised model and prints a report. Measured against 5.15.1:

| checkpoint keys | missing | unexpected | weights restored | raised? |
|---|---:|---:|---|---|
| untouched | 0 | 0 | yes | — |
| `model.language_model.*` | 0 | 0 | yes (remapped) | — |
| `garbage.*` | 56 | 55 | **no** | **no** |
| half the layers removed | 25 | 0 | partial | **no** |

A 27B teacher loaded in either of the last two states generates fluent text, produces a
plausible loss curve, and distils a student that learns nothing. Nothing downstream would
reveal it.

So `load_verified_teacher` treats a non-empty `missing_keys` or `mismatched_keys` as
**fatal**, naming the first few and saying why. This is the single most important behaviour
in the module.

The second row is the good news: the real checkpoint stores its text tower under
`model.language_model.*` and `transformers` remaps it correctly. Reading shards *directly*
does not remap, so `SafetensorsSource` carries `TEXT_TOWER_PREFIXES` for the transfer path —
without it a transfer plan against the real teacher would find every tensor missing.

---

## 5. Memory: estimates, not measurements

**None of these figures has been measured on hardware.** They come from
`qwen_distill.architecture.memory`, the same analytical estimator used for the students,
via `teacher_memory_estimate()` — which reports `measured: false` alongside every number.
Treat them as sizing guidance for choosing a GPU class, not as a hardware requirement.
The first real load will produce measured figures; until then these are arithmetic.

At 4k context, components broken out because they scale and confirm differently:

| scheme | weights | KV cache | recurrent | activations | runtime | total |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 47.32 | 0.25 | 0.14 | 0.30 | 0.90 | **48.91** |
| int8 | 24.63 | 0.25 | 0.14 | 0.30 | 0.90 | **26.23** |
| 4-bit NF4 | 14.71 | 0.25 | 0.14 | 0.30 | 0.90 | **16.31** |

*Weights* are arithmetic and the most trustworthy row. *KV cache* grows linearly with
context — 0.25 GiB at 4k becomes ~4 GiB at 64k. *Runtime overhead* is the term an
analytical model predicts least well, and the one most likely to be wrong; a real load may
exceed it, particularly under `accelerate` sharding or bitsandbytes.

**Practical GPU class**, treating the estimates as lower bounds and leaving headroom:

| scheme | suggested minimum | rationale |
|---|---|---|
| 4-bit NF4 | one 24 GB card | ~16 GiB estimated leaves room for the overhead being wrong |
| int8 | one 40–48 GB card | ~26 GiB estimated |
| bf16 | 2×40 GB or 1×80 GB | ~49 GiB estimated |

### This does not constrain the 16 GB student objective

The teacher's footprint and the student's target are **separate budgets**. The teacher runs
once, on larger rented hardware, to produce the distillation signal; only the student has to
fit 16 GB. That the teacher needs ~24 GB at 4-bit says nothing about whether a student fits
a 16 GB card, and no student candidate should be shrunk because of it.

The one place they meet is *online* KD, where teacher and student are resident together —
which is why that is scoped to rented hardware, and why the offline path exists as the
alternative.

What runs where:

| | local / T4 | 24 GB rented | 40–80 GB rented |
|---|---|---|---|
| unit tests (`test_real_teacher.py`) | ✅ | ✅ | ✅ |
| smoke test on a small checkpoint | ✅ | ✅ | ✅ |
| **smoke test on the real teacher** | ❌ | ✅ 4-bit | ✅ int8 / bf16 |
| online KD (teacher + student resident) | ❌ | ✗ tight | ✅ |
| teacher dataset generation | ❌ | ✅ slow | ✅ |

`--device auto` with `accelerate` shards across visible GPUs; `--max-memory` and
`--offload-folder` bound each device and spill the remainder.

## 6. Running the smoke test

Ten checks, no dataset, nothing trained:

```bash
# the real teacher on a 24 GB card
python scripts/teacher_smoke_test.py --quantization 4bit --revision <sha>

# unquantised across two GPUs
python scripts/teacher_smoke_test.py --dtype bfloat16 --device auto --revision <sha>

# bounded, with CPU spill
python scripts/teacher_smoke_test.py --quantization 4bit \
    --max-memory '{"0":"22GiB","cpu":"64GiB"}' --offload-folder /tmp/offload

# the mechanism only, on any machine
python scripts/teacher_smoke_test.py --local-path ./small-qwen3_5 \
    --model tiny/qwen3_5 --dtype float32 --device cpu --lenient-architecture
```

It loads, tokenizes, renders every reasoning mode, generates, takes logits, builds a
`TeacherSignal`, verifies dimensions and alignment, checks provenance, and unloads. Exit
`0` means the teacher is operational.

The alignment check is the substantive one: it asserts the teacher's own logits have zero
divergence from its own signal, and that logits for a prefix match the same positions of
the full sequence. Together those establish that position *t* is the prediction for the
token a student predicts at *t*.

---

## 7. Provenance, and the one gap

Every operation records: model id, revision, `config.json` SHA-256, `chat_template` SHA-256,
`tokenizer_config.json` SHA-256, reasoning mode and its template kwargs, dtype,
quantisation, device map, the load report, tokenizer facts, and package versions.

**The revision is not pinned.** No commit SHA was supplied with the vendored metadata, and
`huggingface.co` is unreachable from this development sandbox, so it could not be resolved
here. This is recorded as a reproducibility gap, not treated as acceptable: `TeacherIdentity`
reports `is_pinned: false` and attaches a note, and both the smoke test and the generation
script print a warning.

**Pass `--revision <sha>` on the first real run and record it.** Until then a dataset is
reproducible only while the repo id keeps serving the same weights.

---

## 8. What this does not do

- **No offline teacher-logit format.** The KD loss and the capture format are ready
  (top-k logits plus the full-vocabulary logsumexp), but the on-disk layout stays unchosen
  until a real run reports its tail mass at a candidate *k*. The smoke test prints exactly
  that number.
- **No dataset generation.** `scripts/generate_teacher_data.py` now loads the real teacher,
  but running it at scale is a separate, paid decision.
- **No student trained.** See `DISTILLATION_ROADMAP.md`.
