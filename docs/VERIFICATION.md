# Verification Status

Every claim this project relies on, classified by **what evidence backs it**:

| Tier | Meaning |
|---|---|
| **VERIFIED — CHECKPOINT METADATA** | Read from the teacher's own `config.json` / tokenizer / chat template. The authoritative source. |
| **VERIFIED — REFERENCE IMPLEMENTATION** | Mechanically checked against the code that runs the model (`transformers`, vLLM). Establishes how the *architecture family* behaves. |
| **CORROBORATED** | Multiple independent secondary sources agree and the claim is self-consistent with things we did verify. **A secondary article never makes a claim VERIFIED.** |
| **NOT YET VERIFIED / UNKNOWN** | Not established. |

The middle two tiers are genuinely different, and the distinction carries weight here.
Our formulas provably reproduce what `transformers` builds *for a config with the values
we assumed*. That does **not** establish that the teacher's config carries those values.
Both statements would hold even if the real config differed — we would simply be
modelling the wrong model correctly.

Last updated: 2026-08-23 (Phase 1, offline ingestion).

## Current status: no checkpoint metadata has been supplied

`vendor/qwen38-metadata/` is **empty**. Egress to `huggingface.co` remains blocked in
the authoring environment, and per instruction no attempt was made to route around it.
The repository now consumes a locally supplied metadata directory instead — see
[`vendor/README.md`](../vendor/README.md).

**No verification status changed in this pass.** The checkpoint-metadata tier below is
empty, and every question needing the config, tokenizer, template or licence is still
UNKNOWN. The tooling to close them is built and tested; it is waiting on the files.

### VERIFIED — CHECKPOINT METADATA

*(nothing yet — no metadata supplied)*

## Verification attempt log

The three metadata-only verification commands were **executed**, not merely specified.
All three failed at the network layer:

| Command | Result |
|---|---|
| `inspect_teacher.py --repo-id Qwen/Qwen3.8-27B --config-only` | `ProxyError: 403 Forbidden` |
| `verify_teacher_loader.py --model Qwen/Qwen3.8-27B --config-only` | `VERDICT: FAILED` — `OSError: Can't load the configuration of 'Qwen/Qwen3.8-27B'` |
| `benchmark_reasoning.py --model Qwen/Qwen3.8-27B --template-only` | same `OSError` |

No teacher artifact was produced, so none is committed. The attempt itself exposed two
tooling defects, now fixed: `inspect_teacher.py` exited **0** despite failing (a
verification script reporting success for work it never did), and two of the three
scripts leaked raw tracebacks instead of an actionable diagnosis.

## The blocking constraint

`huggingface.co` is **blocked by this environment's egress policy** — confirmed again
in Phase 1 for direct HTTPS and for the fetch tooling (`gateway answered 403 to
CONNECT`). Also blocked: `recipes.vllm.ai`, `northflank.com`, `modelscope.cn`,
`hf-mirror.com`, `qwen.ai`.

A filesystem-wide search found **no local copy** of the checkpoint — no HF cache, no
mounted model directory, no stray `config.json`. The machine also has **no GPU**
(CPU-only, 4 cores).

Consequently the following remain unread: the upstream **model card, `config.json`,
`tokenizer.json`, chat template, and `LICENSE`**. No teacher weights were obtained, so
no teacher measurement of any kind was possible.

What Phase 1 *could* do instead was verify the **reference implementation** — the code
that actually runs the model — which is distributed on PyPI and reachable. That turned
out to settle considerably more than expected, including several questions that
secondary sources could only guess at.

## Phase 1 question ledger

| # | Question | Status | Evidence |
|---|---|---|---|
| 1 | Exact `config.json` of Qwen3.8-27B | **UNKNOWN** | `huggingface.co` blocked; no local copy |
| 2 | Exact `model_type` | **UNKNOWN** | requires the config; see the narrowing argument below |
| 3 | Class Transformers loads | **UNKNOWN** for 3.8; **VERIFIED** for the family | `Qwen3_5ForCausalLM` / `Qwen3_5ForConditionalGeneration` resolve natively for `model_type: qwen3_5*` |
| 4 | Stock transformers vs `trust_remote_code` | **UNKNOWN** | `scripts/verify_teacher_loader.py` answers this in one command once reachable |
| 5 | Exact layer types | **VERIFIED (family)** | expansion rule read from source *and* checked against a built model: 48 linear / 16 full at interval 4, full attention always last in each group |
| 6 | Exact attention dimensions | **VERIFIED (family)** | `q_proj` emits `n_heads·head_dim·2` (query + sigmoid gate); `head_dim` 256; `partial_rotary_factor` 0.25 → RoPE 64 |
| 7 | Exact DeltaNet dimensions | **VERIFIED (family)** | `conv_dim = 2·key_dim + value_dim`; projections `in_proj_qkv/z/b/a`, `out_proj`; depthwise conv kernel 4 |
| 8 | Exact FFN structure | **VERIFIED (family)** | SwiGLU: `gate_proj`, `up_proj`, `down_proj`, no bias |
| 9 | Exact vocabulary size | **CORROBORATED** | 248320 is the `Qwen3_5TextConfig` default and matches secondary reports; the 3.8 checkpoint's own value is unread |
| 10 | Embeddings tied? | **UNKNOWN** | `tie_word_embeddings` defaults `False` in the family; the checkpoint's value is unread |
| 11 | What the checkpoint contains | **UNKNOWN** | requires the checkpoint |
| 12 | How MTP is represented | **VERIFIED** | see the MTP section below |
| 13 | Is MTP used by the inference path | **VERIFIED** | stock transformers **discards** it; vLLM loads it **as a speculative-decoding draft model** |
| 14 | Text-only or multimodal | **CORROBORATED** (multimodal), **VERIFIED** that the text tower loads alone | `Qwen3_5ForCausalLM` ignores `^model.visual.*`, so a text-only student is a supported path |
| 15 | Exact tokenizer | **UNKNOWN** | requires the checkpoint |
| 16 | Exact chat template | **UNKNOWN** | requires the checkpoint |
| 17 | Exact reasoning controls | **CORROBORATED** | `low` / `medium` / `xhigh` plus a thinking on/off switch |
| 18 | Actual default reasoning behaviour | **CORROBORATED** | reported default `xhigh`, `preserve_thinking` on |
| 19 | Is `medium` really a no-op | **UNKNOWN** — but now **mechanically decidable** | `scripts/benchmark_reasoning.py --template-only` renders every setting and diffs; byte-identical prompts prove a no-op by construction. Needs only the tokenizer. Detector validated against a known-positive fixture. |
| 20 | Exact license | **UNKNOWN** | **must be read before any release** |
| 21 | Naming / attribution obligations | **UNKNOWN** | same |
| 22 | Frameworks supporting the checkpoint | **VERIFIED (family)** | transformers 5.15.1 and vLLM 0.27.1 both implement `qwen3_5`; vLLM registers text, multimodal, MoE and MTP variants |
| 23 | Can the model load under the intended stack | **UNKNOWN** for 3.8 | `verify_teacher_loader.py --probe` performs a real generation, since an import proves nothing |
| 24 | Does the analytical parameter model match | **VERIFIED — exactly** | see below |
| 25 | Does the analytical VRAM model match | **VERIFIED for the cache terms**; **UNKNOWN for end-to-end peak** | cache/state terms match a real forward pass byte-for-byte; no GPU here, so runtime overhead is uncalibrated |

## VERIFIED — REFERENCE IMPLEMENTATION: the analytical model matches exactly

`python scripts/validate_analytical_model.py --teacher` — reproducible, no GPU, no
network, no checkpoint. Artifact: `evaluations/baselines/analytical_model_validation.json`.

The full 27B architecture was instantiated with `transformers` on the `meta` device
(shapes and dtypes, zero storage) and compared against our formulas **component by
component** — a total can match by cancelling errors, so components are checked
individually:

| Component | transformers | analytical | Δ |
|---|---:|---:|---:|
| embedding | 1,271,398,400 | 1,271,398,400 | 0 |
| lm_head | 1,271,398,400 | 1,271,398,400 | 0 |
| final_norm | 5,120 | 5,120 | 0 |
| layer_norms | 655,360 | 655,360 | 0 |
| mlp | 17,112,760,320 | 17,112,760,320 | 0 |
| full_attention | 1,677,729,792 | 1,677,729,792 | 0 |
| linear_attention | 5,562,051,072 | 5,562,051,072 | 0 |
| **total** | **26,895,998,464** | **26,895,998,464** | **0** |

851 parameter tensors, 0 unmatched. `model_class = Qwen3_5ForCausalLM`, 16
full-attention layers.

This is the strongest evidence available without the checkpoint. It proves our
formulas are correct **for a config with these values**. It does not prove the teacher
has these values — question 1 remains open.

### Cache and state terms, measured on a real forward pass

Small model, CPU, real generation, cache objects inspected directly:

| Quantity | measured | analytical | Δ |
|---|---:|---:|---:|
| KV cache bytes | 75,776 | 75,776 | 0 |
| recurrent state bytes | 147,456 | 147,456 | 0 |
| conv state bytes | 30,720 | 30,720 | 0 |

Also confirmed by measurement, not assumption:

- **KV cache exists on exactly the full-attention layers** (`[3, 7]` of 8) — the
  property the entire long-context argument rests on.
- **Recurrent state shape is `(batch, num_v_heads, head_k_dim, head_v_dim)`** and is
  **byte-identical at sequence length 8 and 64**, while the KV cache grows.
- Conv state shape is `(batch, conv_dim, kernel_size)`.
- The reference path holds these states in **fp32**, confirming that our conservative
  fp32 default is the right one.

### What is still uncalibrated

The **runtime overhead** and **activation** terms in `qwen_distill.architecture.memory`
are engineering estimates and have **not** been checked against a real GPU, because
this machine has none. `scripts/benchmark_memory.py` computes the measured/estimated
calibration factor and refuses to emit zeros on a CPU-only host rather than producing
numbers that could be mistaken for measurements. Until it runs on real 16 GB hardware,
**peak-VRAM figures in this repository are estimates, not measurements.**

## VERIFIED — REFERENCE IMPLEMENTATION: what MTP actually is

From `vllm/model_executor/models/qwen3_5_mtp.py` (vLLM 0.27.1) and
`transformers/models/qwen3_5/modeling_qwen3_5.py` (5.15.1):

- `Qwen3_5MTP` is registered in vLLM's **`_SPECULATIVE_DECODING_MODELS`**. MTP is a
  **draft model for speculative decoding**, not part of the base LM forward pass. It
  does not change the base model's logits.
- Structure: `fc` (`Linear(hidden·2 → hidden, bias=False)`), `mtp_num_hidden_layers`
  decoder layers (default **1**), and three RMSNorms (`norm`, `pre_fc_norm_hidden`,
  `pre_fc_norm_embedding`).
- **MTP layers are always constructed with `layer_type="full_attention"`** — never
  DeltaNet.
- Forward: `concat(norm(embed(token)), norm(hidden)) → fc → decoder layer → norm → lm_head`.
- **`embed_tokens` and `lm_head` are shared with the base model, not duplicated** —
  vLLM's loader routes only `mtp.*` tensors into the draft model and reuses the main
  checkpoint's embedding and head.
- **Stock `transformers` discards MTP entirely**:
  `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`.

Cost, computed by `qwen_distill.architecture.params.mtp_params`: **424,699,392
parameters ≈ 1.58%** of a 27B model. Cheap enough that carrying an MTP head in the
student is a real option — but only worth it if the backends we target can use it,
which is an ablation, not an assumption.

## The `model_type` narrowing argument

We cannot read the config, so question 2 is UNKNOWN. But the evidence narrows it, and
the release timing matters:

- **`transformers` 5.15.1 was uploaded to PyPI on 2026-08-19** — five days *after* the
  reported Qwen3.8-27B release of 2026-08-14. It contains `qwen3_5`, `qwen3_5_moe`,
  `qwen3_next`, `qwen3_vl` — and **no `qwen3_8` module**.
- **vLLM 0.27.1 was uploaded on 2026-08-11**, *before* the release. Its lack of
  `qwen3_8` code therefore proves nothing and must not be cited as evidence.

A `transformers` release five days after the launch that adds no new architecture is
most consistent with **Qwen3.8-27B reusing `model_type: qwen3_5`** — a new checkpoint
of an existing architecture family, as `Qwen3.5-27B` / `Qwen3.6-27B` / `Qwen3.8-27B`
would naturally be.

**This is inference, not verification.** It is recorded as the leading hypothesis
because it determines whether the community can run our student with stock
`transformers`. One command settles it:

```bash
python scripts/verify_teacher_loader.py --model Qwen/Qwen3.8-27B --config-only
```

## Still blocked, and what unblocks it

| Blocked question | Unblocked by |
|---|---|
| 1, 2, 4, 9, 10, 11, 15, 16, 23 | `config.json` + tokenizer files (a few MB, not the weights) |
| 17, 18, 19 | the chat template alone — `benchmark_reasoning.py --template-only` |
| 20, 21 | reading the upstream `LICENSE` and model card |
| 25 (end-to-end peak VRAM) | any CUDA GPU; ideally the 16 GB target hardware |
| Teacher baseline, all reasoning measurements | the weights, plus a GPU large enough to run 27B |

**The cheapest unblock by far is a few megabytes of metadata, supplied locally.**
No weights, and no network access from this repository. Obtain the files however suits
your network (see [`vendor/README.md`](../vendor/README.md)), place them in
`vendor/qwen38-metadata/`, and run:

```bash
python scripts/validate_teacher_metadata.py --path vendor/qwen38-metadata
python scripts/inspect_teacher.py --path vendor/qwen38-metadata --config-only \
    --json evaluations/baselines/teacher_config_report.json \
    --save-spec configs/teacher/qwen3_8_27b.verified.json
python scripts/verify_teacher_loader.py --model vendor/qwen38-metadata --config-only
python scripts/inspect_chat_template.py --path vendor/qwen38-metadata
```

All four run offline: they set `HF_HUB_OFFLINE`, and tests sever `socket` and assert no
connection is attempted.

Together these close questions 1, 2, 4, 5-10, 15, 16, 17 and 19 (at the template level).
If `cross_check` reports MISMATCH, fix `src/qwen_distill/architecture/params.py` before
trusting any estimate in this repository.

### What metadata still cannot settle

These stay UNKNOWN however complete the metadata is. The tooling reports them as UNKNOWN
rather than MISSING, so they are never mistaken for a gap in the supplied files:

| Fact | Why |
|---|---|
| state-dict parameter count | requires the weights |
| whether the checkpoint loads and generates | a config that parses is **not** proof the model runs |
| whether `medium` changes a *trained model's* behaviour | the template diff can prove it cannot act via the prompt; any other route needs a runtime experiment |
| peak VRAM | requires a GPU |
| upstream licence | requires the upstream `LICENSE` file to be supplied |
