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

## Current status: checkpoint metadata supplied and verified

`vendor/qwen38-metadata/` now contains the upstream `config.json`, `tokenizer_config.json`,
`generation_config.json`, `chat_template.jinja` and `LICENSE`. Everything below in the
checkpoint-metadata tier was read from those files.

Pinned by SHA-256 in `configs/teacher/qwen3_8_27b.verified.json`:

| File | SHA-256 |
|---|---|
| `config.json` | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| `tokenizer_config.json` | `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27` |
| `generation_config.json` | `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e` |
| `chat_template.jinja` | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| `LICENSE` | `50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32` |

**The weights are still absent**, so everything requiring runtime remains UNKNOWN.

### VERIFIED — CHECKPOINT METADATA

Read directly from the supplied files. Regenerate with
`scripts/validate_teacher_metadata.py --path vendor/qwen38-metadata --save-verified ...`.

| Claim | Value | Source |
|---|---|---|
| `model_type` | `qwen3_5` (text: `qwen3_5_text`) | `config.json` |
| declared architecture | `Qwen3_5ForConditionalGeneration` | `config.json` |
| resolved causal-LM class | `Qwen3_5ForCausalLM`, native `transformers` | resolution against 5.15.1 |
| `trust_remote_code` required | **no** — no `auto_map`, no bundled `.py` | `config.json` + directory |
| hidden_size / layers / FFN | 5120 / 64 / 17408 | `config.json` |
| vocab_size | 248320 | `config.json` |
| attention | 24 query / 4 KV heads, head_dim 256 | `config.json` |
| `attn_output_gate` | `true` | `config.json` |
| DeltaNet | 48 value / 16 key heads, dims 128/128, conv kernel 4 | `config.json` |
| layer layout | **explicit** 64-entry list: 48 linear, 16 full | `config.json` |
| `full_attention_interval` | 4 (consistent with the explicit list) | `config.json` |
| `partial_rotary_factor` | 0.25 → RoPE dim 64 | `config.json` |
| RoPE | `rope_theta` 1e7, mRoPE sections `[11, 11, 10]` | `config.json` |
| `rope_scaling` | **absent** — 262144 is native, not YaRN-extended | `config.json` |
| `max_position_embeddings` | 262144 | `config.json` |
| `tie_word_embeddings` | `false` | `config.json` |
| MTP | `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false` | `config.json` |
| vision tower | present: depth 27, hidden 1152, out_hidden 5120 | `config.json` |
| tokenizer class | `Qwen2Tokenizer` | `tokenizer_config.json` |
| eos / pad / bos | `<|im_end|>` / `<|endoftext|>` / **none** (`add_bos_token: false`) | `tokenizer_config.json` |
| `model_max_length` | 262144 | `tokenizer_config.json` |
| sampling defaults | `do_sample: true`, temp 1.0, top_p 0.95, top_k 20 | `generation_config.json` |
| **licence** | **Apache-2.0** | `LICENSE` |
| supported `reasoning_effort` | exactly `xhigh`, `medium`, `low` — `high` raises | `chat_template.jinja` |
| default reasoning effort | `xhigh` (`reasoning_effort|default('xhigh')`) | `chat_template.jinja` |
| tool-call formatting | supported; changes the prompt | `chat_template.jinja` |

**Parameter count from the actual config: 26,895,998,464.** Computed by feeding the
supplied `config.json` through `HybridArchSpec.from_hf_config` → `count_parameters`,
with no preset involved. A test varies each architecture field and asserts the estimate
moves, so no dimension is silently hard-coded.

### Config keys the installed `transformers` does not read

The checkpoint was written by `transformers 5.8.0.dev0` and carries keys absent from
5.15.1. Not errors — but a key that looks like it controls behaviour may not, and a
future release could start honouring one:

| Key | Value | What 5.15.1 does |
|---|---|---|
| `attn_output_gate` | `true` | builds the doubled `q_proj` unconditionally; would ignore `false` |
| `output_gate_type` | `"swish"` | applies **sigmoid** gating unconditionally |
| `mamba_ssm_dtype` | `"float32"` | not read; the reference path accumulates the recurrent state in fp32 anyway, which independently confirms our conservative default |
| `mtp_use_dedicated_embeddings` | `false` | discards `mtp.*` entirely; vLLM reuses the base embedding and head |
| `language_model_only` | `false` | not read |

`output_gate_type: "swish"` versus 5.15.1's sigmoid is a genuine behavioural
discrepancy. It does not change the parameter count, but it means the installed
implementation may not reproduce the checkpoint's intended activation.

## What remains blocked

The metadata is supplied and verified. Two things still are not:

**The weights.** No checkpoint tensors were obtained, so the state-dict parameter count,
successful loading, real generation, reasoning-token behaviour and benchmark capability
are all unmeasured.

**A GPU.** This machine is CPU-only (4 cores), so no VRAM figure can be measured.

Egress to `huggingface.co` remains blocked in this environment and was not circumvented;
the metadata reached the repository by being supplied locally, which is the intended
route (see [`vendor/README.md`](../vendor/README.md)).

## Phase 1 question ledger

M = VERIFIED from checkpoint metadata; R = VERIFIED from the reference implementation.

| # | Question | Status | Evidence |
|---|---|---|---|
| 1 | Exact `config.json` | **VERIFIED (M)** | supplied and hashed |
| 2 | Exact `model_type` | **VERIFIED (M)** | `qwen3_5` / text `qwen3_5_text` |
| 3 | Class Transformers loads | **VERIFIED (M+R)** | `Qwen3_5ForCausalLM`, native module |
| 4 | Stock transformers vs `trust_remote_code` | **VERIFIED (M)** | stock; no `auto_map`, no bundled `.py` |
| 5 | Exact layer types | **VERIFIED (M)** | explicit 64-entry list: 48 linear / 16 full |
| 6 | Exact attention dimensions | **VERIFIED (M)** | 24 q / 4 kv, head_dim 256, `attn_output_gate: true` |
| 7 | Exact DeltaNet dimensions | **VERIFIED (M)** | 48 v / 16 k, 128/128, conv kernel 4 |
| 8 | Exact FFN structure | **VERIFIED (M+R)** | SwiGLU, intermediate 17408, no bias |
| 9 | Exact vocabulary size | **VERIFIED (M)** | 248320 |
| 10 | Embeddings tied? | **VERIFIED (M)** | `tie_word_embeddings: false` |
| 11 | What the checkpoint contains | **PARTIAL** | metadata declares text + vision + MTP; tensor inventory needs the weights |
| 12 | How MTP is represented | **VERIFIED (M+R)** | `mtp_num_hidden_layers: 1`; structure from vLLM |
| 13 | Is MTP used by the inference path | **VERIFIED (R)** | transformers discards it; vLLM uses it as a speculative draft model |
| 14 | Text-only or multimodal | **VERIFIED (M)** | multimodal; the text tower loads alone |
| 15 | Exact tokenizer | **VERIFIED (M)** | `Qwen2Tokenizer`, eos `<\|im_end\|>`, no BOS |
| 16 | Exact chat template | **VERIFIED (M)** | `chat_template.jinja`, hashed |
| 17 | Exact reasoning controls | **VERIFIED (M)** | exactly `xhigh`, `medium`, `low`; `high` raises |
| 18 | Actual default reasoning behaviour | **VERIFIED (M)** | `default('xhigh')`; renders identically to explicit `xhigh` |
| 19 | Is `medium` a no-op | **REFUTED at template level (M)** | renders a *distinct*, shorter prompt — see below |
| 20 | Exact licence | **VERIFIED (M)** | Apache-2.0 |
| 21 | Naming / attribution obligations | **PARTIAL** | Apache-2.0 terms apply; no separate naming policy was supplied |
| 22 | Frameworks supporting the checkpoint | **VERIFIED (R)** | transformers 5.15.1 and vLLM 0.27.1 both implement `qwen3_5` |
| 23 | Can the model load under the intended stack | **NOT YET VERIFIED** | Stage 1 resolves; Stage 2 needs the weights |
| 24 | Does the analytical parameter model match | **VERIFIED (M+R)** | 26,895,998,464 from the real config; exact match vs a `transformers` build |
| 25 | Does the analytical VRAM model match | **PARTIAL** | cache terms exact vs a real forward pass; end-to-end peak needs a GPU |

## Reasoning controls: what the template actually does

This corrects a hypothesis carried from secondary sources in earlier phases.

The template's own logic:

```jinja
{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
    {{- raise_exception('Unexpected reasoning effort ... Supported types are xhigh (default), medium, and low.') }}
{%- if resolved_reasoning_effort == 'xhigh' %}   -> long "think carefully" instruction
{%- elif resolved_reasoning_effort == 'low' %}   -> "keep your thinking brief" instruction
```

`medium` has no branch: it sets **no** reasoning instruction. Rendered prompts for
"What is 15 * 7?":

| Setting | SHA-256 (first 12) | chars |
|---|---|---|
| no control | `7f1de0c2b7fd` | 310 |
| `enable_thinking: true` | `7f1de0c2b7fd` | 310 |
| `reasoning_effort: xhigh` | `7f1de0c2b7fd` | 310 |
| `reasoning_effort: low` | `51f41ace41f5` | 239 |
| `reasoning_effort: medium` | `20ba983e045c` | **73** |
| `enable_thinking: false` | `8475fd3ecb78` | 84 |

**`medium` is not a no-op.** It produces a distinct — in fact the shortest — prompt,
because it omits the xhigh instruction the default injects. The earlier secondary-source
claim conflated "adds no instruction of its own" with "changes nothing"; those differ
precisely because the default is `xhigh`, not `medium`.

What *is* confirmed is that **no control == `xhigh`**: a caller who sets nothing receives
the high-effort instruction.

### Three levels of claim, kept separate

1. **Template behaviour** — VERIFIED above. Which prompts differ, byte for byte.
2. **Generation behaviour** — NOT MEASURED. Whether a distinct prompt produces
   materially different reasoning length or content requires running the model.
3. **Benchmark behaviour** — NOT MEASURED. Whether any difference in generation
   changes task accuracy requires a full evaluation.

A difference at level 1 does not imply one at level 2, and neither implies level 3.
`scripts/benchmark_reasoning.py` records the rendered-prompt hash alongside measured
token counts precisely so levels 1 and 2 can be told apart in the results.

### On the default, stated carefully

The template requests high reasoning effort by default. That is a verified fact about
the template. It is **not** evidence that the resulting token expenditure is wasted —
that would require showing the extra reasoning does not improve task performance, which
is a measurement this project has not yet made. The project's motivating question is
therefore:

> The upstream template requests high reasoning effort by default; this project will
> measure whether the resulting compute and token expenditure is justified by improved
> task performance.

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

## The `model_type` hypothesis: confirmed

Earlier phases inferred, from `transformers` 5.15.1 shipping five days after the
Qwen3.8 release without adding a `qwen3_8` module, that the checkpoint most likely
reused `model_type: qwen3_5`. That inference was recorded as a hypothesis, not a fact.

**The supplied `config.json` confirms it:** `model_type: qwen3_5`, text sub-config
`qwen3_5_text`, architecture `Qwen3_5ForConditionalGeneration`. Stock `transformers`
resolves it natively with no `trust_remote_code`.

The practical consequence: a student built in this family is loadable by anyone with
stock `transformers`, which was the open question with the largest effect on the
project's plan.

## Still blocked, and what unblocks it

| Blocked | Unblocked by |
|---|---|
| state-dict parameter count | the weights (`inspect_teacher.py --path <checkpoint>`) |
| checkpoint loads and generates | the weights (`verify_teacher_loader.py --probe`) |
| reasoning behaviour at each effort level | the weights + a GPU (`benchmark_reasoning.py`) |
| benchmark capability | the weights + a GPU large enough for 27B |
| peak VRAM | any CUDA GPU (`benchmark_memory.py`) |
| long-context retrieval curve | the weights + memory for 128k context |
| upstream revision pin | the commit SHA the metadata was taken from |

The teacher does not fit 16 GB — its Q4_K_M weights alone are 15.85 GiB — so the
baseline needs a larger GPU, rented or borrowed. That is a one-time cost: the baseline
is produced once, committed, and reused for every student comparison.

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
