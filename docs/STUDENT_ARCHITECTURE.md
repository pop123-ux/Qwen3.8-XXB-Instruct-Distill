# Student architecture

Canonical definition: `src/qwen_distill/architecture/moe_student.py`. This document
explains it; the code is the specification. `FROZEN_STUDENT` is a frozen dataclass, and
`tests/test_moe_student_spec.py` fails if any field changes — the tests are a tripwire, not
a design review.

## The specification

```
name                    qwen38_19b_h5120_l48_moe
model_type              qwen3_5_moe_text            (Qwen3_5MoeForCausalLM)
hidden_size             5120
num_hidden_layers       48
vocab_size              248320
max_position_embeddings 262144
tie_word_embeddings     false
rms_norm_eps            1e-6

attention               24 query heads, 2 KV heads, head_dim 256
                        bias false, dropout 0, output gate enabled, swish
                        partial_rotary_factor 0.25  ->  rotary dim 64
                        rope theta 10,000,000

DeltaNet                16 key heads, 48 value heads, key/value head dim 128
                        conv kernel 4, state dtype float32

MoE                     24 routed experts, 2 active
                        1 shared expert, always active
                        expert intermediate 768, shared intermediate 768
                        top-k routing, router jitter false
                        router aux loss true, coefficient 0.001

layout                  48 blocks = 36 DeltaNet + 12 FullAttention
                        [DeltaNet, DeltaNet, DeltaNet, FullAttention] x 12

MTP                     1 layer declared — see "MTP" below
```

Three details are easy to get wrong and are worth stating explicitly, because each one is
silently accepted by the config object and then does the wrong thing:

- **`partial_rotary_factor` lives inside `rope_parameters`**, not at the top level. Set as
  a top-level field it is dropped and the model runs full rotary.
- **`layer_types` is an explicit list.** There is no `full_attention_interval` field; the
  48-entry list is generated from the group pattern and passed verbatim.
- **`attn_output_gate` is not a config field.** The gate exists anyway, fused into
  `q_proj`, which is `2 * heads * head_dim` wide rather than `heads * head_dim`. The
  projection width is the only evidence the gate is there, and a test asserts it.

## What it weighs

Measured by constructing the model on meta tensors and summing real parameters —
`python scripts/student_report.py --section architecture`.

| component | parameters | share |
|---|---:|---:|
| `embedding` | 1,271,398,400 | 5.76% |
| `lm_head` | 1,271,398,400 | 5.76% |
| `attention` | 1,195,382,784 | 5.42% |
| `deltanet` | 4,171,538,304 | 18.90% |
| `routed_experts` | 13,589,544,960 | 61.57% |
| `shared_expert` | 566,476,800 | 2.57% |
| `router` | 5,898,240 | 0.03% |
| `norms` | 496,640 | 0.00% |
| `mtp` | 0 | not built |
| **total** | **22,072,134,528** | |

```
difference_from_19B    +3,072,134,528  (+16.2%)
non_embedding          19,529,337,728
active_per_token        9,615,051,648   (43.6% of stored)
```

The name says 19B and the model weighs 22.07B. The architecture is frozen, so it is not
adjusted to match the label; 19.53B non-embedding is the most likely origin of the name.

Two consequences follow from the table and drive everything downstream:

**Experts are the model.** 61.57% of all parameters are routed experts that each token
mostly does not use. Any memory or quantisation decision that is not primarily about the
experts is working on the other 38%.

**Sparsity is real.** 9.6B active against 22.07B stored. That ratio is the entire argument
for choosing MoE here rather than a dense model of similar quality.

## Why this shape

**36 DeltaNet, 12 attention.** Only the 12 full-attention layers keep a KV cache, at
24 KiB per token in fp16 — against roughly 96 KiB/token if all 48 layers cached. The 36
DeltaNet layers keep a fixed-size recurrent state of about 113 MiB *in total*, identical at
2K and at 262K. This is what makes a 262,144-token window arguable at all on a consumer
card, and it is why the context work in
[CONTEXT_SPECIALIZATION.md](CONTEXT_SPECIALIZATION.md) is about what the recurrent state
learns to carry.

**Depth-only compression, 64 -> 48.** Width, vocabulary, head dimension and the hybrid
pattern all match the teacher. Holding them fixed means every transferred tensor is a copy
rather than a projection, and it means the depth reduction can be studied on its own.

**2 KV heads instead of 4.** Halves the KV cache. It is also a real information loss, and
`measure_kv_merge` reports how much — see
[INITIALIZATION_METHOD.md](INITIALIZATION_METHOD.md).

## MTP

Declared in the specification, **not built by the runtime**. `qwen3_5_moe_text` in
transformers 5.15.1 has no `mtp_num_hidden_layers` field and constructs no MTP head; the
teacher's `mtp.*` tensors are discarded on load by
`Qwen3_5ForCausalLM._keys_to_ignore_on_load_unexpected`. So:

- the student has no MTP tensors and contributes zero MTP parameters;
- no MTP loss can be trained, and the loss term raises rather than approximating one;
- the architecture field is kept as the extension point;
- any MTP result reported today would be fabricated.

`MTP_STATUS` states this in code and `test_mtp_is_declared_but_not_built` pins it, so a
future transformers release that *does* build the head fails loudly here instead of
silently changing the parameter count.

## Structural validation

A 22B model cannot be forward/backward tested in this environment, so `tiny_fixture()`
builds a model of the *same architecture family* — same hybrid pattern, same gated
attention, same GQA and DeltaNet head ratios, same top-k routing with a shared expert,
untied embeddings — with every size scaled down. `tests/test_moe_student_structure.py`
runs it and checks, on the real modules:

- forward, backward with gradients reaching **every** parameter including the experts;
- the hybrid pattern places DeltaNet and attention blocks where `layer_types` says;
- `q_proj` is double width (the output gate) and `k_proj`/`v_proj` are GQA-narrow;
- the router selects exactly top-k distinct experts with weights summing to one;
- zeroing the shared expert changes **every** token; zeroing one routed expert changes
  **exactly** the tokens routed to it;
- both mixer types actually move the residual stream — a silently no-op block would pass a
  shape test;
- step-by-step cached decoding reproduces a single-pass forward to 2e-4, which is the
  strongest available evidence that the recurrent DeltaNet state and the 12-layer KV cache
  are both correctly threaded;
- no MTP module exists.

## The historical baseline

The earlier candidate — `h5120 L40`, **17,763,549,760** parameters, dense — is retained, not
deleted: `research/baselines.py`, derived from the teacher preset rather than hard-coded so
it stays transfer-compatible.

It is a *good* control, not merely an old idea. Its transfer plan from the teacher is 533
pure tensor copies, 100% coverage, zero warnings, no width reduction anywhere — which
removes the `slice`-baseline assumption (that a teacher's parameters are ordered by
importance, which nothing guarantees) from the comparison entirely. And it fits 16 GB at
13.18 GiB at 32K, where the MoE target does not, which makes it the fallback if the
expert-budget decision in [PARETO_EVALUATION.md](PARETO_EVALUATION.md) goes the other way.

**The project's first scientific comparison is 17.76B dense L40 against 22.07B sparse-MoE
L48.** Held constant: teacher and revision, hidden size 5120, vocabulary 248,320, head_dim
256, the 3:1 hybrid pattern, the objective and the token budget. Varying: depth 40 vs 48,
and dense FFN vs 24-expert top-2 MoE. The question is whether routing 22.07B stored
parameters at 9.6B active per token beats 17.76B dense at the same width from the same
teacher — and 9.6B active is *less* than the dense baseline's total, which is what makes
the comparison interesting rather than a foregone conclusion.

Neither model has been distilled. `comparison()` states the design so it can be run.
