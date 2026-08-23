# Qwen3.8-XXB-Instruct-Distill

Research infrastructure for compressing **Qwen3.8-27B** into an instruct model that
runs comfortably on a **single 16 GB consumer GPU**, keeps a genuinely large context
window, and spends far fewer tokens thinking about easy questions.

> **Project status: Phase 1 — verification and evaluation infrastructure.**
> **No final student model has been trained.** A 4.03M Level-1 hybrid prototype has
> been trained successfully on a Tesla T4 for infrastructure validation — it proves the
> pipeline executes and optimises, and nothing about language quality. No benchmark has
> been run against the teacher, and the `XXB` in the name is a placeholder: the final
> parameter count is a research result, not a design input.

## Why this project exists

Qwen3.8-27B is a remarkably capable open-weight model for its size. But it does not
fit a 16 GB card — and not by a small margin.

Running our estimator against the published configuration:

```
$ python scripts/estimate_vram.py --preset teacher --matrix

peak VRAM (GiB) by context x quantisation, budget 15.0 GiB
  quant            8k      32k      64k     128k     256k
  bf16          49.2*    50.7*    52.7*    56.7*    64.7*
  int8          26.5*    28.0*    30.0*    34.0*    42.0*
  q4_k_m        17.7*    19.2*    21.2*    25.2*    33.2*
  (* exceeds budget)
```

At 4-bit the **weights alone are 15.85 GiB** — over budget before a single token of
context. That is the gap this project tries to close.

## What the analysis already tells us

Three findings, all reproducible from this repository, all analytical rather than
measured:

**1. The bottleneck is weights, not context.** Qwen3.8's hybrid design puts Gated
DeltaNet on 48 of 64 layers, and those layers carry a *constant-size* recurrent state
regardless of sequence length. Only the 16 full-attention layers hold a growing KV
cache. At 128k context that is 8 GiB of KV against a fixed 0.15 GiB of recurrent
state. **Long context is already cheap here** — the problem is purely parameter count.

**2. The FFN is where the parameters are.**

| Component | Share of 26.90B |
|---|---:|
| **MLP (SwiGLU)** | **63.6%** |
| Gated DeltaNet | 20.7% |
| Embedding + LM head | 9.4% |
| Gated attention | 6.2% |

**3. The student should be much larger than the obvious guess.** Ranking
architectures by capacity subject to fitting 15.0 GiB at 4-bit:

| Required context | Largest feasible model |
|---|---|
| 32k | ~21B |
| 128k | ~16.5B |
| 262k | ~13.5B |

The answer to "how big?" is not 7B or 10B — it is **13–21B**, and the choice is
governed almost entirely by how much context you demand. Going from 32k to 262k costs
roughly 7.7B parameters of capacity.

These are calculations, not measurements. They narrow the search space; they do not
tell us what a trained model will score.

## Level 1 result — infrastructure validated on a real T4

The first rung of the [development ladder](docs/TRAINING_ON_LIMITED_HARDWARE.md) is done.

| | |
|---|---|
| Model | 4.03M params, 4 layers (3 DeltaNet + 1 full attention), hidden 256, FFN 704 |
| Hardware | Tesla T4, 14.56 GiB, CC 7.5, **fp16 (no bf16)** |
| Run | 200 steps, seq 256, full training, AdamW, ~56 s |
| Train loss | 8.2565 → 2.1008 |
| Validation loss | 4.4904 → 2.0910 |

**What this proves:** the hybrid DeltaNet/attention architecture instantiates, trains and
optimises on real CUDA hardware; checkpoints write and reload; the pipeline is sound.

**What it does not prove:** anything about language quality, distillation, or teacher
capability retention. The task was a synthetic induction task chosen because it is
*learnable* — a loss that falls on it means the optimizer works, not that the model
understands language. Treating 2.09 as a capability number would be a category error.

Level 2 (~100M, real text) is the experiment that starts to answer the language question.

## What can my GPU run?

```bash
python scripts/hardware_info.py --recommend
```

Detects your accelerator (NVIDIA, AMD/ROCm, or none), classifies it, and derives from
the project's memory model — not a lookup table — which models fit at which
quantisation and context, and which training experiments are plausible. Works CPU-only,
where "no GPU" is a normal answer rather than an error.

```bash
# preview a machine you don't have yet
python scripts/hardware_info.py --simulate-vram 16 --simulate-name "Tesla T4" --matrix --recommend

# will this training run fit? seconds, no weights loaded
python scripts/train_student.py --config configs/experiments/t4_prototype.yaml --dry-run
```

See [`docs/HARDWARE.md`](docs/HARDWARE.md) for tiers and rented-GPU guidance, and
[`docs/TRAINING_ON_LIMITED_HARDWARE.md`](docs/TRAINING_ON_LIMITED_HARDWARE.md) for the
development ladder from CPU to the final run.

## Quick start

```bash
pip install -e ".[dev]"
pytest                                    # 71 tests, no GPU required

python scripts/estimate_vram.py --preset teacher --matrix --max-context
python scripts/search_architectures.py --vram 16 --context 32768 --top 15
python scripts/inspect_teacher.py --repo-id Qwen/Qwen3.8-27B --config-only
```

Everything runs on CPU in seconds. No weights are downloaded unless you ask.

## Repository layout

```
src/qwen_distill/
  architecture/   spec, exact parameter accounting, VRAM model, FLOPs, search
  teacher/        checkpoint inspection and cross-checking
  evaluation/     (planned)
  data/           (planned)
scripts/          inspect_teacher, estimate_vram, search_architectures
docs/             plans and analysis (start with VERIFICATION.md)
experiments/      architecture search outputs
tests/            71 tests pinning every formula
```

## Verification status

Architectural formulas here were derived from the **reference implementation**
(`transformers==5.15.1`, `models/qwen3_5/`) — the code that actually runs the model —
and then **checked against it empirically**.

`python scripts/validate_analytical_model.py --teacher` instantiates the full 27B
architecture on PyTorch's `meta` device (shapes only, zero storage, so it runs on a
laptop) and compares component by component:

| Component | transformers | analytical | Δ |
|---|---:|---:|---:|
| mlp | 17,112,760,320 | 17,112,760,320 | 0 |
| linear_attention | 5,562,051,072 | 5,562,051,072 | 0 |
| full_attention | 1,677,729,792 | 1,677,729,792 | 0 |
| embedding + lm_head | 2,542,796,800 | 2,542,796,800 | 0 |
| **total** | **26,895,998,464** | **26,895,998,464** | **0** |

The memory model's cache terms are likewise verified against a **real forward pass**:
KV cache, DeltaNet recurrent state and conv state all match byte-for-byte, the KV cache
appears on exactly the full-attention layers, and the recurrent state is byte-identical
at sequence length 8 and 64 while the KV cache grows.

**What is still unverified:** the upstream **`config.json`, tokenizer, chat template and
licence have never been read** — egress to `huggingface.co` is blocked in the authoring
environment, and no attempt was made to route around it. The formulas are proven correct
*for a config with these values*; that the teacher **has** these values is corroborated,
not verified. Peak VRAM is also uncalibrated — no GPU.

The repository now ingests this metadata from a local directory instead, so anyone who
can obtain the files can close those questions without this environment needing network
access at all.

Every claim is classified VERIFIED / CORROBORATED / UNKNOWN, question by question, in
[`docs/VERIFICATION.md`](docs/VERIFICATION.md).

## The reasoning-efficiency goal

Qwen3.8-27B defaults to `reasoning_effort: xhigh` and reportedly spends thousands of
thinking tokens on simple requests. There is also a report that the `medium` setting
is a **no-op**, silently leaving users on maximum effort.

The objective here is *not* to suppress reasoning. It is to make reasoning
proportional to difficulty: near-zero for "what is 15 × 7", extensive for a hard
proof. A model that answers everything instantly is a failure, and every efficiency
experiment therefore reports hard-task accuracy alongside token counts — so that a
capability regression cannot be presented as an efficiency win. See
[`docs/reasoning-efficiency.md`](docs/reasoning-efficiency.md).

## Documentation

| Document | Contents |
|---|---|
| [VERIFICATION.md](docs/VERIFICATION.md) | Every claim classified VERIFIED / CORROBORATED / UNKNOWN |
| [TEACHER_BASELINE.md](docs/TEACHER_BASELINE.md) | The measuring instrument: modes, suites, determinism |
| [REASONING_BASELINE.md](docs/REASONING_BASELINE.md) | Measuring what the reasoning controls actually do |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Where parameters, memory and compute go |
| [PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | Phases, decision gates, and failure modes |
| [EVALUATION_PLAN.md](docs/EVALUATION_PLAN.md) | Tiers, baselines, contamination, reporting standards |
| [DEPLOYMENT_PLAN.md](docs/DEPLOYMENT_PLAN.md) | The 16 GB envelope and measurement methodology |
| [reasoning-efficiency.md](docs/reasoning-efficiency.md) | Adaptive reasoning research direction |
| [HARDWARE.md](docs/HARDWARE.md) | Capability tiers, what fits where, rented-GPU options |
| [TRAINING_ON_LIMITED_HARDWARE.md](docs/TRAINING_ON_LIMITED_HARDWARE.md) | The development ladder: CPU → T4 → rented |
| [COMPUTE_STRATEGY.md](docs/COMPUTE_STRATEGY.md) | When to rent, and what |

## Reporting standards

This repository does not publish numbers it has not produced.

- Analytical estimates are labelled as estimates.
- Unmeasured quantities are `TBD` / `Not yet evaluated`.
- Teacher and student comparisons come from the same harness, or they are not made.
- Upstream published scores are cited separately and never blended with ours.

## Licensing

This repository's code is MIT (see [`LICENSE`](LICENSE)).

The upstream model's license and any naming or attribution requirements **must be
verified against the upstream repository before any weights are released or any public
model name is chosen** — this has not yet been possible from this environment. See
[`THIRD_PARTY.md`](THIRD_PARTY.md).

## Teacher verification status

The upstream metadata has been supplied and verified. What that establishes, and what it
does not, are kept strictly apart.

### Directly verified from checkpoint metadata

Read from the teacher's own `config.json`, `tokenizer_config.json`, `chat_template.jinja`
and `LICENSE`, all pinned by SHA-256 in
[`configs/teacher/qwen3_8_27b.verified.json`](configs/teacher/qwen3_8_27b.verified.json):

| | |
|---|---|
| model type | `qwen3_5` (text: `qwen3_5_text`) |
| architecture | `Qwen3_5ForConditionalGeneration`, multimodal |
| loads natively | yes — `Qwen3_5ForCausalLM`, **no `trust_remote_code`** |
| dimensions | hidden 5120, 64 layers, FFN 17408, vocab 248320 |
| attention | 24 query / 4 KV heads, head_dim 256, output gate declared |
| DeltaNet | 48 value / 16 key heads, dims 128/128 |
| layer layout | explicit 64-entry list: **48 linear, 16 full** |
| context | 262144 native — **no `rope_scaling`**, so not YaRN-extended |
| MTP | declared, 1 layer, shares the base embedding |
| tokenizer | `Qwen2Tokenizer`, eos `<\|im_end\|>`, no BOS |
| reasoning controls | exactly `xhigh`, `medium`, `low`; `high` raises |
| default effort | **`xhigh`** — set nothing and you get the high-effort instruction |
| licence | **Apache-2.0** |

**Parameter count from the actual config: 26,895,998,464.** Computed by feeding the
supplied `config.json` through the analytical model — no preset, no hard-coding. A test
varies each architecture field and asserts the estimate moves.

### Runtime computation verified

The checkpoint declares `output_gate_type: "swish"` while `transformers` contains a
hard-coded `torch.sigmoid(gate)`. That looks like a contradiction and an earlier phase
recorded it as one. **It is not** — they refer to two different gates:

| Gate | Config key | transformers 5.15.1 | vLLM 0.27.1 |
|---|---|---|---|
| Gated DeltaNet output | `output_gate_type` | `silu` | `output_gate_type`, mapping `swish`→`silu` |
| Full attention output | `attn_output_gate` | `sigmoid` | `sigmoid` |

Swish (β=1) and SiLU are the same function, so the declaration is satisfied. Verdict:
**`VERIFIED_CORRECT`** — verified against the installed source and cross-checked against
vLLM, with a regression test that would fail if a checkpoint ever declared `sigmoid`.

### Not yet runtime-verified

Nothing below has been measured, and configuration resolving is **not** evidence for any
of it:

- state-dict parameter count (needs the weights)
- successful checkpoint loading and real generation (needs the weights)
- benchmark capability (needs a runtime baseline)
- actual reasoning-token behaviour at each effort level (needs generation)
- actual peak VRAM (needs a GPU)

`scripts/verify_teacher_loader.py` reports these as two explicit stages and prints
`STAGE 2: RUNTIME VERIFICATION — NOT PERFORMED` until real weights are supplied.

### One correction worth flagging

Earlier phases carried a secondary-source claim that the `medium` reasoning setting was
a no-op. **The real template refutes it**: `medium` renders a distinct — in fact the
shortest — prompt, because it injects no reasoning instruction while the default injects
the long `xhigh` one. The confusion was between "adds no instruction of its own" and
"changes nothing"; those differ precisely because the default is `xhigh`.

### Reproducing the verification

```bash
python scripts/validate_teacher_metadata.py --path vendor/qwen38-metadata \
    --save-verified configs/teacher/qwen3_8_27b.verified.json
python scripts/inspect_teacher.py --path vendor/qwen38-metadata --config-only
python scripts/verify_teacher_loader.py --model vendor/qwen38-metadata --config-only
python scripts/inspect_chat_template.py --path vendor/qwen38-metadata
```

All four run offline. See [`vendor/README.md`](vendor/README.md) for how to obtain the
metadata files.
