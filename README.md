# Qwen3.8-XXB-Instruct-Distill

Research infrastructure for compressing **Qwen3.8-27B** into an instruct model that
runs comfortably on a **single 16 GB consumer GPU**, keeps a genuinely large context
window, and spends far fewer tokens thinking about easy questions.

> **Project status: Phase 0 — analysis infrastructure.**
> No model has been trained. No benchmark has been run. There are no capability
> results in this repository, and the `XXB` in the name is a placeholder: the final
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

## Verification, and an honest limitation

Architectural formulas in this repository were transcribed from the **reference
implementation** (`transformers==5.15.1`, `models/qwen3_5/`) — the code that actually
runs the model — rather than from prose descriptions.

The anchor check: applying those formulas to the published Qwen3.8-27B configuration
yields **26,895,998,464 parameters (26.90B)**, within 0.4% of the advertised 27B. That
is pinned as a regression test.

**However:** the environment that authored this repository could not reach
`huggingface.co` (blocked by egress policy). So the upstream **model card,
`config.json`, tokenizer, chat template and license were never read directly**, and no
checkpoint cross-check was performed. Several claims — the exact config values, the
Apache-2.0 license, the `reasoning_effort` behaviour — are corroborated by multiple
secondary sources but are **not primary-verified**.

`scripts/inspect_teacher.py` exists to close that gap, and it will report a MISMATCH
if our analytical model disagrees with the real checkpoint. **Run it before relying on
any number here.** Full detail, claim by claim, in [`docs/VERIFICATION.md`](docs/VERIFICATION.md).

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
| [VERIFICATION.md](docs/VERIFICATION.md) | What is verified, what is not, and how to close the gap |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Where parameters, memory and compute go |
| [PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | Phases, decision gates, and failure modes |
| [EVALUATION_PLAN.md](docs/EVALUATION_PLAN.md) | Tiers, baselines, contamination, reporting standards |
| [DEPLOYMENT_PLAN.md](docs/DEPLOYMENT_PLAN.md) | The 16 GB envelope and measurement methodology |
| [reasoning-efficiency.md](docs/reasoning-efficiency.md) | Adaptive reasoning research direction |

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

## Contributing

The most valuable contribution right now is **primary-source verification**: run
`scripts/inspect_teacher.py` against the real checkpoint and open a PR with the report.
That unblocks the architecture decision.
