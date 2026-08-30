# Qwen3.8-XXB-Instruct-Distill

Distilling **Qwen3.8-27B** into a hybrid sparse-MoE student that runs on a **single 16 GB
consumer GPU**, keeps a genuinely large context window, and is studied carefully enough to
publish.

Two goals, held at the same time — see [PROJECT_DIRECTION.md](docs/PROJECT_DIRECTION.md):

- **a paper** — *Beyond Layer Matching: Distilling Computational Behavior in Hybrid
  Language Models*, plus a second component on context-length specialisation, with a
  falsifier written down for every claim before the run;
- **a release** — the strongest deployable model under a strict 16 GB end-to-end VRAM
  constraint.

> **Status: architecture frozen, initialisation implemented and measured, nothing trained.**
> No student has been distilled from the real teacher. No benchmark has been run. No
> external compute has been used — everything here was produced in a browser session with
> no GPU, so every number is either an audit of a locally constructed model, an analytical
> estimate labelled as one, or a third-party figure with a source.

## The student

One frozen target, `qwen38_19b_h5120_l48_moe`. Not a ladder, not a search space.

```
48 layers = 36 Gated DeltaNet + 12 full attention     [D, D, D, A] x 12
hidden 5120,  vocab 248,320,  context 262,144,  untied embeddings
attention   24 query heads, 2 KV heads, head_dim 256, output gate, rotary dim 64
DeltaNet    16 key heads, 48 value heads, head dim 128, conv kernel 4
MoE         24 routed experts (top-2) + 1 shared expert, expert width 768
```

Compression from the teacher is depth-only plus sparsity: 64 -> 48 layers, dense
17408-wide FFN -> 24 experts of 768, 4 KV heads -> 2. Width, vocabulary, head dimensions
and the hybrid pattern are unchanged, which is what makes each reduction attributable.
Full specification: [STUDENT_ARCHITECTURE.md](docs/STUDENT_ARCHITECTURE.md).

## Two measured findings that changed the plan

Both were run before any training code was written. Both returned inconvenient answers, and
both are reported rather than worked around.

### It weighs 22.07B, not 19B

```
$ python scripts/student_report.py --section architecture

    component               parameters    share
    embedding            1,271,398,400    5.76%
    lm_head              1,271,398,400    5.76%
    attention            1,195,382,784    5.42%
    deltanet             4,171,538,304   18.90%
    routed_experts      13,589,544,960   61.57%
    shared_expert          566,476,800    2.57%
    router                   5,898,240    0.03%
    norms                      496,640    0.00%
    TOTAL               22,072,134,528

    difference_from_19B       +3,072,134,528     (+16.2%)
    non_embedding             19,529,337,728
    active_per_token           9,615,051,648
```

The name is the label the project inherited; 19.53B non-embedding is its likely origin. The
architecture is frozen, so it is not adjusted to make the label true. **61.57% of the model
is routed experts** — the number that drives every memory decision downstream.

### It does not fit 16 GB

End to end, fully GPU-resident, at Q4/Q5/Q6, at every context length from 2,048 tokens up:

| quant | context | weights | KV | state | acts | runtime | total | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Q4 | 2,048 | 13.09 | 0.05 | 0.111 | 0.28 | 0.90 | **14.44** | DOES NOT FIT |
| Q4 | 32,768 | 13.09 | 0.75 | 0.111 | 0.28 | 0.90 | **15.14** | DOES NOT FIT |
| Q4 | 262,144 | 13.09 | 6.00 | 0.111 | 0.28 | 0.90 | **20.39** | DOES NOT FIT |

The cheapest all-Q4 configuration is **13.93 GiB at 2,048 tokens against 13.56 GiB usable —
short by 0.37 GiB** before a single long-context token is cached. Fits begin one precision
step below the release set, at 3-bit experts, reaching 64K context.

A 16 GB card reports 14.56 GiB, and planning against the nominal 16.0 GiB instead makes Q4
appear to reach 65,536 tokens. That gap is the whole difference between a plan that works
and an OOM on hardware that obviously had room.

**This is an open decision on the critical path**: ship at 3-bit experts and report the
quality cost, or reduce the expert budget. The repository does not make that call
unilaterally. [PARETO_EVALUATION.md](docs/PARETO_EVALUATION.md).

## The research

**Behavioural distillation** — [BEHAVIORAL_DISTILLATION.md](docs/BEHAVIORAL_DISTILLATION.md).
Conventional layer matching asks the student to visit the teacher's intermediate hidden
states. But removing 16 of 64 layers removes *work*, and pointwise matching never says who
took over. Residual contributions telescope — the total work of a teacher span `[a, b)` is
exactly `h_b - h_a` — so each student layer can instead be trained to reproduce the combined
contribution of every teacher layer it replaced, removed ones included, at no extra cost.

**Context specialisation** — [CONTEXT_SPECIALIZATION.md](docs/CONTEXT_SPECIALIZATION.md).
`max_position_embeddings = 262144` is a config field and is free. What is measurable is the
context-performance curve and where its knee sits. In a hybrid model the 12 attention layers
can look back arbitrarily far and pay linearly; the 36 DeltaNet layers keep a fixed state and
cannot. What that state learns to carry comes from the lengths in the training data — which
is the mechanism by which a training-time choice moves a deployment-time capability.

**The ablation matrix** — `research/ablations.py`. Family A is a 2x2 factorial over
{pointwise matching} x {behavioural delta}; family B is four sequence-length curricula where
B2 and B3 share a token budget exactly, isolating ordering from exposure. Every arm carries
a `falsified_if` written before the run, and `no_measurable_effect` is a reachable, tested
verdict.

```bash
python scripts/student_report.py                 # architecture, memory, context, ablations
python scripts/student_report.py --json out.json --ledger
```

## Reporting standards

1. **Never call an estimate a measurement.** The ledger enforces a closed provenance set —
   `measured_here`, `reported_by_third_party`, `estimated` — where an estimate must carry its
   method and a third-party number must carry its source.
   [EXPERIMENT_LEDGER.md](docs/EXPERIMENT_LEDGER.md)
2. **Never claim a benchmark before running it.** Qwen3.5-9B's published figures are the
   target, not evidence about us. This project holds no benchmark results of its own.
3. **A negative result is a result.** If the central claim fails, it is reported as failed.
4. **16 GB is end-to-end and GPU-resident.** CPU offload is a different product and never
   counts as compliance.
5. **Nothing degrades silently.** Unavailable objectives raise with the reason; a mismatched
   checkpoint is fatal, not a warning. MTP is declared by the specification and **not built**
   by the runtime, so no MTP result is claimed and the loss term refuses to run.

---

# Historical record

Everything below predates the direction reset and is retained, not deleted. The
from-scratch ladder (Levels 1 / 2 / 2R / 3) validated the training mechanism; the dense
`h5120 L40` 17.76B candidate remains the baseline against which the sparse MoE student is
first compared.

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

Level 2 (94.48M, real text) is the experiment that starts to answer the language
question.

## The experiment ladder

| level | model | corpus | status |
|---|---|---|---|
| **1** | 4.03M prototype | synthetic tokens, vocab 4096 | **complete** — validated the training mechanism |
| **2** | 94.48M hybrid | procedural byte text, 8 MB | **complete** — validated the stack; generated `"and and and"` |
| **2R** | 94.48M hybrid | real public-domain English, 44.1 MB | **complete** — the first meaningful language-learning experiment |
| **3** | 236.24M, width-scaled | the same Level-2R corpus | **ready to run — NOT RUN** |
| distillation | — | teacher outputs | infrastructure built; **teacher generation not run, logit KD not implemented** |

Each level is a separate record under [`experiments/runs/`](experiments/runs/), with its
claims split into what it establishes, what it does not, and what is unknown.

## Level 2R — the first real-language result

**COMPLETE.** 2000/2000 steps on a T4 in 18,963.8 s, 32,768,000 tokens, final validation
**1.797 bits/byte** against an 8.0 uniform-byte baseline — **4.45x compression**.

> The 94.48M hybrid learns non-trivial natural-language structure from real English and
> avoids catastrophic unigram-style collapse, but its generation remains repetitive and
> semantically weak at this scale.

Greedy, from the committed [`sanity.json`](experiments/runs/t4_level2r_100m_real_english/sanity.json):

| prompt | continuation |
|---|---|
| `"Yesterday, I"` | ` should have thought it a little too much to be a man of the service of the property of the pros` |
| `"In the beginning "` | `of the state of the present day, and the conversation was the same thing that had been so much a` |
| `"When the sun "` | `was standing on the steps the street was standing before the street was standing before the stre` |

Real English words, function words in grammatical positions, local clause structure that
scans — against the identical architecture's `"and and and and"` on procedural text. And
also: phrase-level looping, motif fixation on *stranger*, *street*, *thing*, *man*, and no
semantic thread. Measured, 3-gram repetition is **12.4% mean** with 4 of 11 generations
looping, against **83%** for Level 2's collapse. Nothing memorised.

It is **not** fluent, useful, instruction-following, benchmark-competitive or
teacher-equivalent, and must not be cited as any of those.

**The run is undertrained, not converged.** It saw **0.826 epochs** — 17.4% of the corpus
was never read — and `OneCycleLR` put the final 200 steps at **0.9% of peak learning rate**.
The flat tail is the schedule running out, not the model saturating. 2000 steps was a
budget, and it now serves as a controlled baseline.

Full record:
[`experiments/runs/t4_level2r_100m_real_english/`](experiments/runs/t4_level2r_100m_real_english/).

## Level 3 — ready to run, not run

One variable changes: **width**. Hidden 640 -> 1024, 94.48M -> **236.24M** parameters
(x2.50). Depth stays 16 layers, the layout stays 12 DeltaNet + 4 full attention (the
teacher's 3:1), the byte vocabulary stays 256, and **all 23 training fields are identical**
— same corpus bytes, sequence length, batch, accumulation, optimizer, learning rate,
schedule, precision and gradient checkpointing. The estimator puts it at **6.18 GiB with
7.38 GiB spare** on a T4, so nothing is forced to change and nothing does.

> Does increasing model capacity above 94.48M produce a material improvement in
> real-language modeling under the same controlled setup?

Estimated ~12.9 h and ~28.3 GB of Drive for ten checkpoints. The stopping rule, the
continuation rule and the reading of every outcome are **written down before the run**:
[`docs/experiments/level3_plan.md`](docs/experiments/level3_plan.md).

## Choosing what comes after Level 3

The destination is a Qwen3.8-27B alternative that runs on **16 GB**, with a path to
**12 GB**. Which architecture gets there is a question to answer with numbers before it
costs GPU hours:

```bash
python scripts/architecture_report.py --list                       # known architectures
python scripts/architecture_report.py --sweep level2r level3 teacher
python scripts/architecture_report.py --presets level3:hidden_size=1280
python scripts/architecture_report.py --summary <run-a> <run-b>    # the decision report
```

A sweep reports parameters, training memory, and 16 GB / 12 GB feasibility **with the
longest context each supports** — because "fits on 16 GB" is true of a long-context model
at 4K and can be false at 256K. Nothing is trained; rejecting a candidate costs
milliseconds.

The comparison refuses what the data does not support: no delta across different
validation corpora, no invented capability score, and no conclusion that assumes a bigger
model is a better one. Full loop:
[ARCHITECTURE_RESEARCH.md](docs/ARCHITECTURE_RESEARCH.md).

## Level 2 — the procedural control

| | |
|---|---|
| Model | **94.48M** params, 16 layers (12 DeltaNet + 4 full attention), hidden 640, FFN 2176 |
| Ratios | 3:1 hybrid layout and 3.40x FFN expansion, both matching the teacher |
| Vocabulary | **256** — byte-level, so nearly every parameter is in the layers rather than an embedding table |
| Objective | causal LM on real text, scored in **bits per byte** (8.0 = learned nothing) |
| Target | Tesla T4, 14.56 GiB, fp16 autocast, seq 1024, 2000 steps |
| Estimate | **3.57 GiB** with the corrected model (the original 4.53 GiB estimate was wrong — see below) |

Byte-level tokenization is the deliberate choice here: no tokenizer to download or
version, any text file works unchanged, and the loss is directly comparable across runs
because it does not depend on a tokenizer's compression rate.

**Level 2 is complete: 2000/2000 steps, no OOM, 2,089.2 tok/s run-wide, final checkpoint
validated bit-for-bit.** Validation BPB 1.270 against an 8.0 uniform-byte baseline — and
greedy generation from that same checkpoint is `"and and and and…"`.

Both are correct, and they are the same fact. The corpus was procedural text with a
Zipfian word distribution and **no syntax, no semantics, no long-range dependency**; the
optimal model for it predicts common words forever. Validation BPB reached 1.279 by step
400 and only 1.270 by step 2000 — 80% of the run bought under 1% of the improvement,
because the corpus was exhausted.

**Level 2 establishes that the architecture trains, checkpoints, resumes and persists. It
establishes nothing about language capability, and must not be cited as if it did.** Full
result: [`experiments/runs/t4_level2_100m_ckpt_complete/`](experiments/runs/t4_level2_100m_ckpt_complete/);
what to run next: [`docs/experiments/level2_report.md`](docs/experiments/level2_report.md).

Level 2's 1.270 and Level 2R's 1.797 are **not comparable** — different corpora, different
intrinsic entropy. `scripts/compare_runs.py` refuses that delta and says why:
[`docs/experiments/level2_vs_level2r.md`](docs/experiments/level2_vs_level2r.md).

### Earlier attempts

**Second attempt: it trains.** No OOM, ~2100 tokens/s, validation bits-per-byte down
from 1.317 at step 200 to 1.279 at step 400 (8.0 = learned nothing). Then the Colab
runtime disconnected at ~step 500 and took the ephemeral filesystem with it — see
[`experiments/runs/t4_level2_100m_ckpt_interrupted_2026-08-24/`](experiments/runs/t4_level2_100m_ckpt_interrupted_2026-08-24/).
**That is ~25% of a 2000-step run, not a finished experiment**, and it says nothing
about final model quality.

Training is now interruption-safe: checkpoints are written atomically with a `COMPLETE`
marker and a verified `latest.json` pointer, and they carry everything a resume needs —
scheduler, GradScaler, RNG, data position, tokens seen. Previously 11 of 17 required
items were not persisted, so even a surviving checkpoint would have restarted the
one-cycle schedule and rewound the data to epoch 0.

```bash
python scripts/train_student.py --config configs/experiments/t4_level2_100m_ckpt.yaml --status
python scripts/train_student.py --config configs/experiments/t4_level2_100m_ckpt.yaml --resume latest
```

A crash may lose the step in flight; it can no longer invalidate the last completed
checkpoint. Full design and the Colab recovery workflow:
[`docs/experiments/t4_level2_resumability.md`](docs/experiments/t4_level2_resumability.md).

**The first attempt OOMed on a real T4** — predicted 4.53 GiB, demanded ~24.8 GiB, died
in the forward pass with zero steps completed. The failure is kept as an artifact in
`experiments/runs/t4_level2_100m_oom_2026-08-24/` and its config is preserved unchanged.

The cause was not batch size. The estimator had **no Gated DeltaNet activation term at
all**, and those 12 layers held 66% of every retained activation: `transformers`'
pure-torch DeltaNet kernel force-upcasts to fp32 and runs a 63-iteration loop that
retains O(chunk²) clones per chunk. The traceback named `scaled_dot_product_attention`,
which was the next allocation rather than the cause — attention is ~3% of the total.
[Full analysis](docs/TRAINING_ON_LIMITED_HARDWARE.md#what-the-oom-taught-us).

The revised config adds gradient checkpointing (measured 67x less retained activation)
and keeps the architecture, sequence length and effective batch identical:

```bash
python scripts/probe_activations.py --config configs/experiments/t4_level2_100m_ckpt.yaml --scaling
python scripts/train_student.py --config configs/experiments/t4_level2_100m_ckpt.yaml --dry-run
python scripts/train_student.py --config configs/experiments/t4_level2_100m_ckpt.yaml
python scripts/validate_checkpoint.py experiments/runs/t4_level2_100m_ckpt/final
python scripts/hardware_info.py --calibrate-run experiments/runs/t4_level2_100m_ckpt/summary.json
```

The last command is the point of the exercise: it compares each estimated term against
what was actually measured and names which term is wrong. See
[Memory: measured, not multiplied](#memory-measured-not-multiplied).

### Working in Colab

Colab runtimes are ephemeral, and checkpoints are too large for git. After a run:

```bash
python scripts/backup_colab_to_drive.py --dry-run   # see exactly what would move
python scripts/backup_colab_to_drive.py
```

It never follows symlinks, never copies credential-shaped files (`.env`, `*.pem`,
`id_rsa*`, `*token*.json`, `.ssh/`, ...), and never deletes anything unless you pass
both `--delete-extraneous` and `--yes`. Code and configs belong in git; checkpoints and
experiment artifacts belong on Drive.

## Distillation infrastructure — built, never run

The teacher has **never been loaded**. No teacher data has been generated, no benchmark
run, no student distilled. What exists is the machinery for doing it once, cleanly:

```bash
python scripts/generate_teacher_data.py --input prompts.jsonl --output data/teacher \
    --reasoning-mode xhigh --dry-run          # validates; loads nothing
python scripts/train_distilled_student.py --list-objectives
```

Three rules it enforces, each protecting a result that would otherwise be worthless:

- **A fake teacher can never stand in for a real one.** The mock backend must be named
  explicitly; the real backend raises rather than degrading. A synthetic dataset that
  looks real would train a student, produce numbers, and mean nothing.
- **KD is never silently SFT.** `logit_kd` is declared and marked `NOT_IMPLEMENTED` —
  no teacher logits exist yet — and refuses to run. Substituting SFT would make the
  project's central question, *does KD beat SFT?*, unanswerable.
- **A partial shard is never read as a complete one.** Generation is resumable by prompt
  id, shards are checksummed on close, and the loader reads only what the manifest calls
  complete.

Reasoning modes are validated against the *real* chat template: `xhigh`, `medium`, `low`
and `thinking_disabled`. `high` is rejected because the template rejects it, and `medium`
is not a no-op — it renders the shortest prompt, since the default it replaces is `xhigh`.
See [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md).

## Memory: measured, not multiplied

An earlier calibration reported a measured/estimated ratio of ~2.85. That number was
not usable, and it is worth saying why: it divided **peak reserved** — which includes
the CUDA context and every block the caching allocator holds — by **modelled tensors
with the overhead term deliberately zeroed**. Those are different quantities, and on a
small probe model the CUDA context alone rivals the whole tensor footprint. Multiplying
future estimates by 2.85 would have baked that confusion in permanently.

What the estimator does instead:

- **Predicts two quantities**, matching the two the probe measures: live tensors
  (`predicted_allocated_gib`) and process footprint (`predicted_reserved_gib`).
- **Attributes every term separately** — weights, gradients, optimizer moments,
  activations, logits — so a total that is right because two errors cancelled is still
  reported as two errors.
- **Models the precision that will actually run.** `torch.autocast` keeps fp32 weights
  and gradients while computing in fp16; pure-bf16 training keeps bf16 weights with an
  fp32 master. Both total 16 bytes/parameter for AdamW, so only a per-component
  breakdown can tell them apart.
- **Counts the loss path properly.** `ForCausalLMLoss` does `logits = logits.float()`
  and `cross_entropy` retains a second fp32 log-softmax buffer, so the logits are held
  three times over — a 2.5-3x correction that is measured in gigabytes at a 248k vocab.

`--calibrate-run` reports per-term residuals and names which term to fix. It never emits
a global multiplier.

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
pytest                                    # 990 tests, no GPU required

python scripts/estimate_vram.py --preset teacher --matrix --max-context
python scripts/search_architectures.py --vram 16 --context 32768 --top 15
python scripts/inspect_teacher.py --repo-id Qwen/Qwen3.8-27B --config-only
```

Everything runs on CPU in seconds. No weights are downloaded unless you ask.

## Repository layout

```
src/qwen_distill/
  architecture/   spec, exact parameter accounting, VRAM model, FLOPs, search
  teacher/        checkpoint inspection, cross-checking, runtime compatibility
  diagnostics/    device detection, capability tiers, fit analysis, calibration
  training/       config, feasibility gate, trainer, corpora, memory probe
  evaluation/     harness, backends, reasoning-effort sweeps
  analysis/       post-hoc run analysis, cross-experiment comparison, scaling candidates
scripts/          hardware_info, train_student, estimate_vram, validate_checkpoint, ...
docs/             plans and analysis (start with VERIFICATION.md)
experiments/      architecture search outputs
vendor/           teacher metadata, supplied out-of-band
tests/            990 tests pinning every formula
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

The teacher's **`config.json`, tokenizer config, generation config, chat template and
licence** are now in the repository under `vendor/qwen38-metadata/`, supplied out-of-band
rather than downloaded: egress to `huggingface.co` is blocked in the authoring
environment, and no attempt was made to route around it. Every field is recorded with a
SHA-256 of its source file in `configs/teacher/qwen3_8_27b.verified.json`.

**What is still unverified:** anything that needs the *weights* or a *GPU* — state-dict
parameter count, real generation, benchmark capability, and measured peak VRAM. The
memory model is analytical throughout.

Every claim is classified VERIFIED / CORROBORATED / UNKNOWN, question by question, in
[`docs/VERIFICATION.md`](docs/VERIFICATION.md).

## The reasoning-efficiency goal

Qwen3.8-27B defaults to `reasoning_effort: xhigh` and reportedly spends thousands of
thinking tokens on simple requests. A widely repeated secondary claim that the `medium`
setting is a **no-op** is *refuted* by the real chat template: `medium` renders a
distinct — in fact the shortest — prompt. See
[below](#one-correction-worth-flagging).

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
| [ARCHITECTURE_RESEARCH.md](docs/ARCHITECTURE_RESEARCH.md) | **The research loop: define, estimate, sweep, compare — and how 16/12 GB feasibility is decided** |
| [PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | Phases, decision gates, and failure modes |
| [EVALUATION_PLAN.md](docs/EVALUATION_PLAN.md) | Tiers, baselines, contamination, reporting standards |
| [DEPLOYMENT_PLAN.md](docs/DEPLOYMENT_PLAN.md) | The 16 GB envelope and measurement methodology |
| [reasoning-efficiency.md](docs/reasoning-efficiency.md) | Adaptive reasoning research direction |
| [CHECKPOINT_RECOVERY_GUARANTEE.md](docs/CHECKPOINT_RECOVERY_GUARANTEE.md) | What "persisted" means, and how a deleted checkpoint is caught |
| [HARDWARE.md](docs/HARDWARE.md) | Capability tiers, what fits where, rented-GPU options |
| [TRAINING_ON_LIMITED_HARDWARE.md](docs/TRAINING_ON_LIMITED_HARDWARE.md) | The development ladder: CPU → T4 → rented |
| [COMPUTE_STRATEGY.md](docs/COMPUTE_STRATEGY.md) | When to rent, and what |

### Experiment protocol and analysis

| Document | Contents |
|---|---|
| [POST_RUN_CHECKLIST.md](docs/experiments/POST_RUN_CHECKLIST.md) | Reaching `max_steps` is not completion. What is. |
| [level2_report.md](docs/experiments/level2_report.md) | The Level-2 result, formalised |
| [level2r_plan.md](docs/experiments/level2r_plan.md) | Level 2R: the plan as written before the run |
| [level3_plan.md](docs/experiments/level3_plan.md) | **Level 3: candidates, choice, stopping rule, evaluation — pre-registered** |
| [level2_vs_level2r.md](docs/experiments/level2_vs_level2r.md) | How to compare them — and the BPB delta that must not be reported |
| [SCALING_STUDY.md](docs/experiments/SCALING_STUDY.md) | Protocol for 4M → 500M, and why two points are not a law |
| [DISTILLATION_DATA_REQUIREMENTS.md](docs/experiments/DISTILLATION_DATA_REQUIREMENTS.md) | Teacher-output storage, and the vocabulary mismatch that blocks logit KD |
| [12GB_DISTILLATION_EXTENSION.md](docs/experiments/12GB_DISTILLATION_EXTENSION.md) | What changes on a 12 GB card |
| [t4_level2_resumability.md](docs/experiments/t4_level2_resumability.md) | Interruption safety and the checkpoint contract |

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
