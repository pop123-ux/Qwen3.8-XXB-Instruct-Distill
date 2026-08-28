# Teacher → student distillation: what exists, and what it decides

The project's target is the strongest model anyone can run in 16 GB of VRAM, with a
separate architecture for 12 GB. This document covers the machinery that gets a student
*started* from Qwen3.8-27B and then trains it against the teacher's distribution.

**Parameter count is an optimisation variable here, not a target.** Nothing below assumes
a size. The sizes that appear are points in a search space, and the space is bounded by
the deployment envelope rather than by a number anyone picked in advance — see
[COMPETITIVE_OBJECTIVE.md](COMPETITIVE_OBJECTIVE.md) for what the envelope is and who
already occupies it.

Everything below is implemented and tested. Where a decision is still open it is marked
**OPEN**, with the measurement that should settle it rather than an opinion.

---

## 1. Why the student keeps the teacher's tokenizer

The Level-2/2R/3 experiments used a 256-entry byte vocabulary, which was the right choice
for from-scratch runs on a T4: no tokenizer to obtain, no embedding cost worth mentioning
at 94M parameters.

It is the wrong choice for a distilled student, and not marginally:

- **Logit KD becomes structurally impossible.** Matching a distribution requires the two
  models to index the same outcome space. Across different vocabularies there is no
  token-level correspondence to match, only an approximation nobody has to believe.
- **Embedding transfer becomes meaningless.** An embedding row is a token identity, not a
  feature vector. Row 40,132 of the teacher's table is whatever token 40,132 is; under a
  different tokenizer it is nothing.
- **The cost argument reverses with scale.** At 94M the 248,320-entry embedding would be
  63% of the model. At 4B it is ~17%, at 9B ~8% — the price of admission to every
  teacher-derived initialisation and to exact KD, and one that falls as the student grows
  into its envelope.

`student_from_teacher()` therefore does not expose `vocab_size` as a parameter, and
`apply_transfer_plan()` raises on a vocabulary mismatch rather than transferring anything.

**Still missing:** `vendor/qwen38-metadata/` carries `config.json`, `tokenizer_config.json`
and the chat template, but **no `tokenizer.json`**. The corpus pipeline is byte-level, so
a Qwen-vocabulary student has no tokenised corpus yet. `scripts/distill_pilot.py` detects
this and stops with the reason rather than training against mis-encoded text.

---

## 2. Layer selection: group alignment

Qwen3.8-27B is a period-4 hybrid — three Gated DeltaNet layers then one gated full-
attention layer, 48 + 16 over 64 layers. A layer's *position in the period* is its block
type, and the two block types do not share tensor names.

That makes the obvious selection strategies fail in opposite directions. Measured, teacher
64 layers → student N:

| N | strategy | tensor coverage | wrong block type | teacher depth spanned |
|---:|---|---:|---:|---|
| 28 | `first` | 100% | 0 | layers 0–27 only |
| 28 | `last` | 100% | 0 | layers 36–63 only |
| 28 | `uniform` | 97.2% | 8 | 0–63 |
| 28 | `interleave` | 97.7% | 7 | 0–54 |
| 28 | **`group`** | **100%** | **0** | **0–63** |

`uniform` spreads across the depth at a non-integer stride, so student layers land on
teacher layers of the wrong type. `interleave` degenerates toward `first` whenever
`teacher // student == 1`. `first`/`last` keep the layout but see one end of the model.

`group` is `uniform` applied to whole 4-layer groups with position preserved inside each
one. It is the only strategy that spans the full depth with zero type mismatches, at every
student depth tested (16, 24, 28, 32). It requires the student to have the teacher's
`full_attention_interval` and a layer count divisible by it; both are enforced.

This says nothing about which initialisation *trains* best — that is still an experiment.
It says the others are broken before the experiment starts.

---

## 3. Materialising the weights

`architecture/transfer.py` produces a plan. `architecture/materialize.py` applies it.
They are separate because their failure modes are: a wrong plan is visible by inspection,
a wrong reduction produces a student that loads cleanly, trains without error, and is
quietly worse than random initialisation.

Three layout facts, read from `transformers.models.qwen3_5.modeling_qwen3_5` rather than
assumed, drive the implementation:

1. **`q_proj` interleaves query and gate per head.** It is viewed as
   `(..., num_heads, head_dim * 2)` and chunked, so head *h* owns rows
   `[h·2d : (h+1)·2d]` as `[query | gate]`. Selecting heads in blocks of `head_dim` keeps
   half as many heads as intended and pairs each with its own gate.
2. **`in_proj_qkv` is three concatenated segments** — `[q (key_dim) | k (key_dim) |
   v (value_dim)]` — each independently head-structured, with `conv1d.weight` depthwise
   over the same concatenation. Row-slicing it drops the value segment entirely.
3. **A head only means anything with its group.** Head selection picks whole GQA groups
   (teacher ratio 6 query heads per KV head) and whole DeltaNet groups (3 value heads per
   key head).

Consistency is enforced across tensors, not just within them: every axis carrying the same
*role* in the same scope keeps the same teacher indices, so `down_proj`'s input columns are
the same neurons `gate_proj`/`up_proj` write to. Choosing independently per tensor would
leave every shape correct and the model scrambled.

**Verified:** transferring a model into an architecturally identical student reproduces the
teacher's logits with a maximum absolute delta of **0.0**.

### What it refuses

Refusals are per-model (raise) or per-tensor (skip and count), never silent:

| change | behaviour | why |
|---|---|---|
| `vocab_size` differs | raise | embedding rows are identities, not features |
| `head_dim` differs | raise | changes what a head *is*; interacts with `partial_rotary_factor` |
| GQA ratio differs | raise | a KV head is shared by exactly its group |
| DeltaNet value/key ratio differs | raise | same regrouping problem |
| any dimension **grows** | skip, counted | transfer selects; it cannot invent |
| teacher tensor absent | skip, counted | reported, not dropped |

### Reduction methods

`slice` (leading rows) is the labelled **baseline**, not a method — it assumes the teacher's
parameters are ordered by importance, which nothing guarantees. `mean_pool` uses adaptive
average pooling so non-integer ratios (5120 → 3072) work at all. `importance` selects by
per-unit weight energy from a single reference tensor per role. Which one wins is
**OPEN** and is a benchmark question.

### Coverage is reported in parameters

Tensor coverage flatters a transfer: the embedding is one tensor and 8–18% of a
multi-billion-parameter student. `TransferReport` reports both, and parameter coverage is the honest one.

---

## 4. The KD objective

`distillation/kd_loss.py`. The trap it exists to close:

> **Top-k logits alone do not determine the teacher's distribution.**

`softmax` needs the sum over the whole vocabulary and the top *k* values do not contain it.
An implementation that renormalises the top-k and calls the result "the teacher's
distribution" has silently changed the objective — it stops penalising the student for
mass on the 248,256 tokens the teacher rejected. That is the difference between *be like
the teacher* and *rank the teacher's shortlist the way the teacher does*.

So a sparse signal carries one extra number per token: the full-vocabulary `logsumexp`.
With it the tail mass `1 − Σ top-k` is exact.

Both treatments are implemented, and the difference is measured rather than argued. With
a student that dumps probability on one token outside the teacher's top-4:

| | student | student + mass off-support |
|---|---:|---:|
| `bucket` | 2.0586 | **4.4530** |
| `renormalize` | 1.1153 | **1.1153** |

`renormalize` cannot see it, to seven decimal places. `bucket` is the default.

**Verified:** the sparse path at `k = vocab` equals the dense path to 1e-5 at every
temperature tested (0.5, 1, 2, 4); self-distillation is exactly 0; the divergence is
non-negative over random pairs under both treatments; `alpha=0` reproduces
`torch.nn.functional.cross_entropy` exactly.

### Diagnostics logged every step

- `teacher_tail_mass` — mean teacher probability outside the stored top-k. **This is the
  empirical answer to "is k large enough".**
- `top1_agreement` — fraction of positions where the student's argmax matches the teacher's.
- `teacher_entropy` — near zero means a near-deterministic teacher, whose distribution
  carries little more than its argmax, and KD is then close to SFT.
- `kd_loss` and `ce_loss` reported separately, always. A combined number cannot distinguish
  "KD is working" from "alpha is small and this is SFT with extra steps".

All five land in `metrics.jsonl` per step and, as first/final/mean endpoints, in the run
summary's `distillation` block alongside the teacher's provenance. Without that block a KD
run's artifact is indistinguishable from an SFT run's apart from a config echo, and "the
config said `logit_kd`" is not evidence that a teacher distribution was ever reached.

---

## 5. Where the teacher distribution comes from — **OPEN**

`distillation/teacher_signal.py` keeps this separate from the objective, because it is a
cost decision and the loss does not care.

| | online (resident teacher) | offline (stored top-k) |
|---|---|---|
| status | **implemented** | loss and capture format ready; **reader not implemented** |
| VRAM | ~15 GB at 4-bit for 27B, *on top of* the student | none |
| storage | none | 388 B/token at k=64 → 3.9 GB per 10M tokens |
| exactness | exact at any temperature | exact at the captured temperature |
| flexibility | on-policy data, change k freely | corpus fixed in advance |
| reuse | pay again per run | one corpus trains many students |

Both produce an identical artifact through `capture_signal()`, so switching changes only
the bill.

**The measurement that should decide it:** run the online path and read `teacher_tail_mass`
at a candidate *k*. If k=64 leaves little mass outside it on real text, an offline corpus at
k=64 loses almost nothing and the offline path is strictly cheaper. If it leaves a lot, the
storage cost of a larger k is the real number to compare against renting the teacher. The
on-disk layout is deliberately unchosen until that number exists.

`build_provider("offline")` raises with this reasoning rather than returning something that
half-works.

---

## 6. Safety rails against the failure that matters

The failure this whole area is built to prevent: **a KD run that is really SFT.** Nothing
in the artifacts would show it — the loss falls, the checkpoints validate, the summary says
`logit_kd`.

- `train()` raises if the objective is not `sft` and no teacher provider was passed.
- A teacher capturing at a different temperature than `training.kd_temperature` is caught at
  setup, not a thousand tokens in — the captured normaliser is not convertible.
- `tail="bucket"` without a `logsumexp` raises rather than falling back to `renormalize`.
- `training.objective` and the KD hyperparameters are **notable** resume keys: switching
  them across a resume is allowed but always reported.
- `signal_source="dataset"` is refused with the reason, not silently ignored.
- KD against the synthetic induction corpus is refused: its tokens come from a rule, so
  "what the teacher believes comes next" is not a question about anything the teacher models.

KD over a plain **text** corpus with a resident teacher *is* allowed, and is the cheapest
real distillation available — it needs no teacher-generated answers at all.

---

## 7. The pilot

`scripts/distill_pilot.py` runs the whole chain in one command: plan → materialise →
evaluate cold → distil → checkpoint.

**Stage 0 — `--stand-in`.** A small randomly-initialised teacher, structured like the real
one (period-4 hybrid, teacher GQA ratio, teacher DeltaNet ratio) so every constraint is
exercised; only the scale is fake. Runs on a T4 or a laptop in minutes. It proves the
mechanism and **nothing whatsoever about capability** — the teacher knows nothing, so a
student that matches it has learned nothing.

**Stage 1 — `--teacher DIR`.** The real teacher, same code path, on hardware that holds it.

The transferred student is always written to `<output>/transferred/` before training, and
that checkpoint — not the architecture spec — is what training loads. This is not a
convenience: the trainer builds its student from the config, so passing it a spec would
rebuild a random model and discard the transfer, while still printing a transfer report, a
cold evaluation and a falling KD loss. Every number would look right and the run would
mean nothing. `tests/test_distill_pilot.py` pins it. Keeping the artifact also matters at
Stage 1 for its own sake: a 54 GB transfer is not something to redo per run, and
`--transfer-only` produces it without training.

Both take the cheapest informative measurement first and for free: the transferred student
is evaluated **before any training**, against a randomly-initialised student of the same
architecture, and again afterwards. If a 6× reduction leaves the transfer at chance, the
transfer strategy is the problem, and no distillation budget will surface that faster.

On a Stage-0 run those numbers have a known correct answer, which is what makes them a
check on the harness rather than a result: the transfer must be worth **≈ 0** (a random
teacher carries no information), and validation loss must **rise** during KD (the student
is pulled toward a random distribution and away from the corpus). A measured Stage-0 run
gives −0.0049 nats for the transfer and validation 5.5822 → 5.5954 across 40 steps. The
rise is the clearest available evidence that the KD term is real rather than falling
through to cross-entropy — an SFT run on the same corpus goes the other way.

### Sizing the student

`student_from_teacher()` builds a student the teacher can actually be transferred into,
with the teacher's ratios fixed by construction. Reducing `kv_heads` and `dn_key_heads`
alongside `hidden` matters: keeping all 24 query heads at `head_dim` 256 against a
3072-wide residual stream makes attention 2× over-complete.

These are **points in a search space, not a shortlist.** The size that wins is the one
that maximises measured capability inside the envelope, and that is an experimental
question. What the envelope says today (q4_k_m, 8K context, 13.56 GiB usable) is that the
smaller end of this table leaves most of the budget unspent while the model it has to beat
occupies roughly half of it:

| hidden | layers | ffn | kv heads | DN key heads | tied | params | embedding | total VRAM |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 3072 | 28 | 10240 | 2 | 8 | yes | 4.36B | 17.5% | 3.85 GiB |
| 3072 | 32 | 10240 | 2 | 8 | yes | 4.87B | 15.7% | — |
| 3584 | 32 | 12160 | 2 | 8 | yes | 6.34B | 14.0% | — |
| 4096 | 32 | 13824 | 3 | 12 | yes | 8.63B | 11.8% | 6.50 GiB |

(Analytical counts from `architecture/params.py`; the teacher is 26.90B. VRAM from
`scripts/competitive_report.py`, which places these rows beside the competition.)

Run `python scripts/competitive_report.py --candidate hidden,layers,ffn,kv,dnk` to place
any point in the same frame as the field.

---

## 8. Known blockers

1. **Teacher weights are unobtainable in the sandboxed environment** (~54 GB, restricted
   egress). Every Stage-1 run is remote. Not a code problem.
2. **No `tokenizer.json` in `vendor/`**, so no Qwen-vocabulary corpus. Blocks Stage 1, not
   Stage 0.
3. **No benchmark suite — now the critical path.** The objective is stated entirely in
   numbers this repository cannot produce: MMLU-Pro, GPQA Diamond, IFEval,
   LiveCodeBench v6, LongBench v2, BFCL v4, TAU2-Bench. None is implemented. Until at
   least one is, "better" is unmeasurable and every architecture decision after the pilot
   is taken blind. See [COMPETITIVE_OBJECTIVE.md](COMPETITIVE_OBJECTIVE.md).
4. **`slice` is a baseline, and the reduction ratio is a variable too.** 27B → 4.4B is 6×,
   more aggressive than the 2–4× where pruning-plus-distillation usually recovers well;
   27B → 9B is 3×, which is inside it. That is one more argument for not fixing the size in
   advance. The cold evaluation is the early warning either way.
5. **Offline signal reader unimplemented** — see §5; deliberately gated on a measurement.
