# Level 3 — 236M, width-scaled, on the Level-2R corpus

**STATUS: READY TO RUN. NOT RUN.**

Written before the run, so the analysis cannot be chosen after the data arrives.

```
config      configs/experiments/t4_level3_236m_real_english.yaml
output      experiments/runs/t4_level3_236m_real_english
baseline    experiments/runs/t4_level2r_100m_real_english
```

---

## 1. The question

> **Does increasing model capacity above 94.48M produce a material improvement in
> real-language modeling under the same controlled setup?**

Level 2R established that the 94.48M hybrid learns real English structure and avoids
Level 2's collapse, while remaining repetitive and semantically weak. Two explanations
survive that result, and they call for different next steps:

- **capacity** — 94.48M is too small to hold more than local structure;
- **budget** — 32.8M tokens over 0.826 epochs, with the learning rate at ~0 for the last
  block, is too little training for any size.

Level 3 tests the first at a fixed budget. It **cannot** distinguish the second; that
needs a longer horizon, which is Stage 2 (§4).

## 2. What changes, and what does not

Verified programmatically against the two config files: **all 23 training fields
identical, all data fields identical, five architecture fields differ and all five are
width.**

| | Level 2R | Level 3 | |
|---|---|---|---|
| hidden_size | 640 | **1024** | ×1.60 |
| intermediate_size | 2176 | **3456** | ×1.59 (3.40 → 3.375 expansion) |
| num_attention_heads | 10 | **16** | ×1.60, head_dim stays 64 |
| linear_num_key_heads | 4 | **6** | ×1.50 |
| linear_num_value_heads | 12 | **18** | ×1.50 |
| **parameters (measured)** | 94,476,448 | **236,237,488** | ×2.50 |

Held identical: `num_hidden_layers` 16, `full_attention_interval` 4 (12 DeltaNet + 4 full
attention, the teacher's 3:1), `head_dim` 64, DeltaNet head dims 64, conv kernel 4,
`vocab_size` 256, `max_position_embeddings` 4096, tied embeddings — and every training
variable: sequence 1024, batch 4, accumulation 4, effective batch 16, AdamW, LR 6e-4,
OneCycleLR, warmup 100, fp16 autocast, gradient checkpointing on, objective `sft`, seed 0,
`max_steps` 2000, `eval_every` 200, and the same corpus bytes in the same order.

**Nothing is forced to change.** The memory estimator puts 236M at **6.18 GiB with 7.38
GiB spare** on a T4, so batch size and accumulation stay at their Level-2R values rather
than being adjusted for memory. This is a single-variable experiment in the strict sense.

### Variables that cannot remain identical

- **Wall-clock.** ~12.9 h estimated against Level 2R's measured 5.27 h. Unavoidable and
  not a confound: it measures the same work taking longer.
- **Tokens per parameter.** 0.347 → **0.139**. Fixing the token budget necessarily
  un-fixes this. It is the defining property of a fixed-budget comparison and the reason
  §6 forbids reading the result as "what a 236M model can do".
- **Checkpoint size.** 1.13 GB → 2.83 GB, so storage planning differs (§7). Not a
  scientific variable.

## 3. Candidates considered

Memory from `qwen_distill.diagnostics.fit`, the estimator that reproduces Level-2R's
recorded 3.57 GiB exactly. Throughput extrapolated by FLOP ratio from **Level 2R's
measured run-wide 1,727.9 tok/s** — not Level 2's 2,089.2, which excluded real-corpus I/O
and verified Drive persistence and would understate every runtime here by ~21%.

| candidate | params | VRAM @ b4 | FLOPs | est. tok/s | est. runtime | ckpt | 10 ckpts |
|---|---|---|---|---|---|---|---|
| Level 2R (measured) | 94,476,448 | 3.57 GiB | 1.00× | **1,728** | **5.27 h** | 1.13 GB | 11.3 GB |
| **236M h1024 L16** | **236,237,488** | **6.18 GiB** | 2.45× | ~704 | ~12.9 h | 2.83 GB | 28.3 GB |
| 269M h960 L20 | 269,106,460 | 6.68 GiB | 2.80× | ~617 | ~14.7 h | 3.23 GB | 32.3 GB |
| 354M h1024 L24 | 354,224,648 | 8.00 GiB | 3.68× | ~470 | ~19.4 h | 4.25 GB | 42.5 GB |
| 472M h1280 L20 | 471,678,480 | 10.19 GiB | 4.87× | ~355 | ~25.6 h | 5.66 GB | 56.6 GB |
| 1B h1792 L24 *(reference only)* | 1,108,753,700 | — | 11.36× | ~152 | ~59.8 h | 13.31 GB | 133 GB |

**Every throughput, runtime and VRAM figure above except Level 2R's is an ESTIMATE.**
Runtimes come from a FLOP ratio against one measured point and ignore memory bandwidth,
kernel efficiency and occupancy. Treat them as order-of-magnitude. VRAM comes from an
estimator calibrated at ~100M and never checked above it.

### Why 236M

1. **It is the only pure width scaling.** Depth, layout and the 3:1 ratio are untouched, so
   a difference in the result is attributable to width alone. The 269M alternative changes
   depth as well (L=20), and every larger candidate changes both.
2. **Memory is not the constraint; time and storage are.** 472M fits a T4 at batch 4. It
   needs ~25.6 h and 56.6 GB of Drive. 236M at ~12.9 h is the largest step completable in a
   few free-tier sessions.
3. **2.5× capacity is a decisive step size.** Large enough that "no material improvement"
   is an informative negative result, small enough that the token budget per parameter has
   not collapsed to nothing.
4. **The negative case stays interpretable.** At 472M and 0.069 tokens/parameter, a null
   result would be unreadable — indistinguishable from starvation.

The largest number was available and was not chosen.

## 4. Stopping rule — pre-registered

### Initial horizon: 2000 steps

**Not** "because Level 2 did 2000". Because **Level 2R did 2000, and a matched comparison
needs a matched budget**: the same 32,768,000 tokens, the same OneCycleLR shape over the
same horizon, the same data order. `OneCycleLR`'s learning rate at any step depends on
`total_steps`, so changing the horizon changes the LR trajectory and the two runs stop
being comparable at *every* step, not just the last one.

Level 2R's curve justifies this as a *baseline* and explicitly not as an endpoint: it was
undertrained at 2000 (0.826 epochs, 17.4% of the corpus unread, final block at 0.9% of
peak LR). Level 3 will be undertrained too, more so per parameter. That is accepted, it is
what a fixed-budget comparison means, and §6 forbids reading the result otherwise.

### Abort conditions — stop early, record as a result

Stop and write up the failure if **any** holds:

| condition | check |
|---|---|
| loss is NaN or inf | any logged step |
| validation BPB rises at two consecutive eval points | e.g. 1000 → 1200 → 1400 |
| train/validation gap exceeds 0.60 **and** is widening over three eval points | Level 2R stayed in 0.272–0.398 |
| generation is degenerate at the step-1000 checkpoint | `sanity_generate.py`, any prompt collapsing |
| a checkpoint fails destination verification twice at the same step | persistence is broken, not the model |

A mid-run sanity check at **step 1000** is mandatory. Level 2 reached a good-looking BPB
while generating `"and and and"`; a loss curve cannot see that.

### Continuation rule — Stage 2, conditional

Run Stage 2 **only if all four** hold after Stage 1 completes and is analysed:

1. Stage 1 finished 2000/2000 with no abort condition triggered;
2. Level 3 validation BPB at step 2000 is **at least 0.05 BPB below** Level 2R's 1.797
   (i.e. ≤ 1.747) — see the threshold note below;
3. generation at step 2000 is non-degenerate and its mean 3-gram repetition is **not worse**
   than Level 2R's 12.4%;
4. persistent storage has room for the longer run (§7).

Stage 2 extends the horizon to **4000 steps** — 65,536,000 tokens, ~1.65 epochs over the
train split, the first point at which the model has seen the whole corpus. It
**must extend Level 2R to 4000 as well**, or the comparison is void. Both models are
resumed from their step-2000 checkpoints with `total_steps` rebuilt to 4000, which
`qwen_distill.training.resume_compat.rebuild_schedule` supports and all ten verified
Level-2R checkpoints permit. Cost: **~5.3 h for Level 2R plus ~12.9 h for Level 3** — the
additional 2000 steps each, not a retrain.

One property of that has to be stated rather than discovered later. `OneCycleLR` computes
its rate from `total_steps`, so a run that trained steps 0–2000 under a 2000-step schedule
and steps 2000–4000 under a 4000-step schedule has a **hybrid LR trajectory**: it decays to
~0 at step 2000, jumps back up, and decays again. That is not the same curve as a clean
4000-step run. Both models receive the identical treatment, so **they remain comparable to
each other** — which is what the question needs — but neither is comparable to a
from-scratch 4000-step run, and neither should be described as one. Retraining both from
scratch at 4000 steps would be clean and costs ~10.5 h plus ~25.8 h; it is the better
experiment and it is not proposed here on cost grounds.

If condition 2 fails, **do not extend.** A width increase that buys nothing at a fixed
budget is the answer to the question asked, and the next experiment is a data-budget one,
not a bigger model.

### On the 0.05 BPB threshold

It is a **judgment made in advance, not a measured noise floor.** This project has no
run-to-run variance estimate — no seed has ever been repeated. 0.05 is set at roughly
twice Level 2R's largest late-run per-block improvement (0.024) and ~10% of its total
improvement (0.483), so it is comfortably outside step-to-step wobble.

Reading, fixed now:

| Level 3 validation BPB at 2000 | conclusion |
|---|---|
| ≤ 1.747 | **material improvement** from capacity at this budget |
| 1.747 – 1.777 | **inconclusive** at this budget; do not extend, do not claim |
| ≥ 1.777 | **no material improvement**; capacity is not the binding constraint here |
| > 1.797 | width **hurt** at this budget — a real and publishable result |

The cheapest way to replace this judgment with evidence is a **seed repeat of Level 2R**
(~5.3 h, seed 1, everything else identical). If two seeds differ by more than 0.05 the
threshold above is too tight and must be widened before Level 3 is interpreted. This is
recommended and is not a prerequisite.

## 5. Evaluation protocol

Both models are evaluated on **the same held-out corpus** — the same
`data/level2r/validation.txt` bytes, verified by digest. This is what makes validation BPB
a legitimate comparison here, unlike Level 2 vs Level 2R where the corpora differed and
`compare_runs.py` refuses the delta.

**Record the corpus manifest into both run directories** so the comparison tooling can
affirm benchmark identity rather than reporting UNKNOWN:

```bash
python scripts/verify_corpus.py data/level2r --level2r --json corpus_manifest.json
cp corpus_manifest.json experiments/runs/t4_level3_236m_real_english/
```

Then, after the run:

```bash
# 1. curves, throughput at three scopes, plateau, checkpoint timeline
python scripts/analyze_training_run.py experiments/runs/t4_level3_236m_real_english --json

# 2. the controlled comparison
python scripts/compare_runs.py \
    experiments/runs/t4_level2r_100m_real_english \
    experiments/runs/t4_level3_236m_real_english

# 3. generation, identical prompts, identical decoding
python scripts/sanity_generate.py experiments/runs/t4_level3_236m_real_english \
    --training-text data/level2r/train.txt \
    --json experiments/runs/t4_level3_236m_real_english/sanity.json

# 4. checkpoint integrity and behaviour
python scripts/validate_checkpoint.py experiments/runs/t4_level3_236m_real_english --behaviour
```

The ten axes, and where each comes from:

| # | axis | source | comparable? |
|---|---|---|---|
| 1 | held-out validation BPB | `analyze_training_run.py` | **yes** — same validation bytes |
| 2 | training BPB | same | yes — same training bytes and order |
| 3 | generation sanity | `sanity_generate.py`, prompt set v2.0 | yes — same 11 prompts, greedy, 96 tokens |
| 4 | repetition | `repeated_ngram_share`, n=3 and 4 | yes — Level 2R baseline is **12.4% mean 3-gram** |
| 5 | deterministic generation | `validate_checkpoint.py --behaviour` | yes |
| 6 | throughput | `analyze_training_run.py`, run-wide scope | yes — same hardware class |
| 7 | peak VRAM | `benchmark_memory.py` on the real card | **Level 2R never measured this** |
| 8 | parameter count | `count_parameters` | yes |
| 9 | tokens seen | run record | identical by construction: 32,768,000 |
| 10 | training time | run record | yes |

Axis 7 is a gap in the baseline, not in the protocol: Level 2R recorded only the 3.57 GiB
estimate. Measuring Level 3's peak is still worth doing — it is the first real check of the
estimator above 100M — but the two cannot be compared. Measuring Level 2R's peak would
require re-running it and is not proposed.

Axis 4 exists because the sanity checker's degeneracy thresholds were calibrated on Level
2's single-token collapse and do not fire on phrase-level looping. Level 2R scored 0/11
degenerate while 4 of its 11 generations loop. The per-prompt baseline is in the run
README; the number to beat is **mean 3-gram repetition 12.4%**.

## 6. Pre-registered readings

**If Level 3 improves materially (≤ 1.747):** capacity above 94.48M helps at a fixed 32.8M
token budget. It says nothing about how much of the remaining gap is capacity, and nothing
about what either model reaches given more data.

**If Level 3 does not improve (≥ 1.777):** capacity is not the binding constraint at this
budget. The next experiment is about data — a longer horizon at 94.48M — not a bigger
model. This would be a genuinely useful negative result and must be published as one.

**If Level 3 is worse (> 1.797):** at 0.139 tokens/parameter the larger model is too
undertrained to exploit its capacity. Also a real result, and a strong argument against
scaling further before the token budget grows.

**In every case**, the following may not be claimed: that either model is fluent, useful,
instruction-following, benchmark-competitive or teacher-equivalent; that the result
transfers to another corpus or tokenizer; that it predicts anything about distillation,
which is a different objective with a different curve; or that a curve which flattens under
a decaying LR has converged.

Work [POST_RUN_CHECKLIST.md](POST_RUN_CHECKLIST.md) before Level 3 counts as complete.
Reaching `max_steps` is not a result.

## 7. Operating notes

**Storage is the practical blocker.** Ten 236M checkpoints are ~28.3 GB, which exceeds a
free 15 GB Drive tier — and Level 2R's ten already occupy ~11.3 GB. Before starting, either
free the space or plan to prune. `save_every` stays at 200 because raising it trades disk
for work-at-risk, and at ~704 tok/s estimated, 200 steps is already ~1.3 h of exposure.

Pruning is manual and deliberate — delete old `step_NNNNNN` directories on Drive, then:

```bash
python scripts/validate_checkpoint.py <drive-run> --persistent
```

which re-verifies what remains and re-points `latest` at the newest that validates. Keep at
least the two most recent, plus step_002000 once it exists.

**Sessions.** ~12.9 h is several free-tier Colab sessions. The persistence stack is built
for this: every checkpoint is verified at the destination before `latest.json` moves, and
`--restore --resume latest` re-validates and falls back with a stated reason. See
[CHECKPOINT_RECOVERY_GUARANTEE.md](../CHECKPOINT_RECOVERY_GUARANTEE.md).

**Use a fresh persistent directory.** Do not reuse Level 2R's — that run is complete and
its checkpoints are its record.

## 8. Ready-to-run checklist

- [ ] `pip install -r requirements/training.txt`, GPU is a T4
- [ ] `python scripts/prepare_level2r_dataset.py --output data/level2r`
- [ ] `python scripts/verify_corpus.py data/level2r --level2r` → digest prefix
      **`4094c48fdd13266c`**. **If it differs, stop** — the comparison is void.
- [ ] `python scripts/estimate_vram.py --config configs/experiments/t4_level3_236m_real_english.yaml`
- [ ] Drive mounted, fresh directory, ≥ 30 GB free (or a pruning plan)
- [ ] uncomment `persistent_backup` in the config, or pass `--persistent-dir`
- [ ] `python scripts/test_persistence.py --destination <drive-path>` passes
- [ ] the stopping rule above has been read

Then, and only then:

```bash
python scripts/train_student.py \
    --config configs/experiments/t4_level3_236m_real_english.yaml \
    --persistent-dir /content/drive/MyDrive/t4_level3_236m_real_english
```
