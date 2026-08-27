> **STATUS: BOTH RUNS ARE NOW COMPLETE.** This document was written before Level 2R
> finished, to fix the analysis in advance. Its central refusal held:
>
> Level 2 scored **1.270** on procedural text; Level 2R scored **1.797** on real English.
> `scripts/compare_runs.py` still reports **no delta** between them, and it is right to.
> The 0.527 difference is dominated by the corpora, not the models.
>
> The comparison that *did* resolve is the qualitative one this document named as the
> real question — and it resolved in Level 2R's favour: Level 2 generated `"and and and"`
> (83% 3-gram repetition); Level 2R generates English at 12.4% mean 3-gram repetition with
> nothing memorised. The tooling reports exactly that finding from the committed records.
>
> A comparison where validation BPB *is* legitimate — same corpus, same held-out bytes —
> is Level 2R vs Level 3: [level3_plan.md](level3_plan.md) §5.

---

# Level 2 vs Level 2R — how to compare them, decided before the data arrives

**Status:** Level 2 is COMPLETE and published. Level 2R is **RUNNING**. This document is
written *now*, while its final numbers are unknown, so that the analysis is fixed before
anyone can pick the reading that flatters the result.

Run the comparison with:

```bash
python scripts/compare_runs.py \
    experiments/runs/t4_level2_100m_ckpt_complete \
    experiments/runs/t4_level2r_100m_real_english
```

---

## 1. The one comparison that must not be made

Level 2 finished at **validation bits-per-byte 1.270**.

Level 2R will finish at some other number. It will almost certainly be **higher**.

That is not a regression, and the difference between the two numbers must never be
reported as one. They are not measurements of the same quantity:

| | Level 2 | Level 2R |
|---|---|---|
| validation text | contiguous 5% tail of one procedurally generated text | 8 whole public-domain books, held out at document level |
| how it was made | words drawn independently from a fixed Zipfian distribution | written by human authors |
| syntax | none | yes |
| semantics | none | yes |
| long-range dependency | none | yes |
| conditional entropy | **low by construction** — once the frequency table is learned there is nothing left to predict | genuinely high |

Level 2's 1.270 and its `"and and and"` generations are **the same fact seen twice**. The
optimal model for that corpus predicts common words forever, and 1.270 is roughly what
doing so costs. The number was low because the corpus had a low floor, not because the
model was good.

So the floor Level 2R is measured against is a different floor. Subtracting gives a
number with no referent.

> `scripts/compare_runs.py` does not compute this delta. Not "computes it with a
> warning" — the field is `None` and there is no flag to override it.

### What a naïve comparison would have said

At step 75 of Level 2R, the training bits-per-byte was **2.458** — nearly double Level 2's
*final* 1.270. Read as a head-to-head that says Level 2R is failing badly. Read correctly
it says almost nothing yet, except that the model is compressing real English to under a
third of the 8.0 uniform-byte baseline after 75 steps.

---

## 2. What *is* comparable, and why

Level 2R changes **exactly one variable: the corpus**. Architecture, parameter count,
layer layout, sequence length, batch sizes, optimizer, precision, gradient checkpointing,
objective, scheduler, checkpoint and persistence systems are all held at their Level-2
values. The config diff is three lines: `data.text_path`, `data.validation_path`,
`runtime.output_dir`.

That control is what keeps the process comparisons valid.

| metric | scope | across a corpus change | why |
|---|---|---|---|
| run-wide tokens/second | process | **COMPARABLE** | same architecture, same shapes, same GPU class. Byte-level tokenisation means a token is a byte in both runs. |
| peak VRAM | process | **COMPARABLE** | footprint depends on shape, not content |
| parameter count | process | **COMPARABLE** | identical by construction — 94,480,000 both sides |
| training stability | process | **COMPARABLE** | a property of the optimisation |
| generation degeneracy | capability | **COMPARABLE** | "does it emit one token forever?" is corpus-independent — **and it is the question the experiment exists to answer** |
| validation bits/byte | data | **NOT COMPARABLE** | different held-out text, different intrinsic entropy |
| validation loss | data | **NOT COMPARABLE** | the same quantity before the change of base |
| final train bits/byte | data | **NOT COMPARABLE** | different training text; also reflects how often each corpus was consumed |
| steps to plateau | data | **COMPARABLE IF** read as curve shape, never as a ratio | each plateau is defined against that run's own best value |
| epochs over the corpus | data | **COMPARABLE IF** read as a property of each run | 8 MB consumed ~4× vs a far larger corpus consumed once |

The classification lives in code, in `CROSS_CORPUS_RULES`
(`src/qwen_distill/analysis/compare.py`), so it cannot drift away from what the tool
actually does. A rule that refuses a comparison is required to name the remedy that
would restore it — enforced at construction.

---

## 3. What would make the BPB numbers comparable

Evaluate **both** checkpoints on **one** shared held-out corpus:

1. Pick a held-out corpus disjoint from **both** runs' training data — Level 2's
   procedural corpus and Level 2R's Gutenberg training split.
2. Fix its SHA-256 and record it with the result. Two evaluations on "the same books"
   that differ by a header are two different benchmarks.
3. Segment it identically for both models: same sequence length, same stride, same batch
   size. Byte-level loss is a mean over sequences, so segmentation moves it.
4. Evaluate both under the same precision. fp16 autocast and fp32 do not agree to three
   decimals, and that gap is comparable to the effect being measured.
5. Report both numbers with the corpus digest attached, never as a bare delta.
6. State what it still cannot settle: two models trained on different data and scored on
   a third corpus differ by their training data **and** by whatever that corpus favours.

Even done perfectly, this is a weaker claim than it looks. Level 2's model was trained on
text with no syntax; scoring it on English measures how badly a Zipfian frequency model
does on prose. That is a real number, and it is not evidence that the *architecture*
improved — only that the data did.

---

## 4. The comparison that actually answers the research question

**Level 2R exists to answer one question:** *can this hybrid DeltaNet/attention
architecture at 94.48M parameters learn meaningful natural-language structure?*

No bits-per-byte value answers it. Level 2 proved that: 1.270 looked healthy and the
model emitted `"and and and"`.

The answer is qualitative, and it comes from generation:

```bash
python scripts/sanity_generate.py experiments/runs/t4_level2r_100m_real_english \
    --training-text data/level2r/train.txt --json sanity.json
```

Level 2's baseline, from the published record:

| prompt | Level 2 generated |
|---|---|
| `"The "` | `and and and and ...` |
| `"In the beginning "` | `and and and ...` |
| `"It was "` | `and and and ...` |
| `"and the "` | `and and and ...` |

Any Level-2R output that is **not** that is the result. The bar for "learned something" is
low and it is the right bar: word boundaries that survive, function words in plausible
positions, punctuation that opens and closes, clauses that end. The bar for "is a useful
language model" is far higher and 94.48M parameters on ~60 MB of text will not clear it.

`sanity_generate.py` also checks generations against the training corpus for verbatim
memorisation, because a model that has memorised its input produces fluent text and has
learned nothing generalisable.

---

## 5. Pre-registered readings

Fixed in advance so the conclusion cannot be chosen after the fact.

**If Level 2R generates non-degenerate English-shaped text:**
the architecture can learn natural-language structure at this scale. That is the result.
It says nothing about the model being *useful*, nothing about distillation, and nothing
about the teacher.

**If Level 2R still generates degenerate text:**
the corpus was not the (only) limiting factor. Candidate causes, none of them established
by this run alone: 94.48M is too small for byte-level English; 2000 steps × 16,384 bytes
= 32.8 MB of tokens is far too little data; the learning rate or schedule is wrong; or
something in the architecture as implemented does not learn long-range structure. Ruling
between them is another experiment, not a paragraph in this one.

**If validation BPB plateaus early again:**
report *where*, and check `epochs_seen`. A plateau after one pass over 60 MB means
something different from a plateau after four passes over 8 MB. Note the tool will not
say which — a flat curve looks identical whether the model converged, the corpus ran out,
or the learning rate decayed away.

**If throughput differs materially from Level 2's 2,089.2 tok/s:**
that is a real, comparable finding about the data pipeline, since nothing else changed.
Reading real text from disk is not the same work as generating procedural bytes in
memory. Interim in-flight observations were ~2,028–2,062 tok/s, which is the same
neighbourhood.

**In every case:** reaching `max_steps` is not a result. See
[POST_RUN_CHECKLIST.md](POST_RUN_CHECKLIST.md).

---

## 6. Level 2's numbers, for reference

Every value below is **VERIFIED** — measured, and published in
`experiments/runs/t4_level2_100m_ckpt_complete/RESULT.json`.

| quantity | value |
|---|---|
| parameters | 94,480,000 |
| layers | 16 (12 Gated DeltaNet + 4 full attention) |
| steps completed | 2000 / 2000 |
| tokens seen | 32,768,000 |
| total training time | 15,684.6 s |
| run-wide throughput | 2,089.2 tok/s |
| final train loss / BPB | 0.8717 / 1.258 |
| final validation loss / BPB | 0.8806 / 1.270 |
| validation BPB at step 200 / 400 / 1800 / 2000 | 1.317 / 1.279 / 1.271 / 1.270 |
| uniform byte baseline | 8.0 |
| generations | degenerate (`"and and and"`) |
| corpus | procedural byte-level, 8 MB, consumed ~4× |

80% of the run bought under 1% of the BPB improvement: 1.279 at step 400 to 1.270 at step
2000.

### Level 2R in-flight observations

**CORROBORATED, not verified** — reported from a live console during the run, not read by
this analysis from a file. Recorded with their scope stated, because a rate without a
scope is how the Level-2 throughput bug survived to the end of a run.

| step | train loss | train BPB | rate (interval scope) |
|---|---|---|---|
| 25 | 2.5534 | 3.684 | ~2,028 tok/s |
| 50 | 1.9566 | 2.823 | ~2,062 tok/s |
| 75 | 1.7040 | 2.458 | ~2,046 tok/s |

These are **training** figures at `log_every`, not validation, and the rate is the
**interval** rate between log records — not the run-wide figure comparable to Level 2's
2,089.2. They will be superseded by `scripts/analyze_training_run.py` reading the run's
own `metrics.jsonl` once it finishes.

Nothing at step 75 supports any conclusion about whether the model is learning English.
