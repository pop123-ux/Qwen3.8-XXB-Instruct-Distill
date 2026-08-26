# Level 2R — 94.48M hybrid on real public-domain English

**COMPLETE.** 2000/2000 steps, Tesla T4, 18,963.8 s.

This is the project's **first meaningful language-learning experiment**. Level 1 validated
the training mechanism on synthetic tokens; Level 2 validated the stack on procedural byte
text and produced `"and and and"`. Level 2R changed exactly one thing — the corpus — and
asked whether the architecture can learn real English at this scale.

## The result

> The 94.48M hybrid learns non-trivial natural-language structure from real English and
> avoids catastrophic unigram-style collapse, but its generation remains repetitive and
> semantically weak at this scale.

That sentence is the whole finding. It is deliberately not "fluent", not "useful", not
"instruction-following", not "benchmark-competitive", and not "teacher-equivalent".

## What it generates

Greedy, 96 new tokens, from `sanity.json` (authoritative):

| prompt | continuation |
|---|---|
| `"The "` | `stranger was a little strange to him. He was a little strange to him and he was a little strange` |
| `"In the beginning "` | `of the state of the present day, and the conversation was the same thing that had been so much a` |
| `"Yesterday, I"` | ` should have thought it a little too much to be a man of the service of the property of the pros` |
| `"When the sun "` | `was standing on the steps the street was standing before the street was standing before the stre` |
| `"In the middle of the"` | ` road, and the street was still and the stranger was still and the stranger was still and the st` |

Read those two ways, because both are true:

**What is there.** Real English words throughout — no non-words. Articles, prepositions
and auxiliaries in grammatical positions. Local clause structure that scans
(`"should have thought it a little too much to be a man of"`). Capitalisation after a full
stop. Compare Level 2's output on the identical architecture: `"and and and and"`.

**What is not.** Phrase-level repetition, sometimes locking a whole clause into a loop.
Motif fixation on a handful of nouns — *stranger*, *street*, *thing*, *man*, *property*,
*service*. No semantic thread across clauses. Nothing that could be called content.

`0 / 11 degenerate` from the sanity checker means **not collapsed**, not **not repetitive**.
Its thresholds were calibrated on Level 2's single-token collapse — >50% one token, <15%
distinct words, a repeating character cycle — and phrase-level repetition clears all three.
Distinct-word ratios ranged 0.42–0.80; the three lowest are the looping generations.
Memorisation checking was enabled and found nothing.

## Training

| | |
|---|---|
| parameters | 94,476,448 (reported as 94.48M) |
| layers | 16 — 12 Gated DeltaNet + 4 full attention (3:1) |
| hidden / FFN | 640 / 2176 |
| vocabulary | 256, byte-level |
| sequence × batch × accum | 1024 × 4 × 4 (effective 16) |
| precision / optimizer | fp16 autocast / AdamW, OneCycleLR, peak 6e-4 |
| gradient checkpointing | on |
| steps / tokens | 2000 / 32,768,000 |
| runtime | 18,963.8 s |
| VRAM estimate | 3.57 GiB analytical, on a 14.56 GiB T4, no OOM |

**Corpus** — Project Gutenberg, document-level split (whole works to train or validation,
never both). 44,113,924 bytes total; 39,677,723 train; 4,436,201 validation; digest prefix
`4094c48fdd13266c`. Not committed: regenerate with `scripts/prepare_level2r_dataset.py`
and verify with `scripts/verify_corpus.py`.

## The curve

| step | train BPB | validation BPB | Δ val | share of total |
|---|---|---|---|---|
| 200 | 1.946 | 2.280 | — | — |
| 400 | 1.753 | 2.064 | −0.216 | 44.7% |
| 600 | 1.716 | 1.988 | −0.076 | 15.7% |
| 800 | 1.578 | 1.930 | −0.058 | 12.0% |
| 1000 | 1.512 | 1.894 | −0.036 | 7.5% |
| 1200 | 1.538 | 1.865 | −0.029 | 6.0% |
| 1400 | 1.486 | 1.837 | −0.028 | 5.8% |
| 1600 | 1.454 | 1.813 | −0.024 | 5.0% |
| 1800 | 1.402 | 1.800 | −0.013 | 2.7% |
| 2000 | 1.475 | **1.797** | −0.003 | 0.6% |

Against the 8.0 uniform-byte baseline that is **4.45× compression**. Half the total
improvement arrived by step 600.

## The run is undertrained, not saturated

The last block bought 0.003 BPB — superficially the shape of Level 2's flat tail. The
cause is the opposite, and three independent facts say so:

1. **0.826 epochs.** 32,768,000 tokens against a 39,677,723-byte train split. **17.4% of
   the corpus was never read.** Level 2 had consumed its 8 MB corpus 4.1 times.
2. **The learning rate ran out.** `OneCycleLR(max_lr=6e-4, total_steps=2000)` decays to
   ~0 by construction: the final 200 steps averaged **0.9% of peak LR**, ending at
   2.4e-09. A model that is barely being updated cannot improve, whatever its capacity or
   data. The flattening is confounded with the schedule and this run cannot separate them.
3. **No overfitting.** The train/validation gap sat between 0.272 and 0.398 with no trend.

Validation decreased monotonically through the final block. Nothing plateaued and nothing
turned.

**2000 steps was a budget inherited from Level 2, not a scientific endpoint.** It is a
sound *controlled baseline*, which is exactly how Level 3 uses it.

## Throughput — three scopes, not one number

| scope | value |
|---|---|
| **run-wide** | **1,727.9 tok/s** — 32,768,000 tokens / 18,963.8 s |
| interval (console) | ~2,100 tok/s, CORROBORATED not read from a log |
| Level 2, same architecture and hardware | 2,089.2 tok/s run-wide |

Level 2R took **20.9% more wall-clock for an identical token count**. Nothing about the
model changed, so the difference is real-corpus I/O, validation on a 4.4 MB held-out set,
and ten verified Drive persists. **Level-3 runtime planning uses 1,727.9, not 2,089.2** —
the optimistic figure would understate a Level-3 run by a fifth.

## Checkpoints

Ten persisted, **all ten verified resumable**. The final one:

```
model.safetensors   377,929,584 bytes
optimizer.pt        755,990,779 bytes
checksum            PASS
```

Behavioural checks on `step_002000`: state-dict reload PASS, all tensors bit-for-bit
identical PASS, independent logits identical PASS, greedy generation reproducible PASS,
resume step and history PASS.

Binaries are not committed. See
[CHECKPOINT_RECOVERY_GUARANTEE.md](../../../docs/CHECKPOINT_RECOVERY_GUARANTEE.md).

## What this does not establish

- Not fluent, useful, instruction-following, benchmark-competitive or teacher-equivalent.
- Not that 94.48M is a sufficient or appropriate size for anything.
- Not that the curve converged — it ended with LR at ~0 and 17.4% of the corpus unread.
- Not anything about distillation, the teacher, or knowledge transfer.
- **Not** that 1.797 is comparable to Level 2's 1.270. Different corpora, different
  intrinsic entropy; `scripts/compare_runs.py` refuses that delta and explains why. See
  [level2_vs_level2r.md](../../../docs/experiments/level2_vs_level2r.md).

## What is unknown

Peak VRAM (never measured — only the 3.57 GiB estimate). The git commit the run was
launched from. Where the curve goes with a longer horizon. And whether the repetition is a
capacity limit, a data-budget limit, or an artefact of greedy decoding.

The first of those is what **Level 3** tests:
[level3_plan.md](../../../docs/experiments/level3_plan.md).

## Files

| file | what it is |
|---|---|
| `sanity.json` | **authoritative** generation record, 11 prompts, greedy, 96 tokens |
| `RESULT.json` | the full machine-readable record |
| `README.md` | this file |

`metrics.jsonl` and `summary.json` live on persistent storage with the checkpoints and are
not committed; the trajectory above is the authoritative record supplied with the run.
