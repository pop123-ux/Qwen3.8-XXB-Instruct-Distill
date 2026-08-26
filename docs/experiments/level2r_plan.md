# Level 2R — the same 94.48M model, on real English

**Prepared, not run.** Everything below is ready to launch; nothing has been trained.

## The question

> Can this 94.48M-parameter hybrid DeltaNet/attention architecture learn meaningful
> natural-language structure from a sufficiently large corpus of unseen public-domain
> English?

Level 2 could not ask it. Its corpus was procedural: words drawn independently from a
fixed Zipfian distribution, with word frequencies and no syntax. It reached validation
BPB 1.270 and generated `"and and and"` — the same fact seen twice, because the optimal
model for that corpus predicts common words forever.

The curve says it too. Validation BPB went 1.317 → 1.279 by step 400, then 1.279 → 1.270
over the remaining 1600 steps. **80% of the run bought under 1% of the improvement.** The
corpus was exhausted; the architecture was never tested.

## Level 2 vs Level 2R

| | Level 2 | Level 2R |
|---|---|---|
| Corpus | 8 MB procedural byte text | real public-domain English, tens of MB |
| Split | contiguous tail of one text | **whole documents**, held out |
| Purpose | verify architecture, stability, memory, checkpoint/resume | **does it learn language?** |
| Result | infrastructure works; no language capability | not yet run |

Level 2R is **not** the final model and **not** a benchmark claim. It is one controlled
experiment with one variable changed.

## What is held fixed

94.48M parameters · 16 layers, 12 DeltaNet + 4 full attention · hidden 640 · FFN 2176 ·
seq 1024 · batch 4 × accum 4 (effective 16) · fp16 autocast · AdamW · gradient
checkpointing ON · byte-level vocab 256 · OneCycle schedule · the checkpoint, persistence
and resume systems.

The config diff against Level 2 is exactly three lines:

```
data.text_path        None -> data/level2r/train.txt
data.validation_path  None -> data/level2r/validation.txt
runtime.output_dir    .../t4_level2_100m_ckpt -> .../t4_level2r_100m_real_english
```

Level 2 is the control, and it stops being one the moment anything else moves. The
`--dry-run` estimate is unchanged at **3.57 GiB** of a T4's 13.56 GiB budget.

## The corpus

Project Gutenberg English prose — pre-1929 publications whose copyright has expired,
distributed by Gutenberg as public domain. ~50 training works across a range of authors,
periods and registers; **8 works held out entirely**.

**The corpus is never committed to git.** `scripts/prepare_level2r_dataset.py`
reconstructs it from catalogue ids and writes a manifest recording every hash, so a
result is tied to the exact bytes that produced it.

### Normalisation, and why each step exists

| step | reason |
|---|---|
| Gutenberg licence header/footer removed | identical across every book; would be the most predictable text in the corpus |
| residual `Title:`/`Author:`/`Release Date:` lines dropped from the top | metadata, not prose |
| Unicode NFC | the same character as one code point or two is different bytes; modelling both wastes capacity on an encoding artefact |
| CRLF/CR → LF | otherwise the same book hashes differently per machine and the corpus stops being reproducible |
| trailing whitespace stripped, 3+ blank lines collapsed to 2 | Gutenberg spacing is irregular; long blank runs are trivially predictable filler that would flatter BPB |

**Deliberately not done:** lowercasing, punctuation stripping, sentence splitting. The
experiment is whether the model learns English as written.

### The split

**Document level.** Whole works go to train or validation and never both.

Level 2 held out a contiguous tail of one concatenated text — that measures how well a
model continues a passage it has been reading. Holding out whole books measures
generalisation to prose it has never seen: different author, different subject, different
century. It is a harder and far more meaningful target, and the number it produces means
something a tail-split number does not.

The assignment is by **explicit catalogue id**, not by fraction. A fraction shifts when
the document list is edited; a named set cannot. The split is then written once into
`train.txt` and `validation.txt` — nothing is recomputed at train time, so no seed,
fraction or ordering can change the validation set between Colab sessions. It is fixed by
construction rather than by convention.

Preparation also samples validation passages and checks none appears verbatim in the
training text. A document-level split should make that impossible, so a hit means a work
is duplicated under two ids.

### Contamination

- Literary prose only. No benchmark dataset, question bank, exam set or evaluation suite
  was included, deliberately.
- Validation works are by authors held out of training wherever possible, so a low
  validation BPB cannot be explained by having learned one author's habits.
- These are famous works a large pretrained model would likely have memorised. Irrelevant
  here — this model trains from scratch on this corpus alone — but it would matter if
  these texts were ever reused to evaluate a *distilled* student whose teacher saw them.
  Noted now so it is not discovered later.

## Training length

**2000 steps is a starting point, not the endpoint.** Level 2's 2000 was an arbitrary
round number and 80% of it bought nothing.

There is no early-stopping mechanism, and none was invented. The procedure is manual:

1. Train. Validation runs every 200 steps, as in Level 2.
2. Watch validation BPB and its **rate of improvement**, not its absolute value.
3. Run `scripts/sanity_generate.py` on checkpoints — generations degrade or improve well
   before BPB moves much.
4. Still improving? Resume with a higher `--max-steps` and continue, across sessions.
5. Clearly plateaued? Stop.

**Where it plateaus is itself the result.** Level 2 plateaued at ~400 steps on 8 MB. If
Level 2R plateaus at a similar BPB despite a much larger corpus, that is a finding about
the architecture at this scale — and a far more useful one than a round step count.

At the measured **2,089 tok/s** and 16,384 tokens/step, 1000 steps is **2.18 h**. A 50 MB
training corpus is one epoch per ~3,050 steps.

## Generation sanity checks

Level 2's central lesson: a healthy loss curve and useless generation coexist perfectly.

`scripts/sanity_generate.py` generates greedily from six fixed prompts and looks for the
failure modes a byte-level model at this scale actually produces: one token forever
(Level 2's exact failure), a short repeating cycle, collapse to a handful of characters,
empty output, and verbatim reproduction of training text.

It is **not a benchmark**. Passing means *not obviously broken*; it does not establish
language capability. Failing means something is wrong, cheaply and early.

## Commands

```bash
export DRIVE=/content/drive/MyDrive/qwen-distill/t4_level2r_100m_real_english

# 1. build the corpus (needs network; run in Colab)
python scripts/prepare_level2r_dataset.py --output data/level2r

# 2. check it fits before spending GPU time
python scripts/train_student.py --config configs/experiments/t4_level2r_100m_real_english.yaml \
    --persistent-dir "$DRIVE" --dry-run

# 3. train
python scripts/train_student.py --config configs/experiments/t4_level2r_100m_real_english.yaml \
    --persistent-dir "$DRIVE"

# 4. from a fresh Colab session: what survived, then continue
python scripts/train_student.py --config configs/experiments/t4_level2r_100m_real_english.yaml \
    --persistent-dir "$DRIVE" --status
python scripts/train_student.py --config configs/experiments/t4_level2r_100m_real_english.yaml \
    --persistent-dir "$DRIVE" --restore --resume latest

# 5. continue past the initial horizon while BPB is still falling
python scripts/train_student.py --config configs/experiments/t4_level2r_100m_real_english.yaml \
    --persistent-dir "$DRIVE" --resume latest --max-steps 6000

# 6. inspect what it actually writes
python scripts/sanity_generate.py experiments/runs/t4_level2r_100m_real_english \
    --training-text data/level2r/train.txt
```

## Reading the result

| observation | reading |
|---|---|
| BPB keeps falling well past Level 2's plateau shape | the architecture is learning real structure — proceed toward distillation |
| generations become word-like, then phrase-like | syntax is being acquired at this scale |
| BPB plateaus early again at a similar value | ~100M is too small for English, or the recipe is wrong — diagnose before scaling |
| BPB falls but generations stay degenerate | a real problem; investigate before spending anything on a teacher |

Published byte-level models on English reach roughly **1.0–1.5 BPB**. Level 2 reached
1.270 on *procedural* text. **A similar number on real English would mean something
entirely different**, and that comparison is the experiment.

## Caveats

- **Validation BPB across a document-level split is not comparable to Level 2's
  tail-split BPB.** Different measurement, harder target. Comparing the two numbers
  directly would be wrong.
- The corpus is literary prose, weighted toward 19th-century fiction. A model trained on
  it learns that register, not modern English generally.
- Byte-level sequences cover ~4× less text per token than BPE, so a 1024-byte window is
  roughly a 250-token window. Long-range structure beyond that is out of reach by
  construction.
- Some works are translations (Tolstoy, Dostoyevsky, Dumas, Homer). Translated English is
  still English, but it is a distinct register.
- The catalogue ids are believed correct but were **not verified against the live
  service** — this environment has no access to it. Preparation records the title each
  file actually declares, so a wrong id shows up in the manifest rather than silently
  substituting a different book. **Check the manifest after preparing.**
