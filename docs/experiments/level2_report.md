# Level 2 complete — what it proved, and what to run next

Full result: [`experiments/runs/t4_level2_100m_ckpt_complete/`](../../experiments/runs/t4_level2_100m_ckpt_complete/).

## Summary in one table

| claim | verdict |
|---|---|
| The hybrid architecture trains stably at ~100M on a 16 GB T4 | **established** |
| Checkpoint / resume / Drive persistence work end to end | **established** |
| ~2,090 tok/s at seq 1024, batch 4, gradient checkpointing ON | **established** |
| The architecture can learn real-language structure | **not tested** |
| 94.48M is a useful model size | **not tested** |

2000/2000 steps, 32,768,000 tokens, 15,684.6 s, no OOM, resumed 1600 → 2000 across Colab
sessions, final checkpoint validated bit-for-bit.

Validation BPB reached 1.270 against an 8.0 uniform-byte baseline — and greedy generation
from that same validated checkpoint is `"and and and and…"`. Both facts are correct and
they are the same fact: the corpus was procedural text with a Zipfian word distribution
and **no syntax, no semantics, no long-range dependency**. The optimal model for it
predicts common words forever.

The tell is in the curve. Validation BPB went 1.317 → 1.279 by step 400, then 1.279 →
1.270 over the remaining 1600 steps. **80% of the compute bought under 1% of the
improvement.** The model had exhausted the corpus by step 400; the rest was the optimizer
finding nothing left to learn.

## The throughput bug

Post-resume logs claimed 139,256 tok/s against a true 2,090, decaying as 1/n to 10,605.
Root cause: **cumulative tokens across all sessions ÷ this session's elapsed time**. The
first log after resuming at step 1600 divided 26.6M tokens by ~196 s.

Reproduced exactly — predicted 135,797 vs reported 139,256 at step 1625; predicted 10,446
vs reported 10,605 at step 2000. The measurement was never wrong; two scopes were mixed.

Now three separate quantities, because collapsing any two is the bug:

| quantity | definition | answers |
|---|---|---|
| interval | tokens since last log ÷ time since last log | did it just get slower? |
| session | this process's tokens ÷ this process's time | what is this GPU doing? |
| run-wide | all tokens ÷ all time, every session | what did the experiment cost? |

Only run-wide is comparable to a headline figure. Existing logs are repairable, because
every record still carries cumulative `tokens_seen` and `elapsed_s`:

```bash
python scripts/recompute_throughput.py experiments/runs/t4_level2_100m_ckpt/metrics.jsonl
```

---

# Recommendation: Level 2R — the same model, real text

**One variable changes: the corpus.**

Level 2 answered "does this architecture train?". It could not answer "can it learn
language?", because the data contained none. The next experiment asks exactly that, and
changing anything else would make the comparison unreadable.

## Hold fixed

94.48M parameters · 16 layers, 12 DeltaNet + 4 full attention · seq 1024 · batch 4 ×
accum 4 · fp16 · gradient checkpointing ON · AdamW · byte-level vocab 256

Every one of these is now a measured quantity on this hardware. Keeping them makes Level
2 the control: same architecture, same budget, same throughput expectation.

## Change

**Corpus: procedural → real public-domain English.** Project Gutenberg is the obvious
source — no licensing question, no download gate, and `load_text_file` already accepts
any UTF-8 file. Target 50–200 MB, well past the ~8 MB procedural corpus, so the model
cannot exhaust it in 400 steps.

**Steps: raise until BPB stops falling, not to a round number.** Level 2's flat tail was
the corpus running out. On real text the curve should keep descending; the run should end
when it stops, and that step count is itself the result.

## What the result will mean

| observation | reading |
|---|---|
| BPB falls well below Level 2's plateau shape and keeps falling | the architecture learns real structure — proceed to distillation |
| generation produces word-like, then phrase-like text | syntax is being acquired at this scale |
| BPB plateaus early again around 1.2–1.5 | ~100M is too small for English, or the recipe is wrong — diagnose before scaling |
| generation stays degenerate despite falling BPB | a real problem: investigate before spending anything on a teacher |

Published byte-level models on English reach roughly 1.0–1.5 BPB, so **a similar number
on real text would mean something entirely different from the same number on procedural
text.** That comparison is the experiment.

## Cost

Same hardware, same throughput. At ~2,090 tok/s, 5000 steps is ~11 hours of T4 time,
across sessions — which the persistence layer now handles. No rented GPU, no teacher, no
new infrastructure.

## Deliberately not next

- **Distillation.** Requires knowing the architecture can learn language at all. Its
  infrastructure is built and tested; it should stay unused until 2R answers that.
- **Scaling past 100M.** Scaling a model that has not been shown to learn language buys a
  more expensive unknown.
- **Architecture changes.** Level 2 measured this architecture. Changing it discards the
  control.
- **Reasoning-efficiency work.** Needs a model with reasoning to be efficient about.

## Command

```bash
# 1. obtain a real corpus (any UTF-8 text file; nothing in-repo needs to change)
# 2. point the config at it, keeping every other field
python scripts/train_student.py --config configs/experiments/t4_level2r_realtext.yaml --dry-run
```

That config does not exist yet — writing it is a one-field change from
`t4_level2_100m_ckpt.yaml`, and it should be written when the corpus is chosen, not
before.
