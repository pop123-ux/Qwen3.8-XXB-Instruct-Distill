# Level 2 — COMPLETE. 94.48M hybrid, 2000 steps on a Tesla T4.

**Read the three sections below as separate claims.** The engineering worked, the
training measurements are sound, and the model cannot use language. All three are true,
and collapsing them is the only way to get this result wrong.

---

## A. Verified engineering result

What the infrastructure demonstrably does. Independently checked, not inferred.

| | |
|---|---|
| Model | 94.48M params, 16 layers = 12 Gated DeltaNet + 4 full attention |
| Sequence length | 1024 |
| Batch | 4 physical × 4 accumulation = **16 effective** |
| Precision | fp16 autocast |
| Gradient checkpointing | ON |
| Hardware | Tesla T4, 14.56 GiB |
| Steps | **2000 / 2000 completed** |
| OOM | **none** |
| Sessions | multiple Colab sessions, checkpoints persisted to Drive |
| Resume | step 1600 → 2000 **succeeded** |

Final-checkpoint validation, all **PASS**:

- `state_dict` loads into a freshly built model — 0 missing, 0 unexpected
- every parameter tensor bit-for-bit identical after reload
- two independent reloads produce identical logits
- greedy generation reproducible
- resume restores step 2000 correctly, history preserved

**This is the strongest claim in the report.** A hybrid DeltaNet/attention architecture
at ~100M scale trains to completion on free consumer hardware, survives runtime death,
and reloads exactly. Three earlier attempts failed — an OOM at 24.8 GiB demanded, a
disconnect at ~step 500, and a disconnect at step 1925 with nothing persisted. This one
finished.

## B. Measured training result

What the numbers say, taking them at face value.

| step | train loss | train BPB | validation loss | validation BPB |
|---:|---:|---:|---:|---:|
| 200 | — | — | — | 1.317 |
| 400 | — | — | — | 1.279 |
| 1800 | — | — | — | 1.271 |
| 2000 | 0.8717 | 1.258 | 0.8806 | **1.270** |

| | |
|---|---|
| Tokens seen | 32,768,000 |
| Total training time | 15,684.6 s (~4.4 h) |
| **Run-wide throughput** | **2,090 tok/s** |

Loss fell and validation tracked training closely — a gap of 0.009 nats at step 2000, so
no meaningful overfitting. Against the 8.0 BPB uniform-byte baseline, 1.270 is a large
reduction.

**But almost all of that reduction happened before step 400.** From 1.279 at step 400 to
1.270 at step 2000 is 0.009 BPB across 1600 steps — 80% of the run bought under 1% of the
improvement. The model had extracted essentially everything the corpus contains by step
400. That is a statement about the corpus, not about the architecture's capacity.

### Throughput reporting was wrong, and is now fixed

The logs after resume reported 139,256 → 70,945 → 48,117 → 36,623 → … → 10,605 tok/s
against a true 2,090. The 1/n decay was the diagnosis: **cumulative tokens across all
sessions divided by this session's elapsed time**. Reproduced exactly — at step 1625 the
buggy arithmetic gives 135,797 against a reported 139,256, and at step 2000 it gives
10,446 against a reported 10,605.

The measurement was fine; the arithmetic mixed two scopes. Fixed in
`training/throughput.py`, which now reports interval, session and run-wide rates
separately. Existing logs are repairable — every affected record still carries cumulative
`tokens_seen` and `elapsed_s`:

```bash
python scripts/recompute_throughput.py experiments/runs/t4_level2_100m_ckpt/metrics.jsonl
```

## C. Model-quality conclusion

**Level 2 does NOT establish useful general-purpose language capability. It does not come
close, and nothing here should be cited as evidence that it does.**

Deterministic greedy generation from the validated final checkpoint:

```
"The "               -> "and and and and ..."
"In the beginning "  -> "and and and ..."
"It was "            -> "and and and ..."
"and the "           -> "and and and ..."
```

The model emits the highest-frequency token, forever. That is what a language model looks
like when it has learned unigram statistics and nothing else.

**This is the expected result, not a failure.** The corpus was `generate_procedural_text`:
words drawn independently from a fixed Zipfian distribution, assembled into sentences with
punctuation. It has word boundaries, a frequency profile and sentence shape — and **no
syntax, no semantics and no long-range dependency**, because none was ever put in. The
optimal model for that corpus *is* one that predicts common words. 1.270 BPB is close to
what the corpus's own entropy allows.

So the low BPB and the degenerate generation are the same fact seen twice. The model
learned the corpus. The corpus does not contain language.

### What Level 2 establishes, precisely

| claim | status |
|---|---|
| The hybrid architecture trains stably at ~100M on a 16 GB GPU | **established** |
| The training/checkpoint/resume/persistence stack is correct | **established** |
| Byte-level BPB is a working, comparable optimisation target | **established** |
| ~2,090 tok/s on a T4 at seq 1024, batch 4, checkpointing ON | **established** |
| The architecture can learn real-language structure | **not tested** |
| 94.48M is a useful size for anything | **not tested** |
| Anything about distillation, or the teacher | **not tested** |

A model that reaches 1.270 BPB on procedural text and generates `"and and and"` tells you
the optimizer works. It tells you nothing about whether this architecture can model
English, because it was never asked to.

---

## Reproducing

```bash
python scripts/train_student.py --config configs/experiments/t4_level2_100m_ckpt.yaml \
    --persistent-dir /content/drive/MyDrive/qwen-distill/t4_level2_100m_ckpt
python scripts/validate_checkpoint.py experiments/runs/t4_level2_100m_ckpt/checkpoints/step_002000
python scripts/recompute_throughput.py experiments/runs/t4_level2_100m_ckpt/metrics.jsonl
```

## Next

The open question is the one Level 2 could not ask: **can this architecture learn real
language structure at this scale?** See
[`docs/experiments/level2_report.md`](../../../docs/experiments/level2_report.md) for the
recommendation.
