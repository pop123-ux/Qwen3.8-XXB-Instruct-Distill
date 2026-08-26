# Post-run checklist

**Reaching `max_steps` means the training loop exited. It means nothing else.**

Level 2 reached 2000/2000 steps, validation bits-per-byte 1.270 against an 8.0 baseline,
zero OOMs, every checkpoint verified — and generated `"and and and and"`. Every box that
gets automatically ticked was ticked. The experiment was not complete; it was over.

An experiment is complete when its result is *stated, supported and bounded*: what was
measured, what that establishes, and what it does not. This checklist is the gap between
those two things.

Work through it in order. Each section says what to run, what to record, and what would
make you stop.

---

## 0. Before anything — do not lose the run

- [ ] **Checkpoints are on persistent storage, not `/content`.** A previous Level-2 run
      reached step 1925 of 2000 with no OOM at ~2050 tok/s and lost all of it. The
      checkpoints were written correctly, atomically, to a filesystem that then ceased to
      exist.
- [ ] `python scripts/analyze_training_run.py <run-dir>` reports **0 incomplete staging
      directories** and the newest complete checkpoint is **not far behind** the newest
      log record.
- [ ] `metrics.jsonl`, `summary.json` and `progress/latest.json` are copied off the
      machine. They are kilobytes and they are the entire record.

> Anything below can be redone from the artefacts. Nothing below can be done without
> them.

---

## 1. Did the loop actually do what the config said?

```bash
python scripts/analyze_training_run.py <run-dir> --json analysis.json
```

- [ ] `steps_completed == max_steps`, or the early stop is explained.
- [ ] The **run-wide** throughput is stated with its scope. Not the interval rate, not the
      session rate. If the report shows a `logged_vs_recomputed` disagreement, the logs
      have the resume bug — repair them with `scripts/recompute_throughput.py` and use the
      recomputed figure.
- [ ] The number of **sessions** is recorded. A run that resumed twice is a different
      artefact from one that ran straight through, and its wall-clock includes two
      restarts.
- [ ] `epochs_seen` is recorded. This is the difference between "converged" and "ran out
      of data", and it is the single most useful number for reading a flat curve.
- [ ] The config that ran is the config in git. Compare `config_sha256` in the metrics
      records against the committed file.

---

## 2. Is the checkpoint trustworthy?

```bash
python scripts/validate_checkpoint.py <run-dir>
```

- [ ] Weights reload **bit-identically** — 0 missing, 0 unexpected.
- [ ] Two reloads produce **identical logits** on fixed input.
- [ ] Greedy generation is **reproducible**.
- [ ] Resume from the final checkpoint reaches the same step with history preserved.

A checkpoint that writes without error but reloads into a different model is silent and
invalidates every result produced after it.

---

## 3. Is the model obviously broken?

**This is the step Level 2 skipped, and it is the one that would have caught it.**

```bash
python scripts/sanity_generate.py <run-dir> \
    --training-text <the training corpus> --json sanity.json
```

- [ ] Run against **the final checkpoint**, and at least one **earlier** one. A model that
      was fine at step 400 and degenerate at step 2000 tells you something a single
      snapshot cannot.
- [ ] No prompt collapses to one repeated token.
- [ ] No prompt produces a short repeating cycle.
- [ ] `--training-text` was supplied, so the memorisation check actually ran. Without it
      the report says NOT CHECKED, and that is not the same as "no memorisation".
- [ ] The generations are **pasted into the run's README verbatim**, not summarised. "The
      model produced reasonable text" is not a record.

> Passing means *not obviously broken*. It does not mean good. Failing means broken,
> which is a much cheaper thing to learn.

---

## 4. Does the loss curve say what you think it says?

- [ ] The **plateau step** is recorded, along with what fraction of the run came after it.
      Level 2: 80% of the run bought under 1% of the improvement.
- [ ] Validation is measured on **held-out** text, and you can say which bytes. Record the
      SHA-256.
- [ ] The **baseline** is stated next to the metric. 1.270 bits/byte means nothing without
      "against a uniform-byte baseline of 8.0".
- [ ] You have **not** concluded *why* the curve flattened. Convergence, corpus
      exhaustion, a decayed learning rate and a saturated objective produce the same
      shape. Say when it flattened; say what you checked.

**A good loss on a corpus with a low entropy floor is not a good model.** That sentence is
the entire Level-2 result.

---

## 5. Is the comparison you are about to make valid?

```bash
python scripts/compare_runs.py <baseline-run> <this-run>
```

- [ ] If the corpus changed, **no validation BPB delta is reported**. Different held-out
      text has different intrinsic entropy; the difference between the numbers is
      dominated by the corpora. See [level2_vs_level2r.md](level2_vs_level2r.md).
- [ ] If anything besides the intended variable changed, the comparison is **confounded**
      and says so — including the process metrics.
- [ ] Comparable metrics (throughput, memory, parameter count, stability) are reported
      with their deltas; incomparable ones are reported as two values and no delta.
- [ ] To compare two models' language modelling, evaluate **both** on **one** shared
      held-out corpus, with matched segmentation and precision.

---

## 6. Write the result down

Every run gets `experiments/runs/<name>/RESULT.json` and a `README.md`, with claims in
three tiers:

- [ ] **`establishes`** — what the measurements support. Each item traceable to a number
      in the record.
- [ ] **`does_not_establish`** — what a reader might reasonably infer and should not.
      Level 2's list included *"any general-purpose language capability whatsoever"*.
- [ ] **`unknown`** — what was not measured. Silence reads as absence; say it.

And:

- [ ] Hardware, VRAM, precision, and whether an OOM occurred.
- [ ] Corpus name, size, SHA-256 of both splits, and the split rule.
- [ ] Parameter count — **measured from the built model**, never from the config arithmetic.
- [ ] Every failure and surprise, including ones that were fixed. A run that OOMed and
      then succeeded establishes a bound that the successful run alone does not.
- [ ] No number in the README that is not in the record.

---

## 7. Stop conditions — when the honest answer is "this run is done, and it failed"

Say so plainly and record it. A negative result recorded is worth more than a positive
result asserted.

- **Generations are degenerate.** The run is over. More steps will not fix a corpus or a
  scale problem, and the loss curve will keep looking fine.
- **Validation BPB plateaued in the first 20% of the run.** The remaining 80% is buying
  nothing. Stop it, record where the plateau was, and change something.
- **Generations reproduce training text verbatim.** The model has memorised rather than
  modelled. More training makes this worse.
- **Validation diverges from training loss.** Overfitting. Record the step where they
  separated.
- **The corpus was consumed more than a few times.** Whatever the curve does after that is
  about the data budget, not the architecture.

---

## 8. What "complete" actually requires

An experiment is complete when all of these are true:

1. The training loop finished, or the reason it stopped is recorded.
2. The checkpoint reloads bit-identically and generates reproducibly.
3. Generation has been inspected and the output is written down verbatim.
4. The result is stated in three tiers: establishes / does not establish / unknown.
5. Every number in the write-up traces to a file in the run directory.
6. Any comparison to another run has been checked for validity, not assumed.

Steps 1 and 2 are automatic. **Steps 3–6 are the experiment.**
