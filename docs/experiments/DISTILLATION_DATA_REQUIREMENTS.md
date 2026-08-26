# What distillation actually needs, before any of it is generated

Nothing in this document has been generated. It exists because the arithmetic below
changes what the project should build, and doing it after generating a dataset would be
the expensive order.

Two findings dominate everything else:

1. **Logit KD from this teacher to a byte-level student is not possible as currently
   designed.** The teacher's vocabulary is 248,320 BPE tokens; the student's is 256 bytes.
   Two distributions over different symbol sets have no KL divergence between them.
   This is a **design blocker, not a storage problem**, and it is unstated anywhere else
   in the repository.
2. **Adopting the teacher's vocabulary to fix that makes the embedding 63% of a
   94.5M-class student.** The fix costs more than the model.

Neither has been decided. Both must be, before a token of teacher output is generated.

---

## 1. The vocabulary mismatch

| | teacher | student (current) |
|---|---|---|
| model | Qwen3.8-27B, 26,895,998,464 params (VERIFIED) | Level-2 hybrid, 94,476,448 params (VERIFIED) |
| vocabulary | **248,320** BPE tokens | **256** bytes |
| a "token" is | a subword | one byte |

Logit KD minimises `KL(P_teacher(·|prefix) ‖ P_student(·|prefix))`. That requires both
distributions over **the same symbol set**. They are not. A 248,320-way distribution and a
256-way distribution are not comparable, and no temperature or weighting fixes it.

`objectives.py` already refuses `logit_kd` and says the blocker is storage — *"full
distributions over a ~248k vocabulary are prohibitive to store, so top-k is the intended
first step"*. That is true and it is not the whole blocker. **Top-k storage does not make
the vocabularies match.** The module's refusal is correct; its stated reason is incomplete.

### The four ways out, with their costs

**A. Give the student the teacher's vocabulary.** Logit KD then works directly.

| student | byte-level total | with tied 248,320 embedding | embedding share |
|---|---|---|---|
| Level 2 (94.5M) | 94,476,448 | **253,237,408** (2.68×) | **63%** |
| 250M class | 236,237,488 | 490,255,024 (2.08×) | 52% |
| 500M class | 471,678,480 | 789,200,400 (1.67×) | 40% |

Untied embeddings double the addition: a 94.5M student becomes 412M, of which **77% is
embedding**. At the sizes this project can train, the student would be mostly a lookup
table. It also abandons byte-level bits-per-byte, so every result so far becomes
incomparable with everything after.

**B. Sequence-level KD — train on the teacher's output text.** No vocabulary alignment
needed: the teacher's text is decoded to bytes and the student learns it as bytes. This is
exactly what `objectives.sft` already implements. It is the only objective that works
today, and it is a real distillation method, not a fallback — but it transfers only the
argmax path, not the distribution.

**C. Cross-tokenizer distillation.** Align teacher tokens to byte spans and project the
distribution onto bytes. Published methods exist; all are approximate, and the
approximation error is unmeasured here. **Would need its own validation experiment before
any result depending on it could be trusted.**

**D. A small shared vocabulary.** Retrain a ~8k–32k BPE tokenizer, run the teacher with it
— impossible without retraining the teacher. Rejected.

**Recommendation: B now, C as a separate research question, A only if the student is
≥1B.** And whichever is chosen, say so in every artifact: a run labelled "distillation"
that is sequence-level KD must not be readable as logit KD.

---

## 2. Storage, if logit KD ever becomes possible

Per response token, at vocab 248,320:

| format | bytes/token | vs full |
|---|---|---|
| full fp16 distribution | 496,640 (485 KiB) | 1× |
| top-256 (int32 id + fp16) | 1,536 | 323× smaller |
| top-128 | 768 | 647× smaller |
| **top-64** | **384** | **1,293× smaller** |
| top-16 | 96 | 5,173× smaller |
| text only (UTF-8) | ~4 | ~124,000× smaller |

Dataset totals:

| response tokens | text only | top-64 | top-128 | full fp16 |
|---|---|---|---|---|
| 1M | 4 MB | 0.38 GB | 0.77 GB | **0.50 TB** |
| 10M | 40 MB | 3.84 GB | 7.68 GB | **4.97 TB** |
| 100M | 400 MB | 38.4 GB | 76.8 GB | **49.7 TB** |

Full distributions are out of the question at every scale. Top-64 at 10M tokens is 3.84 GB
— feasible on Drive, but note it is **96× the size of the text alone**, so the decision to
store logits is the decision that dominates the storage budget.

**k is not free to choose later.** Truncating to top-k discards the tail, and the tail is
where a large model's calibration lives. Whatever k is generated with is the k the dataset
has forever; regenerating means re-running the teacher.

---

## 3. How much data

Unknown, and this project has no basis to estimate it. What is known:

- Level 2 trained on 32,768,000 tokens and produced `"and and and"`. That is a lower bound
  on "not enough", for from-scratch byte-level LM.
- Level 2R's corpus is ~60 MB. Whether that is enough is the question it is running to
  answer, and the answer is not in yet.
- Instruction distillation is a different task from LM pretraining and the budgets do not
  transfer.

**Do not pick a number and defend it.** Generate in tranches, measure after each, and stop
when the curve says to:

| tranche | examples | why |
|---|---|---|
| pilot | 1,000 | shakes out the pipeline, the template, the modes; costs almost nothing |
| first measurement | 10,000 | first point on a data-scaling curve |
| second | 30,000 | second point — a direction, not yet a law |
| third | 100,000 | third point — curvature |

Three points make a direction. Four make an argument. See
[SCALING_STUDY.md](SCALING_STUDY.md) §2 for why fewer does not.

---

## 4. Reasoning modes multiply the token count

The teacher's reasoning modes are `xhigh`, `medium`, `low`, `thinking_disabled`
(`reasoning_modes.py`; `high` is **REJECTED** — it is not a mode this teacher has). Each
produces a different number of tokens for the same prompt, and thinking traces dominate
the total.

Requirements:

- **Record the mode on every example.** A dataset mixing modes without labels cannot
  support the comparison the project exists to make.
- **Decide whether thinking traces are trained on.** Training the student to emit thinking
  it cannot use is a real risk at 100M–500M parameters. Both choices are defensible;
  silently doing one is not.
- **Budget per mode.** `xhigh` can be many times `thinking_disabled` for one prompt, so
  "100,000 examples" means nothing without the mode mix.
- **Generate the same prompts across modes** for at least a subset, so mode is a
  controlled variable rather than confounded with prompt difficulty.

---

## 5. Provenance — required on every example

`distillation/provenance.py` and `manifest.py` already carry this. The requirement is that
nothing is generated without it:

- teacher model id and revision/commit
- exact generation settings: temperature, top_p, top_k, max_new_tokens, seed
- reasoning mode
- the chat template used, by digest — a template change silently changes the data
- prompt source and licence
- the backend that produced it — **and never the mock backend implicitly**
  (`backends.py` already enforces this)
- generation timestamp and library versions
- a digest over the example itself

A dataset without this cannot be reproduced, and a result trained on it cannot be defended.

---

## 6. What must never enter the repository

Per the standing constraint: **the repository must not contain large checkpoint files,
teacher weights, or large teacher-output datasets.**

| in git | not in git |
|---|---|
| generation scripts | generated data |
| the manifest schema | the generated dataset |
| a **manifest** with digests, counts, settings | the shards |
| a small fixture (≤1 MB) for tests | anything above that |
| the prompt list, if licensing allows | teacher weights, ever |

Data lives on Drive or object storage; the repository holds the manifest that says what it
is and how to regenerate it. The same rule that keeps 1.1 GB checkpoints out.

**And: no API keys, no tokens, no credentials, in any file, including manifests.** A
provenance record names the model and the settings, never the key used to reach it.

---

## 7. Contamination

- **Evaluation prompts must never be generated on.** Fix the eval suite and its digest
  *before* generating; `evaluation/benchmark.py` already refuses to compare across a
  changed suite digest.
- **Deduplicate prompts before generation**, not after — the teacher call is the expensive
  part.
- **Record the prompt-source digest** so overlap with a future eval set can be checked
  retrospectively.
- **The Level-2R corpus is public-domain literature and the eval prompts are not drawn
  from it.** If that ever stops being true, it needs stating.

---

## 8. Before generating anything

- [ ] Decide the objective — B (sequence-level) unless something changes — and say so in
      the config, the manifest and the write-up.
- [ ] If logit KD is chosen: settle the vocabulary question (§1) **first**, then k, then
      the storage format. In that order; each decision constrains the next.
- [ ] Fix the eval suite and record its digest.
- [ ] Generate the 1,000-example pilot and inspect it **by reading it**, not by counting
      it.
- [ ] Confirm the storage target has room for the tranche after the one being generated.
- [ ] Confirm provenance is complete on every pilot example before scaling up.
- [ ] Confirm the repository has picked up **none** of the generated data
      (`git status` before committing anything).

---

## 9. Status of every claim here

| claim | status |
|---|---|
| teacher vocabulary is 248,320 | **VERIFIED** — read from the checkpoint config |
| teacher is 26,895,998,464 params | **VERIFIED** — counted from a meta-device build |
| student is byte-level, vocab 256 | **VERIFIED** — the Level-2 config |
| Level 2 = 94,476,448 params | **VERIFIED** — `count_parameters` |
| logit KD needs a shared symbol set | **VERIFIED** — definitional |
| embedding costs above | **VERIFIED** — arithmetic on verified vocabulary and hidden sizes |
| storage table | **VERIFIED** — arithmetic |
| tranche sizes in §3 | **UNKNOWN** — a plan, not a measurement. No data supports these numbers. |
| whether any of this produces a good student | **UNKNOWN** |
