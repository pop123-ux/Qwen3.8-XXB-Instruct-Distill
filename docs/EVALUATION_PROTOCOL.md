# Evaluation protocol

Rules for producing a number this project is willing to publish. They exist because the
cheapest way to "beat the teacher" is to measure badly, and the second cheapest is to
measure something that quietly changed between runs.

Nothing here has been executed yet. This phase built the machinery; the teacher has
never been loaded and no benchmark has been run.

## 1. Contamination

**Training data must not contain evaluation questions.** The failure is not usually
deliberate — it is a prompt set assembled from a public dataset that a benchmark also
draws from, and it inflates every number afterwards with no symptom.

| rule | enforced by |
|---|---|
| Benchmark prompts are **immutable once published** | `BenchmarkSuite.digest` over ids and prompt content; `BenchmarkSuite.read()` refuses a file whose recorded digest no longer matches |
| Every result records the suite digest it was measured against | `BenchmarkRun.suite_digest` |
| Two runs over different suite contents are **not comparable** | `evaluation.benchmark.comparable()` refuses, rather than returning a table |
| Teacher-generation prompt sets are versioned and hashed | `DatasetManifest.prompt_set_sha256` |
| Every training record names the dataset it came from | `DistillationExample.source`, `dataset_version` |
| Every result records the exact model revision | `BenchmarkRun.model_revision`, `TeacherIdentity.revision` |
| A suite with no contamination statement is not usable evidence | `BenchmarkSuite.validate()` reports it as a problem |

An unpinned model revision is recorded as a **gap**, not waved through: the same repo id
serves different weights over time, so a result from an unpinned id is not reproducible
from that id alone. `RunManifest.gaps()` lists it and `fully_reproducible` goes false.

## 2. Token accounting

The project's central claim will be about cost, so the counts have to mean something
specific.

Four numbers, kept separate:

| number | what it counts |
|---|---|
| `input_tokens` | the rendered prompt, after the chat template |
| `reasoning_tokens` | tokens inside the `<think>`...`</think>` span |
| `answer_tokens` | tokens after `</think>` |
| `total_output_tokens` | `reasoning_tokens + answer_tokens` |

**How the split is computed.** `evaluation.runner.split_thinking` handles the three
shapes that occur: `<think>…</think>answer` (normal), `…</think>answer` (the template
already emitted the opening tag, so the model's output starts inside the block), and no
tags at all (everything is answer). An **unterminated** `<think>` — generation hit the
token limit mid-reasoning — yields an empty answer, which is the correct reading: the
model never answered. Counting it as a short answer would flatter the model.

`DistillationExample.validate()` rejects a record whose parts do not sum to its total,
because every reasoning-cost number downstream is derived from those three fields.

A backend that cannot split reasoning from answer must say so — `TeacherResponse`
carries `token_counting_method`, and the mock's says `whitespace (mock)` so its counts
are never mistaken for tokenizer counts.

## 3. Reasoning modes

Read off the verified chat template, not from documentation:

| mode | how it reaches the template | note |
|---|---|---|
| `thinking_disabled` | `enable_thinking=False` | a different control from `reasoning_effort` |
| `low` | `reasoning_effort="low"` | |
| `medium` | `reasoning_effort="medium"` | **not a no-op** — see below |
| `xhigh` | `reasoning_effort="xhigh"` | the template's default when nothing is passed |

`high` is **rejected**. It reads like it should work and raises inside the template.
`reasoning_modes.REJECTED_MODES` names it explicitly so the failure is answerable.

**`medium` is not a no-op.** A widely repeated secondary claim says it is. The real
template refutes it: `medium` injects no reasoning instruction, which makes it the
*shortest* rendered prompt — precisely because the default it replaces is `xhigh`.

**Prompt-level control is not behavioural control.** Setting a mode changes the rendered
prompt. Whether the model then reasons less is an empirical question that needs
generation to answer, and no claim in this repository may assume it.

## 4. Paired evaluation

Teacher and student answer **exactly the same prompts**, and every pairing is stored per
example. Aggregates are computed from those records rather than replacing them: "matches
accuracy at 40% of the tokens" is a claim about a distribution, and a mean can be
produced by a model that is fine on easy items and catastrophically verbose on hard ones.

Reported side by side, raw before any ratio:

```
  metric                       teacher       student
  accuracy                       80.0%         75.0%
  median reasoning tokens          800           200
  median total output              900           300
  truncation rate                  0.0%          0.0%
```

**No opaque efficiency score.** Every derived metric appears in
`evaluation.paired.METRIC_DEFINITIONS` with its formula. Two that are easy to get wrong:

- `accuracy_per_1k_reasoning_tokens` is **null**, not infinite, when mean reasoning
  tokens is zero. Dividing accuracy by no reasoning is a different question, not a very
  good score.
- `accuracy_at_token_budget` reports **coverage alongside accuracy**. A model that
  answers 10% of prompts within 128 tokens and gets them all right is not a 100% model,
  and one number alone says it is.

An ungraded item returns `None` and is excluded from accuracy rather than counted wrong —
otherwise accuracy silently becomes a measure of how much of the suite has answer keys.

## 5. Reproducibility manifest

Every generation and benchmark run records, via `distillation.provenance.RunManifest`:

git commit and whether the tree was dirty · model path and revision · tokenizer,
chat-template and config hashes · generation config and its hash · reasoning mode ·
dataset manifest and checksums · library versions · hardware · timestamp

`gaps()` lists what would stop someone reproducing the run. A manifest that reports gaps
is still written — the run happened — but the result carries the caveat.

## 6. Storage

| where | what |
|---|---|
| git | code, configs, schemas, manifests, tests, docs, small metadata |
| Drive / artifact storage | teacher generations, logits, checkpoints, large datasets |

Teacher weights are never committed. Teacher-output datasets are never committed. What
git holds is enough to *regenerate* a dataset given the GPU; what Drive holds is the
result of having spent it.
