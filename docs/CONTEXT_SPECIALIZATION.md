# Context-length specialization

The paper's second major component. Code: `src/qwen_distill/research/context.py`.
Tests: `tests/test_research_context.py`.

## The question

> Does the distribution of sequence lengths seen during distillation change *where* the
> student's context-performance curve breaks, independently of the architecture's nominal
> context window?

The nominal window is a configuration field — 262,144 — and it is free. It says nothing
about whether the model can use those positions. The measurable quantity is the
**context-performance curve**: accuracy on a held-out probe as a function of input length.
Its shape has a knee. This component is about moving that knee.

## Why the hybrid layout makes it interesting

The student is 36 DeltaNet layers and 12 full-attention layers, and the two fail differently
as length grows:

- the **12 attention layers** keep an exact record of every token, at a KV cost linear in
  length — they can look back arbitrarily far, and pay for it;
- the **36 DeltaNet layers** keep a fixed-size recurrent state whose cost does not grow at
  all — they cannot look back arbitrarily far, and do not pay for it.

So a hybrid model's effective context is not one number. It is whatever the 12 attention
layers can still resolve, **plus whatever the fixed DeltaNet state managed to carry
forward**. The second term is learned, and it is learned from whatever lengths the training
data contained. That is the mechanism by which a training-time choice could move a
deployment-time capability, and it is the specific claim under test.

## Regimes

Bands are chosen so a result inside one is attributable. "Long context is worse" is not a
finding; "retrieval survives to 128K but multi-hop reasoning breaks at 32K" is, and it needs
bands that separate the two.

| regime | tokens | what changes | probe |
|---|---|---|---|
| `local` | 0 – 2,047 | everything fits in the attention span and the conv reach; nothing is compressed yet | instruction following, single-turn reasoning |
| `short` | 2,048 – 8,191 | ordinary chat range; the DeltaNet state starts summarising rather than recording | multi-turn coherence, single-file edits |
| `medium` | 8,192 – 32,767 | KV becomes a material fraction of the budget; the state must now discard | document QA, needle retrieval |
| `long` | 32,768 – 131,071 | beyond plausible pretraining length; DeltaNet is outside the regime it was fitted on | multi-document synthesis, repo-scale code |
| `ultra` | 131,072 – 262,144 | RoPE at the edge of extrapolation; KV dominates memory outright | full-window retrieval, position invariance |

Evaluation lengths are one per octave — 2K through 262K — so the curve is legible on a log
axis and the KV cost doubles between adjacent points, letting capability and memory be read
off the same x-axis.

## Curricula

Four arms. B1 is the control; the others each change one thing relative to it.

| arm | policy | tokens at 4K / 16K / 64K / 256K | hypothesis |
|---|---|---|---|
| **B1** | `short_only`, staged | 100% / – / – / – | control: conventional 4K distillation |
| **B2** | `progressive_lengthening`, staged | 4% / 11% / 29% / 57% | length is a curriculum; learn what to keep, then keep it longer |
| **B3** | `length_balanced_mixture`, interleaved | 4% / 11% / 29% / 57% | exposure matters, order does not; short data is never stale |
| **B4** | `long_weighted`, interleaved | 1% / 3% / 22% / 75% | prices the long-for-short trade deliberately |

**B2 and B3 share a token budget exactly** and differ only in ordering, which is what makes
"ordering or exposure?" a controlled comparison. A test asserts the budgets are identical
and that the interleaved schedule really does revisit 4K data after step 800 while the
staged one does not.

**Token share, not step share.** B2 is 40% of *steps* at 4K but under 4% of *tokens*.
Reporting step fractions alone would badly overstate how much short-context data these
schedules contain, so `token_share()` exists and the docs quote it.

## The result schema

`ContextCurve` carries points and derives what the paper reports.

**`effective_context(threshold=0.90)`** — the longest length at which the model still
retains 90% of **its own** short-context score, and at which every shorter length also
passed. Two design decisions:

- *Relative to itself*, so a model that is simply better everywhere does not get credit for
  context handling it does not have. A test confirms a uniformly weaker model with the same
  curve shape reports the same effective context.
- *And every score before it*, because a curve that dips at 32K and recovers at 64K has not
  earned a 64K claim. Taking the maximum passing length would award it one.

It is emphatically **not** `max_position_embeddings`.

`degradation_onset` gives the first failing length. `by_regime` gives the coarse summary for
tables that cannot fit a curve. `direction` handles `lower_is_better` metrics like
perplexity so that rising perplexity is scored as degradation.

## Comparing arms

`compare_curves` returns a blunt `verdict`, and one of its four values is
**`no_measurable_effect`**. That is the outcome which says context specialisation did not
work here, it is reachable, it is tested, and it would be reported. The comparison also
always returns `short_context_delta` and `long_context_delta` separately, because B4's
predicted outcome is a *trade* — better long, worse short — and reporting only the win
would misprice it.

## What refutes the component

If the curves produced by the four curricula are indistinguishable — if a model distilled
only on 4K sequences degrades at the same length as one distilled on a length-balanced
mixture — then context specialisation does nothing for this architecture and gets reported
as ineffective. Nothing in the comparison code is written to favour either outcome.
