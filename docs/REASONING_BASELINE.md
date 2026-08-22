# Reasoning Baseline

**Status: NOT YET MEASURED against the real teacher.** The measurement pipeline is
built and validated end to end against a synthetic checkpoint; the teacher was not
reachable (see [`VERIFICATION.md`](VERIFICATION.md)).

## What is being established

The claim motivating this project is that Qwen3.8-27B spends far more reasoning than
its tasks require, by default. That claim currently rests on community reports. Before
building a student around fixing it, we measure it ourselves.

Two checks, cheapest first, because they answer different questions and can disagree.

## Check 1 — Template diff (free, no weights, no GPU)

```bash
python scripts/benchmark_reasoning.py --model Qwen/Qwen3.8-27B --template-only
```

Renders the chat template at every reasoning setting and hashes each result. If two
settings produce **byte-identical prompts**, one is a no-op *by construction* — not
"appears ineffective", but cannot possibly differ, because the model receives the same
input. That is a proof, and it needs only the tokenizer.

This is the decisive test for the reported `medium` behaviour (question 19). Sample
output, from the detector running against a fixture built to reproduce that behaviour:

```
setting               sha256[:12]       chars
(default)             fe335e0579dc         71
low                   ddf2edee1799         85
medium                fe335e0579dc         71
xhigh                 a586e3ffe4f7        130
thinking_disabled     0e6506e4c9e4         64

distinct prompts    : 4 of 5
IDENTICAL PROMPTS   : (default) == medium
=> these settings are indistinguishable at the template level:
   selecting one over another cannot change behaviour via the prompt.
```

**That output is from a synthetic fixture, not from Qwen3.8.** The fixture template was
written to reproduce the reported behaviour so the detector could be tested against a
known-positive case. It demonstrates the tool works; it says nothing about the real
model until run against the real tokenizer.

## Check 2 — Generation sweep

```bash
python scripts/benchmark_reasoning.py --model Qwen/Qwen3.8-27B \
    --output evaluations/baselines/teacher/reasoning
```

Runs the difficulty-stratified dev set at each setting and compares measured thinking
tokens. Needed because a control can change the prompt yet barely change behaviour, or
(for a model trained on control tokens) change behaviour without changing the prompt.
`indistinguishable_settings()` flags any pair whose mean thinking cost differs by less
than 5%.

## The shape we are looking for

Not "fewer tokens". A *proportional* curve:

| Difficulty | Desired reasoning cost |
|---|---|
| trivial ("what is 15 × 7?") | near zero |
| easy | brief |
| moderate | focused analysis |
| hard | substantial |
| very hard (a proof) | extensive — and **correctly so** |

A model that answers everything instantly is a failure. The dev set therefore contains
`very_hard` items where long reasoning is the right behaviour, so that a
reasoning-suppression regression shows up as lost accuracy rather than looking like a win.

## The guard against self-deception

`qwen_distill.evaluation.metrics.compare` encodes the rule from
[`reasoning-efficiency.md`](reasoning-efficiency.md) as executable logic:

```
efficiency_win = thinking_token_ratio < 1.0 AND hard_stratum_accuracy_delta >= -0.02
```

A student that uses 40% of the teacher's reasoning tokens but loses hard-task accuracy
returns `efficiency_win: False`, and `paired_eval.py` prints:

> WARNING: the student used fewer reasoning tokens but hard-task accuracy dropped.
> This is a capability regression, not an efficiency win.

This is deliberately not left to the reader's judgement. The temptation to report a
token saving as a success is exactly what the metric has to resist, so the rule is
enforced in code and covered by tests
(`test_compare_rejects_token_saving_that_costs_hard_accuracy`).

The ±2% tolerance absorbs sampling noise on a small set. It is not a licence to lose
capability: on the current 19-item dev set a single hard item is worth far more than
2%, so any real hard-task regression fails the check.

## Results

| Setting | Mean thinking tokens | Accuracy | Trivial-item thinking tokens | Hard-item accuracy | Status |
|---|---|---|---|---|---|
| (default) | TBD | TBD | TBD | TBD | Not yet measured |
| low | TBD | TBD | TBD | TBD | Not yet measured |
| medium | TBD | TBD | TBD | TBD | Not yet measured |
| xhigh | TBD | TBD | TBD | TBD | Not yet measured |
| thinking disabled | TBD | TBD | TBD | TBD | Not yet measured |

**Template-level `medium` no-op:** Not yet checked against the real chat template.
