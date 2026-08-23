# Reasoning Efficiency

**Status: research direction. Nothing here has been implemented or measured.**

## The problem

Qwen3.8-27B ships with `reasoning_effort` defaulting to `xhigh`, the most expensive
setting, with `preserve_thinking` on. Community reports describe very long thinking
traces on simple requests, extensive planning before straightforward coding tasks, and
context consumed by reasoning rather than content. One widely-cited example: a simple
SVG drawing request consuming ~22,000 reasoning tokens over ~21 minutes, versus ~3,700
output tokens in ~137 seconds with thinking disabled.

An earlier version of this document repeated a secondary-source claim that **`medium`
is a no-op**. The supplied chat template **refutes it**: `medium` renders a distinct —
in fact the shortest — prompt, because it omits the reasoning instruction that the
default injects. See [`VERIFICATION.md`](VERIFICATION.md) for the rendered-prompt hashes.

What the template *does* establish is that **no control renders identically to `xhigh`**:
a caller who sets nothing receives the high-effort instruction.

That is a fact about the template, not about behaviour. Whether the extra reasoning the
default requests is *justified* by better task performance is exactly what this project
has to measure — and has not measured yet. The community reports above are second-hand
and describe generation behaviour; Phase 2 measures it on our own harness.

## What we are optimising

Not this:

```
minimise thinking tokens
```

This:

```
minimise unnecessary thinking while preserving necessary reasoning
```

The target is a *shape*:

| Task | Desired behaviour |
|---|---|
| "What is 15 × 7?" | answer immediately; near-zero reasoning |
| "Refactor this function and explain the bug" | moderate, focused analysis |
| "Prove this theorem" | extensive reasoning is correct and should not be penalised |

A model that answers everything instantly is a **failure**, not a success, and it will
look like a success on any metric that only counts tokens.

## The measurement that prevents self-deception

Every reasoning-efficiency experiment reports accuracy **stratified by difficulty**,
alongside token counts. Specifically:

- easy-task token reduction (the thing we want)
- **hard-task accuracy delta (the thing we must not lose)**
- the full accuracy-vs-tokens curve, not a single operating point

A result is only an improvement if hard-task accuracy is preserved. A token reduction
accompanied by a hard-task accuracy drop is a **capability regression reported in
efficiency clothing**, and the reporting format is designed to make that visible
rather than hideable.

## Approaches to investigate

Roughly in order of increasing complexity. Prefer the simplest that works.

1. **Budget-conditioned distillation.** Generate teacher responses at multiple
   `reasoning_effort` levels; train the student on the *shortest correct* response per
   prompt. Simple, uses existing controls, no architecture change.
2. **Difficulty-labelled training.** Label prompts by difficulty (teacher agreement,
   solve rate, verifier outcome) and condition the target reasoning length on it.
3. **Explicit budget control tokens.** Give the student a first-class budget interface.
   The teacher's three levels (`xhigh` / `medium` / `low`) are coarse and, since the
   default is `xhigh`, a user gets maximum effort unless they opt out; a finer or
   better-defaulted interface is worth exploring.
4. **Learned budget prediction.** The student predicts the budget it needs before
   reasoning.
5. **Confidence-gated early exit.** Draft an answer, check confidence, allocate more
   reasoning only if needed.
6. **Auxiliary confidence head.** Architectural change; only if 1–5 are insufficient.

Options 5 and 6 add inference complexity that must be supported by the serving stack
we target. A technique that only works in a bespoke harness is not deployable, and
deployability is a project requirement.

## Metrics

Per evaluation item: task accuracy, thinking tokens, answer tokens, total tokens,
time-to-first-token, total latency, tokens/sec, tool calls, context consumed.

Aggregate: accuracy per 1,000 reasoning tokens; accuracy per second; correct answers
per million generated tokens; and the accuracy-vs-token-budget curve per difficulty
stratum.

## The result we would like to be able to state

> Under our evaluation configuration, the student achieved X on benchmark B versus Y
> for the teacher, using Z% of the reasoning tokens on the same task distribution,
> with hard-stratum accuracy within W points.

Every term in that sentence must be measured, on one harness, before it is written.
