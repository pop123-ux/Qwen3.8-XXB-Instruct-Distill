# Evaluation Plan

**Status: nothing has been evaluated. This document specifies how evaluation will
work; it contains no results.**

## Principles

1. **The teacher baseline comes first.** A student number without a teacher number
   measured on the same harness is uninterpretable.
2. **Never mix incompatible numbers.** Published Qwen scores were produced with
   different prompts, harnesses, shot counts, sampling and reasoning settings. They
   are context, not comparanda. We report *our* teacher measurement next to *our*
   student measurement, and cite upstream numbers separately and explicitly labelled.
3. **Reasoning cost is a first-class metric.** Every reasoning-enabled evaluation
   records tokens and latency alongside accuracy. A model that matches accuracy at a
   third of the tokens is a better model, and our reporting must be able to say so.
4. **Long context is a curve, not a checkbox.** Report score vs context length. If
   performance collapses below the advertised window, that is documented.

## Tiers

### Tier 1 — Local, every checkpoint

Cheap enough to run continuously. Purpose: catch regressions fast.

- Training and validation loss; perplexity on held-out data
- Small subsets: math, coding, reasoning, instruction following (~100–200 items each)
- Short long-context probes (needle-in-haystack at 8k/16k)
- Reasoning-token counts on a fixed easy/medium/hard triage set

Not a quality claim — a regression detector.

### Tier 2 — Single cheap GPU, promising checkpoints

Representative benchmark subsets, fixed and versioned so results are comparable
across checkpoints. Purpose: decide whether a checkpoint earns a Tier 3 run.

Must work on a 16 GB card, since that is what we have and what our users have:
quantized evaluation, batch-size auto-detection, resumable runs, JSONL output.

### Tier 3 — Final candidates only

Full suite under the teacher-matched configuration, on rented or borrowed hardware.

## Benchmark categories

Selected once the harness is chosen; candidates by category:

| Category | Purpose |
|---|---|
| General knowledge | broad capability retention |
| Instruction following | practical usefulness |
| Mathematics | reasoning under verification |
| Science / hard reasoning | the capability most at risk from compression |
| Coding | executable verification available |
| Long context | retrieval, multi-hop, synthesis, position robustness |
| Agentic / tool use | where overthinking is most costly |

**Multimodal:** evaluated only if the student preserves vision. If the student is
text-only — the current expectation, since `Qwen3_5ForCausalLM` can load the text
tower alone — then **we must never imply parity with a multimodal teacher.** Any
comparison table states the student is text-only.

## Harness

Prefer mature open-source infrastructure over reimplementing benchmark logic:
lm-evaluation-harness, LightEval, vLLM, llama.cpp, Transformers.

**Verify hybrid-architecture support before committing to a backend.** Gated DeltaNet
needs a recurrent-state cache; a backend that assumes standard KV caching may load the
model and produce silently wrong results at long context. Test each backend on a
retrieval task at a context where the recurrent state matters, and compare against the
Transformers reference path.

## Teacher baseline: `teacher_baseline_v1`

Recorded per run:

```
benchmark, version, split, sample count
prompt template, system prompt
reasoning_effort (including disabled), preserve_thinking
temperature, top_p, top_k, max tokens, seed
context length, quantization, inference engine
GPU, software versions, date, git commit
```

Run at **every** `reasoning_effort` level. This produces the teacher's own
accuracy-vs-tokens curve, which is the reference the student must beat on efficiency
and match on capability.

If the reported `medium`-is-a-no-op behaviour (see `VERIFICATION.md`) is real, it
should be visible here as `medium` and `xhigh` producing near-identical token counts.
That would be worth reporting on its own.

## Paired teacher/student evaluation

For reasoning efficiency, aggregate scores are not enough. Run both models on the
same prompts and record, per prompt:

```
teacher answer / student answer
teacher reasoning tokens / student reasoning tokens
teacher total tokens / student total tokens
teacher latency / student latency
teacher correctness / student correctness
```

This supports statements like *"the student matched the teacher on 94% of items while
using 38% of the reasoning tokens, and lost 6% concentrated in the hardest quartile"*
— which is far more informative than two accuracy numbers.

## Derived metrics

Report with their assumptions stated, never as standalone headlines:

- Capability retention: `S_student / S_teacher` (same harness, same settings)
- Reasoning efficiency: `accuracy / reasoning tokens`
- Latency efficiency: `accuracy / latency`
- Memory efficiency: `benchmark score / peak VRAM GiB`

These are ratios of quantities measured under one configuration. They are useful for
comparing *our own* checkpoints and misleading across papers.

## Contamination

Because this is intended for public release:

- Never knowingly train on benchmark test items.
- Run n-gram/near-duplicate overlap checks between training data and every test set
  used for reporting; commit the results.
- Synthetic teacher data is a contamination vector — the teacher may reproduce
  memorised benchmark items. Filter generated data against test sets too.
- Document dataset provenance, filtering, and generation process alongside results.
- If contamination cannot be ruled out for a benchmark, **do not report a score
  improvement on it.**

## Reporting standard

Every published table states, per row: benchmark version, evaluation method, reasoning
setting, quantization, context length, sample count, hardware, inference engine.

Placeholders are `TBD` or `Not yet evaluated`. **No hypothetical number is ever
written in a form that could be mistaken for a measurement.**

## Planned experiments

| ID | Question |
|---|---|
| A | Can a smaller compatible model learn teacher behaviour via basic SFT? |
| B | Does logit KD improve over SFT at matched compute? |
| C | Does reasoning KD improve hard-task performance? |
| D | Does reasoning KD make the student overthink easy tasks? |
| E | Can shorter solutions be taught for easy tasks without hurting hard ones? |
| F | **How much recall is lost by reducing full-attention layers (interval 4 vs 6)?** |
| G | How does training context length affect the score-vs-context curve? |
| H | Does aggressive quantization harm long-context performance disproportionately? |
| I | Does a larger structurally-efficient model beat a smaller dense one at equal VRAM? |

Experiment F is the highest-priority architecture experiment: the analytical search
prefers interval 6 for reasons that have nothing to do with retrieval quality, and
that preference must be validated or rejected before it influences the design.

Experiments C and D are a matched pair and must be run together — C alone will look
like a success while silently causing D.
