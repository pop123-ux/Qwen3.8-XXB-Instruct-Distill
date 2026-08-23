# Teacher Baseline

**Status: NOT YET RUN. This document contains no measurements.**

The teacher checkpoint was not reachable from the authoring environment and this
machine has no GPU (see [`VERIFICATION.md`](VERIFICATION.md)). The pipeline below is
built, tested end to end against a synthetic checkpoint, and ready to run.

**Metadata-only verification no longer needs network access from this repository.**
Supply `vendor/qwen38-metadata/` (see [`vendor/README.md`](../vendor/README.md)) and the
config, tokenizer, chat template and reasoning controls can all be verified offline. The
baseline *measurements* below still require the weights and a large GPU.

## Why the baseline comes first

The student will eventually be compared against the teacher. If the teacher's numbers
come from a different harness, prompt format, sampling configuration or reasoning
setting than the student's, the comparison is meaningless — and the failure is silent,
because both numbers look fine on their own.

So the teacher is the **measuring instrument**, and it gets calibrated before anything
is measured with it. Published Qwen scores are *context*, never comparanda: they were
produced with different prompts, shot counts and reasoning settings, and mixing them
with ours would be exactly the kind of invalid comparison this project has committed
to avoiding.

## Configuration

[`configs/evaluation/teacher_baseline_v1.yaml`](../configs/evaluation/teacher_baseline_v1.yaml)
holds every value. Nothing is hard-coded in Python. It pins model revision, dtype,
quantisation, tokenizer source, backend, sampling parameters, seed, context length,
reasoning modes, suites, and the per-generation fields recorded.

Two fields deserve attention:

- **`revision: null`** — must be pinned to a commit SHA before the real run. An
  unpinned baseline is not reproducible.
- **`backend.validated_against_reference: false`** — vLLM is much faster than
  `transformers`, but a backend that mishandles the DeltaNet recurrent state can
  produce perfectly plausible short-context output and *wrong* long-context output.
  That failure is quiet. Until vLLM is checked against the `transformers` reference on
  a long-context retrieval task, `transformers` is the baseline backend.

## The four modes

The project's premise is that reasoning budget matters, so a single "teacher score" is
not a meaningful baseline. Four modes are run:

| Mode | Setting | Why |
|---|---|---|
| A | thinking disabled | the floor: capability with no reasoning at all |
| B | no control set | what a user actually gets by default |
| C | `reasoning_effort: low` | cheapest explicit budget |
| D | `reasoning_effort: xhigh` | most expensive explicit budget |

`medium` is swept separately by `benchmark_reasoning.py` rather than given its own
mode, because it is reported to be a no-op and the template diff settles that before
any generation runs.

Mode B matters most for the project's motivation: if the default is `xhigh`, then the
overthinking users complain about is the *out-of-the-box* behaviour, not something
they opted into.

## Evaluation suites

| Suite | Contents | Purpose |
|---|---|---|
| `tier1` | reasoning dev set + short long-context probes | cheap regression detection, run continuously |
| `reasoning` | difficulty-stratified dev set | the reasoning-cost curve |
| `long_context` | needle-in-haystack across lengths × depths | retrieval vs context length and needle position |

The reasoning dev set is **hand-written for this project** — 19 items spanning
`trivial` → `very_hard`, each with a mechanical checker. Nothing is copied from a
public benchmark, so it cannot leak a test set into training data. It is a
*development* set: it measures reasoning cost, not capability. Capability numbers need
a real benchmark.

## Recorded per generation

```
task_id, category, difficulty, reasoning setting,
prompt_tokens, thinking_tokens, answer_tokens, total_generated_tokens,
latency_s, tokens_per_second, correct
```

Thinking and answer tokens are split **on token IDs** at `</think>`, not by
re-tokenising decoded text — re-tokenisation is lossy and the parts need not sum to the
whole. An unterminated `<think>` yields an empty answer, which is the correct reading:
the model never answered.

## Determinism

- Greedy decoding (`temperature: 0.0`) and a fixed seed.
- `--limit` takes a **stable prefix**, never a shuffle, so two runs of the same command
  are comparable.
- Every run records its task IDs, so subset selection is auditable after the fact.

## Running it

```bash
# 0. prove the backend works. an import succeeding is not evidence.
python scripts/evaluate.py --model Qwen/Qwen3.8-27B --probe-only

# 1. the four modes
python scripts/evaluate.py --model Qwen/Qwen3.8-27B --suite tier1 --no-thinking \
    --output evaluations/baselines/teacher/mode_a_no_reasoning
python scripts/evaluate.py --model Qwen/Qwen3.8-27B --suite tier1 \
    --output evaluations/baselines/teacher/mode_b_default
python scripts/evaluate.py --model Qwen/Qwen3.8-27B --suite tier1 --reasoning-effort low \
    --output evaluations/baselines/teacher/mode_c_low
python scripts/evaluate.py --model Qwen/Qwen3.8-27B --suite tier1 --reasoning-effort xhigh \
    --output evaluations/baselines/teacher/mode_d_xhigh

# 2. the reasoning sweep, including the medium no-op check
python scripts/benchmark_reasoning.py --model Qwen/Qwen3.8-27B \
    --output evaluations/baselines/teacher/reasoning
```

Each run writes `benchmark_results.json`, `metadata.json` (including a full hardware
report) and `generations.jsonl`. Results are committed; `generations.jsonl` is
gitignored, because bulk generated text does not belong in git.

## Hardware requirement

The teacher does not fit 16 GB — its Q4_K_M weights alone are 15.85 GiB
([`ARCHITECTURE.md`](ARCHITECTURE.md)). Baselining needs a larger GPU, rented or
borrowed. This is a **one-time cost**: the baseline is produced once, committed, and
reused for every subsequent student comparison.

## Results

| Mode | Suite | Accuracy | Mean thinking tokens | Mean latency | Status |
|---|---|---|---|---|---|
| A — no reasoning | tier1 | TBD | TBD | TBD | Not yet measured |
| B — default | tier1 | TBD | TBD | TBD | Not yet measured |
| C — low | tier1 | TBD | TBD | TBD | Not yet measured |
| D — xhigh | tier1 | TBD | TBD | TBD | Not yet measured |
