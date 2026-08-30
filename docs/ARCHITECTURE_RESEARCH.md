# The architecture research loop

The destination is a **Qwen3.8-27B alternative that runs locally on 16 GB of VRAM**, with
a credible path to **12 GB**. Not a small language model in general — a useful
approximation of *that design's* capabilities under consumer VRAM.

This document is how to ask the next architecture question cheaply, before it becomes
expensive to answer.

```
        define an architecture          presets.derive(...)
                  |
        parameter analysis              count_parameters, cross-checked against
                  |                     the model transformers actually builds
        memory analysis                 deployment.assess(...)
             /         \
        16 GB fit?   12 GB fit?         at 4K ... 256K, per precision
             \         /
             train the candidate        only if it can be served
                  |
        evaluate capability             validation BPB, generation, repetition
                  |
        compare against 2R / 3          compare_runs, which refuses invalid deltas
                  |
        analyse scaling gain            returns.build_step(...)
                  |
        choose the next model           or conclude that scale is the wrong lever
```

Everything before "train the candidate" costs milliseconds. That is the point.

---

## 1. Define an architecture

Architectures are **derived from a baseline**, never written from scratch, so the diff
against the thing they are being compared to is explicit:

```python
from qwen_distill.architecture.presets import derive, diff, get_spec

wider = derive("level2r", name="wider", hidden_size=1024, intermediate_size=3456,
               num_attention_heads=16, linear_num_key_heads=6, linear_num_value_heads=18)
diff(get_spec("level2r"), wider)
# {'hidden_size': (640, 1024), 'intermediate_size': (2176, 3456), ...}
```

That dict is the experiment. If it has five entries, the result is attributable to five
things at once — which is why Level 3 changed width only.

Nothing is adjusted for you. A change that makes the GQA head counts indivisible raises,
rather than being rounded into a model that trains but is not the one you asked for.

### The presets

| preset | parameters | what it is |
|---|---|---|
| `prototype` | 4,029,700 | Level 1. Vocab **4096**, synthetic data — its loss is not on the byte-level scale |
| `level2` | 94,476,448 | procedural byte text; validated the stack, no language capability |
| `level2r` | 94,476,448 | **the real-English baseline**, validation 1.797 bits/byte |
| `level3` | 236,237,488 | width-scaled from level2r; **running**, no result yet |
| `teacher` | 26,895,998,464 | Qwen3.8-27B, the reference point |

Presets are **experiments that ran**, plus the teacher. There is deliberately no
`level4`: the next architecture is a decision Level 3's result has to inform, and a
registry that already contained the answer would be pre-empting it. Future variants are
`derive`d, and become presets once they have a result.

```bash
python scripts/architecture_report.py --list
```

Every preset is pinned by a test against its config file, field by field. The first draft
of `prototype` hard-coded a DeltaNet head dim of 64 where the real config uses 32 and
reported 4,574,308 parameters against the record's 4,029,700 — a preset that does not
reproduce its experiment is a second, wrong description of it.

## 2. Parameter analysis

`count_parameters` gives total, embedding and non-embedding counts from the spec alone.
It is cross-checked against the model `transformers` actually builds, on the meta device
so shapes are allocated and storage is not:

```bash
python -m pytest tests/test_presets.py -k analytical_and_constructed
python scripts/validate_analytical_model.py           # the same check, as a report
```

If those ever diverge, every parameter count, memory estimate and feasibility verdict in
the repository is wrong and nothing downstream would notice on its own.

## 3. Memory analysis, and how feasibility is decided

```bash
python scripts/architecture_report.py --presets level3
python scripts/architecture_report.py --presets level3:max_position_embeddings=262144
```

The estimate separates **weights**, **KV cache**, **DeltaNet recurrent state**, **conv
state**, **activations** and **runtime overhead**, at **fp16 / int8 / 4-bit**, across
**4K, 8K, 16K, 32K, 64K, 128K, 256K**.

**Targets.** Two, and only two by default:

| target | nominal | reports | usable | source |
|---|---|---|---|---|
| primary | 16 GB | 14.56 GiB | **13.56 GiB** | measured on the Level-2 T4 |
| secondary | 12 GB | 11.76 GiB | **10.76 GiB** | vendor spec converted; not measured |

`usable` subtracts 1 GiB for driver, display and compositor. A "16 GB" card is not 16 GiB,
and planning against the marketing number is how a configuration that obviously fits OOMs.
A sweep that quietly reported "fits on an A100" would answer a question nobody asked.

**Verdicts.**

| verdict | meaning |
|---|---|
| `FIT` | fits with more than 1.5 GiB spare |
| `BORDERLINE` | fits, with less — one background process takes it away |
| `DOES NOT FIT` | over budget |

**A verdict is always paired with a context length.** `max_fitting_context` is the number
that matters: "fits on 16 GB" is true of a long-context model at 4K and can be false at
256K, and for this project those are different models. An architecture that declares a
short `max_position_embeddings` has its ladder truncated and is reported as **UNKNOWN
above that**, not as good — estimating 256K for a model that declares 4K would describe a
model that cannot run there.

**Every figure is an estimate.** The analytical model is not a benchmark. Confirm with
`scripts/benchmark_memory.py` on the real card before committing GPU hours.

## 4. Sweep before you train

```bash
python scripts/architecture_report.py --sweep level2r level3 teacher
```

One row per candidate: parameters, **training** memory at the Level-2R recipe, and the
16 GB and 12 GB status with the best precision and longest context each supports.

Training memory is shown because at these sizes it is the binding constraint and
inference is not — 236M serves in ~1.4 GiB and needs ~6.2 GiB to train. A sweep that
showed only inference would hide the constraint that actually decides what can be run.

Nothing in a sweep is trained. Rejecting a candidate here costs milliseconds; discovering
the same thing eight hours into a run costs eight hours.

## 5. Create an experiment

Copy the nearest config, change the architecture block, and change nothing else you do
not have to. `configs/experiments/t4_level3_236m_real_english.yaml` is the worked example:
its training and data blocks are byte-identical to Level 2R's, verified by a test, so the
only difference is width.

**The corpus is part of the experiment's identity.** Level 2R's corpus is a controlled
dataset with a recorded digest (`4094c48fdd13266c`). A future scaling experiment must use
the same bytes, or its validation BPB is not comparable and the comparison is void:

```bash
python scripts/prepare_level2r_dataset.py --output data/level2r
python scripts/verify_corpus.py data/level2r --level2r
```

If an experiment changes the corpus, that has to be explicit and deliberate — otherwise
`architecture A + corpus X` versus `architecture B + corpus Y` gets reported as an
architectural improvement, which is the single easiest way to publish a wrong result here.

## 6. Compare, and analyse the gain

```bash
python scripts/architecture_report.py --summary \
    experiments/runs/t4_level2r_100m_real_english \
    experiments/runs/t4_level3_236m_real_english
```

Produces the **ARCHITECTURE RESEARCH SUMMARY**: each rung's measured capability and
estimated deployment cost, then the step between them — parameter ratio, relative
improvement, memory increase, and a conclusion drawn from those numbers.

Three things it will not do:

- **No universal capability score.** There is no single number for how good a language
  model is, and inventing one lets a bad architecture win by construction. Comparisons
  are per-metric, and an unmeasured metric stays absent.
- **No comparison across validation corpora.** A step whose two runs disagree on the
  validation bytes has its delta **refused**, not annotated.
- **No assumption that bigger is better.** "2.5× the parameters for 0.4% of the loss" is
  reported as diminishing returns and as an argument for a different lever.

The materiality threshold is **2% relative improvement**, and it is a **stated judgment,
not a measured noise floor** — this project has never repeated a seed, so run-to-run
variance is unknown. A seed repeat of Level 2R (~5.3 h) would replace the judgment with
evidence and is the cheapest way to make every later comparison sharper.

## 7. What an experiment must record

`RESULT.json` per run. Missing values stay missing; nothing is fabricated.

| | |
|---|---|
| architecture | every field, plus measured parameter count |
| training | steps, tokens, sequence length, batch, accumulation, optimizer, scheduler, precision |
| data | corpus name, byte counts, **digest**, split rule |
| capability | validation BPB, train BPB, generation record, repetition |
| deployment | inference estimate, 16 GB and 12 GB feasibility |
| provenance | git commit, hardware, checkpoint verification |

The deployment fields can be produced for any architecture without training it, so a
future experiment should carry them from the start:

```python
from qwen_distill.analysis.deployment import assess
assess(spec).summary_row()
```

## 8. What is still missing

Stated plainly, because infrastructure that hides its gaps is worse than none:

- **No capability evaluation beyond bits-per-byte and generation sanity.** The evaluation
  package has the harness, task and benchmark scaffolding; no benchmark has been run and
  no benchmark data is committed. Adding a real task suite is the next evaluation step,
  and `evaluation/benchmark.py` already refuses to compare results across a changed suite.
- **No measured peak VRAM for any run.** Level 2R recorded only the 3.57 GiB estimate.
  The first `benchmark_memory.py` run on a real card is also the first check of the
  estimator above 100M.
- **No run-to-run variance.** See the materiality threshold above.
- **No long-context measurement.** Every student so far declares 4K, so the long-context
  column is UNKNOWN for all of them. That is a property of the configs, not the estimator.
