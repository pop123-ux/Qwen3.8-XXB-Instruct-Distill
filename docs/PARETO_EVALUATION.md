# Pareto evaluation and the 16 GB constraint

Code: `src/qwen_distill/research/memory.py`. Tests: `tests/test_research_memory.py`.
Report: `python scripts/student_report.py --section memory`.

## The constraint

16 GB, **end to end, fully GPU-resident**. The complete workload:

```
weights + KV cache + DeltaNet recurrent state + conv state + activations + runtime overhead
```

A quantisation table showing 11 GiB of weights and stopping there has answered a question
nobody deploys against. Two terms are frequently omitted and are the reason "it fits" turns
into an OOM: the ~0.9 GiB a PyTorch+CUDA process costs before a single model tensor is
allocated, and the fp32 logits over a 248,320-token vocabulary.

**The ceiling is not 16.0 GiB.** A 16 GB card reports **14.56 GiB** (measured on the
Level-2 T4), and a process that must coexist with a display server or another CUDA context
should not plan on the last gigabyte. Usable: **13.56 GiB**.

**CPU offload is never used to claim compliance.** An offloaded configuration has different
latency and is a different product. If one ships, it is reported separately and labelled.

## The three terms the hybrid layout changes

**KV cache — 12 layers, not 48.** `2 (K and V) x 2 kv heads x 256 head_dim x 12 layers` =
**24,576 bytes per token** in fp16. A dense 48-layer model with the same head configuration
would cost 98,304 bytes/token. At the full 262,144-token window that is 6.00 GiB instead of
24 GiB, and it is the hybrid layout's entire deployment argument.

**Recurrent state — constant.** `36 layers x 48 value heads x 128 x 128` in fp32 =
**108 MiB**, identical at 2K and at 262K. Shapes verified against a running model of this
architecture family, not assumed, and confirmed unchanged between a 32-token and a
128-token forward pass.

**Conv state — 5.6 MiB.** `(2 x key_dim + value_dim) x kernel` per DeltaNet layer, using the
runtime's own `conv_dim` formula, confirmed against the built module.

All state together is smaller than a single octave of KV cache at 8K.

## Weights are bucketed, not uniform

Routed experts are 34.8% of the model and each token touches 2 of 8, so `expert_quant` is a
separate knob from `dense_quant` (attention, DeltaNet, router, norms — the always-active
path) and `embedding_quant`. Quantising the experts harder moves more memory per unit of
damage than a uniform setting.

They were **61.6%** before the expert-budget correction, and that is precisely what made the
first implementation undeployable. Weight counts come from the parameter audit rather than
being recomputed, so the memory table and the parameter table cannot disagree.

## Inactive experts are not free

9.61B of the student's 13.01B parameters are active per token. **That is a compute number
and never a memory number.** Every expert is resident in VRAM for the whole run, whether or
not a token routes to it, so every figure below counts all 13.01B.

Sizing an MoE against its active count is exactly how the rejected 22.07B architecture
looked deployable on paper: 9.6B active reads like a 10B model, and it needed the VRAM of a
22B one. `test_inactive_experts_are_not_free` pins this.

## Quantisation overhead is counted separately

A nominal 4.9 bits per parameter is a **file-size** number. The runtime also pays tensor
alignment padding, per-tensor scale metadata the quoted bpw does not fully cover, and
dequantisation scratch. The model adds an explicit **3% allowance** on top of weight bytes,
labelled as an allowance rather than a measurement — treating bpw as the VRAM figure is a
standard way to be wrong by a third of a gigabyte.

## The result

Q4 / Q5 / Q6 with embeddings at Q6 (what GGUF/AWQ packers habitually do), batch 1, fully
GPU-resident, against 13.56 GiB usable:

| quant | context | weights | quant | KV | state | acts | runtime | **total** | headroom | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Q4 | 32,768 | 7.92 | 0.24 | 0.75 | 0.111 | 0.28 | 0.90 | **10.21** | 3.35 | FIT |
| Q4 | 131,072 | 7.92 | 0.24 | 3.00 | 0.111 | 0.28 | 0.90 | **12.46** | 1.10 | FIT |
| Q5 | 32,768 | 8.90 | 0.27 | 0.75 | 0.111 | 0.28 | 0.90 | **11.21** | 2.35 | FIT |
| Q5 | 131,072 | 8.90 | 0.27 | 3.00 | 0.111 | 0.28 | 0.90 | **13.46** | 0.10 | BORDERLINE |
| Q6 | 32,768 | 9.99 | 0.30 | 0.75 | 0.111 | 0.28 | 0.90 | **12.34** | 1.22 | FIT |
| Q6 | 131,072 | 9.99 | 0.30 | 3.00 | 0.111 | 0.28 | 0.90 | **14.59** | −1.03 | DOES NOT FIT |

Longest context clearing a 0.5 GiB reserve — the number a release should quote, because a
fit with less headroom than that is real but not safe:

| precision | weights (all-quant) | longest context |
|---|---:|---:|
| **Q4** | 7.42 GiB | **131,072** |
| **Q5** | 8.63 GiB | **65,536** |
| **Q6** | 9.99 GiB | **32,768** |

Full ladder at 4K / 8K / 16K / 32K / 64K / 128K / 262K:
`python scripts/student_report.py --section memory`.

### The full 262K window

Does not fit at any precision with an fp16 KV cache — KV alone is 6.00 GiB there. At Q4 with
an **8-bit KV cache** it fits at 11.94 GiB.

KV quantisation costs retrieval accuracy, so it is reported as its own row and never folded
into the headline. The context-performance curve in
[CONTEXT_SPECIALIZATION.md](CONTEXT_SPECIALIZATION.md) must be measured under whichever KV
precision a release actually ships — a 262K claim resting on an unmeasured 8-bit cache would
be exactly the kind of unsupported number this project refuses.

## What the correction changed

| | rejected | corrected |
|---|---:|---:|
| routed experts | 24 | **8** |
| total parameters | 22,072,134,528 | **13,008,505,728** |
| active per token | 9,615,051,648 | **9,611,119,488** |
| routed-expert share | 61.57% | **34.82%** |
| fraction of the 26.90B teacher | 82% | **48%** |
| Q4 longest context | none — did not fit | **131,072** |
| Q5 longest context | none — did not fit | **65,536** |
| Q6 longest context | none — did not fit | **32,768** |

One field moved. Per-token capacity is unchanged to within 0.04%, all of it the smaller
router: a token was only ever using two experts, so the other sixteen cost VRAM and
contributed nothing. The rejected configuration and three evaluated alternatives are kept in
`REJECTED` in `architecture/moe_student.py`, each with the measurement that rejected it.

## The parameter budget

`PARAMETER_BUDGET = 15,000,000,000`, derived rather than chosen: a Q5 release at 32,768
tokens must land under 13.56 GiB with a gigabyte to spare, which allows about 10.2 GiB of
weights, which at 5.7 bits per parameter is about 15.4B parameters. A test fails if the
student exceeds it, so a future edit adding experts, widening the FFN or untying something
cannot silently reintroduce a model that does not deploy.

## Competitor comparison

`analysis/competition.py` holds third-party figures with provenance. The measured fact worth
repeating: of the three named comparison models, **only Qwen3.5-9B actually fits 16 GB**
(7.56 GiB at 32K). Qwen3-14B needs 14.43 GiB and Gemma-3-27B 19.31 GiB.

For scale: the corrected student is **13.01B total / 9.61B active** and uses 7.42 GiB of
weights at Q4, against Qwen3.5-9B's 7.56 GiB at 32K. The two are in the same deployment
class, which is the point — and no capability claim follows from that, because none has been
measured.

Two rules apply to that table and are enforced by the ledger:

- **Competitor results are never fabricated.** Every third-party number carries a source; an
  entry without one is refused.
- **No claim is made before benchmarking.** Qwen3.5-9B's reported figures — MMLU-Pro 82.5,
  GPQA Diamond 81.7, IFEval 91.5, LiveCodeBench v6 65.6, LongBench v2 55.2, BFCL v4 66.1,
  TAU2-Bench 79.1 — are the target to beat. They are not evidence about our model, and this
  project holds no benchmark results of its own yet.

## The Pareto frontier the release optimises

Four axes, not one:

1. **benchmark capability** — not yet measured;
2. **VRAM** — accounted analytically above, and now the axis the architecture was corrected
   against rather than the axis it failed;
3. **inference speed** — not yet measured; the sparse FFN and the 12-layer KV cache both
   argue for it, and neither argument is evidence;
4. **effective context capability** — defined and schematised in
   [CONTEXT_SPECIALIZATION.md](CONTEXT_SPECIALIZATION.md), not yet measured.

Only axis 2 has numbers today. Any statement ranking this model against another on axes 1,
3 or 4 would be unsupported.
