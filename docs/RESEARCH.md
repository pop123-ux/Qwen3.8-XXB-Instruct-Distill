# Research protocol

The two hypotheses this project exists to test, how they are measured, and — stated
plainly — what our compute budget does not let us answer.

Nothing in this document reports a result. No distillation run has happened.

---

## Hypothesis A — Beyond layer matching

> When teacher and student have materially different computational topologies, forcing
> correspondence primarily through **layer matching** may be an inferior abstraction to
> transferring the **behaviour** the computation produces.

This is a hypothesis. It is not established, and this repository does not claim it.

### Why this teacher/student pair is a fair test

The topologies genuinely differ:

| | teacher | student |
|---|---|---|
| layers | 64 | 48 |
| FFN | dense, 17408 wide | 8 experts x 768, top-2 + 1 shared |
| KV heads | 4 | 2 |
| mixer layout | 48 DeltaNet + 16 attention | 36 DeltaNet + 12 attention |
| parameters | 26,895,998,464 | 13,008,505,728 |

16 of the teacher's 64 layers have no student layer at all. Pointwise matching asks the
student to *visit* the teacher's intermediate states; it never says who performs the work
of the layers that were deleted. Residual contributions telescope — the total work of a
teacher span `[a, b)` is exactly `h_b - h_a` — so a student layer can instead be trained
to reproduce the combined contribution of every teacher layer it replaced. That is the
alternative abstraction under test. Mechanism:
[BEHAVIORAL_DISTILLATION.md](BEHAVIORAL_DISTILLATION.md).

### The four objectives, kept distinguishable

| arm | objective | terms |
|---|---|---|
| **A0** | CE only | `ce` |
| **A2** | CE + logit KD | `ce`, `logit_kd` |
| **A1** | CE + intermediate/layer KD | `ce`, `logit_kd`, `hidden_pointwise` |
| **A3** | CE + computational-behaviour KD | `ce`, `logit_kd`, `hidden_delta` |
| A4 | both matching terms (interaction cell) | `ce`, `logit_kd`, `hidden_pointwise`, `hidden_delta` |

`router_balance` is on in every arm: without the architecture's own load-balancing loss an
arm measures expert collapse rather than what it meant to measure.

A0 is the floor. Without it, every margin is relative to another KD arm rather than to *no
teacher at all*. A2 and A3 carry the same number of loss terms, so a difference between
them is attributable to the **kind** of supervision rather than its quantity.

Definitions and falsifiers: `src/qwen_distill/research/ablations.py`. Every arm carries a
`falsified_if` written before any run.

### Behaviour signals, and which are actually available

| signal | status |
|---|---|
| logits | **implemented** — full-vocabulary KL with exact tail mass |
| hidden states (pointwise) | **implemented** |
| hidden states (residual delta over spans) | **implemented** |
| attention maps | **implemented** — head-marginalised KL on the 12 attention layers |
| FFN / MoE function output | **implemented** — measured at initialisation |
| DeltaNet recurrent state | **NOT AVAILABLE** |
| MTP | **NOT AVAILABLE** |

The two unavailable signals are unavailable for concrete reasons, not for lack of effort:

- **DeltaNet state.** Student and teacher recurrent states are different shapes, and any
  comparison needs a projection that would itself be an untested modelling choice. The
  delta term measures the same layers at the hidden-size interface, where the shapes do
  agree. Requesting the state term raises rather than silently substituting the proxy.
- **MTP.** `transformers` 5.15.1 builds no MTP head for this architecture, so there are no
  student tensors to train and the teacher's `mtp.*` weights are discarded on load. The
  architecture field is kept as the extension point. **Any MTP result reported today would
  be fabricated.**

---

## Hypothesis B — Context-length specialisation

> The optimal distribution of context lengths used during distillation may differ from the
> deployment context, and deliberate specialisation may improve the quality/compute
> trade-off.

Also a hypothesis, also unestablished.

### Four different context numbers, never conflated

| quantity | value | what it means |
|---|---|---|
| architectural maximum | 262,144 | a config field. Free, and evidence of nothing |
| training context | per arm, 4K–262K | what the model actually saw |
| evaluation context | 4K–262K | where the curve is measured |
| **demonstrated deployment context** | **none yet** | needs measured accuracy *and* measured VRAM |

A release may only claim the fourth.

### The mixtures

| arm | mixture | token share 4K / 16K / 64K / 256K |
|---|---|---|
| **B0** | uniform (equal tokens per length) | 25 / 25 / 25 / 25 |
| **B1** | short-only — the conventional control | 100 / – / – / – |
| **B2** | progressive curriculum, staged | 4 / 11 / 29 / 57 |
| **B3** | length-balanced, interleaved — B2's budget, reordered | 4 / 11 / 29 / 57 |
| **B4** | long-heavy | 1 / 3 / 22 / 75 |
| **B5** | medium-heavy | 8 / 40 / 26 / 26 |

**Token share, not step share.** A step at 262,144 carries 64x the tokens of one at 4,096,
so an even split of *steps* puts 57% of the tokens at the longest length. B0 exists to make
that distinction concrete, and `token_share()` reports it.

B2 and B3 share a token budget exactly and differ only in ordering, which is what isolates
ordering from exposure.

Curves are scored by `effective_context` — the longest length at which the model retains a
stated fraction of **its own** short-context score, with every shorter length also passing.
`compare_curves` can return `no_measurable_effect`, and that verdict would be reported.
Details: [CONTEXT_SPECIALIZATION.md](CONTEXT_SPECIALIZATION.md).

---

## The experiment sequence

In order. Each step is cheap enough to catch the failure the next one would waste money on.

| # | step | what it establishes | needs |
|---|---|---|---|
| 1 | teacher reproduction | the pinned checkpoint loads with no missing weights | GPU ≥24 GB |
| 2 | teacher fingerprint | config/template hashes, tail mass per top-*k* | same |
| 3 | student instantiation | the 13.01B student builds | CPU |
| 4 | materialisation | teacher → student, coverage reported | CPU + checkpoint |
| 5 | initialisation audit | reconstruction error per component | CPU |
| 6 | one-step real KD | a real gradient from real teacher logits | GPU |
| 7 | tiny controlled KD | the chain trains and resumes | GPU |
| 8 | layer-mapping experiment | group vs importance-aware | GPU |
| 9 | MoE reconstruction | decomposition vs alternatives | GPU |
| 10 | first context experiment | one B arm against the control | GPU |
| 11 | larger distillation | only after 1–10 | budget decision |

**The training budget is an experimental decision, not a constant.** No token count is
hard-coded anywhere. Steps 1–10 tell us what step 11 should cost.

---

## Further Questions and Future Work

The honest boundary of this work. It is stated here so the paper can report controlled
evidence without implying the programme is complete, and so that no reader mistakes an
untested question for a negative result.

### Validated in this work

Everything here is reproducible from this repository today, on CPU, with no teacher
download.

- **Exact parameter accounting.** 13,008,505,728 total, 9,611,119,488 active per token,
  derived two independent ways — closed form from the specification and a tensor sum over
  the instantiated model — that agree on every component bucket.
- **Structural validity.** A scaled fixture of the same architecture family runs forward
  and backward with gradients reaching every parameter, routes top-2 of 8, and reproduces a
  single-pass forward under step-by-step cached decoding to 2e-4 — which is the strongest
  available evidence that the hybrid cache and recurrent state are correctly threaded.
- **16 GB feasibility, analytically.** Weights + quantisation overhead + KV + DeltaNet
  state + activations + runtime, at Q4/Q5/Q6 across 4K–262K, fully GPU-resident, counting
  every stored expert.
- **The architecture correction.** 24 experts (22.07B) did not fit at any release precision
  or any context; 8 experts (13.01B) does, at essentially unchanged per-token capacity.
- **Initialisation behaviour.** The FFN decomposition beats random initialisation on MSE,
  cosine and relative-norm error on the real MoE block; the routing-weight compensation
  restores the teacher's output scale exactly; exactly-uniform router logits leave 6 of 8
  experts dead.
- **Teacher-path safety.** A mismatched checkpoint cannot masquerade as the teacher, and an
  unpinned Hub revision is refused before any download.

### Not yet tested — blocked by compute

These are open questions. **None of them is a negative result**, and none may be reported
as one.

- Whether behaviour matching (A3) beats layer matching (A1). **The paper's central claim is
  untested.**
- Whether any KD arm beats CE-only (A0) at this compression ratio.
- Whether importance-aware layer mapping beats the group baseline.
- What the 60.3% of teacher FFN channels the decomposition does not transfer actually costs.
- Whether the context mixtures move the curve's knee at all.
- Every benchmark number. This project holds none of its own.
- Whether the analytical 16 GB accounting matches measured peak VRAM on real hardware.
- Seed-to-seed variance, which is unmeasured — so **no margin can yet be called
  significant**, and the ablation matrix records this in `not_controlled`.

### Future work — needs significantly more compute

- Large-scale teacher-data generation for offline KD.
- Distillation at tens to hundreds of billions of tokens. Our budget does not reach
  convergence, so results will be reported as controlled comparisons under a fixed budget,
  not as converged model quality.
- Extensive reasoning distillation across the teacher's thinking modes.
- Full context specialisation at 262K, which is expensive in both tokens and memory.
- RL and other post-training.
- Broad competitor benchmarking under one harness.
- Wide architecture and ablation sweeps; the 2x2 factorial here is the affordable design.
- Measured latency and throughput, without which the Pareto frontier has only one real axis.

### What the paper may honestly say

That limited compute prevented exhaustive and fully converged distillation, **and** that
the controlled comparisons actually run are reported in full — including the ones that came
out negative or null. A fixed, equal budget across arms is a legitimate experimental design.
Presenting an under-trained run as either a finished result or a refuted hypothesis is not.

---

## Provenance

Every experiment is traceable through `experiments/ledger.jsonl`: an append-only JSONL
record with a closed provenance set (`measured_here`, `reported_by_third_party`,
`estimated`), where an estimate must carry its method and a third-party number must carry
its source. Results are superseded by retraction, never edited.
[EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md).

Figures are generated by `plots/` and indexed by [plots/REGISTRY.md](../plots/REGISTRY.md),
which records for every figure its scientific question, research-question linkage, source
experiments, source artifact paths, source metric fields, plotting script and status (real /
partial / schematic / unavailable). A figure with no artifact behind it exits 2 naming what
would produce it, rather than drawing a plausible curve. Cross-objective comparisons admit a
run only when its protocol matches the reference arm's in every field except the objective,
so a mechanism-validation run cannot drift into a performance curve.
