# The competitive objective

**Build the strongest model a person can actually run on a 16 GB GPU. Separately, the
strongest one that runs on a 12 GB GPU.**

That is the whole objective. Two things follow from it that are easy to get wrong:

**Parameter count is an optimisation variable, not a target.** Not 4B, not 6B, not 9B, not
14B, not 20B. The fixed quantity is the deployment envelope; the thing being maximised is
measured capability inside it. Any document, config or preset in this repository that
reads as though a size has been chosen is out of date, and should be corrected rather than
worked around.

**The question is not "can a Qwen3.8-27B student fit in 16 GB?"** It is "can we beat the
strongest existing 16 GB-deployable models on the benchmarks that matter?" Fitting is table
stakes; several strong models already fit. Distillation from Qwen3.8-27B is a *strategy*
for getting there, and it is on trial like any other.

The 12 GB model has the same objective under the stricter constraint, and its own
architecture. It is not the 16 GB model quantised harder.

---

## 1. The measurements every serious candidate needs

For each candidate — ours or a competitor's — the objective names twelve quantities:

| | quantity | status here |
|---|---|---|
| 1 | total parameters | computed (`architecture/params.py`) |
| 2 | active parameters, if MoE | schema exists (`Competitor.active_parameters`); we have no MoE candidate |
| 3 | quantised weight memory | computed (`architecture/memory.py`, `analysis/competition.py`) |
| 4 | KV-cache memory at relevant contexts | computed for our hybrid; **needs each competitor's geometry** |
| 5 | total VRAM footprint | computed, with unknowns named rather than zeroed |
| 6 | inference throughput | **not measured**; schema exists, always paired with the hardware |
| 7 | benchmark performance | **not measurable — no harness exists** |
| 8 | reasoning | ↑ |
| 9 | coding | ↑ |
| 10 | instruction following | ↑ |
| 11 | long context | ↑ |
| 12 | multilingual, where relevant | ↑, **and no benchmark is chosen** |

Rows 7–12 are the objective's actual scorecard, and none of them can be produced today.
That is the single most important fact about the current state of the project.

---

## 2. The competitive field

`scripts/competitive_report.py` puts the field and our candidates in one frame.
At **q4_k_m weights, 32K context, batch 1, 13.56 GiB usable on a 16 GB card**:

| model | params | weights | KV + state | total | verdict |
|---|---:|---:|---:|---:|---|
| **Qwen3.5-9B** | 8.95B | 5.51 | 1.05 | **7.56** | FITS |
| Qwen3-14B | 14.80B | 8.44 | 5.00 | 14.44 | UNKNOWN |
| Gemma-3-27B | 27.00B | 15.40 | 15.50 | 31.90 | UNKNOWN |
| *ours*, h3072 L28 | 4.36B | 2.64 | 0.44 | 4.18 | FITS |
| *ours*, h4096 L32 | 8.63B | 5.12 | 0.75 | 7.06 | FITS |

The Qwen3.5-9B row is computed with **our own hybrid-aware estimator**, not a
parameter-count approximation — see §2.1. The two `UNKNOWN` rows are the tooling refusing
to conclude on unverified geometry, not a formatting artefact; both are almost certainly
overstated, because neither model's actual attention layout has been read.

**The headline number: the model we have to beat uses 7.56 of 13.56 GiB at 32K context.
Roughly 6 GiB of the envelope is unspent.** Even at 128K it reaches only 10.56 GiB. Sizing
has been happening below the envelope rather than against it, and the obvious competitive
move — a larger model of the same family in the same envelope — has not been tried.

### 2.1 Qwen3.5-9B is a close relative, not a stranger

Verification turned up something that changes the strategic picture. Qwen3.5-9B is
**the same architecture family as our teacher**:

| | Qwen3.8-27B (teacher) | Qwen3.5-9B (target) |
|---|---|---|
| layout | period-4 hybrid, 3 DeltaNet : 1 gated attention | **same** |
| layers | 64 (48 + 16) | 32 (24 + 8) |
| hidden | 5120 | 4096 |
| attention | 24 Q / 4 KV, head_dim 256 | 16 Q / 4 KV, head_dim 256 |
| DeltaNet | 48 V / 16 K, dim 128 | 32 V / 16 K, dim 128 |
| FFN | 17408 | 12288 |
| vocabulary | 248,320 | 248,320 (inferred, see below) |
| context | 262,144 | 262,144 |

The vocabulary and embedding tying are not stated in any source reached here; they are
**inferred from the parameter count**. With the teacher's 248,320-entry vocabulary and
untied embeddings the spec computes to 8.95B, which is what "9B" means. No other
combination tried lands as close — tied embeddings give 7.94B, a 151,936 vocabulary gives
8.16B. That is corroboration by arithmetic, not a reading of the config, and
`tests/test_competition.py` pins it as such.

Two consequences:

1. **The tokenizer decision is confirmed from a second direction.** The target model
   already uses the vocabulary we chose for the student.
2. **A direct teacher → Qwen3.5-9B-shaped transfer is refused**, correctly. The teacher has
   6 query heads per KV head and 3 DeltaNet value heads per key head; the target has 4 and
   2. `materialize.py` raises rather than regrouping heads that share nothing, so this
   shows up as an error rather than as a quietly scrambled student.

### 2.2 What could not be verified, and why

The Hugging Face model card is **blocked by this environment's egress proxy**. That is
sandbox policy and not something to work around, so nothing here has been read from a
primary source. Architecture and six of seven scores are *corroborated* — seen in
independent secondary sources and, for the architecture, cross-checked arithmetically.
That is better than hearsay and short of a citation.

## 3. The target board

The first concrete bar is Qwen3.5-9B:

| benchmark | target | provenance | measures | harness status |
|---|---:|---|---|---|
| MMLU-Pro | 82.5 | corroborated | knowledge, reasoning | not implemented |
| GPQA Diamond | 81.7 | corroborated | reasoning | not implemented |
| IFEval | 91.5 | corroborated | instruction following | not implemented |
| LiveCodeBench v6 | 65.6 | corroborated ⚠ | coding | not implemented |
| LongBench v2 | 55.2 | **unverified** | long context | not implemented |
| BFCL v4 | 66.1 | corroborated | tool use | not implemented |
| TAU2-Bench | 79.1 | corroborated | agentic, tool use | not implemented |

> ⚠ **LiveCodeBench v6 came back two different ways.** One search summary gave 82.7, a
> second gave 65.6, matching the figure supplied with the objective. The conflict is
> recorded rather than resolved by preference — it is precisely why the primary source and
> the evaluation protocol both matter. LongBench v2 is the one score no independent source
> confirmed.
>
> **No evaluation protocol is recorded for any of these**, so even a corroborated number
> cannot yet serve as a bar: two scores on the same benchmark under different protocols are
> different quantities. `analysis/competition.py` blocks the comparison on exactly this.

**No multilingual benchmark is on this board.** The objective asks for multilingual
performance where relevant; until a benchmark is chosen (MMMLU, Belebele, Flores-200 and
MGSM are the usual candidates) that capability is unmeasured, which is not the same as
passing.

---

## 4. Why the tooling refuses to declare wins

The tempting failure is publishing *"we beat Qwen3.5-9B on MMLU-Pro"* on the strength of a
number nobody verified, produced under a protocol nobody matched. It would look exactly
like a result, and it would be worthless.

So `analysis/competition.py` returns `INCOMPARABLE`, with reasons, unless **all** of:

- our score is `MEASURED` — produced by this repository, with a committed artifact;
- their score is better than `UNVERIFIED`;
- both protocols are recorded, and identical.

And an envelope reports `UNKNOWN` rather than a verdict whenever something the total
depends on is missing. An unknown KV cache is never treated as zero: Gemma 3 interleaves
sliding-window and global attention, so assuming uniform global attention would overstate
its cache several-fold and hand us a flattering, false comparison — the error would point
in our favour, which is exactly when to be most careful.

`python scripts/competitive_report.py --strict` exits non-zero while anything is
unverified. It currently reports **16 items**.

---

## 5. What this changes about the plan

Three consequences, in order of how much they should affect what happens next.

**1. The benchmark harness is now the critical path, not an adjacent task.** The objective
is defined entirely in numbers we cannot produce. Every architecture decision after the
pilot — size, depth/width ratio, quantisation, hybrid ratio, whether distillation beats the
alternatives — is taken blind until at least one benchmark runs. IFEval is the cheapest
starting point: its constraints are checked programmatically, with no model judge, no gated
dataset and no code sandbox.

**2. Size selection becomes an experiment with a real budget to spend.** The envelope
supports far more than the sizes discussed so far. The reduction ratio matters too:
27B → 4.4B is 6×, more aggressive than the 2–4× where pruning-plus-distillation usually
recovers well; 27B → 9B is 3×, inside it. Neither the size nor the ratio should be fixed in
advance.

**3. The unspent envelope is the clearest opening.** Qwen3.5-9B leaves ~6 GiB of a 16 GB
card unused at 32K context. A model of the same family sized *into* that headroom — call it
13–15B at q4_k_m — is a straightforward competitive play that nothing in the project has
tried, and the hybrid layout makes its KV cost cheap enough that long context stays
affordable. The 12 GB target has less room: Qwen3.5-9B already fits there too, at 7.56 GiB
of 10.76.

**4. Distillation from Qwen3.8-27B is one strategy, and the honest read is that it starts
behind.** A student initialised by pruning the teacher begins at *no* capability and has to
reach ~82 MMLU-Pro on a rented-GPU budget. Qwen3.5-9B is already there, was trained on far
more compute than this project will ever have, and fits the card with 6 GiB to spare. The
competing arms deserve to be on the board explicitly:

- **grow into the headroom**: a same-family model larger than 9B, distilled from the 27B
  teacher, exploiting the envelope the incumbent leaves unused;
- **start from strength**: distil 27B → a student *initialised from an existing strong
  open-weight base*, keeping that base's capability as the floor rather than starting at
  zero;
- **improve the envelope**: take a strong 9–14B base and buy quality per gigabyte through
  quantisation-aware training, KV reduction or structured pruning;
- **prune the teacher**: what is built today.

The last is what exists. The others reuse most of it — the KD loss, the transfer machinery
and the trainer do not care where the student's initial weights came from. Which arm wins
is exactly what the benchmark harness exists to answer, and it cannot be settled by
argument.

## 6. Success criterion

The 16 GB model is successful only when it demonstrates **competitive or superior measured
benchmark performance against the strongest 16 GB-deployable alternatives, while remaining
genuinely practical on a 16 GB GPU** — real throughput, real context length, real
quantisation. The 12 GB model, the same under the stricter constraint.

Architecture novelty is optional. Parameter count is flexible. Measured capability under
the VRAM constraint is the only thing that counts.
