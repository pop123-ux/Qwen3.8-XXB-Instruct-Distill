# Experiment: Analytical Architecture Search (pass 1)

**ID:** `arch-search-001`
**Date:** 2026-08-22
**Type:** Analytical (no training, no measurement)
**Status:** Complete — produces a shortlist, not a conclusion

## Hypothesis

Stated before running: under a 16 GB envelope, the feasible student is substantially
larger than the 7–10B range typical of "distilled" releases, because Qwen3.8's hybrid
architecture already removes most of the long-context memory cost, leaving weight size
as the sole binding constraint.

## Method

`scripts/search_architectures.py` enumerates architectures over a grid of hidden size,
depth, FFN multiplier, attention interval and embedding tying; prunes any that cannot
hold the required context within 15.0 GiB usable at Q4_K_M; and ranks survivors by
non-embedding parameter count.

```bash
python scripts/search_architectures.py --context 32768  --top 15 --json pass1_ctx32k.json
python scripts/search_architectures.py --context 131072 --top 8  --json pass2_ctx128k.json
python scripts/search_architectures.py --context 262144 --top 5
```

Grid: hidden ∈ {2560…5120}, layers ∈ {24…64}, FFN multiplier ∈ {2.0, 2.5, 3.0, 3.4},
attention interval ∈ {2, 3, 4, 6}, tied ∈ {yes, no}. 1152 candidates.

## Results

Feasible counts and the capacity ceiling by required context:

| Required context | Feasible / 1152 | Largest feasible (total / non-emb) |
|---|---:|---|
| 32k | 1076 | 21.16B / 19.89B |
| 128k | 692 | 16.54B / 15.26B |
| 262k | 311 | 13.48B / 12.21B |

Top candidate at each context target:

| Context | Architecture | Params | Peak VRAM | Max ctx | tok/s ceiling |
|---|---|---|---|---|---|
| 32k | h5120 / 64L / FFN 12800 / interval 6 / tied | 21.16B | 14.92 GiB | 34,778 | 25.9 |
| 128k | h5120 / 40L / FFN 17408 / interval 6 / tied | 16.54B | 13.99 GiB | 175,275 | 33.2 |
| 262k | h5120 / 32L / FFN 17408 / interval 6 / tied | 13.48B | 14.22 GiB | 262,144 | 40.7 |

Conservative alternative — keeps the teacher's 3:1 attention ratio and FFN width:

| Architecture | Params | Peak VRAM @32k | Max ctx | tok/s ceiling (3060 Ti) |
|---|---|---|---|---|
| h5120 / 48L / FFN 17408 / interval 4 / tied | 19.54B | 14.21 GiB | 50,118 | 28.1 |

## Findings

1. **The hypothesis holds.** The feasible ceiling is 13–21B, not 7–10B.
2. **Context is expensive in capacity, not in cache.** Raising the requirement from
   32k to 262k costs ~7.7B parameters of capacity.
3. **Tied embeddings dominate every ranking.** At hidden 5120 they free 1.27B
   parameters — a large fraction of a student's budget.
4. **A conservative candidate is competitive.** Keeping the teacher's 3:1 ratio and
   3.4x FFN at 48 layers yields 19.54B fitting in 14.21 GiB with 50k context. It is
   within ~1.6B non-embedding parameters of the top-ranked candidate while changing
   far less about the teacher — which matters for weight transfer and for risk.

## Caveats — read before using this shortlist

**The ranking objective is a capacity proxy, not capability.** Non-embedding parameter
count correlates with capacity under scaling laws; it does not predict post-training
benchmark scores. Nothing here has been trained.

**The ranking is biased toward `interval 6`, and that bias is a known hazard.** Fewer
full-attention layers means a smaller KV cache, which frees budget for parameters, so
the search reaches for it in every configuration. But full attention is what provides
exact long-range recall — precisely the capability a long-context model is judged on.
The proxy cannot see this trade-off at all.

**Do not adopt interval 6 without running Experiment F** (`docs/EVALUATION_PLAN.md`):
a long-context retrieval comparison of interval 4 vs 6 at matched VRAM. If interval 6
degrades retrieval, the correct reading of this table is the conservative candidate,
not the top-ranked one.

This is the failure mode the project plan names explicitly: optimising a proxy into a
real regression. The shortlist is a hypothesis generator. The next step is training.

## Reproduction

```bash
git checkout <commit>
pip install -e ".[dev]"
python scripts/search_architectures.py --context 32768 --top 15
```

Deterministic: no sampling, no randomness.
