# Run015 / Run016 preregistration — seed-2 replication of composite A3 raw vs normalised

**Status:** PREREGISTERED. Committed and pushed before either arm trains.
**Launcher:** `scripts/run015_016_A3_seed2.py`

## 1. Question

At seed 1 the effect of normalising the residual-delta hidden term depends on supervision context:

| context | raw | normalised | normalised − raw |
|---|---|---|---|
| pure hidden-KD (Run008 / Run011) | 7.41131 | 10.82203 | **+3.41072** (raw far better) |
| composite A3 (Run013) | 4.936469793319702 | 4.838203148408369 | **−0.09826664491133297** (normalised better) |

> **Does the raw-vs-normalised ordering under the composite CE + logit-KD + residual-transition
> objective replicate at seed 2?**

## 2. Arms and preregistered order

1. `run015_A3_behavioural_delta_raw_seed2` — replicates `run013_A3_behavioural_delta_raw_seed1`
2. `run016_A3_behavioural_delta_normalised_seed2` — replicates `run013_A3_behavioural_delta_normalised_seed1`

Run in that order, in one process with the teacher loaded once (as the seed-1 matrix did). Both
configs are built and verified before the first arm trains; the first result cannot alter the second.

## 3. Protocol equality (enforced by the launcher, proven by tests)

Configs are produced by the canonical seed-1 construction path — `make()` in
`experiments/_session_2026-09-11/run_matrix_composite.py` — and then exactly three fields change:
`training.seed` 1 → 2, `name`, `runtime.output_dir`.

The launcher refuses unless the diff against BOTH the freshly built seed-1 config AND the archived
executed seed-1 config (`summary.json`, 64 resolved fields) is exactly those three fields, and unless
seed-2 raw vs seed-2 normalised differ only in `training.layer_kd_normalise`, `name`, `output_dir`.

Held fixed (verified): corpus and packed SHA `e11ca38bb099fc89c2f74e96f5d2f1209def6a16f6a8432d4e9972acd50c100d`;
teacher `Qwen/Qwen3.8-27B` @ `dbdc473dea0d6a9763042881cc33d6058d1742d2`, 4-bit; transferred student
(`model.safetensors` SHA-256 `30a10bdb…`); group mapping (48 pairs); sequence length 1536; max tokens
700000; 128 steps; batch 1; grad accumulation 1; AdamW; LR 2e-4; cosine; warmup 10; bf16; gradient
checkpointing; QLoRA r16 / α32 / dropout 0.05; clip-norm 1.0 (hardcoded in the trainer); composite
weights ce 0.1, logit_kd 1.0, hidden_delta 0.5, router_balance 1.0 (router aux coef 0.001); KD T 2.0;
top-k 64; chunk pairs 4; direction weight 1.0; eval 32 / save 64 / log 1; raw and normalised loss
definitions.

Implementation identity: `trainer.py`, `behavioral.py`, `kd_run.py`, `ablations.py`, `config.py` have
identical git blob SHAs at `e3c525c` (seed-1 raw), `0f3ec06` (seed-1 normalised) and the preparation
commit; no existing file under `src/` was modified since seed 1.

Known, recorded difference: process history. Seed-1 raw ran as the last arm of the A0/A2/A1/A3 matrix
process; seed-1 normalised ran separately. `build_model` reseeds the global RNG and the sampler uses a
seeded generator, so per-arm randomness is determined by the seed, but CUDA kernel non-determinism across
process histories is not controlled.

## 4. Primary endpoint and quantity

Held-out validation loss at step 128 (`final_validation_loss`), plain CE under `no_grad`.

**Primary quantity:** `c2 = val(run016 normalised) − val(run015 raw)`. Seed-1 value: `c1 = −0.09826664491133297`.

## 5. Interpretation rule (frozen as formulas before any result)

Seed-to-seed spread, measured from this pair:
`S = max(|raw_seed2 − raw_seed1|, |normalised_seed2 − normalised_seed1|)`
i.e. how far a single arm moves when only the seed changes. A contrast no larger than that is not
practically distinguishable from seed variation.

| condition | classification | statement |
|---|---|---|
| `c2 < 0` and `|c2| > S` | REPLICATED_DIRECTION | replicated directional evidence (2 seeds) that normalisation helps under composite supervision — evidence for supervision-context dependence, not a universal law |
| `c2 < 0` and `|c2| ≤ S` | SAME_DIRECTION_WITHIN_SPREAD | direction repeats but the effect is not practically clear relative to seed variation |
| `c2 > 0` | NOT_REPLICATED_ORDERING_CHANGED | composite normalisation effect unstable / not replicated (reported with whether `|c2| > S`) |
| `c2 = 0` | NO_PRACTICALLY_CLEAR_EFFECT | no normalisation effect |

Also reported, descriptively only: `mean(c1, c2)`, both individual seed contrasts, `S`.

No statistical significance is claimed from two seeds. The pure hidden-KD side of the interaction has
only seed 1, so this pair can replicate the composite ordering but cannot by itself replicate the full
interaction.

## 6. Refusals

Dirty worktree; any config diff beyond the permitted fields; raw vs normalised differing beyond
normalisation; canonical `make()` producing an unexpected seed-1 id; existing completed `summary.json`
for either arm; a non-zero return from the first arm stops the pair.

No tuning, no reruns, no changes after the first arm's result.
