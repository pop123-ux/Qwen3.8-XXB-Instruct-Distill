# Initialization method

Code: `src/qwen_distill/architecture/moe_init.py`. Tests: `tests/test_moe_init.py`.

Three reductions turn the dense 64-layer teacher into the sparse 48-layer student. Each is
a research question rather than a mechanical copy, and each has a *measurement* function
next to it, because after training the differences are unattributable.

| reduction | teacher | student |
|---|---|---|
| depth | 64 layers | 48 layers |
| FFN | dense, 17408 wide | 24 experts x 768, top-2 |
| KV heads | 4 | 2 |

---

## 1. Depth: 64 -> 48

**The failure this avoids.** Deleting every fourth teacher layer is the obvious move and it
is wrong here. The teacher's layout is `[DeltaNet, DeltaNet, DeltaNet, FullAttention] x 16`.
Striding through it rotates the pattern, so DeltaNet weights land in the student's attention
slots and vice versa. `test_naive_stride_deletion_is_what_the_group_mapping_avoids` pins the
counter-example so the design choice is evidenced rather than asserted.

**Baseline: `group`.** Whole 4-layer hybrid groups are selected evenly across the teacher's
depth and copied position-for-position. 12 of the teacher's 16 groups are kept. Properties
checked by tests: the mapping is injective, order-preserving, spans both ends of the
teacher's depth (layer 0 and layer 63 are both kept), and every student layer lands on a
teacher layer of its own type.

**Research: `importance`.** Keeps the groups the teacher uses most, ranked by any *measured*
signal the caller supplies — ablation loss delta, hidden-state change, residual contribution.
The score is per teacher *group*, because a group is the unit that can be removed without
breaking the topology. Calling it without a score raises rather than falling back to a
parameter-count proxy, which would just re-derive the baseline while looking principled.

The mapping is auditable: `to_dict()` emits `student_layer -> teacher_layer` for all 48
plus the 16 removed teacher layers.

---

## 2. Dense FFN -> 24 experts

### The bound, stated first

**Top-2-of-24 experts at width 768 cannot reproduce a 17408-wide dense FFN.** Active FFN
width per token is `2 x 768 + 768 (shared) = 2304` against the teacher's 17408 — a 7.6x
reduction. Even a perfect decomposition reconstructs at most about 13% of the teacher's
per-token FFN capacity. The method's job is to choose *which* 13% and to scale it sensibly.
**A method reporting near-zero reconstruction error would indicate a bug, not success**, and
a test asserts the error stays above a floor for exactly that reason.

### What the method is

Two things are explicitly not done: the whole teacher FFN is not duplicated into every
expert, and the experts are not randomly initialised and called a transfer. Tests assert
both — no two experts receive identical channel sets, no expert receives the full teacher
FFN, and pairwise overlap between experts stays under 50%.

What happens instead, `importance_partition`:

1. **Score every teacher intermediate channel.** With activations, importance is the mean
   absolute contribution the channel actually makes on real text. Without them it falls back
   to weight energy, labelled as a proxy by the caller.
2. **The shared expert takes the globally strongest 768 channels.** It runs on every token,
   so it should carry what every token needs.
3. **The routed experts deal the remainder round-robin in descending importance.** Expert
   `e` gets ranks `e, e+24, e+48, ...`, so each expert receives a comparable mix of strong
   and weak channels rather than one expert getting all the good ones and the rest arriving
   dead. A test checks the weakest expert holds at least 80% of the strongest expert's total
   importance.

### The routing-weight correction

The runtime's MoE block does **not** sum expert outputs:

```
y = sum_k w_k . E_{i_k}(x)  +  sigmoid(g.x) . S(x),     sum_k w_k = 1
```

The routing weights are a softmax over all experts restricted to the top-k and then
renormalised — a *convex combination*. So a decomposition that copies channel subsets into
the experts arrives attenuated: each routed expert scaled by `w_k` (exactly `1/top_k` under
the near-uniform initial router), and the shared expert by `sigmoid(0) = 0.5`.

The correction is to scale routed `down_proj` by `top_k` and shared `down_proj` by 2 at
initialisation, which makes the initialised block reproduce the plain sum of the channel
subsets it was given. Measured on the real module, the output-norm ratio against the teacher
FFN moves from 0.357 to 0.715 — exactly the factor of 2 — with the cosine similarity
unchanged, because scaling is orthogonal to direction. Without it the block starts at half
the teacher's FFN output scale, which the residual stream reads as a systematically weakened
FFN.

The compensation is exact only while routing is uniform. It is an initialisation choice, not
a training-time correction.

### What it achieves

Measured on the tiny fixture with a synthetic dense teacher FFN at the same width ratio,
running the **real** `Qwen3_5MoeSparseMoeBlock` — not a simulation of it:

| | MSE | cosine | relative norm error | output norm ratio |
|---|---:|---:|---:|---:|
| random initialisation | 0.366 | -0.011 | 1.000 | 0.002 |
| **after transfer** | **0.176** | **0.721** | **0.693** | **0.715** |
| oracle top-2 router (upper bound) | 0.112 | 0.833 | 0.553 | — |

Three things to read from this. The transfer is real — a random block has cosine ≈ 0 and
the transferred block has 0.72. It is lossy, as the bound requires. And the gap between the
transferred row and the oracle row is the price of the router being untrained: that is the
quantity distillation should shrink, and it is measurable from step 0.

Reproduce: `tests/test_moe_init.py::test_initialised_block_beats_a_random_block_on_every_metric`.

---

## 3. KV heads: 4 -> 2

Adjacent pairs merge: `student_i = merge(teacher_2i, teacher_2i+1)`. Averaging is the
baseline, not an established optimum — two heads that attend to different things average
into something that attends to neither. The method is therefore a parameter (`mean`,
`weighted`, `first`) so alternatives can be compared on measured attention error rather than
argued about, and `measure_kv_merge` reports MSE, cosine and relative error of the projected
K and V against the mean-folded teacher on real hidden states.

`mean` reproduces the mean-folded reference exactly (MSE < 1e-10). That identity is what
makes it the reference the other methods are scored against, not evidence that it is
lossless — the loss is in the folding, which is the reduction itself.

---

## 4. Router initialisation

**The measured trap.** Perfectly uniform router logits look ideal: maximal entropy, no
expert favoured. On 4096 hidden states at the frozen width:

| init scale | entropy / max | dead experts | max load share |
|---:|---:|---:|---:|
| `0.0` | 1.000000 | **22 of 24** | 0.5000 |
| `1e-3` | 0.999230 | 0 of 24 | 0.0455 |
| `2e-2` | 0.750482 | 0 of 24 | 0.0455 |

With every logit identical, `torch.topk` breaks the tie by index and sends **every** token
to experts 0 and 1. The other 22 receive no gradient and stay dead. Entropy reports 1.000
throughout — **entropy alone does not detect this failure; only realised load does.**

`DEFAULT_ROUTER_SCALE = 1e-3` breaks the tie randomly while keeping entropy at 99.92% of
maximum. `scale=0.0` stays reachable as the exactly-uniform ablation, and
`measure_router_balance` reports which was used along with routing entropy, tokens per
expert, load share, dead experts and overloaded experts.

The shared expert's gate is zero-initialised, giving `sigmoid(0) = 0.5` — the value the
routing-weight correction above cancels. A randomly initialised gate would make that
correction wrong by an unknown factor, so a test pins it.

---

## Applying it

`build_moe_weights` materialises a plan into the runtime's exact tensor layout, and
`apply_moe_weights` writes it into a live block. Two guards:

- **Shape mismatches raise**, naming the tensor and both shapes.
- **A partial initialisation raises.** After writing, every parameter in the block must have
  been written; any left at its random default is a silent partial transfer and is refused.

The fused expert layout is a real trap: `gate_up_proj` is `(E, 2*width, hidden)` and
`Qwen3_5MoeExperts.forward` chunks it into `(gate, up)` in that order. Swapping the halves
produces correct shapes and wrong arithmetic, so a test checks the halves against the
teacher tensors directly.
