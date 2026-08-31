"""Initialising the sparse student from the dense teacher, and measuring how well it worked.

Three reductions have to be initialised, and each one is a research question rather than a
mechanical copy:

======================  =========================  ==========================
reduction               teacher                    student
======================  =========================  ==========================
depth                   64 layers                  48 layers
FFN                     dense, 17408 wide          8 experts x 768, top-2
KV heads                4                          2
======================  =========================  ==========================

Every routine here returns a *measured* report, not a claim. The point is to know where
the student differs from the teacher **before** training starts, because after training
those differences are unattributable.

A finding that shapes the FFN work, stated up front because it bounds what any
decomposition can achieve:

    **Top-2-of-8 experts at width 768 cannot reproduce a 17408-wide dense FFN.**
    Active FFN width per token is ``2 x 768 + 768 (shared) = 2304`` against the teacher's
    17408 — a 7.6x reduction. Even a perfect decomposition reconstructs at most ~13% of
    the teacher's per-token FFN capacity. The decomposition's job is to choose *which*
    13% and to scale it sensibly; it is not to be lossless, and a method reporting near-zero
    error would indicate a bug, not success.

A second bound, introduced by the expert-budget correction that brought the student inside
16 GB: the 8 experts plus the shared expert hold ``8 x 768 + 768 = 6912`` of the teacher's
17408 channels, so **39.7% of the teacher's FFN is not transferred at all** and has to be
learned. That is a coverage limit, not a per-token one — active width is unchanged — and it
is the price of a model that deploys. It is measured, not estimated, by
:func:`plan_ffn_decomposition`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .moe_student import FROZEN_STUDENT, TEACHER_FFN_INTERMEDIATE, MoEStudentSpec
from .spec import FULL_ATTENTION, LINEAR_ATTENTION

if TYPE_CHECKING:  # pragma: no cover
    pass

LayerMapStrategy = Literal["group", "importance"]
FFNMethod = Literal["importance_partition", "contiguous_partition"]
KVMergeMethod = Literal["mean", "weighted", "first"]


# ---------------------------------------------------------------------------
# 64 -> 48 layer mapping
# ---------------------------------------------------------------------------
@dataclass
class LayerMapping:
    """Which teacher layer each student layer starts from, with block types preserved."""

    strategy: str
    mapping: dict[int, int] = field(default_factory=dict)
    removed_teacher_layers: list[int] = field(default_factory=list)
    student_types: list[str] = field(default_factory=list)
    teacher_types: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def block_types_preserved(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "mapping": {str(k): v for k, v in sorted(self.mapping.items())},
            "removed_teacher_layers": self.removed_teacher_layers,
            "n_removed": len(self.removed_teacher_layers),
            "block_types_preserved": self.block_types_preserved,
            "problems": self.problems,
        }


def _hybrid_types(n_layers: int, deltanet: int, attention: int) -> list[str]:
    group = [LINEAR_ATTENTION] * deltanet + [FULL_ATTENTION] * attention
    return group * (n_layers // len(group))


def map_layers(
    spec: MoEStudentSpec = FROZEN_STUDENT,
    *,
    teacher_layers: int = 64,
    strategy: LayerMapStrategy = "group",
    importance: dict[int, float] | None = None,
) -> LayerMapping:
    """Map 48 student layers onto 64 teacher layers, keeping every block type intact.

    ``group`` is the baseline: whole 4-layer hybrid groups are selected evenly across the
    teacher's depth and copied position-for-position, so a student layer always lands on a
    teacher layer of its own type. Deleting every fourth layer would instead rotate the
    pattern and put DeltaNet weights into attention slots.

    ``importance`` keeps the *groups the teacher uses most*, scored by any measurable
    signal the caller supplies (ablation loss delta, hidden-state change, residual
    contribution). The score is per teacher *group*, because a group is the unit that can
    be removed without breaking the topology.
    """
    group_size = spec.group_size
    if teacher_layers % group_size or spec.num_hidden_layers % group_size:
        raise ValueError(
            f"both depths must be whole {group_size}-layer groups: teacher "
            f"{teacher_layers}, student {spec.num_hidden_layers}"
        )
    teacher_groups = teacher_layers // group_size
    student_groups = spec.num_hidden_layers // group_size

    if strategy == "importance":
        if not importance:
            raise ValueError(
                "strategy='importance' needs a per-teacher-group score. Supply one measured "
                "from ablation, hidden-state change or residual contribution — selecting by "
                "parameter count would just re-derive the group baseline."
            )
        ranked = sorted(importance, key=lambda g: importance[g], reverse=True)
        chosen = sorted(ranked[:student_groups])
    else:
        step = (teacher_groups - 1) / (student_groups - 1) if student_groups > 1 else 0
        chosen = [round(i * step) for i in range(student_groups)]

    mapping = {
        s * group_size + offset: chosen[s] * group_size + offset
        for s in range(student_groups)
        for offset in range(group_size)
    }
    student_types = _hybrid_types(spec.num_hidden_layers, spec.deltanet_per_group, spec.attention_per_group)
    teacher_types = _hybrid_types(teacher_layers, spec.deltanet_per_group, spec.attention_per_group)
    problems = [
        f"student layer {s} is {student_types[s]} but maps to teacher layer {t} "
        f"which is {teacher_types[t]}"
        for s, t in mapping.items()
        if student_types[s] != teacher_types[t]
    ]
    kept = set(mapping.values())
    return LayerMapping(
        strategy=strategy, mapping=mapping,
        removed_teacher_layers=[i for i in range(teacher_layers) if i not in kept],
        student_types=student_types, teacher_types=teacher_types, problems=problems,
    )


# ---------------------------------------------------------------------------
# dense FFN -> sparse MoE
# ---------------------------------------------------------------------------
@dataclass
class FFNDecomposition:
    """Which teacher FFN channels each expert received, and how well it reconstructs."""

    method: str
    expert_channels: list[list[int]]
    shared_channels: list[int]
    teacher_intermediate: int
    expert_intermediate: int
    coverage: float
    active_width: int
    measurements: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_experts": len(self.expert_channels),
            "expert_intermediate": self.expert_intermediate,
            "teacher_intermediate": self.teacher_intermediate,
            "channel_coverage": self.coverage,
            "active_width": self.active_width,
            "active_fraction_of_teacher": self.active_width / self.teacher_intermediate,
            "measurements": self.measurements,
        }


def channel_importance(gate_proj, up_proj, down_proj, activations=None):
    """Per-channel importance of the teacher's FFN intermediate dimension.

    With activations, importance is the mean absolute contribution each channel actually
    makes on real text — the quantity that matters. Without them it falls back to weight
    energy, which is a proxy and is labelled as one by the caller.
    """

    if activations is not None:
        # activations: (tokens, intermediate) post-SwiGLU. Contribution is the activation
        # magnitude scaled by how strongly down_proj reads that channel.
        return (activations.abs().mean(0).float() * down_proj.float().pow(2).sum(0).sqrt())
    return (
        gate_proj.float().pow(2).sum(1).sqrt()
        * up_proj.float().pow(2).sum(1).sqrt()
        * down_proj.float().pow(2).sum(0).sqrt()
    )


def plan_ffn_decomposition(
    spec: MoEStudentSpec = FROZEN_STUDENT,
    *,
    importance=None,
    method: FFNMethod = "importance_partition",
    teacher_intermediate: int = TEACHER_FFN_INTERMEDIATE,
) -> FFNDecomposition:
    """Assign teacher FFN channels to the shared expert and the routed experts.

    ``importance_partition`` gives the shared expert the globally strongest channels —
    it is active for *every* token, so it should carry what every token needs — and then
    deals the remaining channels round-robin in descending importance across the routed
    experts, so each expert receives a comparable share of strong and weak channels rather
    than one expert getting all the good ones and the rest being dead on arrival.
    """
    import torch

    n_experts = spec.num_experts
    width = spec.moe_intermediate_size
    shared_width = spec.shared_expert_intermediate_size

    if importance is None:
        order = torch.arange(teacher_intermediate)
        method = "contiguous_partition"
    else:
        order = torch.argsort(importance.float(), descending=True)

    shared = order[:shared_width].tolist()
    rest = order[shared_width:].tolist()

    # Round-robin in descending importance: expert e receives ranks e, e+n, e+2n, ...
    buckets: list[list[int]] = [[] for _ in range(n_experts)]
    for rank, channel in enumerate(rest):
        buckets[rank % n_experts].append(channel)
    # Pad short buckets by re-using the strongest channels; every expert must be `width`
    # wide, and reusing a strong channel is better than zero-padding a dead one.
    for bucket in buckets:
        i = 0
        while len(bucket) < width:
            bucket.append(shared[i % len(shared)])
            i += 1
        del bucket[width:]

    covered = len(set(shared) | {c for b in buckets for c in b})
    return FFNDecomposition(
        method=method,
        expert_channels=buckets,
        shared_channels=shared,
        teacher_intermediate=teacher_intermediate,
        expert_intermediate=width,
        coverage=covered / teacher_intermediate,
        active_width=spec.num_experts_per_tok * width + shared_width,
    )


def measure_ffn_reconstruction(
    plan: FFNDecomposition, gate_proj, up_proj, down_proj, hidden, *, top_k: int = 2
) -> dict[str, float]:
    """Compare the initialised MoE FFN against the teacher's dense FFN on real hidden states.

    Reports the error a perfect router would leave, by selecting the ``top_k`` experts whose
    channels contribute most for each input — an upper bound on what the real router can
    achieve at initialisation, and the number the paper needs.
    """
    import torch
    import torch.nn.functional as F

    hidden = hidden.float()
    gate, up, down = gate_proj.float(), up_proj.float(), down_proj.float()
    teacher_act = F.silu(hidden @ gate.T) * (hidden @ up.T)      # (tokens, intermediate)
    teacher_out = teacher_act @ down.T

    def subset_out(channels: list[int]):
        idx = torch.tensor(channels, dtype=torch.long)
        return teacher_act[:, idx] @ down[:, idx].T

    shared_out = subset_out(plan.shared_channels)
    expert_outs = torch.stack([subset_out(c) for c in plan.expert_channels])   # (E, T, H)
    # An oracle router: pick the experts that most reduce the residual.
    residual = teacher_out - shared_out
    scores = torch.einsum("eth,th->et", expert_outs, residual)
    chosen = scores.topk(min(top_k, expert_outs.shape[0]), dim=0).indices        # (k, T)
    selected = torch.stack([expert_outs[chosen[i], torch.arange(hidden.shape[0])]
                            for i in range(chosen.shape[0])]).sum(0)
    student_out = shared_out + selected

    error = student_out - teacher_out
    denom = teacher_out.norm() + 1e-12
    return {
        "mse": float((error ** 2).mean()),
        "cosine_similarity": float(
            F.cosine_similarity(student_out.flatten(), teacher_out.flatten(), dim=0)
        ),
        "relative_norm_error": float(error.norm() / denom),
        "teacher_output_norm": float(teacher_out.norm()),
        "student_output_norm": float(student_out.norm()),
        "router": f"oracle top-{top_k} (upper bound on a learned router)",
    }


# ---------------------------------------------------------------------------
# 4 -> 2 KV heads
# ---------------------------------------------------------------------------
def merge_kv_heads(tensor, *, teacher_heads: int = 4, student_heads: int = 2,
                   head_dim: int = 256, method: KVMergeMethod = "mean", weights=None):
    """Merge teacher KV heads pairwise: ``student_i = merge(teacher_2i, teacher_2i+1)``.

    Averaging is the baseline, not an established optimum — two heads that attend to
    different things average into something that attends to neither. The method is a
    parameter so ``weighted`` and future activation-aware merges can be compared on the
    measured attention error rather than argued about.
    """

    if teacher_heads % student_heads:
        raise ValueError(f"{teacher_heads} teacher heads do not group evenly into {student_heads}")
    per = teacher_heads // student_heads
    grouped = tensor.reshape(teacher_heads, head_dim, -1)
    if method == "first":
        merged = grouped.reshape(student_heads, per, head_dim, -1)[:, 0]
    elif method == "weighted":
        if weights is None:
            weights = grouped.float().flatten(1).norm(dim=1)
        w = weights.reshape(student_heads, per, 1, 1).float()
        merged = (grouped.reshape(student_heads, per, head_dim, -1).float() * w).sum(1) / w.sum(1)
        merged = merged.to(tensor.dtype)
    else:
        merged = grouped.reshape(student_heads, per, head_dim, -1).float().mean(1).to(tensor.dtype)
    return merged.reshape(student_heads * head_dim, -1)


def measure_kv_merge(k_proj, v_proj, hidden, *, teacher_heads: int = 4,
                     student_heads: int = 2, head_dim: int = 256,
                     method: KVMergeMethod = "mean") -> dict[str, float]:
    """How far the merged K/V projections move the attention inputs on real hidden states."""
    import torch.nn.functional as F

    out: dict[str, float] = {"method": method}
    for label, weight in (("k", k_proj), ("v", v_proj)):
        teacher = hidden.float() @ weight.float().T
        merged = merge_kv_heads(weight, teacher_heads=teacher_heads,
                                student_heads=student_heads, head_dim=head_dim, method=method)
        student = hidden.float() @ merged.float().T
        # Compare like with like: average the teacher's heads down to the student's count.
        folded = teacher.reshape(teacher.shape[0], teacher_heads, head_dim)
        folded = folded.reshape(teacher.shape[0], student_heads, teacher_heads // student_heads,
                                head_dim).mean(2).reshape(teacher.shape[0], -1)
        error = student - folded
        out[f"{label}_mse"] = float((error ** 2).mean())
        out[f"{label}_cosine"] = float(
            F.cosine_similarity(student.flatten(), folded.flatten(), dim=0)
        )
        out[f"{label}_relative_error"] = float(error.norm() / (folded.norm() + 1e-12))
    return out


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------
#: Router initialisation scale, chosen by measurement rather than convention. Measured on
#: 4096 random hidden states at the frozen width (hidden 5120, 8 experts, top-2)::
#:
#:     scale     entropy / max   dead experts   max load share
#:     0.0       1.000000         6 of 8        0.5000
#:     1e-3      0.997196         0 of 8        0.1292
#:     1e-2      0.900751         0 of 8        0.1292
#:     2e-2      0.707873         0 of 8        0.1292
#:
#: ``scale=0`` is the tempting choice — perfectly uniform *probabilities*, maximal entropy —
#: and it is a trap: with every logit identical, ``torch.topk`` breaks the tie by index and
#: sends **every** token to experts 0 and 1, leaving the rest permanently dead because a
#: never-selected expert receives no gradient. Entropy reports 1.000 throughout, so entropy
#: alone does not detect the failure; only realised load does. ``1e-3`` breaks the tie
#: randomly while keeping entropy at 99.7% of maximum, and is the default for that reason.
#:
#: The pathology is not specific to an expert count — it was first measured at 24 experts,
#: where it left 22 of 24 dead — but the usable range of ``scale`` is: with fewer experts
#: the same logit noise costs more entropy, because maximum entropy is ``ln(E)``. 1e-3 is
#: comfortably inside the safe range at both counts.
DEFAULT_ROUTER_SCALE = 1e-3


def init_router(n_experts: int, hidden_size: int, *,
                scale: float = DEFAULT_ROUTER_SCALE, seed: int = 0):
    """Near-uniform router logits, so no expert wins before any data has been seen.

    See :data:`DEFAULT_ROUTER_SCALE` for why the default is a small non-zero value and not
    zero. ``scale=0.0`` is kept reachable because it is the exactly-uniform ablation, and
    :func:`measure_router_balance` reports which was used.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(n_experts, hidden_size, generator=generator) * scale
    return weight


def measure_router_balance(router_weight, hidden, *, top_k: int = 2) -> dict[str, Any]:
    """Routing entropy, load per expert, dead experts and overload, before training."""
    import torch

    logits = hidden.float() @ router_weight.float().T
    probs = torch.softmax(logits, dim=-1)
    chosen = logits.topk(top_k, dim=-1).indices
    n_experts = router_weight.shape[0]
    counts = torch.bincount(chosen.flatten(), minlength=n_experts).float()
    share = counts / counts.sum()
    uniform = 1.0 / n_experts
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum(-1).mean())
    return {
        "n_experts": n_experts,
        "top_k": top_k,
        "routing_entropy_nats": entropy,
        "max_entropy_nats": float(torch.log(torch.tensor(float(n_experts)))),
        "entropy_fraction_of_uniform": entropy / float(torch.log(torch.tensor(float(n_experts)))),
        "tokens_per_expert": counts.tolist(),
        "load_share": share.tolist(),
        "dead_experts": int((counts == 0).sum()),
        "overloaded_experts": int((share > 3 * uniform).sum()),
        "max_load_share": float(share.max()),
        "min_load_share": float(share.min()),
    }


# ---------------------------------------------------------------------------
# turning a plan into weights the runtime actually uses
# ---------------------------------------------------------------------------
#: What ``Qwen3_5MoeSparseMoeBlock`` computes, which is *not* a sum of expert outputs::
#:
#:     y = sum_k w_k . E_{i_k}(x)  +  sigmoid(g.x) . S(x),      sum_k w_k = 1
#:
#: The routing weights are a softmax over all experts, restricted to the top-k and then
#: renormalised, so they form a convex combination. A decomposition that copies channel
#: subsets into the experts therefore arrives *attenuated*: each routed expert is scaled by
#: ``w_k`` (exactly ``1/top_k`` under the near-uniform initial router) and the shared expert
#: by ``sigmoid(0) = 0.5`` when its gate is zero-initialised.
#:
#: Compensating for that at initialisation — routed ``down_proj`` x ``top_k``, shared
#: ``down_proj`` x 2 — makes the initialised block reproduce the plain sum of the channel
#: subsets it was given, which is the quantity :func:`plan_ffn_decomposition` was designed
#: to control. Without the compensation the block starts at roughly half the teacher's FFN
#: output scale, which the residual stream reads as a systematically weakened FFN.
#:
#: The compensation is exact only while routing is uniform. It is an initialisation choice,
#: not a training-time correction: once the router specialises, ``w_k`` moves and the scale
#: is absorbed by ``down_proj`` like any other learned quantity.
GATE_COMPENSATION_NOTE = (
    "routed experts scaled by top_k, shared expert by 2, to cancel the convex routing "
    "weights and the zero-initialised sigmoid gate"
)


def gate_compensation(spec: MoEStudentSpec = FROZEN_STUDENT,
                      *, compensate: bool = True) -> tuple[float, float]:
    """``(routed_scale, shared_scale)`` — see :data:`GATE_COMPENSATION_NOTE`."""
    if not compensate:
        return 1.0, 1.0
    return float(spec.num_experts_per_tok), 2.0


def build_moe_weights(
    plan: FFNDecomposition,
    gate_proj,
    up_proj,
    down_proj,
    *,
    spec: MoEStudentSpec = FROZEN_STUDENT,
    compensate: bool = True,
    router_scale: float = DEFAULT_ROUTER_SCALE,
    seed: int = 0,
) -> dict[str, Any]:
    """Materialise a decomposition plan as tensors in the runtime's exact layout.

    The teacher tensors are ``nn.Linear`` weights: ``gate_proj``/``up_proj`` are
    ``(intermediate, hidden)`` and ``down_proj`` is ``(hidden, intermediate)``. The runtime
    stores routed experts fused, as ``gate_up_proj (E, 2*width, hidden)`` whose first
    ``width`` rows are the gate and whose last ``width`` rows are the up projection —
    ``Qwen3_5MoeExperts.forward`` chunks it in exactly that order — and
    ``down_proj (E, hidden, width)``.
    """
    import torch

    routed_scale, shared_scale = gate_compensation(spec, compensate=compensate)
    width = plan.expert_intermediate
    hidden_size = gate_proj.shape[1]

    def fuse(channels: list[int], scale: float):
        idx = torch.tensor(channels, dtype=torch.long)
        gate_up = torch.cat([gate_proj[idx], up_proj[idx]], dim=0)      # (2*width, hidden)
        down = down_proj[:, idx] * scale                                # (hidden, width)
        return gate_up, down

    fused = [fuse(c, routed_scale) for c in plan.expert_channels]
    shared_idx = torch.tensor(plan.shared_channels, dtype=torch.long)

    weights = {
        "experts.gate_up_proj": torch.stack([g for g, _ in fused]),
        "experts.down_proj": torch.stack([d for _, d in fused]),
        "shared_expert.gate_proj.weight": gate_proj[shared_idx].clone(),
        "shared_expert.up_proj.weight": up_proj[shared_idx].clone(),
        "shared_expert.down_proj.weight": down_proj[:, shared_idx] * shared_scale,
        "shared_expert_gate.weight": torch.zeros(1, hidden_size),
        "gate.weight": init_router(spec.num_experts, hidden_size, scale=router_scale, seed=seed),
    }
    assert weights["experts.gate_up_proj"].shape == (spec.num_experts, 2 * width, hidden_size)
    assert weights["experts.down_proj"].shape == (spec.num_experts, hidden_size, width)
    return weights


def apply_moe_weights(block, weights: dict[str, Any]) -> list[str]:
    """Copy built weights into a live ``Qwen3_5MoeSparseMoeBlock``.

    Returns the names written, so an initialisation audit can assert that every tensor in
    the block was accounted for rather than trusting that it was.
    """
    import torch

    written: list[str] = []
    with torch.no_grad():
        for name, tensor in weights.items():
            target = block
            *path, leaf = name.split(".")
            for part in path:
                target = getattr(target, part)
            param = getattr(target, leaf)
            if param.shape != tensor.shape:
                raise ValueError(f"{name}: block wants {tuple(param.shape)}, "
                                 f"got {tuple(tensor.shape)}")
            param.copy_(tensor.to(param.dtype))
            written.append(name)
    expected = {n for n, _ in block.named_parameters()}
    missed = sorted(expected - set(written))
    if missed:
        raise ValueError(f"initialisation left these block tensors untouched: {missed}")
    return written


def measure_block_reconstruction(block, gate_proj, up_proj, down_proj, hidden) -> dict[str, Any]:
    """Run the *initialised block itself* against the teacher's dense FFN.

    :func:`measure_ffn_reconstruction` reports an oracle upper bound; this reports what the
    model will actually compute on step 0, routing weights, shared gate and all. The gap
    between the two is the price of the router being untrained, and is the number that
    should shrink during distillation.
    """
    import torch
    import torch.nn.functional as F

    flat = hidden.reshape(-1, hidden.shape[-1])
    teacher_act = F.silu(flat.float() @ gate_proj.float().T) * (flat.float() @ up_proj.float().T)
    teacher_out = teacher_act @ down_proj.float().T
    with torch.no_grad():
        student_out = block(hidden).reshape(-1, hidden.shape[-1]).float()

    error = student_out - teacher_out
    return {
        "mse": float((error ** 2).mean()),
        "cosine_similarity": float(
            F.cosine_similarity(student_out.flatten(), teacher_out.flatten(), dim=0)
        ),
        "relative_norm_error": float(error.norm() / (teacher_out.norm() + 1e-12)),
        "teacher_output_norm": float(teacher_out.norm()),
        "student_output_norm": float(student_out.norm()),
        "norm_ratio": float(student_out.norm() / (teacher_out.norm() + 1e-12)),
        "router": "the block's own initialised router",
    }
