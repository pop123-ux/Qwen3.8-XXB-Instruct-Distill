"""End-to-end VRAM accounting for the frozen MoE student, against a hard 16 GB ceiling.

The constraint is *end to end*, not weight memory. A quantisation table that shows 11 GiB
of weights and stops there has answered a question nobody deploys against. The complete
workload is::

    weights + KV cache + DeltaNet recurrent state + conv state + activations + runtime

and every one of those terms is included below. Two of them are frequently omitted and are
the reason "it fits" turns into an OOM: the ~0.9 GiB a PyTorch+CUDA process costs before a
single model tensor is allocated, and the fp32 logits over a 248,320-token vocabulary.

Three architecture facts drive the numbers
------------------------------------------
**Only 12 of 48 layers have a KV cache.** The 36 DeltaNet layers keep a fixed-size
recurrent state instead. So the cache costs ``2 x 2 heads x 256 dim x 12 layers`` = 24 KiB
per token in fp16 — against roughly 96 KiB/token for a dense 48-layer model with the same
head configuration. That is the hybrid layout's entire deployment argument, and it is what
makes a 262,144-token context arguable at all.

**The recurrent state does not grow.** ``(48 value heads x 128 x 128)`` per layer in fp32,
36 layers — about 113 MiB, identical at 2K and at 262K. Shapes verified against the
running model, not assumed.

**61.6% of the weights are routed experts.** Which is why ``expert_quant`` is a separate
knob: quantising the experts harder than the attention and DeltaNet paths moves far more
memory per unit of damage than a uniform setting, and the experts are the part where each
token only touches 2 of 24. A uniform-quantisation table would hide the one lever that
matters most here.

The rule about offload
----------------------
:data:`GPU_RESIDENT_ONLY` is not a preference. A configuration that reaches 16 GB by moving
experts to host RAM is a different product with different latency, and reporting it as
16 GB compliance would be false. Every verdict here is for a fully GPU-resident model;
:func:`offload_note` exists to state that, not to enable it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..architecture.memory import DTYPE_BYTES, GIB, QUANT_BYTES_PER_PARAM
from ..architecture.moe_student import FROZEN_STUDENT, MoEStudentSpec, audit

#: Every number in this module is for a model held entirely in VRAM.
GPU_RESIDENT_ONLY = True

#: The ceiling, and what is actually addressable underneath it. A "16 GB" card reports
#: 14.56 GiB (measured on the Level-2 T4), and a process that wants to survive a display
#: server or a second CUDA context should not plan to use the last gigabyte of that.
NOMINAL_VRAM_GIB = 16.0
MEASURED_TOTAL_GIB = 14.56
RESERVED_GIB = 1.0
USABLE_GIB = MEASURED_TOTAL_GIB - RESERVED_GIB

#: Below this much headroom the fit is real but not safe.
BORDERLINE_MARGIN_GIB = 1.0

FIT = "FIT"
BORDERLINE = "BORDERLINE"
DOES_NOT_FIT = "DOES NOT FIT"

#: The three release precisions the project is required to report.
RELEASE_QUANTS: tuple[str, ...] = ("q4_k_m", "q5_k_m", "q6_k")
QUANT_LABELS = {"q4_k_m": "Q4", "q5_k_m": "Q5", "q6_k": "Q6"}

#: Context ladder, one octave per step.
CONTEXT_LADDER: tuple[int, ...] = (2_048, 4_096, 8_192, 16_384, 32_768, 65_536,
                                   131_072, 262_144)


@dataclass(frozen=True)
class RuntimeConfig:
    """How the model is being served. Defaults are the conservative reading."""

    context_length: int = 32_768
    batch_size: int = 1
    #: Applied to routed experts and the shared expert.
    expert_quant: str = "q4_k_m"
    #: Applied to attention, DeltaNet, router and norms — the always-active path.
    dense_quant: str | None = None
    #: Embeddings and the LM head; packers habitually keep these higher, and with a
    #: 248,320-token vocabulary the correction is over a gigabyte.
    embedding_quant: str = "q6_k"
    kv_cache_dtype: str = "fp16"
    #: The reference DeltaNet path accumulates in fp32. Measured on the running model.
    recurrent_state_dtype: str = "fp32"
    runtime_overhead_gib: float = 0.9
    activation_safety_factor: float = 1.25
    prefill_chunk_tokens: int = 2_048

    @property
    def resolved_dense_quant(self) -> str:
        return self.dense_quant or self.expert_quant

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_length": self.context_length, "batch_size": self.batch_size,
            "expert_quant": self.expert_quant, "dense_quant": self.resolved_dense_quant,
            "embedding_quant": self.embedding_quant, "kv_cache_dtype": self.kv_cache_dtype,
            "recurrent_state_dtype": self.recurrent_state_dtype,
            "runtime_overhead_gib": self.runtime_overhead_gib,
            "activation_safety_factor": self.activation_safety_factor,
            "prefill_chunk_tokens": self.prefill_chunk_tokens,
        }


def _bytes(quant: str) -> float:
    if quant not in QUANT_BYTES_PER_PARAM:
        raise ValueError(f"unknown quantisation {quant!r}; "
                         f"known: {sorted(QUANT_BYTES_PER_PARAM)}")
    return QUANT_BYTES_PER_PARAM[quant]


def _dtype(name: str) -> float:
    if name not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype {name!r}; known: {sorted(DTYPE_BYTES)}")
    return DTYPE_BYTES[name]


#: Which audited component goes in which quantisation bucket.
EXPERT_COMPONENTS = ("routed_experts", "shared_expert")
EMBEDDING_COMPONENTS = ("embedding", "lm_head")


def weight_bytes(components: dict[str, int], config: RuntimeConfig) -> dict[str, int]:
    """Weight bytes per bucket, from the *audited* component counts.

    Taking the counts from the audit rather than recomputing them analytically means the
    memory table and the parameter table can never disagree — a class of error that is
    invisible until someone checks both.
    """
    buckets = {"experts": 0, "dense": 0, "embeddings": 0}
    for name, count in components.items():
        if name in EXPERT_COMPONENTS:
            buckets["experts"] += int(count * _bytes(config.expert_quant))
        elif name in EMBEDDING_COMPONENTS:
            buckets["embeddings"] += int(count * _bytes(config.embedding_quant))
        else:
            buckets["dense"] += int(count * _bytes(config.resolved_dense_quant))
    return buckets


def kv_cache_bytes(spec: MoEStudentSpec, config: RuntimeConfig) -> int:
    """``2 (K and V) x kv_heads x head_dim`` per token, over the full-attention layers only."""
    per_token_per_layer = 2 * spec.num_key_value_heads * spec.head_dim
    n_full = len(spec.attention_layer_indices)
    elements = per_token_per_layer * n_full * config.context_length * config.batch_size
    return int(elements * _dtype(config.kv_cache_dtype))


def recurrent_state_bytes(spec: MoEStudentSpec, config: RuntimeConfig) -> int:
    """``(batch, value_heads, key_head_dim, value_head_dim)`` per DeltaNet layer.

    Shape verified against a running model of this architecture family and confirmed
    constant between a 32-token and a 128-token forward pass.
    """
    per_layer = (spec.linear_num_value_heads * spec.linear_key_head_dim
                 * spec.linear_value_head_dim)
    elements = per_layer * len(spec.deltanet_layer_indices) * config.batch_size
    return int(elements * _dtype(config.recurrent_state_dtype))


def conv_state_bytes(spec: MoEStudentSpec, config: RuntimeConfig) -> int:
    """``(batch, 2*key_dim + value_dim, kernel)`` per DeltaNet layer — the depthwise conv
    window. The ``2*key_dim + value_dim`` width is the runtime's own ``conv_dim`` formula,
    confirmed against the built module."""
    key_dim = spec.linear_num_key_heads * spec.linear_key_head_dim
    value_dim = spec.linear_num_value_heads * spec.linear_value_head_dim
    conv_dim = 2 * key_dim + value_dim
    elements = (conv_dim * spec.linear_conv_kernel_dim
                * len(spec.deltanet_layer_indices) * config.batch_size)
    return int(elements * _dtype(config.recurrent_state_dtype))


def activation_bytes(spec: MoEStudentSpec, config: RuntimeConfig) -> int:
    """Transient working set for one chunked forward pass.

    Sparsity helps here too: the widest MoE buffer is ``top_k x moe_intermediate`` = 1536,
    not a dense 17408. The DeltaNet projections are then the widest thing in the model, and
    the fp32 logits over 248,320 tokens are a full megabyte per sequence position kept.
    """
    tokens = min(config.prefill_chunk_tokens, config.context_length) * config.batch_size
    act = _dtype("bf16")

    key_dim = spec.linear_num_key_heads * spec.linear_key_head_dim
    value_dim = spec.linear_num_value_heads * spec.linear_value_head_dim
    widest = max(
        spec.num_experts_per_tok * spec.moe_intermediate_size
        + spec.shared_expert_intermediate_size,
        2 * key_dim + value_dim + value_dim,
        2 * spec.num_attention_heads * spec.head_dim,
    )
    per_token = 3 * widest + 2 * spec.hidden_size
    layer_activations = tokens * per_token * act
    # The router must materialise a per-token score for every expert, and the expert
    # dispatch gathers hidden states per expert. Small, but not zero, and it scales with
    # the chunk rather than the model.
    routing = tokens * spec.num_experts * _dtype("fp32") * 2
    logits = config.batch_size * spec.vocab_size * _dtype("fp32")
    return int((layer_activations + routing + logits) * config.activation_safety_factor)


@dataclass
class MemoryAccount:
    """A complete end-to-end accounting, in bytes, with nothing left implicit."""

    spec_name: str
    config: RuntimeConfig
    weights: dict[str, int]
    kv_cache: int
    recurrent_state: int
    conv_state: int
    activations: int
    runtime_overhead: int
    parameters: int = 0

    @property
    def weight_total(self) -> int:
        return sum(self.weights.values())

    @property
    def total(self) -> int:
        return (self.weight_total + self.kv_cache + self.recurrent_state
                + self.conv_state + self.activations + self.runtime_overhead)

    @property
    def total_gib(self) -> float:
        return self.total / GIB

    def headroom_gib(self, usable_gib: float = USABLE_GIB) -> float:
        return usable_gib - self.total_gib

    def verdict(self, usable_gib: float = USABLE_GIB) -> str:
        headroom = self.headroom_gib(usable_gib)
        if headroom < 0:
            return DOES_NOT_FIT
        return BORDERLINE if headroom < BORDERLINE_MARGIN_GIB else FIT

    def to_dict(self, usable_gib: float = USABLE_GIB) -> dict[str, Any]:
        return {
            "spec": self.spec_name,
            "config": self.config.to_dict(),
            "parameters": self.parameters,
            "gpu_resident_only": GPU_RESIDENT_ONLY,
            "bytes": {
                "weights_experts": self.weights.get("experts", 0),
                "weights_dense": self.weights.get("dense", 0),
                "weights_embeddings": self.weights.get("embeddings", 0),
                "kv_cache": self.kv_cache,
                "recurrent_state": self.recurrent_state,
                "conv_state": self.conv_state,
                "activations": self.activations,
                "runtime_overhead": self.runtime_overhead,
                "total": self.total,
            },
            "gib": {
                "weights": self.weight_total / GIB,
                "weights_experts": self.weights.get("experts", 0) / GIB,
                "weights_dense": self.weights.get("dense", 0) / GIB,
                "weights_embeddings": self.weights.get("embeddings", 0) / GIB,
                "kv_cache": self.kv_cache / GIB,
                "recurrent_state": self.recurrent_state / GIB,
                "conv_state": self.conv_state / GIB,
                "activations": self.activations / GIB,
                "runtime_overhead": self.runtime_overhead / GIB,
                "total": self.total_gib,
            },
            "usable_gib": usable_gib,
            "headroom_gib": self.headroom_gib(usable_gib),
            "verdict": self.verdict(usable_gib),
        }


def account(spec: MoEStudentSpec = FROZEN_STUDENT,
            config: RuntimeConfig | None = None,
            components: dict[str, int] | None = None) -> MemoryAccount:
    """Full end-to-end accounting for one runtime configuration."""
    config = config or RuntimeConfig()
    if components is None:
        report = audit(spec)
        components = report["components"]
        parameters = report["exact_parameter_count"]
    else:
        parameters = sum(components.values())
    return MemoryAccount(
        spec_name=spec.name, config=config,
        weights=weight_bytes(components, config),
        kv_cache=kv_cache_bytes(spec, config),
        recurrent_state=recurrent_state_bytes(spec, config),
        conv_state=conv_state_bytes(spec, config),
        activations=activation_bytes(spec, config),
        runtime_overhead=int(config.runtime_overhead_gib * GIB),
        parameters=parameters,
    )


def max_context(spec: MoEStudentSpec = FROZEN_STUDENT,
                config: RuntimeConfig | None = None,
                usable_gib: float = USABLE_GIB) -> int:
    """Longest context on the ladder that still fits. 0 means the weights alone do not."""
    config = config or RuntimeConfig()
    best = 0
    for length in CONTEXT_LADDER:
        trial = account(spec, RuntimeConfig(**(config.to_dict() | {"context_length": length,
                                                                  "dense_quant": config.resolved_dense_quant})))
        if trial.verdict(usable_gib) == DOES_NOT_FIT:
            break
        best = length
    return best


@dataclass
class MemoryTable:
    """The quantisation x context matrix the release has to publish."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    usable_gib: float = USABLE_GIB
    spec_name: str = FROZEN_STUDENT.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec_name,
            "usable_gib": self.usable_gib,
            "nominal_vram_gib": NOMINAL_VRAM_GIB,
            "measured_total_gib": MEASURED_TOTAL_GIB,
            "gpu_resident_only": GPU_RESIDENT_ONLY,
            "offload_note": offload_note(),
            "rows": self.rows,
            "max_context_by_quant": {
                QUANT_LABELS.get(q, q): max(
                    [r["context_length"] for r in self.rows
                     if r["quant"] == q and r["verdict"] != DOES_NOT_FIT] or [0]
                )
                for q in RELEASE_QUANTS
            },
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def build_table(spec: MoEStudentSpec = FROZEN_STUDENT, *,
                quants: tuple[str, ...] = RELEASE_QUANTS,
                lengths: tuple[int, ...] = CONTEXT_LADDER,
                usable_gib: float = USABLE_GIB,
                base: RuntimeConfig | None = None) -> MemoryTable:
    base = base or RuntimeConfig()
    report = audit(spec)
    components, parameters = report["components"], report["exact_parameter_count"]
    rows: list[dict[str, Any]] = []
    for quant in quants:
        for length in lengths:
            config = RuntimeConfig(
                context_length=length, batch_size=base.batch_size,
                expert_quant=quant, dense_quant=quant,
                embedding_quant=base.embedding_quant,
                kv_cache_dtype=base.kv_cache_dtype,
                recurrent_state_dtype=base.recurrent_state_dtype,
                runtime_overhead_gib=base.runtime_overhead_gib,
                activation_safety_factor=base.activation_safety_factor,
                prefill_chunk_tokens=base.prefill_chunk_tokens,
            )
            acc = account(spec, config, components)
            acc.parameters = parameters
            row = acc.to_dict(usable_gib)
            rows.append({
                "quant": quant, "label": QUANT_LABELS.get(quant, quant),
                "context_length": length,
                "weights_gib": row["gib"]["weights"],
                "kv_cache_gib": row["gib"]["kv_cache"],
                "state_gib": row["gib"]["recurrent_state"] + row["gib"]["conv_state"],
                "activations_gib": row["gib"]["activations"],
                "runtime_gib": row["gib"]["runtime_overhead"],
                "total_gib": row["gib"]["total"],
                "headroom_gib": row["headroom_gib"],
                "verdict": row["verdict"],
            })
    return MemoryTable(rows=rows, usable_gib=usable_gib, spec_name=spec.name)


def offload_note() -> str:
    return (
        "Every figure here is for a fully GPU-resident model. CPU offload of experts or KV "
        "would change these totals and must never be used to claim 16 GB compliance: an "
        "offloaded configuration has different latency and is a different product. If a "
        "release ships an offloaded variant it is reported separately and labelled as such."
    )


def render_table(table: MemoryTable | None = None) -> str:
    table = table or build_table()
    data = table.to_dict()
    lines = [
        f"  {data['spec']} — end-to-end VRAM, fully GPU-resident",
        f"  ceiling {NOMINAL_VRAM_GIB:.0f} GB nominal / {MEASURED_TOTAL_GIB:.2f} GiB reported "
        f"/ {data['usable_gib']:.2f} GiB usable",
        "",
        "    quant  context   weights      KV     state    acts   runtime    TOTAL  "
        "headroom  verdict",
    ]
    for row in data["rows"]:
        lines.append(
            f"    {row['label']:<5}  {row['context_length']:>7,}  "
            f"{row['weights_gib']:>7.2f} {row['kv_cache_gib']:>7.2f} "
            f"{row['state_gib']:>9.3f} {row['activations_gib']:>7.2f} "
            f"{row['runtime_gib']:>9.2f} {row['total_gib']:>8.2f} "
            f"{row['headroom_gib']:>9.2f}  {row['verdict']}"
        )
    lines += ["", "    longest fitting context:"]
    for label, length in data["max_context_by_quant"].items():
        lines.append(f"      {label}: {length:,} tokens" if length
                     else f"      {label}: does not fit at any context")
    lines += ["", "    " + offload_note()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the frontier, and the finding it produced
# ---------------------------------------------------------------------------
#: Quantisations worth searching over, coarsest first. ``q3_k_m`` is below the three
#: precisions the release is required to publish and is included so the search can say
#: *how far* below them a fit is, rather than only that there isn't one.
SEARCH_QUANTS: tuple[str, ...] = ("q3_k_m", "int4", "q4_k_m", "q5_k_m", "q6_k", "q8_0")


def frontier(spec: MoEStudentSpec = FROZEN_STUDENT, *,
             usable_gib: float = USABLE_GIB,
             expert_quants: tuple[str, ...] = SEARCH_QUANTS,
             dense_quants: tuple[str, ...] = SEARCH_QUANTS,
             embedding_quants: tuple[str, ...] = SEARCH_QUANTS,
             base: RuntimeConfig | None = None) -> list[dict[str, Any]]:
    """Every quantisation combination that fits, with the longest context each reaches.

    Sorted by context reached, then by how much precision it kept. This is the Pareto view
    the release decision needs: the axes are capability (precision kept, context reached)
    against a fixed memory ceiling, and the frontier is the set of configurations where
    neither can be improved without losing the other.
    """
    base = base or RuntimeConfig()
    report = audit(spec)
    components = report["components"]
    order = {q: i for i, q in enumerate(SEARCH_QUANTS)}
    results: list[dict[str, Any]] = []
    for expert in expert_quants:
        for dense in dense_quants:
            for embedding in embedding_quants:
                reached, floor = 0, None
                for length in CONTEXT_LADDER:
                    config = RuntimeConfig(
                        context_length=length, batch_size=base.batch_size,
                        expert_quant=expert, dense_quant=dense, embedding_quant=embedding,
                        kv_cache_dtype=base.kv_cache_dtype,
                        recurrent_state_dtype=base.recurrent_state_dtype,
                        runtime_overhead_gib=base.runtime_overhead_gib,
                        activation_safety_factor=base.activation_safety_factor,
                        prefill_chunk_tokens=base.prefill_chunk_tokens,
                    )
                    acc = account(spec, config, components)
                    floor = acc.total_gib if floor is None else floor
                    if acc.headroom_gib(usable_gib) < 0:
                        break
                    reached = length
                if reached:
                    results.append({
                        "expert_quant": expert, "dense_quant": dense,
                        "embedding_quant": embedding, "max_context": reached,
                        "total_gib_at_shortest": floor,
                        "precision_rank": order.get(expert, 0) + order.get(dense, 0)
                                          + order.get(embedding, 0),
                        "uses_release_quant_only": all(
                            q in RELEASE_QUANTS for q in (expert, dense, embedding)
                        ),
                    })
    results.sort(key=lambda r: (-r["max_context"], -r["precision_rank"]))
    return results


def headline(spec: MoEStudentSpec = FROZEN_STUDENT,
             usable_gib: float = USABLE_GIB) -> dict[str, Any]:
    """The one-paragraph answer to "does the frozen student meet the 16 GB constraint?"

    It does not, and this returns the numbers that say so along with the closest thing that
    does. The conclusion is reported rather than engineered around: the architecture is
    frozen by the brief, so the finding belongs to the architecture, not to the accounting.
    """
    report = audit(spec)
    components = report["components"]
    fits = frontier(spec, usable_gib=usable_gib)
    release_only = [r for r in fits if r["uses_release_quant_only"]]
    smallest_release = account(
        spec,
        RuntimeConfig(context_length=CONTEXT_LADDER[0], expert_quant="q4_k_m",
                      dense_quant="q4_k_m", embedding_quant="q4_k_m"),
        components,
    )
    naive = [
        {"quant": q, "max_context": max(
            [length for length in CONTEXT_LADDER
             if account(spec, RuntimeConfig(context_length=length, expert_quant=q,
                                            dense_quant=q, embedding_quant=q),
                        components).total_gib <= NOMINAL_VRAM_GIB] or [0])}
        for q in RELEASE_QUANTS
    ]
    return {
        "parameters": report["exact_parameter_count"],
        "usable_gib": usable_gib,
        "fits_at_any_release_quant": bool(release_only),
        "best_case_release_quant_gib": smallest_release.total_gib,
        "shortfall_gib": max(0.0, smallest_release.total_gib - usable_gib),
        "configurations_that_fit": len(fits),
        "best_fitting": fits[0] if fits else None,
        "naive_nominal_16gib_result": naive,
        "finding": (
            "The frozen 22.07B student does not fit a real 16 GB card at Q4, Q5 or Q6 at any "
            "context length. The cheapest all-Q4 configuration needs "
            f"{smallest_release.total_gib:.2f} GiB at 2,048 tokens against {usable_gib:.2f} GiB "
            f"usable — short by {smallest_release.total_gib - usable_gib:.2f} GiB before a "
            "single long-context token is cached. Fits begin one precision step below the "
            "release set, at 3-bit experts. Planning against the nominal 16.0 GiB instead of "
            "a real card's 14.56 GiB reported capacity makes Q4 appear to reach 65,536 tokens; "
            "that difference is entirely the card's own overhead and the 1 GiB left for the "
            "rest of the system, and it is the arithmetic that produces an out-of-memory error "
            "on hardware that 'obviously' had room."
        ),
        "implication": (
            "Two honest options, and the choice belongs to whoever owns the release: publish "
            "the 22.07B target at 3-bit experts and report the quality cost, or reduce the "
            "expert budget — experts are 61.6% of the weights and each token uses 2 of 24, so "
            "expert count and expert width are the only levers with enough mass to close a "
            f"{smallest_release.total_gib - usable_gib:.2f} GiB gap without touching the parts "
            "of the model every token depends on. Nothing here alters the frozen architecture; "
            "it reports what the frozen architecture costs."
        ),
    }
