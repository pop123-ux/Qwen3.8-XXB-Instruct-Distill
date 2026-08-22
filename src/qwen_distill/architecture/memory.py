"""Deployment-envelope VRAM estimation for hybrid DeltaNet/attention models.

The project's hard constraint is a 16 GB consumer GPU, and the mandate is explicit
that "fits" must not mean "the weight file is smaller than the card". This module
therefore models the whole envelope::

    weights + KV cache + recurrent state + conv state + activations + runtime overhead

The structurally important property of this architecture family is that only the
``full_attention`` layers hold a cache that grows with sequence length. The
``linear_attention`` (Gated DeltaNet) layers hold a *constant-size* recurrent state
of shape ``(batch, num_v_heads, head_k_dim, head_v_dim)`` plus a small depthwise
conv state, independent of context length. With a 3:1 layout that removes 75% of
the usual KV-cache growth, which is what makes long context affordable here.

All figures are analytical estimates. They are labelled as such everywhere and are
not a substitute for a measured peak-VRAM run (see ``scripts/benchmark_memory.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .params import ParamBreakdown, count_parameters
from .spec import FULL_ATTENTION, HybridArchSpec

GIB = 1024 ** 3
MIB = 1024 ** 2

#: Effective bytes-per-parameter for common deployment formats.
#: The sub-8-bit entries are *effective* rates for block-quantised schemes
#: (GGUF K-quants, AWQ, GPTQ): they include the per-block scale/zero-point
#: overhead, which is why e.g. Q4_K_M lands near 4.5 bits rather than 4.0.
QUANT_BYTES_PER_PARAM: dict[str, float] = {
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "q8_0": 8.5 / 8,
    "q6_k": 6.6 / 8,
    "q5_k_m": 5.7 / 8,
    "q4_k_m": 4.9 / 8,
    "int4": 4.5 / 8,
    "q3_k_m": 3.9 / 8,
}

DTYPE_BYTES: dict[str, float] = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0}

QuantName = Literal[
    "bf16", "fp16", "fp8", "int8", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "int4", "q3_k_m"
]


@dataclass(frozen=True)
class DeploymentConfig:
    """One row of a deployment matrix: how the model is actually being run."""

    context_length: int = 32768
    batch_size: int = 1
    weight_quant: str = "q4_k_m"
    #: Embeddings and the LM head are commonly left at higher precision by
    #: GGUF/AWQ/GPTQ packers because they are quantisation-sensitive. With a
    #: 248k vocab this is a large correction, so it is modelled explicitly.
    embedding_quant: str | None = "q6_k"
    kv_cache_dtype: str = "fp16"
    #: The torch reference path accumulates the delta-rule state in fp32; fp32 is
    #: the conservative default. Fused kernels may keep it in bf16.
    recurrent_state_dtype: str = "fp32"
    #: Fixed allocator/CUDA-context/framework cost. ~0.6-1.0 GiB is typical for a
    #: PyTorch+CUDA process before any model tensors are allocated.
    runtime_overhead_bytes: int = int(0.9 * GIB)
    #: Multiplier on transient activation working set to cover fragmentation and
    #: framework slack. 1.0 disables the margin.
    activation_safety_factor: float = 1.25
    #: Longest chunk of tokens processed in a single forward pass (prefill chunking).
    #: vLLM and llama.cpp both chunk prefill; this bounds peak activation memory.
    prefill_chunk_tokens: int = 2048


@dataclass(frozen=True)
class MemoryEstimate:
    """Analytical VRAM breakdown, all fields in bytes."""

    spec_name: str
    config: DeploymentConfig
    params: ParamBreakdown
    weights: int
    kv_cache: int
    recurrent_state: int
    conv_state: int
    activations: int
    runtime_overhead: int

    @property
    def total(self) -> int:
        return (
            self.weights
            + self.kv_cache
            + self.recurrent_state
            + self.conv_state
            + self.activations
            + self.runtime_overhead
        )

    @property
    def total_gib(self) -> float:
        return self.total / GIB

    def fits_in(self, vram_gib: float) -> bool:
        return self.total_gib <= vram_gib

    def headroom_gib(self, vram_gib: float) -> float:
        return vram_gib - self.total_gib

    def as_dict(self) -> dict[str, float]:
        return {
            "weights_gib": self.weights / GIB,
            "kv_cache_gib": self.kv_cache / GIB,
            "recurrent_state_gib": self.recurrent_state / GIB,
            "conv_state_gib": self.conv_state / GIB,
            "activations_gib": self.activations / GIB,
            "runtime_overhead_gib": self.runtime_overhead / GIB,
            "total_gib": self.total_gib,
        }


def _bytes_per_param(quant: str) -> float:
    try:
        return QUANT_BYTES_PER_PARAM[quant]
    except KeyError:
        raise ValueError(
            f"unknown weight quantisation {quant!r}; known: {sorted(QUANT_BYTES_PER_PARAM)}"
        ) from None


def _dtype_bytes(dtype: str) -> float:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(DTYPE_BYTES)}") from None


def weight_bytes(spec: HybridArchSpec, config: DeploymentConfig) -> int:
    """Weight memory, allowing embeddings/head to use a different precision."""
    params = count_parameters(spec)
    body_bpp = _bytes_per_param(config.weight_quant)
    embed_params = params.embedding + params.lm_head
    if config.embedding_quant is None:
        return int(params.total * body_bpp)
    embed_bpp = _bytes_per_param(config.embedding_quant)
    return int(params.non_embedding * body_bpp + embed_params * embed_bpp)


def kv_cache_bytes(spec: HybridArchSpec, config: DeploymentConfig) -> int:
    """KV cache over the *full-attention layers only*.

    Per token per full-attention layer: ``2 (K and V) * num_key_value_heads * head_dim``.
    Linear-attention layers contribute nothing here — that is the whole point of the
    hybrid layout.
    """
    per_token_per_layer = 2 * spec.num_key_value_heads * spec.head_dim
    n_full = sum(1 for t in spec.resolved_layer_types() if t == FULL_ATTENTION)
    elements = per_token_per_layer * n_full * config.context_length * config.batch_size
    return int(elements * _dtype_bytes(config.kv_cache_dtype))


def recurrent_state_bytes(spec: HybridArchSpec, config: DeploymentConfig) -> int:
    """Gated DeltaNet recurrent state: ``(batch, num_v_heads, head_k_dim, head_v_dim)`` per layer.

    Constant in context length.
    """
    per_layer = (
        spec.linear_num_value_heads * spec.linear_key_head_dim * spec.linear_value_head_dim
    )
    elements = per_layer * spec.num_linear_attention_layers * config.batch_size
    return int(elements * _dtype_bytes(config.recurrent_state_dtype))


def conv_state_bytes(spec: HybridArchSpec, config: DeploymentConfig) -> int:
    """Depthwise causal-conv1d state: ``(batch, conv_dim, kernel_size)`` per linear layer."""
    per_layer = spec.linear_conv_dim * spec.linear_conv_kernel_dim
    elements = per_layer * spec.num_linear_attention_layers * config.batch_size
    return int(elements * _dtype_bytes(config.recurrent_state_dtype))


def activation_bytes(spec: HybridArchSpec, config: DeploymentConfig) -> int:
    """Transient inference working set for one chunked forward pass.

    Inference activations are not retained across layers, so the peak is driven by
    the widest few tensors live at once within a single layer, over the largest
    chunk of tokens processed together. The dominant terms are the MLP intermediate
    (``intermediate_size``), the DeltaNet projections (``conv_dim + value_dim``) and
    the gated-attention q/gate projection (``2 * n_heads * head_dim``); we take the
    widest, allow a few concurrent buffers, and add the output logits.
    """
    tokens = min(config.prefill_chunk_tokens, config.context_length) * config.batch_size
    act_dtype = _dtype_bytes("bf16")

    widest_layer_tensor = max(
        spec.intermediate_size,
        spec.linear_conv_dim + spec.linear_value_dim,
        2 * spec.num_attention_heads * spec.head_dim,
    )
    # ~3 concurrent buffers of the widest tensor (e.g. gate, up, product) plus the
    # residual stream carried alongside.
    per_token = 3 * widest_layer_tensor + 2 * spec.hidden_size
    layer_activations = tokens * per_token * act_dtype

    # Output logits are materialised over the vocabulary; with a 248k vocab this is
    # significant, and frameworks commonly compute them in fp32.
    logits = config.batch_size * spec.vocab_size * _dtype_bytes("fp32")

    return int((layer_activations + logits) * config.activation_safety_factor)


def estimate_memory(spec: HybridArchSpec, config: DeploymentConfig | None = None) -> MemoryEstimate:
    """Analytical peak-VRAM estimate for ``spec`` under ``config``."""
    config = config or DeploymentConfig()
    return MemoryEstimate(
        spec_name=spec.name,
        config=config,
        params=count_parameters(spec),
        weights=weight_bytes(spec, config),
        kv_cache=kv_cache_bytes(spec, config),
        recurrent_state=recurrent_state_bytes(spec, config),
        conv_state=conv_state_bytes(spec, config),
        activations=activation_bytes(spec, config),
        runtime_overhead=config.runtime_overhead_bytes,
    )


def max_context_within(
    spec: HybridArchSpec,
    vram_gib: float,
    config: DeploymentConfig | None = None,
    *,
    ceiling: int | None = None,
) -> int:
    """Largest context length (in tokens) that fits ``vram_gib``, by bisection.

    Returns 0 if the model does not fit even at a 1k context. The result is capped
    at ``ceiling`` (default: the spec's ``max_position_embeddings``).
    """
    from dataclasses import replace

    base = config or DeploymentConfig()
    hi_cap = ceiling if ceiling is not None else spec.max_position_embeddings

    def fits(ctx: int) -> bool:
        return estimate_memory(spec, replace(base, context_length=ctx)).fits_in(vram_gib)

    if not fits(1024):
        return 0
    if fits(hi_cap):
        return hi_cap

    lo, hi = 1024, hi_cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo
