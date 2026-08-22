"""Forward-pass FLOP accounting for hybrid DeltaNet/attention models.

Counts multiply-accumulate work as ``2 * M * K`` per matrix-vector product (one
multiply + one add), which is the usual convention. Two regimes matter:

* **per-token decode cost** — dominated by weight-matrix GEMVs, independent of
  context for the linear layers and linear-in-context for the attention layers;
* **prefill cost over a sequence** — where the quadratic attention term appears,
  but only in the ``full_attention`` layers.

Memory bandwidth, not FLOPs, is usually the binding constraint for single-stream
decode on a consumer GPU; :func:`decode_bytes_per_token` exposes that separately.
"""

from __future__ import annotations

from dataclasses import dataclass

from .memory import _bytes_per_param
from .params import count_parameters, full_attention_params, linear_attention_params, mlp_params
from .spec import HybridArchSpec


@dataclass(frozen=True)
class FlopBreakdown:
    """FLOPs for a single token at a given context position."""

    mlp: int
    full_attention_proj: int
    full_attention_scores: int
    linear_attention_proj: int
    linear_attention_state: int
    lm_head: int

    @property
    def total(self) -> int:
        return (
            self.mlp
            + self.full_attention_proj
            + self.full_attention_scores
            + self.linear_attention_proj
            + self.linear_attention_state
            + self.lm_head
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "mlp": self.mlp,
            "full_attention_proj": self.full_attention_proj,
            "full_attention_scores": self.full_attention_scores,
            "linear_attention_proj": self.linear_attention_proj,
            "linear_attention_state": self.linear_attention_state,
            "lm_head": self.lm_head,
            "total": self.total,
        }


def decode_flops_per_token(spec: HybridArchSpec, context_length: int = 0) -> FlopBreakdown:
    """FLOPs to decode one token with ``context_length`` tokens already in cache.

    The ``full_attention_scores`` term is the only one that grows with context: each
    full-attention layer scores the new query against every cached key and then mixes
    the values, i.e. ``2 * 2 * n_heads * head_dim * context`` per layer.
    """
    n_full = spec.num_full_attention_layers
    n_linear = spec.num_linear_attention_layers

    mlp = 2 * mlp_params(spec) * spec.num_hidden_layers

    # Projection GEMVs: parameter count minus the tiny norm terms, times 2.
    attn_proj_params = full_attention_params(spec) - 2 * spec.head_dim
    full_proj = 2 * attn_proj_params * n_full

    # QK^T over the cache plus the AV mix, per full-attention layer.
    scores_per_layer = 2 * 2 * spec.num_attention_heads * spec.head_dim * context_length
    full_scores = scores_per_layer * n_full

    lin_proj_params = linear_attention_params(spec) - (
        2 * spec.linear_num_value_heads + spec.linear_value_head_dim
    )
    lin_proj = 2 * lin_proj_params * n_linear

    # Delta-rule state update and readout. The state is
    # (num_v_heads, head_k_dim, head_v_dim); each token performs a handful of
    # elementwise passes over it (decay, kv read, rank-1 write, query readout).
    state_elems = (
        spec.linear_num_value_heads * spec.linear_key_head_dim * spec.linear_value_head_dim
    )
    lin_state = 8 * state_elems * n_linear

    lm_head = 2 * spec.vocab_size * spec.hidden_size

    return FlopBreakdown(
        mlp=mlp,
        full_attention_proj=full_proj,
        full_attention_scores=full_scores,
        linear_attention_proj=lin_proj,
        linear_attention_state=lin_state,
        lm_head=lm_head,
    )


def prefill_flops(spec: HybridArchSpec, sequence_length: int) -> int:
    """Total FLOPs to prefill a sequence of ``sequence_length`` tokens.

    The quadratic term ``O(L^2)`` applies only to the full-attention layers; summing
    the per-position linear term over the sequence gives ``L^2 / 2`` per layer.
    """
    per_token_linear = decode_flops_per_token(spec, context_length=0)
    linear_part = per_token_linear.total * sequence_length

    scores_per_layer_per_pair = 2 * 2 * spec.num_attention_heads * spec.head_dim
    quadratic = (
        scores_per_layer_per_pair
        * spec.num_full_attention_layers
        * (sequence_length * (sequence_length - 1) // 2)
    )
    return linear_part + quadratic


def decode_bytes_per_token(spec: HybridArchSpec, weight_quant: str = "q4_k_m") -> int:
    """Bytes of weights that must be read from VRAM to decode one token.

    Single-stream decode reads essentially every weight once per token, so this
    divided by the GPU's achievable bandwidth is a hard upper bound on tokens/sec —
    usually a tighter bound than the FLOP count on consumer hardware.
    """
    params = count_parameters(spec)
    return int(params.total * _bytes_per_param(weight_quant))


def bandwidth_bound_tokens_per_second(
    spec: HybridArchSpec,
    memory_bandwidth_gb_s: float,
    weight_quant: str = "q4_k_m",
    efficiency: float = 0.75,
) -> float:
    """Upper bound on single-stream decode throughput.

    ``efficiency`` accounts for the fraction of theoretical bandwidth a real kernel
    achieves (0.7-0.85 is typical for well-tuned dequant-GEMV kernels). This is a
    ceiling, not a prediction: it ignores kernel launch overhead, sampling, and the
    cache reads that grow with context.
    """
    bytes_per_token = decode_bytes_per_token(spec, weight_quant)
    if bytes_per_token == 0:
        return float("inf")
    return (memory_bandwidth_gb_s * 1e9 * efficiency) / bytes_per_token


def format_flops(n: float) -> str:
    for threshold, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= threshold:
            return f"{n / threshold:.2f}{suffix}"
    return f"{n:.0f}"
