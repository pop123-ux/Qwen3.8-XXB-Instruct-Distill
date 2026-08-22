"""Exact parameter accounting for the Qwen3.5/3.8 hybrid text tower.

Every term below is transcribed from ``transformers.models.qwen3_5.modeling_qwen3_5``
(v5.15.1). The module-by-module provenance is:

``Qwen3_5MLP``           gate_proj, up_proj: (hidden, inter); down_proj: (inter, hidden); no bias.
``Qwen3_5Attention``     q_proj: (hidden, n_heads * head_dim * 2)  <- the 2x is the output gate,
                         k_proj/v_proj: (hidden, n_kv * head_dim), o_proj: (n_heads*head_dim, hidden),
                         q_norm/k_norm: head_dim each. bias = config.attention_bias (default False).
``Qwen3_5GatedDeltaNet`` in_proj_qkv: (hidden, conv_dim), in_proj_z: (hidden, value_dim),
                         in_proj_b/in_proj_a: (hidden, num_v_heads) each,
                         out_proj: (value_dim, hidden), conv1d: depthwise groups=conv_dim,
                         bias=False -> conv_dim * kernel; dt_bias & A_log: num_v_heads each;
                         norm (RMSNormGated over head_v_dim): head_v_dim.
``Qwen3_5DecoderLayer``  input_layernorm + post_attention_layernorm: hidden each.
``Qwen3_5TextModel``     embed_tokens: (vocab, hidden); final norm: hidden.
``Qwen3_5ForCausalLM``   lm_head: (vocab, hidden) unless tie_word_embeddings.

The gated-attention ``2x`` on ``q_proj`` and the ``in_proj_z`` gate on DeltaNet are the two
terms most often missed by naive estimators; both are material at 27B scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from .spec import FULL_ATTENTION, LINEAR_ATTENTION, HybridArchSpec


@dataclass(frozen=True)
class ParamBreakdown:
    """Parameter counts, in units of parameters (not bytes)."""

    embedding: int
    lm_head: int
    final_norm: int
    layer_norms: int
    mlp: int
    full_attention: int
    linear_attention: int

    @property
    def non_embedding(self) -> int:
        """Everything except the input embedding and the (untied) output head."""
        return self.final_norm + self.layer_norms + self.mlp + self.full_attention + self.linear_attention

    @property
    def total(self) -> int:
        return self.embedding + self.lm_head + self.non_embedding

    def as_dict(self) -> dict[str, int]:
        return {
            "embedding": self.embedding,
            "lm_head": self.lm_head,
            "final_norm": self.final_norm,
            "layer_norms": self.layer_norms,
            "mlp": self.mlp,
            "full_attention": self.full_attention,
            "linear_attention": self.linear_attention,
            "non_embedding": self.non_embedding,
            "total": self.total,
        }

    def shares(self) -> dict[str, float]:
        """Fraction of total parameters held by each component."""
        total = self.total
        if total == 0:
            return {}
        return {k: v / total for k, v in self.as_dict().items() if k not in ("total",)}


def mlp_params(spec: HybridArchSpec) -> int:
    """Per-layer SwiGLU MLP parameters (gate + up + down, no bias)."""
    return 3 * spec.hidden_size * spec.intermediate_size


def full_attention_params(spec: HybridArchSpec) -> int:
    """Per-layer gated-attention parameters.

    ``q_proj`` emits ``n_heads * head_dim * 2`` because the second half is the
    sigmoid output gate applied after attention (``attn_output * sigmoid(gate)``).
    """
    h = spec.hidden_size
    q_out = spec.num_attention_heads * spec.head_dim * 2
    kv_out = spec.num_key_value_heads * spec.head_dim
    total = h * q_out + 2 * (h * kv_out) + spec.attention_query_dim * h
    if spec.attention_bias:
        total += q_out + 2 * kv_out + h
    total += 2 * spec.head_dim  # q_norm + k_norm, over head_dim only
    return total


def linear_attention_params(spec: HybridArchSpec) -> int:
    """Per-layer Gated DeltaNet parameters."""
    h = spec.hidden_size
    conv_dim = spec.linear_conv_dim
    value_dim = spec.linear_value_dim
    n_v = spec.linear_num_value_heads
    total = h * conv_dim  # in_proj_qkv
    total += h * value_dim  # in_proj_z (gate)
    total += 2 * (h * n_v)  # in_proj_b, in_proj_a
    total += value_dim * h  # out_proj
    total += conv_dim * spec.linear_conv_kernel_dim  # depthwise conv1d, bias=False
    total += 2 * n_v  # dt_bias, A_log
    total += spec.linear_value_head_dim  # RMSNormGated over head_v_dim
    return total


def count_parameters(spec: HybridArchSpec) -> ParamBreakdown:
    """Full parameter breakdown for the text tower of ``spec``."""
    layer_types = spec.resolved_layer_types()
    n_full = sum(1 for t in layer_types if t == FULL_ATTENTION)
    n_linear = sum(1 for t in layer_types if t == LINEAR_ATTENTION)

    embedding = spec.vocab_size * spec.hidden_size
    lm_head = 0 if spec.tie_word_embeddings else spec.vocab_size * spec.hidden_size

    return ParamBreakdown(
        embedding=embedding,
        lm_head=lm_head,
        final_norm=spec.hidden_size,
        layer_norms=2 * spec.hidden_size * spec.num_hidden_layers,
        mlp=mlp_params(spec) * spec.num_hidden_layers,
        full_attention=full_attention_params(spec) * n_full,
        linear_attention=linear_attention_params(spec) * n_linear,
    )


def mtp_params(spec: HybridArchSpec, num_mtp_layers: int = 1) -> int:
    """Parameters of the Multi-Token Prediction head, as vLLM builds it.

    Verified against ``vllm/model_executor/models/qwen3_5_mtp.py`` (vLLM 0.27.1),
    where ``Qwen3_5MTP`` is registered under ``_SPECULATIVE_DECODING_MODELS``:

    * ``fc``: ``Linear(hidden * 2 -> hidden, bias=False)`` — the draft head concatenates
      the normalised next-token embedding with the normalised main-model hidden state;
    * ``layers``: ``num_mtp_layers`` full ``Qwen3_5DecoderLayer`` blocks, each
      constructed with ``layer_type="full_attention"`` — MTP layers are **never**
      DeltaNet layers;
    * three RMSNorms: ``norm``, ``pre_fc_norm_hidden``, ``pre_fc_norm_embedding``.

    ``embed_tokens`` and ``lm_head`` are **shared with the base model**, not duplicated:
    vLLM's loader routes only ``mtp.*`` tensors into the draft model and reuses the
    main checkpoint's embedding and head. They are therefore excluded here.

    Defaults to ``num_mtp_layers=1``, matching vLLM's
    ``getattr(config, "mtp_num_hidden_layers", 1)``.
    """
    h = spec.hidden_size
    per_layer = full_attention_params(spec) + mlp_params(spec) + 2 * h
    return 2 * h * h + num_mtp_layers * per_layer + 3 * h


def format_params(n: int) -> str:
    """Human-readable parameter count, e.g. ``26.90B``."""
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= threshold:
            return f"{n / threshold:.2f}{suffix}"
    return str(n)
