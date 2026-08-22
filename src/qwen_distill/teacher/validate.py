"""Empirically validate the analytical model against the reference implementation.

Phase 0 derived parameter and memory formulas by *reading*
``transformers.models.qwen3_5``. Reading can be wrong. This module checks the
formulas by building the real thing and measuring it:

* :func:`validate_parameters` instantiates the architecture with `transformers` on the
  ``meta`` device — shapes and dtypes, but **zero storage** — so even the full 27B
  structure can be built on a laptop with no GPU, then compares
  ``sum(p.numel())`` against :func:`qwen_distill.architecture.params.count_parameters`,
  component by component.
* :func:`validate_cache_shapes` runs a real forward pass on a small model and compares
  the actual KV cache, DeltaNet recurrent state and conv state against the memory model.

Neither requires the teacher checkpoint, so both run in CI. They validate the
*formulas*; they do not validate that the teacher's config values are what we assume
(see :mod:`qwen_distill.teacher.inspect` for that).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..architecture.memory import (
    DeploymentConfig,
    conv_state_bytes,
    kv_cache_bytes,
    recurrent_state_bytes,
)
from ..architecture.params import count_parameters
from ..architecture.spec import HybridArchSpec

#: Component keys compared between the analytical model and the built model.
COMPONENTS = (
    "embedding", "lm_head", "final_norm", "layer_norms",
    "mlp", "full_attention", "linear_attention",
)


@dataclass
class ValidationResult:
    """Outcome of one validation check."""

    name: str
    passed: bool
    comparisons: dict[str, dict[str, int]] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "comparisons": self.comparisons,
            "details": self.details,
            "error": self.error,
        }


def _build_text_config(spec: HybridArchSpec):
    """Build a ``Qwen3_5TextConfig`` from our spec."""
    from transformers import AutoConfig

    fields = {k: v for k, v in spec.to_hf_text_config().items() if k != "model_type"}
    return AutoConfig.for_model("qwen3_5_text", **fields)


def group_parameters(model) -> dict[str, int]:
    """Group a model's parameters into the same components as ``ParamBreakdown``."""
    groups = dict.fromkeys((*COMPONENTS, "other"), 0)
    for name, param in model.named_parameters():
        n = param.numel()
        if "embed_tokens" in name:
            groups["embedding"] += n
        elif "lm_head" in name:
            groups["lm_head"] += n
        elif ".mlp." in name:
            groups["mlp"] += n
        elif ".self_attn." in name:
            groups["full_attention"] += n
        elif ".linear_attn." in name:
            groups["linear_attention"] += n
        elif "input_layernorm" in name or "post_attention_layernorm" in name:
            groups["layer_norms"] += n
        elif name.endswith("norm.weight"):
            groups["final_norm"] += n
        else:
            groups["other"] += n
    return groups


def validate_parameters(spec: HybridArchSpec) -> ValidationResult:
    """Check our parameter formulas against a real `transformers` instantiation.

    Uses the ``meta`` device, so no memory is allocated for weights.
    """
    result = ValidationResult(name=f"parameters[{spec.name}]", passed=False)
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        result.error = f"torch/transformers not installed: {exc}"
        return result

    try:
        config = _build_text_config(spec)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
    except Exception as exc:  # noqa: BLE001 - report verbatim
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    actual_groups = group_parameters(model)
    ours = count_parameters(spec).as_dict()

    for key in COMPONENTS:
        result.comparisons[key] = {
            "measured": actual_groups[key],
            "analytical": ours[key],
            "delta": actual_groups[key] - ours[key],
        }
    measured_total = sum(p.numel() for p in model.parameters())
    result.comparisons["total"] = {
        "measured": measured_total,
        "analytical": ours["total"],
        "delta": measured_total - ours["total"],
    }
    result.details = {
        "model_class": type(model).__name__,
        "model_module": type(model).__module__,
        "config_class": type(config).__name__,
        "unmatched_parameters": actual_groups["other"],
        "n_parameter_tensors": sum(1 for _ in model.named_parameters()),
        "layer_types_head": list(config.layer_types[:8]),
        "n_full_attention": sum(1 for t in config.layer_types if t == "full_attention"),
    }
    result.passed = (
        all(c["delta"] == 0 for c in result.comparisons.values())
        and actual_groups["other"] == 0
    )
    return result


def validate_cache_shapes(spec: HybridArchSpec, sequence_length: int = 37) -> ValidationResult:
    """Check the memory model's cache terms against a real forward pass.

    Runs on CPU with real (small) weights, so ``spec`` must be small enough to
    instantiate. Compares three quantities the deployment envelope depends on:
    the KV cache, the DeltaNet recurrent state, and the depthwise conv state.
    """
    result = ValidationResult(name=f"cache_shapes[{spec.name}]", passed=False)
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        result.error = f"torch/transformers not installed: {exc}"
        return result

    try:
        config = _build_text_config(spec)
        model = AutoModelForCausalLM.from_config(config).eval()
        with torch.no_grad():
            output = model(
                torch.randint(0, spec.vocab_size, (1, sequence_length)), use_cache=True
            )
        cache = output.past_key_values
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    kv_bytes = 0
    kv_layer_indices: list[int] = []
    recurrent_bytes = 0
    conv_bytes = 0
    recurrent_shapes: list[list[int]] = []
    conv_shapes: list[list[int]] = []

    for index, layer in enumerate(cache.layers):
        keys = getattr(layer, "keys", None)
        if keys is not None and keys.numel() > 0:
            kv_layer_indices.append(index)
            kv_bytes += (keys.numel() + layer.values.numel()) * keys.element_size()
        # Linear-attention layers keep their state in per-layer dicts.
        for attribute, store in (
            ("recurrent_states", getattr(layer, "recurrent_states", None)),
            ("conv_states", getattr(layer, "conv_states", None)),
        ):
            if not isinstance(store, dict):
                continue
            for tensor in store.values():
                if tensor is None or not torch.is_tensor(tensor):
                    continue
                size = tensor.numel() * tensor.element_size()
                if attribute == "recurrent_states":
                    recurrent_bytes += size
                    recurrent_shapes.append(list(tensor.shape))
                else:
                    conv_bytes += size
                    conv_shapes.append(list(tensor.shape))

    # Match the dtypes the reference path actually used, so we compare like with like.
    deployment = DeploymentConfig(
        context_length=sequence_length,
        batch_size=1,
        kv_cache_dtype="fp32",
        recurrent_state_dtype="fp32",
    )
    result.comparisons = {
        "kv_cache_bytes": {
            "measured": kv_bytes,
            "analytical": kv_cache_bytes(spec, deployment),
            "delta": kv_bytes - kv_cache_bytes(spec, deployment),
        },
        "recurrent_state_bytes": {
            "measured": recurrent_bytes,
            "analytical": recurrent_state_bytes(spec, deployment),
            "delta": recurrent_bytes - recurrent_state_bytes(spec, deployment),
        },
        "conv_state_bytes": {
            "measured": conv_bytes,
            "analytical": conv_state_bytes(spec, deployment),
            "delta": conv_bytes - conv_state_bytes(spec, deployment),
        },
    }
    expected_kv_layers = [
        i for i, t in enumerate(config.layer_types) if t == "full_attention"
    ]
    result.details = {
        "cache_class": type(cache).__name__,
        "layer_types": list(config.layer_types),
        "layers_with_kv_cache": kv_layer_indices,
        "expected_kv_layers": expected_kv_layers,
        "kv_only_on_full_attention": kv_layer_indices == expected_kv_layers,
        "recurrent_state_shape": recurrent_shapes[0] if recurrent_shapes else None,
        "conv_state_shape": conv_shapes[0] if conv_shapes else None,
        "sequence_length": sequence_length,
    }
    result.passed = (
        all(c["delta"] == 0 for c in result.comparisons.values())
        and kv_layer_indices == expected_kv_layers
    )
    return result


#: A small but structurally faithful model: hybrid layout, GQA, DeltaNet head ratios.
SMALL_SPEC = HybridArchSpec(
    name="validation-mini",
    hidden_size=256,
    num_hidden_layers=8,
    intermediate_size=512,
    vocab_size=1024,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    linear_num_key_heads=2,
    linear_num_value_heads=6,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    full_attention_interval=4,
    max_position_embeddings=2048,
)
