"""Analytical models of the Qwen3.5/3.8 hybrid architecture family."""

from .flops import (
    bandwidth_bound_tokens_per_second,
    decode_bytes_per_token,
    decode_flops_per_token,
    prefill_flops,
)
from .memory import DeploymentConfig, MemoryEstimate, estimate_memory, max_context_within
from .params import ParamBreakdown, count_parameters, format_params
from .search import Candidate, SearchConstraints, evaluate_candidate, generate_grid, search
from .spec import FULL_ATTENTION, LINEAR_ATTENTION, HybridArchSpec, build_layer_types

__all__ = [
    "HybridArchSpec",
    "build_layer_types",
    "FULL_ATTENTION",
    "LINEAR_ATTENTION",
    "ParamBreakdown",
    "count_parameters",
    "format_params",
    "DeploymentConfig",
    "MemoryEstimate",
    "estimate_memory",
    "max_context_within",
    "decode_flops_per_token",
    "prefill_flops",
    "decode_bytes_per_token",
    "bandwidth_bound_tokens_per_second",
    "Candidate",
    "SearchConstraints",
    "evaluate_candidate",
    "generate_grid",
    "search",
]
