"""Analytical models of the Qwen3.5/3.8 hybrid architecture family.

Two student representations live here and they are **not** alternatives:

``moe_student``  the canonical frozen student, ``qwen38_19b_h5120_l48_moe``. This is the
                 project's architecture. :data:`FROZEN_STUDENT` is the specification,
                 :func:`materialise_student` fills it from the teacher.
``spec``         the dense ``HybridArchSpec`` family, which the *teacher* is described in
                 and which the retained ``h5120 L40`` baseline is built from. It is also
                 what the architecture search and the memory estimator operate on.

If you are reaching for a student to distil into, it is ``FROZEN_STUDENT``.
``student_from_teacher`` derives a dense one and is for the baseline and the chain
self-test, not for the experiment.
"""

from .flops import (
    bandwidth_bound_tokens_per_second,
    decode_bytes_per_token,
    decode_flops_per_token,
    prefill_flops,
)
from .materialize import (
    SafetensorsSource,
    StateDictSource,
    TransferReport,
    UnsupportedReduction,
    apply_transfer_plan,
    initialise_student,
)
from .memory import DeploymentConfig, MemoryEstimate, estimate_memory, max_context_within
from .moe_init import (
    LayerMapping,
    MaterialisationReport,
    map_layers,
    materialise_student,
    plan_ffn_decomposition,
)
from .moe_student import (
    FROZEN_STUDENT,
    PARAMETER_BUDGET,
    STUDENT_ID,
    MoEStudentSpec,
    audit,
    parameter_model,
)
from .params import ParamBreakdown, count_parameters, format_params, mtp_params
from .search import Candidate, SearchConstraints, evaluate_candidate, generate_grid, search
from .spec import FULL_ATTENTION, LINEAR_ATTENTION, HybridArchSpec, build_layer_types
from .transfer import (
    TransferPlan,
    build_transfer_plan,
    compare_strategies,
    select_layers,
    student_from_teacher,
)

__all__ = [
    # the canonical student
    "FROZEN_STUDENT",
    "STUDENT_ID",
    "MoEStudentSpec",
    "PARAMETER_BUDGET",
    "audit",
    "parameter_model",
    "map_layers",
    "materialise_student",
    "plan_ffn_decomposition",
    "LayerMapping",
    "MaterialisationReport",
    # the dense family: the teacher, the baseline, the search
    "HybridArchSpec",
    "build_layer_types",
    "FULL_ATTENTION",
    "LINEAR_ATTENTION",
    "ParamBreakdown",
    "count_parameters",
    "format_params",
    "mtp_params",
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
    "TransferPlan",
    "TransferReport",
    "UnsupportedReduction",
    "SafetensorsSource",
    "StateDictSource",
    "apply_transfer_plan",
    "initialise_student",
    "build_transfer_plan",
    "compare_strategies",
    "select_layers",
    "student_from_teacher",
]
