"""Teacher-data generation and student distillation.

The pipeline this package implements, and the reason it is shaped this way:

    prompts.jsonl -> teacher generation (expensive GPU) -> sharded JSONL + manifest
                                                                   |
                              student training (cheap GPU) <-------+

The teacher and the student never need to coexist. Generation runs once on hardware big
enough for a 27B model and produces a durable artifact; training consumes that artifact
on a T4, weeks later, on a different machine. That separation is what makes the project
affordable, and every design choice here protects it.
"""

from .backends import (
    MOCK_BACKEND,
    TRANSFORMERS_BACKEND,
    MockTeacher,
    TeacherBackend,
    TeacherResponse,
    TransformersTeacher,
    make_backend,
)
from .dataset import (
    DatasetFilter,
    DatasetStats,
    TeacherDataset,
    format_sft_example,
    load_teacher_dataset,
)
from .generation import (
    GenerationStats,
    Prompt,
    generate_dataset,
    iter_records,
    read_prompts,
    scan_completed_ids,
    write_prompts,
)
from .manifest import DatasetManifest, ShardRecord, shard_name
from .objectives import (
    IMPLEMENTED,
    LOGIT_KD,
    MIXED_KD,
    NOT_IMPLEMENTED,
    OBJECTIVES,
    SFT,
    ObjectiveConfig,
    ObjectiveSpec,
    ObjectiveUnavailable,
    describe_objectives,
)
from .provenance import RunManifest, TeacherIdentity, sha256_file, sha256_json, sha256_text
from .reasoning_modes import (
    DEFAULT_EFFORT,
    MODES,
    SUPPORTED_MODES,
    THINKING_DISABLED,
    ReasoningMode,
    UnsupportedReasoningMode,
    resolve_mode,
    sweep_modes,
)

__all__ = [
    "DEFAULT_EFFORT", "IMPLEMENTED", "LOGIT_KD", "MIXED_KD", "MOCK_BACKEND", "MODES",
    "NOT_IMPLEMENTED", "OBJECTIVES", "SFT", "SUPPORTED_MODES", "THINKING_DISABLED",
    "TRANSFORMERS_BACKEND", "DatasetFilter", "DatasetManifest", "DatasetStats",
    "GenerationStats", "MockTeacher", "ObjectiveConfig", "ObjectiveSpec",
    "ObjectiveUnavailable", "Prompt", "ReasoningMode", "RunManifest", "ShardRecord",
    "UnsupportedReasoningMode",
    "TeacherBackend", "TeacherDataset", "TeacherIdentity", "TeacherResponse",
    "TransformersTeacher", "describe_objectives", "format_sft_example",
    "generate_dataset", "iter_records", "load_teacher_dataset", "make_backend",
    "read_prompts", "resolve_mode", "scan_completed_ids", "sha256_file", "sha256_json",
    "sha256_text", "shard_name", "sweep_modes", "write_prompts",
]
