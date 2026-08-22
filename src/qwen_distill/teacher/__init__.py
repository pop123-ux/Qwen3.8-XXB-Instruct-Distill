"""Teacher checkpoint inspection, loader verification, and empirical validation."""

from .inspect import TeacherReport, cross_check, inspect_hub, inspect_local, read_safetensors_header
from .loader import LoaderReport, collect_versions, detect_remote_code, verify_loader
from .validate import ValidationResult, validate_cache_shapes, validate_parameters

__all__ = [
    "TeacherReport",
    "inspect_local",
    "inspect_hub",
    "cross_check",
    "read_safetensors_header",
    "LoaderReport",
    "verify_loader",
    "collect_versions",
    "detect_remote_code",
    "ValidationResult",
    "validate_parameters",
    "validate_cache_shapes",
]
