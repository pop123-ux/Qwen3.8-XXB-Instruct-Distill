"""Student training: configuration, feasibility, data, and the training loop."""

from .config import ExperimentConfig, ModelConfig, TrainingConfig
from .data import DistillationExample, read_jsonl, synthetic_corpus, write_jsonl
from .feasibility import FeasibilityReport, check_feasibility

__all__ = [
    "ExperimentConfig", "ModelConfig", "TrainingConfig",
    "FeasibilityReport", "check_feasibility",
    "DistillationExample", "read_jsonl", "write_jsonl", "synthetic_corpus",
]
