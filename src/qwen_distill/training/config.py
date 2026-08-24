"""Experiment configuration for student training, from toy scale to the final run.

One configuration schema spans every level of the development ladder, so a recipe
validated on a T4 is the *same* recipe scaled up rather than a different script. That
matters: a separate training script per scale is how a pipeline silently diverges
between what you tested and what you ran.

Nothing here chooses a student architecture. The spec is supplied per experiment,
because the project has not decided that yet and encoding a guess in the trainer would
quietly make it the default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..architecture.spec import HybridArchSpec

#: Training strategies, cheapest first. See docs/TRAINING_ON_LIMITED_HARDWARE.md for
#: why LoRA is a prototyping tool here rather than a route to the final model.
STRATEGIES = ("full", "lora", "qlora")
OPTIMIZERS = ("adamw", "adamw_8bit", "adafactor", "sgd")
PRECISIONS = ("bf16", "fp16", "fp32")
OBJECTIVES = ("sft", "logit_kd", "mixed_kd")


@dataclass
class ModelConfig:
    """Where the student comes from: a spec, a checkpoint, or a preset."""

    #: Path to a saved HybridArchSpec JSON. Mutually exclusive with `pretrained`.
    spec_path: str | None = None
    #: A pretrained checkpoint to initialise from (the recommended starting point:
    #: see docs/TRAINING_ON_LIMITED_HARDWARE.md on why not to pretrain from scratch).
    pretrained: str | None = None
    #: Inline architecture overrides, used for toy/prototype runs.
    architecture: dict[str, Any] = field(default_factory=dict)
    trust_remote_code: bool = False

    def resolve_spec(self, base_dir: Path | None = None) -> HybridArchSpec | None:
        """Build the student spec, if this config defines one architecturally."""
        if self.spec_path:
            path = Path(self.spec_path)
            if base_dir and not path.is_absolute():
                path = base_dir / path
            return HybridArchSpec.load(path)
        if self.architecture:
            return HybridArchSpec(name="from-config", **self.architecture)
        return None


@dataclass
class DataConfig:
    """Where training examples come from.

    Teacher generation and student training are deliberately separate operations: the
    dataset is produced once (possibly on rented hardware) and consumed offline, so a
    16 GB card never has to hold the teacher and the student at the same time.
    """

    train_path: str | None = None
    validation_path: str | None = None
    #: Use a deterministic synthetic *induction* task instead of a corpus. This is the
    #: Level-1 mechanism test: it proves the optimizer works, not that the model learns
    #: language. Do not use it to claim a language result.
    synthetic: bool = False
    synthetic_examples: int = 256
    #: Byte-level language modelling on real text. `text_path` points at any UTF-8 file;
    #: when `text_corpus` is set without a path, a deterministic procedural corpus is
    #: generated offline. This is the Level-2 objective.
    text_corpus: bool = False
    text_path: str | None = None
    procedural_bytes: int = 2_000_000
    validation_fraction: float = 0.05
    max_corpus_bytes: int | None = None
    max_sequence_length: int = 1024
    streaming: bool = False
    shuffle_seed: int = 0

    @property
    def mode(self) -> str:
        """Which data path this config selects."""
        if self.text_corpus:
            return "text"
        if self.synthetic:
            return "synthetic"
        return "distillation"


@dataclass
class TrainingConfig:
    """Optimisation and memory settings."""

    strategy: str = "qlora"
    optimizer: str = "adamw_8bit"
    precision: str = "bf16"
    objective: str = "sft"

    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 10
    max_steps: int = 100
    scheduler: str = "cosine"

    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    #: Weight on the KD term when objective is mixed_kd. 0 = pure SFT, 1 = pure KD.
    kd_weight: float = 0.5
    kd_temperature: float = 2.0

    seed: int = 0
    eval_every: int = 50
    save_every: int = 100
    log_every: int = 10

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class RuntimeConfig:
    """Where and how the run executes."""

    output_dir: str = "experiments/runs/unnamed"
    device: str = "auto"
    resume_from: str | None = None
    reserved_vram_gib: float = 1.0


@dataclass
class ExperimentConfig:
    """A complete, reproducible experiment."""

    name: str
    description: str = ""
    level: str = ""             # position on the development ladder, e.g. "Level 1"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        """Reject configurations that cannot work, with a reason."""
        errors: list[str] = []
        if self.training.strategy not in STRATEGIES:
            errors.append(f"strategy must be one of {STRATEGIES}, got {self.training.strategy!r}")
        if self.training.optimizer not in OPTIMIZERS:
            errors.append(f"optimizer must be one of {OPTIMIZERS}, got {self.training.optimizer!r}")
        if self.training.precision not in PRECISIONS:
            errors.append(f"precision must be one of {PRECISIONS}, got {self.training.precision!r}")
        if self.training.objective not in OBJECTIVES:
            errors.append(f"objective must be one of {OBJECTIVES}, got {self.training.objective!r}")
        if self.model.spec_path and self.model.pretrained:
            errors.append("set either model.spec_path or model.pretrained, not both")
        if not (self.model.spec_path or self.model.pretrained or self.model.architecture):
            errors.append(
                "model is unset. Supply exactly one of:\n"
                "      model.pretrained    a checkpoint to initialise from "
                "(recommended starting point)\n"
                "      model.spec_path     a saved HybridArchSpec JSON, for a new architecture\n"
                "      model.architecture  inline dimensions, for toy/prototype runs\n"
                "    A config shipped with `pretrained: null` is a template: choose a base "
                "model before running it"
            )
        if not (self.data.train_path or self.data.synthetic or self.data.text_corpus):
            errors.append("data needs one of: train_path, synthetic: true, text_corpus: true")
        if self.data.synthetic and self.data.text_corpus:
            errors.append(
                "data.synthetic and data.text_corpus are different objectives "
                "(induction task vs byte-level language modelling); set exactly one"
            )
        if self.training.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.training.gradient_accumulation_steps < 1:
            errors.append("gradient_accumulation_steps must be >= 1")
        if self.training.objective != "sft" and (self.data.synthetic or self.data.text_corpus):
            errors.append(
                "KD objectives need teacher outputs; synthetic and text corpora "
                "cannot provide them"
            )
        if errors:
            raise ValueError(f"invalid experiment {self.name!r}:\n  - " + "\n  - ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            level=data.get("level", ""),
            model=ModelConfig(**data.get("model", {})),
            data=DataConfig(**data.get("data", {})),
            training=TrainingConfig(**data.get("training", {})),
            runtime=RuntimeConfig(**data.get("runtime", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path} does not contain a YAML mapping")
        config = cls.from_dict(raw)
        config.validate()
        return config
