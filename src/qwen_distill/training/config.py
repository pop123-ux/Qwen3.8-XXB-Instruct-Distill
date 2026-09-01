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
from .tokenized_data import DOCUMENT_SEPARATORS

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

    #: Language modelling on real text through the **teacher's own tokenizer**, rather
    #: than the byte-level encoding above. This is what the canonical student requires:
    #: its embedding has 248,320 rows and a byte stream only ever indexes the first 256.
    #: Reads `text_path` and needs `tokenizer_path`; loads tokenizer files only, never
    #: teacher weights. See qwen_distill.training.tokenized_data.
    tokenized_text: bool = False
    #: A local teacher checkpoint directory, or any directory holding tokenizer files.
    #: The GPU workflow points this at the pinned teacher checkout.
    tokenizer_path: str | None = None
    #: Fail if the tokenizer's vocabulary is not exactly this. Set it to the student's
    #: vocab_size to turn a silent mismatch into a refusal; leave it None to accept
    #: whatever the tokenizer reports.
    expected_vocab_size: int | None = None
    #: How documents are found in the corpus file: "blank_line", "line" or "file".
    #: Each document is followed by an explicit EOS before packing.
    document_separator: str = "blank_line"
    #: Smoke-test limits. Both None for a real run.
    max_documents: int | None = None
    max_tokens: int | None = None

    @property
    def mode(self) -> str:
        """Which data path this config selects."""
        if self.tokenized_text:
            return "tokenized"
        if self.text_corpus:
            return "text"
        if self.synthetic:
            return "synthetic"
        return "distillation"

    @property
    def is_sequence_corpus(self) -> bool:
        """Whether this config yields packed id sequences rather than JSONL records.

        Both corpus paths — byte-level and tokenised — produce ``list[list[int]]`` and
        share the trainer's sampler, validation and resume machinery. They differ only in
        what a token *means*, which is why `mode` still distinguishes them.
        """
        return self.mode in ("text", "tokenized")


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
    #: Ignored by logit_kd, which is pure KD by definition.
    kd_weight: float = 0.5
    kd_temperature: float = 2.0
    #: How the probability mass outside the teacher's top-k is treated. ``bucket`` keeps it
    #: exact using the teacher's full-vocabulary logsumexp; ``renormalize`` discards it,
    #: which stops penalising the student for mass the teacher rejected.
    kd_tail: str = "bucket"
    #: Teacher truncation. ``null`` keeps the full distribution — exact, but it holds a
    #: (batch, positions, 248320) tensor beside the student's own.
    kd_top_k: int | None = 64

    seed: int = 0
    eval_every: int = 50
    #: Full resumable checkpoint interval. Deliberately separate from `progress_every`:
    #: a checkpoint writes ~1.1 GB for a 94.5M model at fp32 with AdamW state, so writing
    #: one per step would make the experiment I/O-bound and hammer Drive. Progress
    #: records are kilobytes and can be frequent.
    save_every: int = 100
    log_every: int = 10
    #: How often to write a lightweight progress record (metrics only, no weights).
    #: Defaults to `log_every` when unset, so logging and progress stay in step.
    progress_every: int | None = None
    #: Copy each completed checkpoint to persistent storage. Off by default: the trainer
    #: must work with no Drive, no network and no mounted volume.
    persistent_backup: str | None = None

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def resolved_progress_every(self) -> int:
        return self.progress_every if self.progress_every else self.log_every


@dataclass
class RuntimeConfig:
    """Where and how the run executes."""

    output_dir: str = "experiments/runs/unnamed"
    device: str = "auto"
    #: A checkpoint directory, a step number, or "latest" to discover the newest
    #: *verified* checkpoint under the run's own checkpoints/ directory.
    resume_from: str | None = None
    reserved_vram_gib: float = 1.0


@dataclass
class ExperimentConfig:
    """A complete, reproducible experiment."""

    name: str
    description: str = ""
    level: str = ""             # position on the development ladder, e.g. "Level 1"
    #: Distillation blocks. Optional and empty by default, so every existing experiment
    #: config is unaffected — but PRESERVED rather than dropped, because a config asking
    #: for `logit_kd` that silently loaded as `sft` would invalidate the one comparison
    #: this project exists to make.
    objective: dict[str, Any] = field(default_factory=dict)
    teacher: dict[str, Any] = field(default_factory=dict)
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
        sources = [
            name for name, on in (
                ("train_path", bool(self.data.train_path)),
                ("synthetic", self.data.synthetic),
                ("text_corpus", self.data.text_corpus),
                ("tokenized_text", self.data.tokenized_text),
            ) if on
        ]
        if not sources:
            errors.append(
                "data needs one of: train_path, synthetic: true, text_corpus: true, "
                "tokenized_text: true"
            )
        elif len(sources) > 1:
            # Each source is a different objective over different tokens. Silently
            # preferring one would make the run something other than the config says.
            errors.append(
                f"data sources {', '.join(sources)} are mutually exclusive; set exactly "
                "one (synthetic is an induction task, text_corpus is byte-level, "
                "tokenized_text is the teacher's tokenizer, train_path is a "
                "teacher-generated dataset)"
            )
        if self.data.tokenized_text:
            if not self.data.text_path:
                errors.append("data.tokenized_text needs data.text_path (a UTF-8 corpus file)")
            if not self.data.tokenizer_path:
                errors.append(
                    "data.tokenized_text needs data.tokenizer_path: a local teacher "
                    "checkpoint directory, or any directory holding tokenizer files. "
                    "Only the tokenizer is read; teacher weights are never loaded."
                )
            if self.data.document_separator not in DOCUMENT_SEPARATORS:
                errors.append(
                    f"data.document_separator must be one of {DOCUMENT_SEPARATORS}, "
                    f"got {self.data.document_separator!r}"
                )
            if self.data.expected_vocab_size is not None and self.data.expected_vocab_size < 1:
                errors.append("data.expected_vocab_size must be >= 1, or null to skip the check")
        if self.training.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.training.gradient_accumulation_steps < 1:
            errors.append("gradient_accumulation_steps must be >= 1")
        if self.training.objective != "sft" and self.data.synthetic:
            errors.append(
                "KD against the synthetic induction corpus is meaningless: its tokens are "
                "generated by a rule, not drawn from any distribution the teacher models"
            )
        # A text corpus *is* valid for KD when a resident teacher supplies the
        # distribution — that is the cheapest real pilot there is, and needs no
        # teacher-generated answers at all. It is only invalid when the signal is expected
        # to come from the records, which a text corpus has none of.
        if (
            self.training.objective != "sft"
            and (self.data.text_corpus or self.data.tokenized_text)
            and (self.objective.get("signal_source") or "dataset") != "online"
        ):
            errors.append(
                "a text corpus carries no stored teacher logits; either set "
                "objective.signal_source='online' or use a teacher-generated dataset"
            )
        if self.training.kd_tail not in ("bucket", "renormalize"):
            errors.append(
                f"training.kd_tail must be 'bucket' or 'renormalize', "
                f"got {self.training.kd_tail!r}"
            )
        if self.training.kd_top_k is not None and self.training.kd_top_k < 1:
            errors.append("training.kd_top_k must be >= 1, or null for the full distribution")
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
            objective=dict(data.get("objective") or {}),
            teacher=dict(data.get("teacher") or {}),
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
