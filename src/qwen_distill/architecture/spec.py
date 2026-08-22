"""Declarative specification of a Qwen3.5/3.8-style hybrid (Gated DeltaNet + gated attention) model.

Every field name and default mirrors ``transformers.models.qwen3_5.configuration_qwen3_5``
so that a spec can be round-tripped to a Hugging Face ``config.json`` without guesswork.
The formulas that consume this spec live in :mod:`qwen_distill.architecture.params`,
:mod:`qwen_distill.architecture.memory` and :mod:`qwen_distill.architecture.flops`;
they were transcribed from the reference implementation rather than inferred from
parameter-count tables. See ``docs/VERIFICATION.md`` for provenance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

LayerType = Literal["linear_attention", "full_attention"]

#: Layer-type strings used by ``Qwen3_5TextConfig.layer_types``.
LINEAR_ATTENTION: LayerType = "linear_attention"
FULL_ATTENTION: LayerType = "full_attention"


def build_layer_types(num_hidden_layers: int, full_attention_interval: int) -> list[LayerType]:
    """Reproduce ``Qwen3_5TextConfig.__post_init__`` layer-type expansion.

    Upstream expands the interval as::

        "linear_attention" if bool((i + 1) % interval) else "full_attention"

    so every ``interval``-th layer (1-indexed) is a full-attention layer and it is
    always the *last* layer of each repeated group.
    """
    if num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}")
    if full_attention_interval <= 0:
        raise ValueError(f"full_attention_interval must be positive, got {full_attention_interval}")
    return [
        LINEAR_ATTENTION if bool((i + 1) % full_attention_interval) else FULL_ATTENTION
        for i in range(num_hidden_layers)
    ]


@dataclass(frozen=True)
class HybridArchSpec:
    """A text-tower architecture in the Qwen3.5/3.8 hybrid family.

    Attributes mirror ``Qwen3_5TextConfig``. ``layer_types`` is derived from
    ``full_attention_interval`` unless given explicitly, matching upstream behaviour.
    """

    name: str = "unnamed"

    # --- core dimensions -------------------------------------------------
    hidden_size: int = 5120
    num_hidden_layers: int = 64
    intermediate_size: int = 17408
    vocab_size: int = 248320
    tie_word_embeddings: bool = False

    # --- gated full attention -------------------------------------------
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    attention_bias: bool = False
    partial_rotary_factor: float = 0.25

    # --- gated deltanet (linear attention) ------------------------------
    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # --- layout ----------------------------------------------------------
    full_attention_interval: int = 4
    layer_types: list[LayerType] | None = None

    # --- context ----------------------------------------------------------
    max_position_embeddings: int = 262144

    # --- bookkeeping -------------------------------------------------------
    notes: str = ""
    provenance: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_types", self.resolved_layer_types())
        self.validate()

    # ------------------------------------------------------------------
    # derived quantities
    # ------------------------------------------------------------------
    def resolved_layer_types(self) -> list[LayerType]:
        if self.layer_types is not None:
            return list(self.layer_types)
        return build_layer_types(self.num_hidden_layers, self.full_attention_interval)

    @property
    def num_full_attention_layers(self) -> int:
        return sum(1 for t in self.resolved_layer_types() if t == FULL_ATTENTION)

    @property
    def num_linear_attention_layers(self) -> int:
        return sum(1 for t in self.resolved_layer_types() if t == LINEAR_ATTENTION)

    @property
    def linear_key_dim(self) -> int:
        """``key_dim`` in ``Qwen3_5GatedDeltaNet.__init__``."""
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        """``value_dim`` in ``Qwen3_5GatedDeltaNet.__init__``."""
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_conv_dim(self) -> int:
        """``conv_dim = key_dim * 2 + value_dim``."""
        return self.linear_key_dim * 2 + self.linear_value_dim

    @property
    def rope_dim(self) -> int:
        """Rotary dimension actually rotated: ``partial_rotary_factor * head_dim``."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def attention_query_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def attention_kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise ``ValueError`` on structurally invalid combinations."""
        errors: list[str] = []
        if self.hidden_size <= 0:
            errors.append("hidden_size must be positive")
        if self.num_hidden_layers <= 0:
            errors.append("num_hidden_layers must be positive")
        if self.num_attention_heads <= 0:
            errors.append("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
            errors.append("num_key_value_heads must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            errors.append(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads}) for GQA"
            )
        if self.linear_num_value_heads % self.linear_num_key_heads:
            errors.append(
                f"linear_num_value_heads ({self.linear_num_value_heads}) must be divisible by "
                f"linear_num_key_heads ({self.linear_num_key_heads})"
            )
        if not 0 < self.partial_rotary_factor <= 1:
            errors.append("partial_rotary_factor must be in (0, 1]")
        if self.rope_dim % 2:
            errors.append(
                f"rope_dim ({self.rope_dim}) must be even; "
                f"head_dim * partial_rotary_factor = {self.head_dim} * {self.partial_rotary_factor}"
            )
        resolved = self.resolved_layer_types()
        if len(resolved) != self.num_hidden_layers:
            errors.append(
                f"layer_types has {len(resolved)} entries but num_hidden_layers is {self.num_hidden_layers}"
            )
        if self.num_full_attention_layers == 0:
            errors.append("at least one full_attention layer is required for exact long-range recall")
        if errors:
            raise ValueError(f"invalid HybridArchSpec {self.name!r}: " + "; ".join(errors))

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HybridArchSpec:
        known = {f for f in cls.__dataclass_fields__}
        unknown = {k: v for k, v in data.items() if k not in known}
        kwargs = {k: v for k, v in data.items() if k in known}
        if unknown:
            kwargs.setdefault("extra", {}).update(unknown)
        return cls(**kwargs)

    def to_hf_text_config(self) -> dict[str, Any]:
        """Emit a ``Qwen3_5TextConfig``-shaped dict.

        ``layer_types`` is written explicitly so the checkpoint is unambiguous even
        if upstream changes its interval-expansion default.
        """
        return {
            "model_type": "qwen3_5_text",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "hidden_act": "silu",
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": self.attention_bias,
            "partial_rotary_factor": self.partial_rotary_factor,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "layer_types": self.resolved_layer_types(),
        }

    @classmethod
    def from_hf_config(cls, config: dict[str, Any], name: str = "from_hf") -> HybridArchSpec:
        """Build a spec from a Hugging Face ``config.json`` dict.

        Accepts either a text-only config or a multimodal config carrying a
        ``text_config`` sub-dict (as Qwen3.5/3.8 checkpoints do).
        """
        text = config.get("text_config", config)
        interval = text.get("full_attention_interval", 4)
        layer_types = text.get("layer_types")
        return cls(
            name=name,
            hidden_size=text["hidden_size"],
            num_hidden_layers=text["num_hidden_layers"],
            intermediate_size=text["intermediate_size"],
            vocab_size=text["vocab_size"],
            tie_word_embeddings=text.get("tie_word_embeddings", False),
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text.get("head_dim", text["hidden_size"] // text["num_attention_heads"]),
            attention_bias=text.get("attention_bias", False),
            partial_rotary_factor=text.get("partial_rotary_factor", 0.25),
            linear_num_value_heads=text.get("linear_num_value_heads", 32),
            linear_num_key_heads=text.get("linear_num_key_heads", 16),
            linear_key_head_dim=text.get("linear_key_head_dim", 128),
            linear_value_head_dim=text.get("linear_value_head_dim", 128),
            linear_conv_kernel_dim=text.get("linear_conv_kernel_dim", 4),
            full_attention_interval=interval,
            layer_types=layer_types,
            max_position_embeddings=text.get("max_position_embeddings", 262144),
            provenance=f"from_hf_config(model_type={config.get('model_type', '?')})",
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> HybridArchSpec:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
