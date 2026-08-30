"""Named architectures, so an experiment is a name rather than fourteen numbers.

Every preset here is either **a real experiment that ran** or **the teacher**. Nothing is
a proposal. The project's next architecture is a decision that Level 3's result has to
inform, and a registry that already contained ``level4`` would be pre-empting it.

Two kinds of entry:

``measured``
    the exact architecture of a completed or running experiment, read from the same
    values its config carries. These are historical records — changing one would rewrite
    what an experiment was, so :func:`get_preset` returns a copy and the tests pin the
    parameter counts.
``reference``
    the teacher, as the thing every reduction is measured against.

Deriving a *new* architecture is what :func:`derive` is for: it takes an existing preset
and changes named fields, so a future variant is expressed as "level2r, but wider" and
the diff against its parent is explicit and small. That is the discipline the scaling
ladder needs — one variable at a time, and a record of which one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .params import count_parameters
from .spec import HybridArchSpec

#: Ratios the Qwen3.8-27B teacher uses, and which every student here inherits. They are
#: recorded so a derived architecture can be checked against them rather than drifting
#: silently: a variant that abandons the 3:1 hybrid layout is a different architecture
#: family, not a scaled one, and should say so.
TEACHER_RATIOS: dict[str, Any] = {
    "full_attention_interval": 4,          # 48 DeltaNet + 16 attention = 3:1
    "deltanet_to_attention": 3.0,
    "ffn_expansion": 17408 / 5120,         # 3.4
    "head_dim": 256,                       # the teacher's; students use 64
    "deltanet_value_per_key_head": 3.0,    # 48 value heads / 16 key heads
}

MEASURED = "measured"
REFERENCE = "reference"


@dataclass(frozen=True)
class Preset:
    """One named architecture and what it is."""

    name: str
    kind: str
    spec: HybridArchSpec
    summary: str
    #: The experiment record this architecture belongs to, when one exists.
    experiment: str | None = None
    config: str | None = None

    @property
    def parameters(self) -> int:
        return count_parameters(self.spec).total

    def to_dict(self) -> dict[str, Any]:
        params = count_parameters(self.spec)
        return {
            "name": self.name,
            "kind": self.kind,
            "summary": self.summary,
            "experiment": self.experiment,
            "config": self.config,
            "parameters": params.total,
            "non_embedding_parameters": params.total - params.embedding,
            "embedding_parameters": params.embedding,
            **architecture_fields(self.spec),
        }


def architecture_fields(spec: HybridArchSpec) -> dict[str, Any]:
    """The architecture, flattened for a table or a manifest.

    Exactly the fields the research loop reports on, in one shape, so a sweep row, an
    experiment record and a decision report cannot describe the same model differently.
    """
    return {
        "hidden_size": spec.hidden_size,
        "num_layers": spec.num_hidden_layers,
        "intermediate_size": spec.intermediate_size,
        "ffn_expansion": round(spec.intermediate_size / spec.hidden_size, 4),
        "deltanet_layers": spec.num_linear_attention_layers,
        "attention_layers": spec.num_full_attention_layers,
        "full_attention_interval": spec.full_attention_interval,
        "deltanet_to_attention": (
            round(spec.num_linear_attention_layers / spec.num_full_attention_layers, 3)
            if spec.num_full_attention_layers else None
        ),
        "attention_heads": spec.num_attention_heads,
        "kv_heads": spec.num_key_value_heads,
        "head_dim": spec.head_dim,
        "deltanet_key_heads": spec.linear_num_key_heads,
        "deltanet_value_heads": spec.linear_num_value_heads,
        "deltanet_key_head_dim": spec.linear_key_head_dim,
        "deltanet_value_head_dim": spec.linear_value_head_dim,
        "vocab_size": spec.vocab_size,
        "context_length": spec.max_position_embeddings,
        "tie_word_embeddings": spec.tie_word_embeddings,
    }


def _student(
    *, name: str, hidden_size: int, num_hidden_layers: int, intermediate_size: int,
    num_attention_heads: int, num_key_value_heads: int,
    linear_num_key_heads: int, linear_num_value_heads: int,
    vocab_size: int = 256, head_dim: int = 64, max_position_embeddings: int = 4096,
    linear_head_dim: int = 64,
) -> HybridArchSpec:
    """A student in the Level-2 family: byte-level, 3:1 hybrid layout.

    Every dimension is a parameter rather than a constant. The first draft hard-coded
    ``linear_head_dim=64`` and silently misdescribed the prototype, which uses 32 — the
    parameter count came out 4,574,308 against the run record's 4,029,700. A preset that
    does not reproduce its experiment's parameter count is not a record of it, which is
    why ``tests/test_presets.py`` loads every config file and compares field by field.
    """
    return HybridArchSpec(
        name=name,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        linear_num_key_heads=linear_num_key_heads,
        linear_num_value_heads=linear_num_value_heads,
        linear_key_head_dim=linear_head_dim,
        linear_value_head_dim=linear_head_dim,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=True,
        provenance=f"configs/experiments/{name}.yaml",
    )


#: Every architecture this project has actually built, plus the teacher. Values are the
#: ones the corresponding config carries; ``tests/test_presets.py`` pins each parameter
#: count so a preset cannot drift away from the experiment it claims to describe.
_PRESETS: dict[str, Preset] = {
    "prototype": Preset(
        name="prototype", kind=MEASURED,
        spec=_student(
            name="t4_prototype", hidden_size=256, num_hidden_layers=4,
            intermediate_size=704, num_attention_heads=4, num_key_value_heads=2,
            linear_num_key_heads=2, linear_num_value_heads=6,
            vocab_size=4096, max_position_embeddings=2048, linear_head_dim=32,
        ),
        summary=(
            "Level 1, 4.03M. Validated the training mechanism on synthetic tokens. "
            "Vocabulary 4096, NOT byte-level — its loss is not comparable with anything "
            "below it."
        ),
        experiment="experiments/runs/t4_prototype",
        config="configs/experiments/t4_prototype.yaml",
    ),
    "level2": Preset(
        name="level2", kind=MEASURED,
        spec=_student(
            name="t4_level2_100m_ckpt", hidden_size=640, num_hidden_layers=16,
            intermediate_size=2176, num_attention_heads=10, num_key_value_heads=2,
            linear_num_key_heads=4, linear_num_value_heads=12,
        ),
        summary=(
            "94.48M on procedural byte text. Validated the training and persistence "
            "stack; generated \"and and and\" and establishes no language capability."
        ),
        experiment="experiments/runs/t4_level2_100m_ckpt_complete",
        config="configs/experiments/t4_level2_100m_ckpt.yaml",
    ),
    "level2r": Preset(
        name="level2r", kind=MEASURED,
        spec=_student(
            name="t4_level2r_100m_real_english", hidden_size=640, num_hidden_layers=16,
            intermediate_size=2176, num_attention_heads=10, num_key_value_heads=2,
            linear_num_key_heads=4, linear_num_value_heads=12,
        ),
        summary=(
            "94.48M on real public-domain English. The project's real-language baseline: "
            "validation 1.797 bits/byte, genuine English structure, repetitive and "
            "semantically weak. Architecturally identical to level2."
        ),
        experiment="experiments/runs/t4_level2r_100m_real_english",
        config="configs/experiments/t4_level2r_100m_real_english.yaml",
    ),
    "level3": Preset(
        name="level3", kind=MEASURED,
        spec=_student(
            name="t4_level3_236m_real_english", hidden_size=1024, num_hidden_layers=16,
            intermediate_size=3456, num_attention_heads=16, num_key_value_heads=2,
            linear_num_key_heads=6, linear_num_value_heads=18,
        ),
        summary=(
            "236.24M, level2r scaled in width only. Tests whether capacity above 94.48M "
            "materially improves real-language modeling. RUNNING — no result yet."
        ),
        experiment="experiments/runs/t4_level3_236m_real_english",
        config="configs/experiments/t4_level3_236m_real_english.yaml",
    ),
    "teacher": Preset(
        name="teacher", kind=REFERENCE,
        spec=HybridArchSpec(name="qwen3.8-27b"),
        summary=(
            "Qwen3.8-27B, 26,895,998,464 parameters. The reference point every reduction "
            "is measured against, not a target to reproduce."
        ),
    ),
}


def preset_names() -> list[str]:
    return list(_PRESETS)


def get_preset(name: str) -> Preset:
    """One named architecture. Raises rather than guessing at an unknown name."""
    try:
        return _PRESETS[name]
    except KeyError:
        raise KeyError(
            f"unknown preset {name!r}; known: {', '.join(_PRESETS)}. Future architectures "
            f"are derived with `derive(...)`, not added here until they have run."
        ) from None


def get_spec(name: str) -> HybridArchSpec:
    """The spec for a preset. A fresh object, so a caller cannot mutate the registry."""
    return replace(get_preset(name).spec)


def derive(base: str | HybridArchSpec, *, name: str, **changes: Any) -> HybridArchSpec:
    """A new architecture expressed as an existing one plus named changes.

    This is how a future variant should be built. ``derive("level2r", name="wider",
    hidden_size=1024, ...)`` states in one line what differs from the baseline, which is
    exactly what a controlled experiment has to be able to say. Building a spec from
    scratch instead makes the diff against the baseline something a reader has to
    reconstruct by eye.

    Nothing is adjusted for you: if a change makes the GQA head counts indivisible the
    spec's own validation raises, rather than the number being quietly rounded into
    something that trains but is not what was asked for.
    """
    spec = base if isinstance(base, HybridArchSpec) else get_spec(base)
    unknown = set(changes) - set(vars(spec))
    if unknown:
        raise ValueError(f"unknown architecture field(s): {sorted(unknown)}")
    # layer_types is derived from the interval; carrying the parent's list would pin the
    # child to the parent's depth.
    return replace(spec, name=name, layer_types=None, **changes)


def diff(left: HybridArchSpec, right: HybridArchSpec) -> dict[str, tuple[Any, Any]]:
    """Which architecture fields differ, and by how much.

    The number this returns is the one that decides whether a comparison is controlled.
    A result attributed to "scale" when five fields moved is not a scaling result.
    """
    skip = {"name", "notes", "provenance", "extra", "layer_types"}
    return {
        field: (getattr(left, field), getattr(right, field))
        for field in sorted(vars(left))
        if field not in skip and getattr(left, field) != getattr(right, field)
    }
