"""Turning a :class:`~qwen_distill.architecture.transfer.TransferPlan` into real weights.

:mod:`.transfer` deliberately stops at a mapping. This module is the other half: it reads
teacher tensors, reduces them to the student's shape, and produces a state dict. It is
kept separate because the two failure modes are different — a wrong *plan* is visible by
inspection, whereas a wrong *reduction* produces a student that loads cleanly, trains
without error, and is quietly worse than random init.

Three layout facts drive the whole design, all read from
``transformers.models.qwen3_5.modeling_qwen3_5`` rather than assumed:

1. ``q_proj`` is viewed as ``(..., num_heads, head_dim * 2)`` and chunked in half, so each
   head's rows are ``[query | gate]`` **interleaved per head**, not ``[all queries | all
   gates]``. Taking the first ``n * head_dim`` rows to keep ``n`` heads therefore keeps
   half as many heads and mixes them with their own gates.
2. ``in_proj_qkv`` is one matrix holding three concatenated segments
   ``[q (key_dim) | k (key_dim) | v (value_dim)]``, each independently head-structured,
   and ``conv1d.weight`` is depthwise over that same concatenation. Row-slicing it drops
   the value segment entirely.
3. A head is only meaningful together with the heads it shares a GQA/DeltaNet group with,
   so head reduction selects whole *groups* and never individual heads.

What this module refuses to do is as important as what it does. It will not reduce
``head_dim`` or the vocabulary: those change what a head *is* and how logits are indexed,
they interact with ``partial_rotary_factor`` and with logit distillation respectively, and
a plausible-looking implementation of either would be a silent capability loss rather than
a visible failure. Both raise.

Nothing here decides *which* reduction is right. ``slice`` remains the labelled baseline;
``mean_pool`` and ``importance`` are the candidates that have to beat it on a benchmark.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .spec import HybridArchSpec
from .transfer import TransferPlan, WidthReduction

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

#: Index-set identities. Two tensor axes carrying the same role in the same scope must
#: keep the *same* teacher indices, or the residual stream stops lining up with itself.
HIDDEN = "hidden"
INTERMEDIATE = "intermediate"
ATTN_KV_HEAD = "attn_kv_head"
ATTN_HEAD = "attn_head"
DN_KEY_HEAD = "dn_key_head"
DN_VALUE_HEAD = "dn_value_head"

#: Roles that are shared by the whole model rather than chosen per layer.
GLOBAL_ROLES = frozenset({HIDDEN})


class UnsupportedReduction(NotImplementedError):
    """A shape change this module will not attempt, with the reason it will not."""


# ---------------------------------------------------------------------------
# where teacher tensors come from
# ---------------------------------------------------------------------------
class WeightSource(Protocol):
    """A read-only, name-addressed collection of teacher tensors."""

    def names(self) -> set[str]: ...

    def get(self, name: str) -> torch.Tensor: ...


class StateDictSource:
    """A teacher already in memory — used by tests and by small stand-in teachers."""

    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        self._state_dict = state_dict

    def names(self) -> set[str]:
        return set(self._state_dict)

    def get(self, name: str) -> torch.Tensor:
        return self._state_dict[name]


class SafetensorsSource:
    """A sharded checkpoint on disk, read one tensor at a time.

    The teacher is ~54 GB; the student is a few. Materialisation therefore never holds
    more than one teacher tensor at a time, which is what makes it runnable on a machine
    that could not load the teacher at all.
    """

    def __init__(self, directory: str | Path, *, device: str = "cpu") -> None:
        from safetensors import safe_open

        self._safe_open = safe_open
        self.directory = Path(directory)
        index = self.directory / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            self._shard_of = {k: self.directory / v for k, v in weight_map.items()}
        else:
            single = self.directory / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(
                    f"no model.safetensors.index.json and no model.safetensors in {self.directory}"
                )
            with safe_open(single, framework="pt", device=device) as handle:
                self._shard_of = dict.fromkeys(handle.keys(), single)
        self._device = device
        self._open_path: Path | None = None
        self._open_handle: Any = None

    def names(self) -> set[str]:
        return set(self._shard_of)

    def get(self, name: str) -> torch.Tensor:
        shard = self._shard_of[name]
        if shard != self._open_path:
            self.close()
            self._open_handle = self._safe_open(shard, framework="pt", device=self._device)
            self._open_path = shard
        return self._open_handle.get_tensor(name)

    def close(self) -> None:
        if self._open_handle is not None:
            self._open_handle.__exit__(None, None, None)
        self._open_handle, self._open_path = None, None

    def __enter__(self) -> SafetensorsSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# how a tensor dimension is structured
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Segment:
    """A run of one axis: ``units`` heads (or channels) of ``block`` elements each."""

    role: str
    block: int
    teacher_units: int
    student_units: int

    @property
    def teacher_size(self) -> int:
        return self.block * self.teacher_units

    @property
    def student_size(self) -> int:
        return self.block * self.student_units


@dataclass(frozen=True)
class Axis:
    """One tensor dimension, as a concatenation of independently structured segments."""

    segments: tuple[Segment, ...]

    @property
    def teacher_size(self) -> int:
        return sum(s.teacher_size for s in self.segments)

    @property
    def student_size(self) -> int:
        return sum(s.student_size for s in self.segments)

    @property
    def unchanged(self) -> bool:
        return all(s.teacher_units == s.student_units for s in self.segments)


def _axis(role: str, block: int, teacher_units: int, student_units: int) -> Axis:
    return Axis((Segment(role, block, teacher_units, student_units),))


def _free(size: int) -> Axis:
    """An axis with no reduction — head_dim, kernel width, the vocabulary."""
    return _axis("fixed", size, 1, 1)


def tensor_axes(
    short_name: str, teacher: HybridArchSpec, student: HybridArchSpec
) -> tuple[Axis, ...] | None:
    """The per-dimension structure of one tensor, or ``None`` if the name is unknown.

    ``short_name`` is the tensor name with any ``model.layers.<i>.`` prefix removed, so
    one table covers every layer.
    """
    hidden = _axis(HIDDEN, 1, teacher.hidden_size, student.hidden_size)
    ffn = _axis(INTERMEDIATE, 1, teacher.intermediate_size, student.intermediate_size)
    vocab = _free(student.vocab_size)

    head_dim = teacher.head_dim
    q_heads = _axis(ATTN_HEAD, head_dim * 2, teacher.num_attention_heads, student.num_attention_heads)
    o_heads = _axis(ATTN_HEAD, head_dim, teacher.num_attention_heads, student.num_attention_heads)
    kv_heads = _axis(
        ATTN_KV_HEAD, head_dim, teacher.num_key_value_heads, student.num_key_value_heads
    )

    k_dim, v_dim = teacher.linear_key_head_dim, teacher.linear_value_head_dim
    key_seg = Segment(DN_KEY_HEAD, k_dim, teacher.linear_num_key_heads, student.linear_num_key_heads)
    value_seg = Segment(
        DN_VALUE_HEAD, v_dim, teacher.linear_num_value_heads, student.linear_num_value_heads
    )
    # [q | k | v] in one matrix, and the depthwise conv over the same concatenation.
    qkv = Axis((key_seg, key_seg, value_seg))
    values = Axis((value_seg,))
    value_scalars = _axis(
        DN_VALUE_HEAD, 1, teacher.linear_num_value_heads, student.linear_num_value_heads
    )

    table: dict[str, tuple[Axis, ...]] = {
        "model.embed_tokens.weight": (vocab, hidden),
        "lm_head.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
        "mlp.gate_proj.weight": (ffn, hidden),
        "mlp.up_proj.weight": (ffn, hidden),
        "mlp.down_proj.weight": (hidden, ffn),
        "self_attn.q_proj.weight": (q_heads, hidden),
        "self_attn.k_proj.weight": (kv_heads, hidden),
        "self_attn.v_proj.weight": (kv_heads, hidden),
        "self_attn.o_proj.weight": (hidden, o_heads),
        "self_attn.q_norm.weight": (_free(student.head_dim),),
        "self_attn.k_norm.weight": (_free(student.head_dim),),
        "linear_attn.in_proj_qkv.weight": (qkv, hidden),
        "linear_attn.in_proj_z.weight": (values, hidden),
        "linear_attn.in_proj_b.weight": (value_scalars, hidden),
        "linear_attn.in_proj_a.weight": (value_scalars, hidden),
        "linear_attn.out_proj.weight": (hidden, values),
        "linear_attn.conv1d.weight": (qkv, _free(1), _free(student.linear_conv_kernel_dim)),
        "linear_attn.dt_bias": (value_scalars,),
        "linear_attn.A_log": (value_scalars,),
        "linear_attn.norm.weight": (_free(student.linear_value_head_dim),),
    }
    return table.get(short_name)


def strip_layer_prefix(name: str) -> tuple[str, int | None]:
    """``model.layers.7.mlp.up_proj.weight`` -> ``("mlp.up_proj.weight", 7)``."""
    parts = name.split(".")
    if len(parts) > 3 and parts[0] == "model" and parts[1] == "layers":
        return ".".join(parts[3:]), int(parts[2])
    return name, None


# ---------------------------------------------------------------------------
# choosing what survives
# ---------------------------------------------------------------------------
ScoreProvider = Callable[[str, int | None], "torch.Tensor | None"]


class Surgeon:
    """Reduces tensors, keeping every axis role's index set consistent across tensors.

    The consistency is the point. Selecting hidden channels independently per tensor
    would give each matrix a different idea of what the residual stream means — every
    shape would still be right, and the model would be scrambled.
    """

    def __init__(
        self,
        method: WidthReduction,
        teacher: HybridArchSpec,
        student: HybridArchSpec,
        *,
        scores: ScoreProvider | None = None,
    ) -> None:
        self.method = method
        self.teacher = teacher
        self.student = student
        self._scores = scores
        self._indices: dict[tuple[str, int | None], torch.Tensor] = {}
        self.notes: list[str] = []

    # -- index sets ------------------------------------------------------
    def _group_derived(self, role: str, layer: int | None) -> tuple[str, int] | None:
        """Roles whose selection is dictated by a coarser role's, and the group size."""
        if role == ATTN_HEAD:
            groups = self.teacher.num_attention_heads // self.teacher.num_key_value_heads
            return ATTN_KV_HEAD, groups
        if role == DN_VALUE_HEAD:
            groups = self.teacher.linear_num_value_heads // self.teacher.linear_num_key_heads
            return DN_KEY_HEAD, groups
        return None

    def _primary_units(self, role: str) -> tuple[int, int]:
        if role == ATTN_KV_HEAD:
            return self.teacher.num_key_value_heads, self.student.num_key_value_heads
        if role == DN_KEY_HEAD:
            return self.teacher.linear_num_key_heads, self.student.linear_num_key_heads
        raise KeyError(role)

    def units(self, role: str, layer: int | None, teacher_units: int, student_units: int) -> torch.Tensor:
        """Which teacher units survive for this role, chosen once and cached."""
        import torch

        key = (role, None if role in GLOBAL_ROLES else layer)
        cached = self._indices.get(key)
        if cached is not None:
            return cached

        derived = self._group_derived(role, layer)
        if derived is not None:
            parent_role, group = derived
            student_group = (
                self.student.num_attention_heads // self.student.num_key_value_heads
                if role == ATTN_HEAD
                else self.student.linear_num_value_heads // self.student.linear_num_key_heads
            )
            if group != student_group:
                raise UnsupportedReduction(
                    f"{role}: the teacher keeps {group} heads per group and the student "
                    f"{student_group}. Changing the group size changes what a group shares, "
                    "so the teacher's heads have no consistent destination; keep the "
                    "teacher's ratio or initialise these layers randomly."
                )
            t_parent, s_parent = self._primary_units(parent_role)
            parent = self.units(parent_role, layer, t_parent, s_parent)
            chosen = (parent.unsqueeze(1) * group + torch.arange(group)).reshape(-1)
        else:
            chosen = self._choose(role, layer, teacher_units, student_units)

        self._indices[key] = chosen
        return chosen

    def _choose(self, role: str, layer: int | None, teacher_units: int, student_units: int) -> torch.Tensor:
        import torch

        if student_units == teacher_units:
            return torch.arange(teacher_units)
        if student_units > teacher_units:
            raise UnsupportedReduction(
                f"{role!r} grows from {teacher_units} to {student_units} units. Transfer "
                "selects from what the teacher has; it cannot invent heads or channels. "
                "Widen the student only where the teacher is at least as wide."
            )
        if self.method == "importance":
            scores = self._scores(role, layer) if self._scores else None
            if scores is None:
                raise UnsupportedReduction(
                    f"width_reduction='importance' needs a score vector for role {role!r} "
                    f"(layer {layer}) and none was provided. Falling back to slicing here "
                    "would report 'importance' while doing something else."
                )
            if scores.numel() != teacher_units:
                raise UnsupportedReduction(
                    f"importance scores for {role!r} have {scores.numel()} entries but the "
                    f"teacher has {teacher_units} units"
                )
            keep = torch.topk(scores.float(), student_units).indices
            return keep.sort().values
        # 'slice' and 'mean_pool' both take the leading units; mean_pool then averages
        # the discarded ones back in, which _reduce_axis handles.
        return torch.arange(student_units)

    # -- tensor surgery ----------------------------------------------------
    def reduce(self, tensor: torch.Tensor, axes: tuple[Axis, ...], layer: int | None) -> torch.Tensor:
        for dim, axis in enumerate(axes):
            if axis.unchanged:
                continue
            tensor = self._reduce_axis(tensor, dim, axis, layer)
        return tensor

    def _reduce_axis(self, tensor: torch.Tensor, dim: int, axis: Axis, layer: int | None) -> torch.Tensor:
        import torch

        pieces, offset = [], 0
        for segment in axis.segments:
            chunk = tensor.narrow(dim, offset, segment.teacher_size)
            offset += segment.teacher_size
            if segment.teacher_units == segment.student_units:
                pieces.append(chunk)
                continue
            if self.method == "mean_pool":
                pieces.append(self._pool(chunk, dim, segment))
            else:
                keep = self.units(segment.role, layer, segment.teacher_units, segment.student_units)
                index = (keep.unsqueeze(1) * segment.block + torch.arange(segment.block)).reshape(-1)
                pieces.append(chunk.index_select(dim, index.to(chunk.device)))
        return torch.cat(pieces, dim=dim) if len(pieces) > 1 else pieces[0]

    def _pool(self, chunk: torch.Tensor, dim: int, segment: Segment) -> torch.Tensor:
        """Average teacher units into student units with adaptive (possibly uneven) windows.

        ``adaptive_avg_pool1d`` is used rather than a reshape-and-mean so that ratios like
        5120 -> 3072 work at all; the windows are then uneven, which is a real property of
        this reduction and not something to hide.
        """
        import torch.nn.functional as F

        if segment.student_units > segment.teacher_units:
            raise UnsupportedReduction(
                f"{segment.role!r} grows from {segment.teacher_units} to "
                f"{segment.student_units} units; adaptive pooling would silently upsample "
                "teacher weights into channels the teacher never had."
            )
        moved = chunk.movedim(dim, -1)
        shape = moved.shape
        flat = moved.reshape(-1, segment.teacher_units, segment.block)
        pooled = F.adaptive_avg_pool1d(
            flat.float().transpose(1, 2), segment.student_units
        ).transpose(1, 2)
        pooled = pooled.reshape(*shape[:-1], segment.student_size).to(chunk.dtype)
        return pooled.movedim(-1, dim)


def default_scores(source: WeightSource, teacher: HybridArchSpec, layer_map: dict[int, int]) -> ScoreProvider:
    """Norm-based importance proxies, each from a single teacher tensor.

    These are *proxies*, not measured importance: a channel that carries large weights is
    not necessarily a channel the model needs. They are cheap (one tensor read per role)
    and defensible enough to be worth measuring against ``slice``, which is the only
    claim being made for them.
    """
    cache: dict[tuple[str, int | None], Any] = {}

    def provider(role: str, layer: int | None) -> torch.Tensor | None:
        key = (role, layer)
        if key in cache:
            return cache[key]
        value = _score_for(source, teacher, layer_map, role, layer)
        cache[key] = value
        return value

    return provider


def _score_for(
    source: WeightSource,
    teacher: HybridArchSpec,
    layer_map: dict[int, int],
    role: str,
    layer: int | None,
) -> torch.Tensor | None:
    names = source.names()
    if role == HIDDEN and "model.embed_tokens.weight" in names:
        # Per-channel energy of the embedding: how much the residual stream is written to.
        return source.get("model.embed_tokens.weight").float().pow(2).sum(0)
    if layer is None or layer not in layer_map:
        return None
    prefix = f"model.layers.{layer_map[layer]}"
    if role == ATTN_KV_HEAD:
        name = f"{prefix}.self_attn.o_proj.weight"
        if name not in names:
            return None
        # Per-head output energy, folded down onto the KV group that feeds it.
        per_head = source.get(name).float().pow(2).sum(0).reshape(teacher.num_attention_heads, -1).sum(1)
        groups = teacher.num_attention_heads // teacher.num_key_value_heads
        return per_head.reshape(teacher.num_key_value_heads, groups).sum(1)
    if role == DN_KEY_HEAD:
        name = f"{prefix}.linear_attn.out_proj.weight"
        if name not in names:
            return None
        per_head = (
            source.get(name).float().pow(2).sum(0).reshape(teacher.linear_num_value_heads, -1).sum(1)
        )
        groups = teacher.linear_num_value_heads // teacher.linear_num_key_heads
        return per_head.reshape(teacher.linear_num_key_heads, groups).sum(1)
    if role == INTERMEDIATE:
        name = f"{prefix}.mlp.down_proj.weight"
        if name not in names:
            return None
        return source.get(name).float().pow(2).sum(0)
    return None


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
@dataclass
class TransferReport:
    """What was actually written, in parameters rather than tensor names.

    Tensor-count coverage flatters a transfer: the embedding is one tensor and anywhere
    from 8% to 18% of a multi-billion-parameter student, depending on its size.
    Parameter coverage is the figure that says how much of the student the teacher
    actually determined.
    """

    strategy: str
    width_reduction: str
    written: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    operations: Counter = field(default_factory=Counter)
    elements_written: int = 0
    elements_skipped: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def parameter_coverage(self) -> float:
        total = self.elements_written + self.elements_skipped
        return self.elements_written / total if total else 0.0

    @property
    def tensor_coverage(self) -> float:
        total = len(self.written) + len(self.skipped)
        return len(self.written) / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "width_reduction": self.width_reduction,
            "n_written": len(self.written),
            "n_skipped": len(self.skipped),
            "elements_written": self.elements_written,
            "elements_skipped": self.elements_skipped,
            "parameter_coverage": self.parameter_coverage,
            "tensor_coverage": self.tensor_coverage,
            "operations": dict(self.operations),
            "skipped": [{"tensor": n, "reason": r} for n, r in self.skipped],
            "warnings": self.warnings,
        }

    def render(self) -> str:
        total = self.elements_written + self.elements_skipped
        lines = [
            "  TRANSFER REPORT",
            f"    strategy            : {self.strategy}",
            f"    width reduction     : {self.width_reduction}",
            f"    tensors written     : {len(self.written):,} of {len(self.written) + len(self.skipped):,}"
            f"  ({self.tensor_coverage:.1%})",
            f"    parameters written  : {self.elements_written:,} of {total:,}"
            f"  ({self.parameter_coverage:.1%})",
        ]
        for operation, count in self.operations.most_common():
            lines.append(f"      {operation:<28}{count:>6} tensor(s)")
        if self.skipped:
            lines.append(f"    left at random init : {len(self.skipped):,} tensor(s)")
            for name, reason in self.skipped[:8]:
                lines.append(f"      {name}: {reason}")
            if len(self.skipped) > 8:
                lines.append(f"      ... and {len(self.skipped) - 8:,} more")
        for warning in self.warnings:
            lines.append(f"    ! {warning}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------
def _check_supported(teacher: HybridArchSpec, student: HybridArchSpec) -> None:
    if student.head_dim != teacher.head_dim:
        raise UnsupportedReduction(
            f"head_dim differs ({student.head_dim} vs {teacher.head_dim}). Reducing within "
            "a head changes what a head is and interacts with partial_rotary_factor "
            f"(rope_dim {student.rope_dim} vs {teacher.rope_dim}); no reduction here would "
            "be honest. Keep the teacher's head_dim and reduce head *count* instead."
        )
    for field_name in ("linear_key_head_dim", "linear_value_head_dim"):
        if getattr(student, field_name) != getattr(teacher, field_name):
            raise UnsupportedReduction(
                f"{field_name} differs ({getattr(student, field_name)} vs "
                f"{getattr(teacher, field_name)}); reduce DeltaNet head count instead."
            )
    teacher_gqa = teacher.num_attention_heads // teacher.num_key_value_heads
    student_gqa = student.num_attention_heads // student.num_key_value_heads
    if teacher_gqa != student_gqa:
        raise UnsupportedReduction(
            f"GQA group size differs ({student_gqa} query heads per KV head vs "
            f"{teacher_gqa}). A KV head's weights are shared by exactly the query heads in "
            "its group, so regrouping gives them no consistent destination. Keep the "
            "teacher's ratio and change head *count*."
        )
    teacher_dn = teacher.linear_num_value_heads // teacher.linear_num_key_heads
    student_dn = student.linear_num_value_heads // student.linear_num_key_heads
    if teacher_dn != student_dn:
        raise UnsupportedReduction(
            f"DeltaNet value-per-key-head ratio differs ({student_dn} vs {teacher_dn}); "
            "the same regrouping problem as GQA. Keep the teacher's ratio."
        )
    if student.linear_conv_kernel_dim != teacher.linear_conv_kernel_dim:
        raise UnsupportedReduction(
            f"linear_conv_kernel_dim differs ({student.linear_conv_kernel_dim} vs "
            f"{teacher.linear_conv_kernel_dim}); the depthwise conv has no meaningful "
            "kernel-width reduction."
        )


def apply_transfer_plan(
    plan: TransferPlan,
    teacher: HybridArchSpec,
    student: HybridArchSpec,
    source: WeightSource,
    *,
    width_reduction: WidthReduction = "slice",
    dtype: torch.dtype | None = None,
    scores: ScoreProvider | None = None,
) -> tuple[dict[str, torch.Tensor], TransferReport]:
    """Materialise the student weights a plan describes.

    Returns ``(state_dict, report)``. Tensors the plan lists as randomly initialised, and
    tensors that cannot be reduced, are **absent** from the state dict rather than filled
    with anything — so loading it with ``strict=False`` leaves them at the model's own
    initialisation and the report says exactly which those were.
    """

    _check_supported(teacher, student)
    if student.vocab_size != teacher.vocab_size:
        raise UnsupportedReduction(
            f"vocabulary differs ({student.vocab_size} vs {teacher.vocab_size}). Embedding "
            "rows are token identities, not features: there is no reduction of them that "
            "preserves meaning, and logit distillation would need a token mapping. Keep "
            "the teacher's tokenizer."
        )

    if width_reduction == "importance" and scores is None:
        scores = default_scores(source, teacher, plan.layer_map)

    surgeon = Surgeon(width_reduction, teacher, student, scores=scores)
    report = TransferReport(strategy=plan.strategy, width_reduction=width_reduction)
    report.warnings.extend(plan.warnings)
    available = source.names()
    state: dict[str, torch.Tensor] = {}

    for name in plan.randomly_initialised:
        report.skipped.append((name, "plan leaves it randomly initialised"))

    for mapping in plan.mappings:
        short, layer = strip_layer_prefix(mapping.student_name)
        expected = mapping.student_shape
        count = 1
        for dimension in expected:
            count *= dimension

        if short == "lm_head.weight" and student.tie_word_embeddings:
            report.skipped.append(
                (mapping.student_name, "tied to model.embed_tokens.weight, not a separate parameter")
            )
            continue

        axes = tensor_axes(short, teacher, student)
        if axes is None:
            report.skipped.append((mapping.student_name, "no known axis structure for this tensor"))
            report.elements_skipped += count
            continue

        teacher_name = mapping.teacher_names[0]
        if teacher_name not in available:
            report.skipped.append((mapping.student_name, f"teacher has no tensor {teacher_name!r}"))
            report.elements_skipped += count
            continue

        try:
            reduced = surgeon.reduce(source.get(teacher_name), axes, layer)
        except UnsupportedReduction as exc:
            report.skipped.append((mapping.student_name, str(exc)))
            report.elements_skipped += count
            continue

        if list(reduced.shape) != list(expected):
            report.skipped.append(
                (
                    mapping.student_name,
                    f"reduced to {tuple(reduced.shape)} but the student wants {tuple(expected)}",
                )
            )
            report.elements_skipped += count
            continue

        if dtype is not None:
            reduced = reduced.to(dtype)
        state[mapping.student_name] = reduced.contiguous().clone()
        report.written.append(mapping.student_name)
        report.operations[mapping.operation] += 1
        report.elements_written += reduced.numel()

    report.warnings.extend(surgeon.notes)
    return state, report


def initialise_student(
    model: Any,
    plan: TransferPlan,
    teacher: HybridArchSpec,
    student: HybridArchSpec,
    source: WeightSource,
    **kwargs: Any,
) -> TransferReport:
    """Copy transferred weights into an already-constructed student, in place.

    Anything the transfer did not produce keeps the model's own initialisation, and
    ``missing``/``unexpected`` from the load are folded into the report so a name that
    silently matched nothing cannot pass for a successful transfer.
    """
    state, report = apply_transfer_plan(plan, teacher, student, source, **kwargs)
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        report.warnings.append(
            f"{len(result.unexpected_keys)} transferred tensor(s) matched nothing in the "
            f"student and were dropped, e.g. {result.unexpected_keys[:3]}"
        )
    return report
