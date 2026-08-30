"""The frozen primary student: ``qwen38_19b_h5120_l48_moe``.

One canonical specification, so the architecture is a named object rather than fourteen
magic numbers scattered across configs and scripts. Every field below is the frozen
research target; nothing here is tuned, and nothing may be quietly adjusted to make a
parameter count come out round.

**The name says 19B. The architecture is 22.07B.** That is not a bug and it has not been
"fixed" by shrinking anything — the specification is frozen and the count is reported
honestly. The most likely reconciliation is that ~19B referred to *non-embedding*
parameters, which come to 19.53B; the 248,320-entry vocabulary with untied embeddings adds
2.54B on top. :func:`audit` prints the full breakdown and the discrepancy, and a test pins
both numbers so neither can drift silently.

The architecture is realisable directly: `transformers` 5.15.1 ships ``qwen3_5_moe_text``,
the MoE variant of the teacher's hybrid family, and it carries every field this target
needs — including ``router_aux_loss_coef`` defaulting to the frozen 0.001. The hybrid
topology is set through explicit ``layer_types`` rather than an interval, so the twelve
full-attention positions are stated rather than derived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .spec import FULL_ATTENTION, LINEAR_ATTENTION, LayerType

#: The registered `transformers` architecture this student instantiates as.
MOE_MODEL_TYPE = "qwen3_5_moe_text"

#: Canonical name. Kept even though the count is 22.07B: renaming it would break every
#: artifact already referring to the frozen target, and the honest number lives in
#: :func:`audit` where it cannot be mistaken for the label.
STUDENT_ID = "qwen38_19b_h5120_l48_moe"

#: The teacher this student is compressed from, and the reduction it represents.
TEACHER_ID = "Qwen/Qwen3.8-27B"
TEACHER_LAYERS = 64
TEACHER_KV_HEADS = 4
TEACHER_FFN_INTERMEDIATE = 17408


@dataclass(frozen=True)
class MoEStudentSpec:
    """The frozen ~19B sparse hybrid student. Field names mirror ``Qwen3_5MoeTextConfig``."""

    name: str = STUDENT_ID

    # --- core dimensions -------------------------------------------------
    hidden_size: int = 5120
    num_hidden_layers: int = 48
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"

    # --- gated full attention -------------------------------------------
    num_attention_heads: int = 24
    num_key_value_heads: int = 2          # teacher has 4; merged 2->1 pairwise
    head_dim: int = 256
    attention_bias: bool = False
    attention_dropout: float = 0.0
    partial_rotary_factor: float = 0.25   # rotary dimension 64 = 256 * 0.25
    rope_theta: int = 10_000_000

    # --- gated DeltaNet ---------------------------------------------------
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # --- sparse MoE FFN ---------------------------------------------------
    num_experts: int = 24
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 768
    shared_expert_intermediate_size: int = 768
    router_aux_loss_coef: float = 0.001
    #: Router jitter is off in the frozen target. Recorded so the absence is deliberate.
    router_jitter: bool = False

    # --- hybrid topology --------------------------------------------------
    #: 3 DeltaNet then 1 full attention, twelve times. Explicit, not derived: the twelve
    #: attention positions are load-bearing for layer mapping and must not move.
    deltanet_per_group: int = 3
    attention_per_group: int = 1

    # --- multi-token prediction -------------------------------------------
    #: Carried as a declared field. `transformers` 5.15.1 does not build an MTP head for
    #: this architecture, so it is an architecture-level intent with no runtime tensors
    #: yet — see :data:`MTP_STATUS`. Never counted as if it existed.
    mtp_num_hidden_layers: int = 1
    mtp_use_dedicated_embeddings: bool = False
    distill_from_teacher_mtp: bool = True

    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        errors: list[str] = []
        group = self.deltanet_per_group + self.attention_per_group
        if self.num_hidden_layers % group:
            errors.append(
                f"num_hidden_layers ({self.num_hidden_layers}) is not a whole number of "
                f"{group}-layer hybrid groups"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            errors.append("num_attention_heads must be divisible by num_key_value_heads")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            errors.append("linear_num_value_heads must be divisible by linear_num_key_heads")
        if self.num_experts_per_tok > self.num_experts:
            errors.append("num_experts_per_tok cannot exceed num_experts")
        if int(self.head_dim * self.partial_rotary_factor) % 2:
            errors.append("rotary dimension must be even")
        if errors:
            raise ValueError(f"invalid {self.name!r}: " + "; ".join(errors))

    # -- topology -------------------------------------------------------
    @property
    def group_size(self) -> int:
        return self.deltanet_per_group + self.attention_per_group

    @property
    def num_groups(self) -> int:
        return self.num_hidden_layers // self.group_size

    def layer_types(self) -> list[LayerType]:
        """``[DeltaNet, DeltaNet, DeltaNet, FullAttention] x 12``, stated explicitly."""
        group = [LINEAR_ATTENTION] * self.deltanet_per_group + [FULL_ATTENTION] * self.attention_per_group
        return group * self.num_groups

    @property
    def attention_layer_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types()) if t == FULL_ATTENTION]

    @property
    def deltanet_layer_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types()) if t == LINEAR_ATTENTION]

    @property
    def rope_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    # -- realisation ----------------------------------------------------
    def to_hf_text_config(self) -> dict[str, Any]:
        """A ``Qwen3_5MoeTextConfig``-shaped dict. Round-trips without guesswork."""
        return {
            "model_type": MOE_MODEL_TYPE,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "hidden_act": self.hidden_act,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": self.attention_bias,
            "attention_dropout": self.attention_dropout,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "moe_intermediate_size": self.moe_intermediate_size,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "router_aux_loss_coef": self.router_aux_loss_coef,
            "layer_types": self.layer_types(),
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        data = asdict(self)
        data["layer_types"] = self.layer_types()
        data["attention_layer_indices"] = self.attention_layer_indices
        data["rope_dim"] = self.rope_dim
        return data


#: The frozen target, as one importable object.
FROZEN_STUDENT = MoEStudentSpec()


def tiny_fixture(**overrides: Any) -> MoEStudentSpec:
    """A scaled-down member of the *same* architecture family as ``FROZEN_STUDENT``.

    Every structural property of the frozen target is preserved — hybrid group pattern,
    gated attention, GQA ratio, DeltaNet key/value head ratio, top-k routing with a shared
    expert, untied embeddings — and only the sizes shrink. That is what makes it a valid
    stand-in: a 22B model cannot be forward/backward tested here, but a model that differs
    from it *only* in width, depth and vocabulary can be, and any structural defect in the
    frozen configuration reproduces in it.

    Ratios held identical to the frozen target:

    ==========================  ================  ==============
    property                    frozen            tiny
    ==========================  ================  ==============
    heads : kv heads            24 : 2            4 : 2  (both GQA)
    deltanet value : key heads  48 : 16 (3x)      6 : 2  (3x)
    group pattern               DDDA x 12         DDDA x 2
    experts, top-k              24, 2             6, 2
    shared expert width         = expert width    = expert width
    tie_word_embeddings         False             False
    ==========================  ================  ==============

    The head:kv ratio is the one number that cannot be held (12x would need 24 heads, which
    is no longer tiny); it stays a GQA ratio > 1, which is the property under test.
    """
    fields: dict[str, Any] = dict(
        name="tiny_" + STUDENT_ID,
        hidden_size=64,
        num_hidden_layers=8,
        vocab_size=256,
        max_position_embeddings=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        num_experts=6,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        notes="scaled fixture; not a release candidate",
    )
    fields.update(overrides)
    return MoEStudentSpec(**fields)


#: Teacher FFN width scaled by the same factor as the fixture's hidden size, so
#: decomposition tests exercise a realistic dense -> sparse ratio instead of a trivial one.
TINY_TEACHER_FFN_INTERMEDIATE = 224

#: What `transformers` 5.15.1 actually builds for MTP on this architecture. Read from the
#: installed config, not assumed: the frozen target declares one MTP layer, and the runtime
#: has no field for it, so the intent is recorded and the tensors are not pretended into
#: existence.
MTP_STATUS = (
    "DECLARED, NOT BUILT: qwen3_5_moe_text in transformers 5.15.1 exposes no "
    "mtp_num_hidden_layers field and constructs no MTP head, so the student has no MTP "
    "tensors and no MTP loss can be trained today. The teacher's own mtp.* tensors are "
    "discarded on load by Qwen3_5ForCausalLM._keys_to_ignore_on_load_unexpected. The "
    "architecture field is kept as the extension point; any MTP result would be fabricated."
)


def build_config(spec: MoEStudentSpec = FROZEN_STUDENT, **overrides: Any):
    """A real `transformers` config for this student."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    fields = {k: v for k, v in spec.to_hf_text_config().items() if k != "model_type"}
    fields.update(overrides)
    return CONFIG_MAPPING[MOE_MODEL_TYPE](**fields)


def build_model(spec: MoEStudentSpec = FROZEN_STUDENT, *, meta: bool = True, **overrides: Any):
    """Instantiate the student. ``meta=True`` allocates no memory, which is how a 22B
    model is inspected on a laptop."""
    import torch
    from transformers import AutoModelForCausalLM

    config = build_config(spec, **overrides)
    if meta:
        with torch.device("meta"):
            return AutoModelForCausalLM.from_config(config)
    return AutoModelForCausalLM.from_config(config)


# ---------------------------------------------------------------------------
# the honest parameter audit
# ---------------------------------------------------------------------------
#: Component buckets, in the order the audit reports them.
COMPONENTS = (
    "embedding", "lm_head", "attention", "deltanet",
    "routed_experts", "shared_expert", "router", "norms", "mtp", "other",
)


def _bucket(name: str) -> str:
    if "embed_tokens" in name:
        return "embedding"
    if name.startswith("lm_head"):
        return "lm_head"
    if "mtp" in name:
        return "mtp"
    if "shared_expert" in name:
        return "shared_expert"
    if ".experts." in name:
        return "routed_experts"
    if "self_attn" in name:
        return "attention"
    if "linear_attn" in name:
        return "deltanet"
    # The router is `mlp.gate` — a bare Linear, distinct from the SwiGLU `gate_proj`.
    if name.endswith("mlp.gate.weight") or name.endswith("mlp.gate.bias"):
        return "router"
    if "norm" in name:
        return "norms"
    return "other"


def audit(spec: MoEStudentSpec = FROZEN_STUDENT) -> dict[str, Any]:
    """Count every parameter of the real instantiated model, by component.

    Built on ``meta`` so it costs nothing, and counted from the actual model's state dict
    rather than from a formula — a formula can agree with itself while disagreeing with
    what `transformers` builds.
    """
    model = build_model(spec, meta=True)
    counts = dict.fromkeys(COMPONENTS, 0)
    for name, tensor in model.state_dict().items():
        counts[_bucket(name)] += tensor.numel()

    total = sum(counts.values())
    embedding_total = counts["embedding"] + counts["lm_head"]
    per_layer_routed = counts["routed_experts"] // spec.num_hidden_layers
    active_routed = per_layer_routed * spec.num_experts_per_tok // spec.num_experts
    active = total - counts["routed_experts"] + spec.num_hidden_layers * active_routed

    return {
        "student_id": spec.name,
        "model_class": type(model).__name__,
        "components": counts,
        "exact_parameter_count": total,
        "difference_from_19B": total - 19_000_000_000,
        "non_embedding_parameters": total - embedding_total,
        "embedding_parameters": embedding_total,
        "active_parameters_per_token": active,
        "num_layers": spec.num_hidden_layers,
        "attention_layers": len(spec.attention_layer_indices),
        "deltanet_layers": len(spec.deltanet_layer_indices),
        "mtp_status": MTP_STATUS,
        "note": (
            "The name says 19B; the architecture is what it is. The frozen specification "
            "was NOT altered to hit a round number. ~19B most likely referred to "
            "non-embedding parameters."
        ),
    }


def render_audit(report: dict[str, Any] | None = None) -> str:
    report = report or audit()
    total = report["exact_parameter_count"]
    lines = [
        f"  {report['student_id']}  ({report['model_class']})",
        f"  {report['num_layers']} layers = {report['deltanet_layers']} DeltaNet + "
        f"{report['attention_layers']} full attention",
        "",
        f"    {'component':<18}{'parameters':>16}{'share':>9}",
    ]
    for name, count in report["components"].items():
        if count:
            lines.append(f"    {name:<18}{count:>16,}{count / total:>9.2%}")
    lines += [
        f"    {'TOTAL':<18}{total:>16,}",
        "",
        f"    exact_parameter_count   {total:>16,}",
        f"    difference_from_19B     {report['difference_from_19B']:>+16,}",
        f"    non_embedding           {report['non_embedding_parameters']:>16,}",
        f"    active_per_token        {report['active_parameters_per_token']:>16,}",
        "",
        f"    MTP: {report['mtp_status'][:76]}...",
    ]
    return "\n".join(lines)
