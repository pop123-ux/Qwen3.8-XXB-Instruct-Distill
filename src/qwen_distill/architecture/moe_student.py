"""The frozen primary student: ``qwen38_19b_h5120_l48_moe``.

One canonical specification, so the architecture is a named object rather than fourteen
magic numbers scattered across configs and scripts. Every field below is the frozen
research target; nothing here is tuned, and nothing may be quietly adjusted to make a
parameter count come out round.

**The name says 19B. The architecture is 13.01B, corrected down from 22.07B.** The first
implementation of this specification used 24 routed experts of width 768 and came to
22,072,134,528 parameters, of which 13.59B — 61.6% of the model — were routed experts. It
did not fit a 16 GB card at any release precision or any context length, so it failed the
project's primary constraint. See :data:`REJECTED` for the full record.

The correction is one field: ``num_experts`` 24 -> 8. Nothing else moved. In particular the
*active* parameter count is essentially unchanged (9,615,051,648 -> 9,611,119,488, a 0.04%
difference from the smaller router), because a token was only ever using two experts. The
16 stored-but-unused experts bought VRAM cost and no per-token capacity.

The arithmetic that settles it, derived in :data:`PARAMETER_MODEL` and asserted by a test::

    total  = BASE + 3.H.L.(E.W + S) + H.L.E + H.L
    active = BASE + 3.H.L.(K.W + S) + H.L.E + H.L

Total depends on the *product* ``E x W``; active depends on ``K x W``. So the split of a
fixed expert budget between count and width is free with respect to memory and is *not*
free with respect to per-token capacity: at a fixed total, wider experts carry more active
parameters. That is why the correction reduces the expert count and leaves the width at 768
rather than the reverse.

The name is unchanged and is now further from the count, not closer. Renaming would break
every artifact referring to the frozen target; :func:`audit` reports the real number and a
test pins it.

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

#: Canonical name. Kept even though the count is 13.01B: renaming it would break every
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
    #: Corrected from 24. Twenty-four experts of width 768 put 13.59B parameters — 61.6%
    #: of the model — into routed experts, of which any token used two. See :data:`REJECTED`.
    num_experts: int = 8
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


#: The closed-form parameter model, as an inspectable function rather than a comment.
#:
#: Derived tensor by tensor from the specification and cross-checked against the
#: instantiated model — both agree to the parameter, on every component bucket.
#:
#: The two invariants that decide the architecture::
#:
#:     total  depends on  E x W     (the expert *product*, not the split)
#:     active depends on  K x W     (so at fixed total, wider experts carry more)
#:
#: Consequences worth stating because they are counter-intuitive: splitting a fixed expert
#: budget as 8x768, 6x1024, 12x512 or 24x256 gives the *same* total parameters and the same
#: VRAM, and the same fraction of the teacher's FFN channels covered by the decomposition.
#: What differs is per-token capacity, which is why the correction cut the count and kept
#: the width.
PARAMETER_MODEL = (
    "total  = BASE + 3.H.L.(E.W + S) + H.L.E + H.L\n"
    "active = BASE + 3.H.L.(K.W + S) + H.L.E + H.L\n"
    "BASE   = embeddings + lm_head + attention + deltanet + norms"
)


def parameter_model(spec: MoEStudentSpec = FROZEN_STUDENT) -> dict[str, int]:
    """Closed-form counts, computed without instantiating anything.

    Independent of :func:`audit`, which builds the real model and sums its tensors. Keeping
    both and asserting they agree is what makes either trustworthy: an error in the spec
    reaches both, but an error in *either derivation* shows up as a mismatch.
    """
    h, layers = spec.hidden_size, spec.num_hidden_layers
    n_attn = len(spec.attention_layer_indices)
    n_dn = len(spec.deltanet_layer_indices)
    q_dim = spec.num_attention_heads * spec.head_dim
    kv_dim = spec.num_key_value_heads * spec.head_dim
    key_dim = spec.linear_num_key_heads * spec.linear_key_head_dim
    value_dim = spec.linear_num_value_heads * spec.linear_value_head_dim
    conv_dim = 2 * key_dim + value_dim

    embedding = spec.vocab_size * h
    lm_head = 0 if spec.tie_word_embeddings else spec.vocab_size * h
    # q_proj is double width: the output gate is fused into it.
    attention = n_attn * (2 * q_dim * h + 2 * kv_dim * h + h * q_dim + 2 * spec.head_dim)
    deltanet = n_dn * (
        conv_dim * h + value_dim * h + 2 * spec.linear_num_value_heads * h
        + conv_dim * spec.linear_conv_kernel_dim + h * value_dim
        + spec.linear_value_head_dim + 2 * spec.linear_num_value_heads
    )
    norms = layers * 2 * h + h

    per_width = 3 * h * layers
    routed = per_width * spec.num_experts * spec.moe_intermediate_size
    shared = per_width * spec.shared_expert_intermediate_size + h * layers
    router = spec.num_experts * h * layers

    base = embedding + lm_head + attention + deltanet + norms
    total = base + routed + shared + router
    active = (base + per_width * spec.num_experts_per_tok * spec.moe_intermediate_size
              + shared + router)
    return {
        "embedding": embedding, "lm_head": lm_head, "attention": attention,
        "deltanet": deltanet, "routed_experts": routed, "shared_expert": shared,
        "router": router, "norms": norms, "mtp": 0,
        "total": total, "active_per_token": active, "base": base,
    }


#: Hard upper bound on total parameters, set by the deployment constraint rather than by
#: taste. Derived, not chosen: 16 GB usable is 13.56 GiB; a Q5 release at 32,768 tokens
#: needs weights + 3% quantisation overhead + 0.9 GiB runtime + 0.111 GiB recurrent state +
#: 0.28 GiB activations + 0.75 GiB KV to land under that with a gigabyte to spare, which
#: allows roughly 10.2 GiB of weights, which at 5.7 bits per parameter is about 15.4B
#: parameters. Rounded down to a round number that leaves the Q6 path alive as well.
#:
#: A test fails if the frozen student exceeds this. The bound exists so that a future edit
#: adding experts, widening the FFN or untying something cannot silently reintroduce a
#: model that does not deploy.
PARAMETER_BUDGET = 15_000_000_000

#: Configurations evaluated and rejected, kept as the research record. Each entry carries
#: the measurement that rejected it, not an opinion.
REJECTED: tuple[dict[str, Any], ...] = (
    {
        "config": "num_experts=24, moe_intermediate_size=768",
        "total_parameters": 22_072_134_528,
        "active_parameters": 9_615_051_648,
        "routed_expert_share": 0.6157,
        "why_rejected": (
            "Does not fit a 16 GB card at Q4, Q5 or Q6 at any context length, including "
            "2,048 tokens. The cheapest all-Q4 configuration needed 13.93 GiB against "
            "13.56 GiB usable. 13.59B of the 22.07B were routed experts, of which a token "
            "used two: the other 16 experts cost VRAM and contributed nothing per token."
        ),
        "status": "rejected — failed the primary deployment constraint",
    },
    {
        "config": "num_experts=12, moe_intermediate_size=768",
        "total_parameters": 15_274_412_928,
        "active_parameters": 9_612_102_528,
        "why_rejected": (
            "Fits Q4 to 65,536 tokens and Q5 to 32,768, but leaves no Q6 path at any "
            "context and halves the Q4 context reach. Active parameters are within 0.01% of "
            "the chosen configuration, so the extra 2.27B buys routing diversity only — not "
            "per-token capacity — at the cost of the long-context regime the context "
            "specialisation research needs."
        ),
        "status": "rejected — feasible but dominated",
    },
    {
        "config": "num_experts=24, moe_intermediate_size=256",
        "total_parameters": 13_012_437_888,
        "active_parameters": 8_860_076_928,
        "why_rejected": (
            "Identical total parameters and identical VRAM to the chosen configuration — "
            "total depends on E x W, not on the split — while carrying 751M fewer active "
            "parameters, because active FFN width per token falls from 2,304 to 1,280. It "
            "keeps 24 experts, which serves the MoE novelty objective, but the project's "
            "stated priority order puts capability above novelty."
        ),
        "status": "rejected — same memory, less per-token capacity",
    },
    {
        "config": "num_experts=6, moe_intermediate_size=1024",
        "total_parameters": 13_008_014_208,
        "active_parameters": 9_988_115_328,
        "why_rejected": (
            "The same total as the chosen configuration with 377M *more* active parameters, "
            "which is genuinely attractive. Not taken because it activates a third of the "
            "experts per token rather than a quarter, changes two fields instead of one, and "
            "moves the expert width the FFN decomposition was measured at. Recorded because "
            "it is the strongest alternative and should be revisited if per-token capacity "
            "turns out to bind before routing diversity does."
        ),
        "status": "rejected — viable alternative, kept on the record",
    },
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
