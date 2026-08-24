"""Does this model fit this device — for inference and for training?

Both answers reuse :func:`qwen_distill.architecture.memory.estimate_memory` rather than
comparing a weight-file size against VRAM, which is the mistake this project has
repeatedly warned against: weights are only one term of the envelope.

Training adds terms inference does not have — gradients, optimizer state, and
activations retained for the backward pass — and those dominate. Full bf16 fine-tuning
of a model costs roughly **16 bytes per parameter** before a single activation, which is
why a 16 GB card cannot full-fine-tune anything close to the sizes this project targets,
and why LoRA/QLoRA exist as options rather than conveniences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..architecture.memory import (
    GIB,
    QUANT_BYTES_PER_PARAM,
    DeploymentConfig,
    estimate_memory,
)
from ..architecture.params import count_parameters
from ..architecture.spec import HybridArchSpec

#: Bytes of *optimizer moment* state per trained parameter — the running statistics
#: only. Gradients and any fp32 master copy are counted separately (see
#: :data:`PRECISION_SCHEMES`) because the memory probe measures them at different
#: stages: gradients appear during ``backward()``, moments only on the first
#: ``optimizer.step()``, since AdamW allocates them lazily. Folding all three into one
#: number made the estimate impossible to check against a measurement component-wise.
OPTIMIZER_MOMENT_BYTES: dict[str, float] = {
    "adamw": 8.0,            # fp32 first and second moments
    "adamw_8bit": 2.0,       # 8-bit quantised moments (bitsandbytes)
    "adafactor": 1.0,        # factored second moment, no first moment
    "sgd": 4.0,              # fp32 momentum buffer
    "sgd_no_momentum": 0.0,  # stateless
}


@dataclass(frozen=True)
class PrecisionScheme:
    """How a training precision actually holds weights, gradients and activations.

    This distinction is not cosmetic. `torch.autocast` — what a T4 run uses — leaves
    **parameters and gradients in fp32** and casts only the compute, whereas a "pure
    bf16" run holds bf16 parameters and gradients with a separate fp32 master copy
    inside the optimizer. Both land near 16 bytes per parameter in total for AdamW, so
    a single total hides the difference; the per-component attribution the memory probe
    produces does not, and would flag a correct total as two wrong components.
    """

    name: str
    weight_bytes: float          # per parameter, as held for training
    gradient_bytes: float        # per trainable parameter
    master_copy_bytes: float     # extra fp32 optimizer-side copy; 0 when weights are fp32
    activation_bytes: float      # per activation element
    description: str


#: Keyed by the user-facing ``precision`` value, plus the explicit scheme names.
PRECISION_SCHEMES: dict[str, PrecisionScheme] = {
    "fp32": PrecisionScheme(
        "fp32", 4.0, 4.0, 0.0, 4.0,
        "no mixed precision: fp32 weights, gradients and activations",
    ),
    "fp16": PrecisionScheme(
        "amp_fp16", 4.0, 4.0, 0.0, 2.0,
        "torch.autocast(fp16) + GradScaler: fp32 weights and gradients, fp16 compute",
    ),
    "bf16": PrecisionScheme(
        "amp_bf16", 4.0, 4.0, 0.0, 2.0,
        "torch.autocast(bf16): fp32 weights and gradients, bf16 compute",
    ),
    "pure_bf16": PrecisionScheme(
        "pure_bf16", 2.0, 2.0, 4.0, 2.0,
        "bf16 weights and gradients with an fp32 master copy in the optimizer",
    ),
}

#: During training the loss path materialises the logits three times over, which a
#: single ``tokens x vocab x 4`` term understates by 2.5-3x. Verified against
#: ``transformers.loss.loss_utils.ForCausalLMLoss``, which does ``logits =
#: logits.float()`` (one fp32 copy), and against the tensors autograd retains for
#: ``cross_entropy`` (a second fp32 log-softmax buffer). The ``lm_head`` output itself
#: is live alongside both, at the activation dtype. On a 248k-vocab model this term is
#: measured in gigabytes, so getting it wrong is not a rounding error.
LOSS_PATH_FP32_LOGIT_COPIES = 2

#: How much of the model is actually trained under each strategy.
#: LoRA ranks typically touch well under 1% of parameters.
TRAINABLE_FRACTION: dict[str, float] = {
    "full": 1.0,
    "lora": 0.005,
    "qlora": 0.005,
}

#: Bytes per parameter for the frozen base weights under each strategy.
BASE_BYTES_PER_PARAM: dict[str, float] = {
    "full": 2.0,                             # bf16
    "lora": 2.0,                             # bf16 frozen base
    "qlora": QUANT_BYTES_PER_PARAM["int4"],  # 4-bit frozen base
}


@dataclass
class InferenceFit:
    """Whether a model can be served on a device at a given context."""

    label: str
    quantization: str
    context_length: int
    weights_gib: float
    kv_cache_gib: float
    state_gib: float
    activations_gib: float
    overhead_gib: float
    total_gib: float
    available_gib: float
    verdict: str                    # FITS | TIGHT | DOES NOT FIT
    headroom_gib: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _verdict(total: float, available: float, tight_margin: float = 1.5) -> str:
    """Three outcomes, because 'fits with 200 MB to spare' is not the same as fits.

    A run that fits only when nothing else touches the card will fail the moment a
    desktop compositor or a second process appears, so it is reported as TIGHT.
    """
    if total > available:
        return "DOES NOT FIT"
    if available - total < tight_margin:
        return "TIGHT"
    return "FITS"


def analyse_inference_fit(
    spec: HybridArchSpec,
    available_gib: float,
    *,
    quantization: str = "q4_k_m",
    context_length: int = 32768,
    label: str | None = None,
    embedding_quant: str | None = "q6_k",
    batch_size: int = 1,
) -> InferenceFit:
    """Full-envelope inference fit for one (quantisation, context) pair."""
    config = DeploymentConfig(
        context_length=context_length,
        batch_size=batch_size,
        weight_quant=quantization,
        embedding_quant=embedding_quant,
    )
    estimate = estimate_memory(spec, config)
    total = estimate.total_gib
    return InferenceFit(
        label=label or spec.name,
        quantization=quantization,
        context_length=context_length,
        weights_gib=estimate.weights / GIB,
        kv_cache_gib=estimate.kv_cache / GIB,
        state_gib=(estimate.recurrent_state + estimate.conv_state) / GIB,
        activations_gib=estimate.activations / GIB,
        overhead_gib=estimate.runtime_overhead / GIB,
        total_gib=total,
        available_gib=available_gib,
        verdict=_verdict(total, available_gib),
        headroom_gib=available_gib - total,
    )


def fit_matrix(
    spec: HybridArchSpec,
    available_gib: float,
    *,
    quantizations: tuple[str, ...] = ("bf16", "int8", "q6_k", "q5_k_m", "q4_k_m"),
    contexts: tuple[int, ...] = (8192, 16384, 32768, 65536, 131072),
    embedding_quant: str | None = "q6_k",
) -> dict[str, dict[int, InferenceFit]]:
    """Quantisation x context grid of inference fits."""
    return {
        quant: {
            ctx: analyse_inference_fit(
                spec, available_gib, quantization=quant, context_length=ctx,
                embedding_quant=embedding_quant,
            )
            for ctx in contexts
        }
        for quant in quantizations
    }


@dataclass
class TrainingFit:
    """Whether a training configuration is plausible on a device."""

    label: str
    strategy: str                 # full | lora | qlora
    optimizer: str
    precision: str
    sequence_length: int
    batch_size: int
    gradient_checkpointing: bool
    total_parameters: int
    trainable_parameters: int
    base_weights_gib: float
    optimizer_state_gib: float
    activations_gib: float
    overhead_gib: float
    total_gib: float
    available_gib: float
    verdict: str
    headroom_gib: float
    #: Gradients, separated from optimizer state because the probe measures them at a
    #: different stage: gradients during ``backward()``, moments on the first ``step()``.
    gradients_gib: float = 0.0
    #: Logits and the two fp32 copies the loss path makes of them, separated because on
    #: a large vocabulary this term rivals the weights and is easy to get wrong alone.
    logits_gib: float = 0.0
    precision_scheme: str = ""
    #: What the estimate predicts for each of the two quantities the probe reports.
    #: ``allocated`` is live tensors — the estimator's own arithmetic, and the one a
    #: correction should be derived from. ``reserved`` adds the CUDA context and
    #: allocator allowance, and is what the process actually occupies.
    predicted_allocated_gib: float = 0.0
    predicted_reserved_gib: float = 0.0
    suggestions: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.verdict != "NOT FEASIBLE"

    def component_estimate(self) -> dict[str, float]:
        """The estimate keyed the way :mod:`qwen_distill.training.memory_probe` keys
        its measurements, so the two can be compared term by term rather than only in
        total. A correct total made of two compensating errors is still two errors."""
        return {
            "weights_gib": self.base_weights_gib,
            "activations_gib": self.activations_gib + self.logits_gib,
            "gradients_gib": self.gradients_gib,
            "optimizer_state_gib": self.optimizer_state_gib,
            "peak_allocated_gib": self.predicted_allocated_gib,
            "peak_reserved_gib": self.predicted_reserved_gib,
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["feasible"] = self.feasible
        return data


def resolve_precision_scheme(precision: str | None) -> PrecisionScheme:
    """Map a configured precision onto how memory is actually held.

    Labels arrive from configs and from run summaries, so they carry decoration —
    ``"fp16 (CPU)"``, ``"4-bit base + bf16 compute"``. The first recognised token wins,
    and an unrecognised label falls back to the AMP bf16 scheme rather than raising:
    a feasibility check must still produce an estimate for a label it has not seen.
    """
    if not precision:
        return PRECISION_SCHEMES["bf16"]
    if precision in PRECISION_SCHEMES:
        return PRECISION_SCHEMES[precision]
    lowered = precision.lower()
    for key in ("pure_bf16", "fp32", "fp16", "bf16"):
        if key in lowered:
            return PRECISION_SCHEMES[key]
    return PRECISION_SCHEMES["bf16"]


def estimate_training_memory(
    spec: HybridArchSpec,
    available_gib: float,
    *,
    strategy: str = "qlora",
    optimizer: str = "adamw_8bit",
    sequence_length: int = 2048,
    batch_size: int = 1,
    gradient_checkpointing: bool = True,
    label: str | None = None,
    runtime_overhead_gib: float = 1.2,
    precision: str | None = None,
) -> TrainingFit:
    """Estimate peak training memory and say whether it is plausible.

    The activation term is the one most sensitive to configuration, and the one users
    most often get wrong. With gradient checkpointing only layer boundaries are
    retained, so activations scale with ``layers`` rather than with the far larger
    per-layer intermediates; without it, the retained set is roughly an order of
    magnitude larger. Both are approximations — the point is to catch an obviously
    doomed configuration before it consumes an hour of rented GPU, not to predict peak
    usage to the megabyte.
    """
    if strategy not in TRAINABLE_FRACTION:
        raise ValueError(f"unknown strategy {strategy!r}; known: {sorted(TRAINABLE_FRACTION)}")
    if optimizer not in OPTIMIZER_MOMENT_BYTES:
        raise ValueError(f"unknown optimizer {optimizer!r}; known: {sorted(OPTIMIZER_MOMENT_BYTES)}")

    scheme = resolve_precision_scheme(precision)

    params = count_parameters(spec)
    total_params = params.total
    trainable = int(total_params * TRAINABLE_FRACTION[strategy])

    # QLoRA freezes a 4-bit base; everything else holds the base at the scheme's
    # training weight precision, which under autocast is fp32 and not the compute dtype.
    base_bytes = total_params * (
        BASE_BYTES_PER_PARAM["qlora"] if strategy == "qlora" else scheme.weight_bytes
    )
    gradient_bytes = trainable * scheme.gradient_bytes
    optimizer_bytes = trainable * (
        scheme.master_copy_bytes + OPTIMIZER_MOMENT_BYTES[optimizer]
    )

    tokens = sequence_length * batch_size
    if gradient_checkpointing:
        # One saved activation per layer boundary, plus a working buffer for the layer
        # currently being recomputed.
        per_token = spec.num_hidden_layers * spec.hidden_size + 4 * spec.intermediate_size
    else:
        # Every layer retains its intermediates for the backward pass.
        per_token = spec.num_hidden_layers * (
            2 * spec.hidden_size + 3 * spec.intermediate_size
        )
    activation_bytes = tokens * per_token * scheme.activation_bytes

    # The loss path holds the logits three times over: the lm_head output at the
    # activation dtype, its fp32 upcast, and cross_entropy's retained log-softmax
    # buffer. See LOSS_PATH_FP32_LOGIT_COPIES.
    logit_bytes = tokens * spec.vocab_size * (
        scheme.activation_bytes + 4.0 * LOSS_PATH_FP32_LOGIT_COPIES
    )

    # Tensors the estimator can actually derive, kept separate from the allowance for
    # things it cannot: the CUDA context and the caching allocator's reserve.
    allocated_bytes = base_bytes + gradient_bytes + optimizer_bytes + activation_bytes + logit_bytes
    overhead_bytes = runtime_overhead_gib * GIB
    total_bytes = allocated_bytes + overhead_bytes
    total_gib = total_bytes / GIB

    verdict = "PLAUSIBLE" if total_gib <= available_gib else "NOT FEASIBLE"
    if verdict == "PLAUSIBLE" and available_gib - total_gib < 1.0:
        verdict = "TIGHT"

    fit = TrainingFit(
        label=label or spec.name,
        strategy=strategy,
        optimizer=optimizer,
        precision=precision or ("4-bit base + bf16 compute" if strategy == "qlora" else "bf16"),
        sequence_length=sequence_length,
        batch_size=batch_size,
        gradient_checkpointing=gradient_checkpointing,
        total_parameters=total_params,
        trainable_parameters=trainable,
        base_weights_gib=base_bytes / GIB,
        optimizer_state_gib=optimizer_bytes / GIB,
        activations_gib=activation_bytes / GIB,
        gradients_gib=gradient_bytes / GIB,
        logits_gib=logit_bytes / GIB,
        precision_scheme=scheme.name,
        predicted_allocated_gib=allocated_bytes / GIB,
        predicted_reserved_gib=total_gib,
        overhead_gib=runtime_overhead_gib,
        total_gib=total_gib,
        available_gib=available_gib,
        verdict=verdict,
        headroom_gib=available_gib - total_gib,
    )
    if not fit.feasible:
        fit.suggestions = _training_suggestions(fit, spec)
    return fit


def _training_suggestions(fit: TrainingFit, spec: HybridArchSpec) -> list[str]:
    """Concrete, ordered changes — cheapest and least damaging first."""
    suggestions: list[str] = []
    if not fit.gradient_checkpointing:
        suggestions.append("enable gradient checkpointing (largest single saving)")
    if fit.batch_size > 1:
        suggestions.append(
            f"reduce batch size {fit.batch_size} -> 1 and recover the effective batch "
            "with gradient accumulation (no quality cost)"
        )
    if fit.sequence_length > 1024:
        suggestions.append(
            f"reduce sequence length {fit.sequence_length} -> {fit.sequence_length // 2}"
        )
    if fit.strategy == "full":
        suggestions.append("switch to LoRA, or QLoRA for a 4-bit frozen base")
    elif fit.strategy == "lora":
        suggestions.append("switch to QLoRA (4-bit frozen base)")
    if fit.optimizer == "adamw":
        suggestions.append("use adamw_8bit to cut optimizer state by ~6 bytes/parameter")
    suggestions.append(
        f"use a larger GPU: this configuration needs about {fit.total_gib:.1f} GiB"
    )
    return suggestions
