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

#: Bytes per *trained* parameter, beyond the frozen weights themselves.
#: Mixed-precision AdamW keeps an fp32 master copy (4) + two fp32 moments (8) and a
#: gradient (2 in bf16). 8-bit optimizers replace the 8 with 2.
OPTIMIZER_BYTES_PER_PARAM: dict[str, float] = {
    "adamw": 4 + 8 + 2,          # fp32 master + m,v + bf16 grad = 14
    "adamw_8bit": 4 + 2 + 2,     # fp32 master + 8-bit m,v + bf16 grad = 8
    "adafactor": 4 + 1 + 2,      # factored second moment
    "sgd": 4 + 4 + 2,            # master + momentum + grad
    "sgd_no_momentum": 4 + 2,
}

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
    suggestions: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.verdict != "NOT FEASIBLE"

    def to_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["feasible"] = self.feasible
        return data


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
    if optimizer not in OPTIMIZER_BYTES_PER_PARAM:
        raise ValueError(f"unknown optimizer {optimizer!r}; known: {sorted(OPTIMIZER_BYTES_PER_PARAM)}")

    params = count_parameters(spec)
    total_params = params.total
    trainable = int(total_params * TRAINABLE_FRACTION[strategy])

    base_bytes = total_params * BASE_BYTES_PER_PARAM[strategy]
    optimizer_bytes = trainable * OPTIMIZER_BYTES_PER_PARAM[optimizer]

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
    activation_bytes = tokens * per_token * 2  # bf16

    # Logits over a large vocabulary are material during training: they are produced in
    # fp32 and retained for the loss.
    activation_bytes += tokens * spec.vocab_size * 4

    overhead_bytes = runtime_overhead_gib * GIB
    total_bytes = base_bytes + optimizer_bytes + activation_bytes + overhead_bytes
    total_gib = total_bytes / GIB

    verdict = "PLAUSIBLE" if total_gib <= available_gib else "NOT FEASIBLE"
    if verdict == "PLAUSIBLE" and available_gib - total_gib < 1.0:
        verdict = "TIGHT"

    fit = TrainingFit(
        label=label or spec.name,
        strategy=strategy,
        optimizer=optimizer,
        precision=(
            precision if precision
            else ("4-bit base + bf16 compute" if strategy == "qlora" else "bf16")
        ),
        sequence_length=sequence_length,
        batch_size=batch_size,
        gradient_checkpointing=gradient_checkpointing,
        total_parameters=total_params,
        trainable_parameters=trainable,
        base_weights_gib=base_bytes / GIB,
        optimizer_state_gib=optimizer_bytes / GIB,
        activations_gib=activation_bytes / GIB,
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
