"""Parameter-efficient training for the canonical student.

Why this module exists
----------------------

The canonical student ``qwen38_19b_h5120_l48_moe`` has 13,008,505,728 parameters. In
bf16 the weights alone are 24.23 GiB, measured. Full-parameter AdamW needs, on top of
that, bf16 gradients (24.23 GiB) and fp32 master/exp_avg/exp_avg_sq (145 GiB). On a
48 GB A40 that is not close, and it is further from closing once the 27B teacher is
resident in the same process to produce an online signal.

So the canonical KD experiment is only executable through a parameter-efficient path,
and the trainer refused every one of them: ``STRATEGIES`` has advertised ``lora`` and
``qlora`` since the config schema was written, while ``_require_supported`` raised
``NotImplementedError`` for anything but ``full``. This module is the smallest thing
that closes that gap honestly. It does not touch the architecture: the base model is
loaded from the materialised checkpoint exactly as written, every base weight is frozen,
and the only new parameters are the LoRA factors.

What is deliberately *not* done
-------------------------------

``peft.prepare_model_for_kbit_training`` is the usual next call and this module does not
make it. It upcasts every non-quantised parameter to fp32, which for this student means
``embed_tokens`` and ``lm_head`` — 2.54B parameters between them — go from 5.09 GiB to
10.17 GiB. Both stay frozen throughout, so the upcast buys nothing and costs more than
the headroom the teacher leaves. The one thing it does that is actually needed here is
:meth:`enable_input_require_grads`, without which gradient checkpointing produces no
gradient at all: the checkpointed segment sees only inputs that do not require grad,
so autograd records nothing to differentiate. That call is made explicitly below.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CANONICAL_LORA_TARGETS",
    "PEFT_STRATEGIES",
    "adapter_state_dict",
    "apply_peft",
    "build_quantization_config",
    "is_peft_model",
    "require_peft",
    "trainable_parameter_report",
]

#: Strategies this module implements. ``full`` is handled by the trainer directly.
PEFT_STRATEGIES = ("lora", "qlora")

#: The projections LoRA adapts, named exactly as they appear in the materialised
#: canonical student. The set is chosen to reach **every one of the 48 layers**:
#:
#: * ``in_proj_qkv`` / ``out_proj`` are the DeltaNet linear-attention projections and
#:   exist in the 36 ``linear_attention`` layers;
#: * ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` are the full-attention
#:   projections and exist in the other 12.
#:
#: Adapting only ``q/k/v/o`` — the reflex for a dense transformer — would leave 36 of
#: 48 layers untouched, so the run would not be a fair test of the canonical stack.
#:
#: The MoE expert matrices are excluded on purpose. Targeting them means 384 adapters
#: per projection and turns a minimal probe into the largest thing in the process; the
#: router would also be adapted with a top-2 gate that sees at most two experts per
#: token, so most of those adapters would receive near-zero gradient. Excluding them is
#: a scope decision about this first controlled run, not an architecture change: the
#: expert weights are present, frozen, and used in every forward pass.
#:
#: ``in_proj_a``, ``in_proj_b`` and ``in_proj_z`` are not listed. peft matches a target
#: against the final dotted segment of a module's name, so these are unaffected by
#: ``in_proj_qkv``; they are small (48x5120) gate/decay projections and are left frozen.
CANONICAL_LORA_TARGETS = (
    "in_proj_qkv",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)


def require_peft() -> Any:
    """Import ``peft``, or explain precisely what is missing and why it is needed."""
    try:
        import peft
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise NotImplementedError(
            "strategies 'lora' and 'qlora' need the `peft` package, which is not "
            "installed. Install it with `pip install peft`. The canonical 13.01B "
            "student cannot be trained with full-parameter AdamW on a single 48 GB "
            "card, so this is not an optional extra for that experiment."
        ) from exc
    return peft


def build_quantization_config(strategy: str, compute_dtype: Any) -> Any | None:
    """The bitsandbytes config for ``strategy``, or ``None`` when the base stays dense.

    ``qlora`` is NF4 with double quantisation, the configuration QLoRA was measured
    with. ``lora`` keeps the base weights in the training precision.
    """
    if strategy != "qlora":
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def is_peft_model(model: Any) -> bool:
    """True when ``model`` carries LoRA adapters."""
    return hasattr(model, "peft_config")


def apply_peft(model: Any, config: Any) -> Any:
    """Freeze the base model and attach LoRA adapters to :data:`CANONICAL_LORA_TARGETS`.

    Returns the wrapped model. The base weights are not modified, moved or re-typed.
    """
    peft = require_peft()

    # A frozen embedding emits activations with requires_grad=False, and a checkpointed
    # segment whose inputs all lack requires_grad records no graph -- the backward pass
    # then finds nothing to differentiate and every adapter gradient comes back None.
    # This hook is what makes gradient checkpointing and PEFT compose, and it is needed
    # for `lora` as much as `qlora`: what matters is that the base is frozen, not that
    # it is quantised. See the module docstring for what is deliberately skipped here.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = peft.LoraConfig(
        r=config.training.lora_rank,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        target_modules=list(CANONICAL_LORA_TARGETS),
        bias="none",
        task_type=peft.TaskType.CAUSAL_LM,
    )
    wrapped = peft.get_peft_model(model, lora_config)

    trainable = [n for n, p in wrapped.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "LoRA attached no trainable parameters: none of "
            f"{CANONICAL_LORA_TARGETS} matched a module in this model. Training would "
            "run, report a loss and change nothing."
        )
    return wrapped


def trainable_parameter_report(model: Any) -> dict[str, Any]:
    """Count what will actually receive gradients, and what is merely resident.

    Recorded in the run summary so a reader can tell a parameter-efficient run from a
    full one without having to trust the strategy label.
    """
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        # bitsandbytes packs two 4-bit values per uint8, so `numel()` on a quantised
        # tensor is half the logical parameter count. Recover it, or the student would
        # be reported as billions of parameters smaller than it is.
        if param.__class__.__name__ == "Params4bit":
            n *= 2
        total += n
        if param.requires_grad:
            trainable += n
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": (trainable / total) if total else 0.0,
        "frozen_parameters": total - trainable,
    }


def adapter_state_dict(model: Any) -> dict[str, Any]:
    """Just the adapter tensors, for a checkpoint that is megabytes rather than 25 GB."""
    peft = require_peft()
    return peft.get_peft_model_state_dict(model)
