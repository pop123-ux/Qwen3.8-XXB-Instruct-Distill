"""Generate experiment recommendations from measured hardware.

Recommendations are **derived**, not written down: every entry below is produced by
running the fit analysis against the device's actual VRAM. That matters because static
prose goes stale the moment a tier boundary or a memory formula changes, and because a
user with an unusual card (10 GB, 20 GB) deserves an answer rather than the nearest
canned tier.

Three buckets, deliberately: what works, what works with care, and what will not. The
third is the most useful one — the project's whole premise is that people waste time
discovering OOMs the hard way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..architecture.spec import HybridArchSpec
from .fit import analyse_inference_fit, estimate_training_memory
from .tiers import Tier, classify

#: Reference student sizes used to probe what a device can hold. These are *probes*,
#: not proposals: the project has not chosen a student architecture.
PROBE_SPECS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("~0.5B toy", dict(hidden_size=1024, num_hidden_layers=12, intermediate_size=2816,
                       num_attention_heads=8, num_key_value_heads=2, head_dim=128,
                       linear_num_key_heads=4, linear_num_value_heads=12,
                       tie_word_embeddings=True)),
    ("~1.8B small", dict(hidden_size=2048, num_hidden_layers=24, intermediate_size=5632,
                         num_attention_heads=16, num_key_value_heads=4, head_dim=128,
                         linear_num_key_heads=8, linear_num_value_heads=24,
                         tie_word_embeddings=True)),
    ("~8B mid", dict(hidden_size=3584, num_hidden_layers=40, intermediate_size=10240,
                     num_attention_heads=16, num_key_value_heads=4, head_dim=224,
                     linear_num_key_heads=8, linear_num_value_heads=24,
                     tie_word_embeddings=True)),
    ("~14B large", dict(hidden_size=4608, num_hidden_layers=48, intermediate_size=13824,
                        num_attention_heads=20, num_key_value_heads=4, head_dim=256,
                        linear_num_key_heads=16, linear_num_value_heads=48,
                        tie_word_embeddings=True)),
)


def _probe(name: str, overrides: dict[str, Any]) -> HybridArchSpec:
    return HybridArchSpec(name=name, vocab_size=248320, **overrides)


@dataclass
class Recommendations:
    """What a specific machine can plausibly do."""

    tier: Tier
    available_gib: float
    device_name: str
    good: list[str] = field(default_factory=list)
    with_care: list[str] = field(default_factory=list)
    not_realistic: list[str] = field(default_factory=list)
    inference: list[str] = field(default_factory=list)
    training: list[str] = field(default_factory=list)
    evaluation: list[str] = field(default_factory=list)
    architecture: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_level": self.tier.level,
            "tier_name": self.tier.name,
            "tier_summary": self.tier.summary,
            "available_gib": self.available_gib,
            "device_name": self.device_name,
            "good": self.good,
            "with_care": self.with_care,
            "not_realistic": self.not_realistic,
            "inference": self.inference,
            "training": self.training,
            "evaluation": self.evaluation,
            "architecture_experiments": self.architecture,
        }


def recommend(
    available_gib: float | None,
    device_name: str = "unknown",
    *,
    reserved_gib: float = 1.0,
    teacher: HybridArchSpec | None = None,
) -> Recommendations:
    """Derive recommendations by running the fit analysis on this device."""
    tier = classify(available_gib)
    usable = max(0.0, (available_gib or 0.0) - reserved_gib)
    result = Recommendations(tier=tier, available_gib=usable, device_name=device_name)

    if tier.level == 0:
        result.good = [
            "unit tests and the full analysis suite (no GPU required)",
            "metadata verification and chat-template inspection",
            "architecture search and VRAM estimation",
            "dataset preparation and filtering",
        ]
        result.not_realistic = [
            "any model inference at useful speed",
            "any training",
        ]
        result.evaluation = ["analysis-only; no generation"]
        result.architecture = ["parameter/memory estimation and meta-device shape checks"]
        return result

    teacher = teacher or HybridArchSpec(name="Qwen3.8-27B")

    # --- inference: probe each size at 4-bit and bf16 --------------------
    for label, overrides in PROBE_SPECS:
        spec = _probe(label, overrides)
        q4 = analyse_inference_fit(spec, usable, quantization="q4_k_m", context_length=8192)
        bf16 = analyse_inference_fit(spec, usable, quantization="bf16", context_length=8192)
        if bf16.verdict == "FITS":
            result.inference.append(f"{label} bf16 @8k ({bf16.total_gib:.1f} GiB)")
        elif q4.verdict == "FITS":
            result.inference.append(f"{label} 4-bit @8k ({q4.total_gib:.1f} GiB)")
        elif q4.verdict == "TIGHT":
            result.with_care.append(f"{label} 4-bit @8k inference ({q4.total_gib:.1f} GiB, tight)")
        else:
            result.not_realistic.append(f"{label} inference ({q4.total_gib:.1f} GiB at 4-bit)")

    teacher_q4 = analyse_inference_fit(teacher, usable, quantization="q4_k_m", context_length=8192)
    teacher_bf16 = analyse_inference_fit(teacher, usable, quantization="bf16", context_length=8192)
    if teacher_bf16.verdict != "FITS":
        result.not_realistic.append(
            f"Qwen3.8-27B bf16 inference (needs {teacher_bf16.total_gib:.1f} GiB)"
        )
    if teacher_q4.verdict == "DOES NOT FIT":
        result.not_realistic.append(
            f"Qwen3.8-27B 4-bit inference (needs {teacher_q4.total_gib:.1f} GiB)"
        )
    elif teacher_q4.verdict == "TIGHT":
        result.with_care.append(
            f"Qwen3.8-27B 4-bit inference at short context ({teacher_q4.total_gib:.1f} GiB, tight)"
        )
    else:
        result.good.append(f"Qwen3.8-27B 4-bit inference ({teacher_q4.total_gib:.1f} GiB)")

    # --- training: probe each size under each strategy -------------------
    for label, overrides in PROBE_SPECS:
        spec = _probe(label, overrides)
        # Strongest strategy first: full-parameter training is the most capable, and a
        # user with the VRAM for it should be told so rather than pushed to QLoRA.
        for strategy in ("full", "lora", "qlora"):
            fit = estimate_training_memory(
                spec, usable, strategy=strategy, sequence_length=2048,
                gradient_checkpointing=True,
            )
            if fit.verdict == "PLAUSIBLE":
                result.training.append(
                    f"{label} {strategy.upper()} @2048 tokens ({fit.total_gib:.1f} GiB)"
                )
                break
            if fit.verdict == "TIGHT":
                result.with_care.append(
                    f"{label} {strategy.upper()} training @2048 ({fit.total_gib:.1f} GiB, tight)"
                )
                break
        else:
            result.not_realistic.append(f"{label} training (even QLoRA exceeds VRAM)")

    full_teacher = estimate_training_memory(teacher, usable, strategy="full", sequence_length=2048)
    if not full_teacher.feasible:
        result.not_realistic.append(
            f"full-parameter training of a 27B-class model "
            f"(needs ~{full_teacher.total_gib:.0f} GiB)"
        )

    # --- evaluation and architecture work --------------------------------
    result.good.extend([
        "the full analysis suite and unit tests",
        "architecture prototyping and shape verification",
        "training-pipeline debugging with toy models",
    ])
    result.evaluation = [
        "Tier 1 development evaluation on models that fit above",
        "reasoning-cost sweeps on small models",
    ]
    if teacher_q4.verdict in ("FITS", "TIGHT"):
        result.evaluation.append("teacher evaluation at short context (quantised)")
    else:
        result.evaluation.append(
            "teacher baseline requires a larger GPU; generate teacher data elsewhere "
            "and train/evaluate the student here"
        )
    result.architecture = [
        "candidate architecture search (CPU-bound, always available)",
        "small hybrid prototypes with real DeltaNet/attention layers",
        "cache and recurrent-state behaviour checks",
    ]
    return result
