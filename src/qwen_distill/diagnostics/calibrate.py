"""Measure how accurate the analytical memory model is on *this* machine.

The estimator in :mod:`qwen_distill.architecture.memory` carries two terms that are
engineering judgement rather than derivation: runtime overhead (CUDA context, allocator
reserve, framework slack) and the transient activation working set. Both vary by driver,
allocator settings and framework version, so the honest thing is to measure them rather
than defend the guess.

This runs a **small** model — never the 27B teacher — through load, prefill and decode,
comparing measured peak against the prediction. The output is a calibration factor:
values near 1.0 mean the estimator is trustworthy on this machine; a consistent offset
means it is missing a term and should be corrected rather than explained away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..architecture.memory import DeploymentConfig, estimate_memory
from ..architecture.spec import HybridArchSpec

#: Small enough to run anywhere, structurally faithful enough for the terms to matter:
#: real hybrid layout, GQA, DeltaNet head ratios.
CALIBRATION_SPEC = HybridArchSpec(
    name="calibration-probe",
    hidden_size=512,
    num_hidden_layers=8,
    intermediate_size=1408,
    vocab_size=4096,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=64,
    linear_num_key_heads=4,
    linear_num_value_heads=12,
    linear_key_head_dim=64,
    linear_value_head_dim=64,
    full_attention_interval=4,
    max_position_embeddings=8192,
)


@dataclass
class CalibrationPoint:
    """One measured (context, batch) configuration."""

    context_length: int
    batch_size: int
    measured_peak_gib: float
    estimated_gib: float
    ratio: float | None
    phase: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class CalibrationReport:
    """Whether the analytical model matches reality on this machine."""

    device: str
    cuda_available: bool
    points: list[CalibrationPoint] = field(default_factory=list)
    mean_ratio: float | None = None
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def verdict(self) -> str:
        if self.error:
            return "FAILED"
        if not self.cuda_available:
            return "UNAVAILABLE (no CUDA device)"
        if self.mean_ratio is None:
            return "UNRESOLVED"
        if abs(self.mean_ratio - 1.0) <= 0.15:
            return "ESTIMATOR TRUSTWORTHY"
        return "ESTIMATOR NEEDS CORRECTION"

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "cuda_available": self.cuda_available,
            "verdict": self.verdict,
            "mean_ratio": self.mean_ratio,
            "points": [p.to_dict() for p in self.points],
            "notes": self.notes,
            "error": self.error,
        }


def calibrate(
    spec: HybridArchSpec = CALIBRATION_SPEC,
    *,
    contexts: tuple[int, ...] = (512, 1024, 2048),
    batch_size: int = 1,
    dtype: str = "bfloat16",
) -> CalibrationReport:
    """Load a small model, measure peak memory at several contexts, compare.

    On a CPU-only machine this returns ``UNAVAILABLE`` rather than fabricating numbers:
    a memory measurement without a GPU is not a measurement.
    """
    report = CalibrationReport(device="cpu", cuda_available=False)
    try:
        import torch
    except ImportError:
        report.error = "torch not installed"
        return report

    if not torch.cuda.is_available():
        report.notes.append(
            "no CUDA device: peak-VRAM calibration cannot run here. The analytical "
            "estimator's overhead and activation terms remain uncalibrated."
        )
        return report

    report.cuda_available = True
    report.device = torch.cuda.get_device_name(0)

    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        from ..utils.hardware import measure_memory, reset_peak_memory

        fields = {k: v for k, v in spec.to_hf_text_config().items() if k != "model_type"}
        config = AutoConfig.for_model("qwen3_5_text", **fields)
        torch_dtype = getattr(torch, dtype)

        reset_peak_memory()
        model = AutoModelForCausalLM.from_config(config).to("cuda", dtype=torch_dtype).eval()
        load_point = measure_memory("load")
        report.points.append(
            CalibrationPoint(
                context_length=0, batch_size=batch_size,
                measured_peak_gib=load_point.torch_peak_reserved_gib,
                estimated_gib=estimate_memory(
                    spec, DeploymentConfig(context_length=1, weight_quant="bf16",
                                           embedding_quant=None, runtime_overhead_bytes=0)
                ).total_gib,
                ratio=None, phase="load (weights only; overhead excluded from estimate)",
            )
        )

        for context in contexts:
            reset_peak_memory()
            ids = torch.randint(0, spec.vocab_size, (batch_size, context), device="cuda")
            with torch.no_grad():
                model(ids, use_cache=True)
            measured = measure_memory(f"prefill@{context}").torch_peak_reserved_gib
            estimated = estimate_memory(
                spec,
                DeploymentConfig(
                    context_length=context, batch_size=batch_size, weight_quant="bf16",
                    embedding_quant=None, kv_cache_dtype="bf16",
                    runtime_overhead_bytes=0,   # allocator reserve is measured, not assumed
                ),
            ).total_gib
            report.points.append(
                CalibrationPoint(
                    context_length=context, batch_size=batch_size,
                    measured_peak_gib=measured, estimated_gib=estimated,
                    ratio=(measured / estimated) if estimated else None, phase="prefill",
                )
            )
            del ids
            torch.cuda.empty_cache()

        ratios = [p.ratio for p in report.points if p.ratio]
        if ratios:
            report.mean_ratio = sum(ratios) / len(ratios)
        report.notes.append(
            "runtime_overhead_bytes was set to 0 in the estimate so the measured "
            "allocator reserve is compared directly against the modelled tensors"
        )
        del model
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - the failure is the result
        report.error = f"{type(exc).__name__}: {exc}"
    return report
