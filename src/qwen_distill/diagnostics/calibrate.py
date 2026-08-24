"""Measure how accurate the analytical memory model is on *this* machine.

The estimator in :mod:`qwen_distill.architecture.memory` carries two terms that are
engineering judgement rather than derivation: runtime overhead (CUDA context, allocator
reserve, framework slack) and the transient activation working set. Both vary by driver,
allocator settings and framework version, so the honest thing is to measure them rather
than defend the guess.

This runs a **small** model — never the 27B teacher — through load and prefill,
comparing measurement against prediction.

An earlier version of this module reported a single "calibration factor" of about 2.85
and that number was not usable. It divided **peak reserved** — which includes the CUDA
context and every block the caching allocator is holding — by **modelled tensors with
the overhead term deliberately zeroed**. Those are different quantities, and on a small
probe model the fixed CUDA context alone is comparable to the entire tensor footprint,
so the "factor" was mostly measuring the context. Multiplying future estimates by it
would have baked that confusion in permanently.

What replaces it: the CUDA context is measured directly, *before any tensor exists*, and
subtracted. Then two ratios are reported separately —

* ``ratio_allocated`` — live tensors against modelled tensors. This is the estimator's
  own arithmetic, and the only one from which a correction should be derived.
* ``ratio_reserved`` — process footprint against modelled tensors plus overhead. This is
  what a deployment claim has to satisfy.

Neither is a multiplier. A ratio away from 1.0 identifies *which term* is wrong, and the
term gets fixed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..architecture.memory import GIB, DeploymentConfig, estimate_memory
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
    """One measured (context, batch) configuration.

    Both measured quantities are kept, never collapsed into one. ``allocated`` is the
    live tensors, which is what the estimator models; ``reserved`` is what the caching
    allocator holds from the driver, which is what the process occupies.
    """

    context_length: int
    batch_size: int
    measured_allocated_gib: float
    measured_reserved_gib: float
    estimated_tensors_gib: float
    ratio_allocated: float | None
    ratio_reserved: float | None
    phase: str

    @property
    def allocator_overhead_gib(self) -> float:
        """What the allocator holds beyond the live tensors."""
        return max(0.0, self.measured_reserved_gib - self.measured_allocated_gib)

    def to_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["allocator_overhead_gib"] = self.allocator_overhead_gib
        return data


@dataclass
class CalibrationReport:
    """Whether the analytical model matches reality on this machine."""

    device: str
    cuda_available: bool
    points: list[CalibrationPoint] = field(default_factory=list)
    #: Mean of the per-point *tensor* ratios. Named for what it compares, so it cannot
    #: be mistaken for a factor that also covers context and allocator reserve.
    mean_ratio_allocated: float | None = None
    mean_ratio_reserved: float | None = None
    #: CUDA context, measured before any tensor is allocated. This is the term the old
    #: single-factor calibration was mostly measuring.
    cuda_context_gib: float | None = None
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def verdict(self) -> str:
        """Judged on the tensor ratio, because that is what the estimator predicts.

        Reserve depends on allocator settings and fragmentation, which the estimator
        does not model and should not be blamed for.
        """
        if self.error:
            return "FAILED"
        if not self.cuda_available:
            return "UNAVAILABLE (no CUDA device)"
        if self.mean_ratio_allocated is None:
            return "UNRESOLVED"
        if abs(self.mean_ratio_allocated - 1.0) <= 0.15:
            return "ESTIMATOR TRUSTWORTHY"
        return "ESTIMATOR NEEDS CORRECTION"

    @property
    def correction(self) -> str:
        """What to actually change — never "multiply everything by X"."""
        if self.mean_ratio_allocated is None:
            return "no measurement: nothing to correct"
        if abs(self.mean_ratio_allocated - 1.0) <= 0.15:
            return "tensor model is within 15% of measurement; no correction warranted"
        direction = "over" if self.mean_ratio_allocated < 1.0 else "under"
        return (
            f"the tensor model {direction}estimates by "
            f"{abs(self.mean_ratio_allocated - 1.0) * 100:.0f}%. Identify which term is "
            "wrong (weights are exact, so it is the activation or cache term) and fix "
            "that term. Do NOT apply this ratio as a global multiplier."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "cuda_available": self.cuda_available,
            "verdict": self.verdict,
            "correction": self.correction,
            "mean_ratio_allocated": self.mean_ratio_allocated,
            "mean_ratio_reserved": self.mean_ratio_reserved,
            "cuda_context_gib": self.cuda_context_gib,
            "points": [p.to_dict() for p in self.points],
            "notes": self.notes,
            "error": self.error,
        }


def _measure_cuda_context_gib() -> float | None:
    """Driver-side memory in use that is not a torch tensor.

    ``torch.cuda.mem_get_info`` reports what the driver sees; subtracting torch's own
    reserve leaves the CUDA context and any other process on the card. Returns ``None``
    rather than 0.0 when it cannot be determined, so a missing measurement is never
    mistaken for a context of zero.
    """
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        used_by_driver = (total - free) / GIB
        torch_reserved = torch.cuda.memory_reserved() / GIB
        return max(0.0, used_by_driver - torch_reserved)
    except Exception:  # noqa: BLE001 - an unavailable measurement is not a failure
        return None


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

        # The CUDA context is allocated by the driver on first use and is invisible to
        # `memory_allocated`, yet it is included in what nvidia-smi and the OS see.
        # Force it to exist, then measure the floor before any model tensor is created.
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        report.cuda_context_gib = _measure_cuda_context_gib()

        reset_peak_memory()
        model = AutoModelForCausalLM.from_config(config).to("cuda", dtype=torch_dtype).eval()
        load = measure_memory("load")
        weights_only = estimate_memory(
            spec, DeploymentConfig(context_length=1, weight_quant="bf16",
                                   embedding_quant=None, runtime_overhead_bytes=0)
        ).total_gib
        report.points.append(
            CalibrationPoint(
                context_length=0, batch_size=batch_size,
                measured_allocated_gib=load.torch_peak_allocated_gib,
                measured_reserved_gib=load.torch_peak_reserved_gib,
                estimated_tensors_gib=weights_only,
                ratio_allocated=(load.torch_peak_allocated_gib / weights_only) if weights_only else None,
                ratio_reserved=(load.torch_peak_reserved_gib / weights_only) if weights_only else None,
                phase="load (weights)",
            )
        )

        for context in contexts:
            reset_peak_memory()
            ids = torch.randint(0, spec.vocab_size, (batch_size, context), device="cuda")
            with torch.no_grad():
                model(ids, use_cache=True)
            measured = measure_memory(f"prefill@{context}")
            estimated = estimate_memory(
                spec,
                DeploymentConfig(
                    context_length=context, batch_size=batch_size, weight_quant="bf16",
                    embedding_quant=None, kv_cache_dtype="bf16",
                    # Zero here on purpose: this term is the thing being measured, and
                    # including it would compare the estimate against itself.
                    runtime_overhead_bytes=0,
                ),
            ).total_gib
            report.points.append(
                CalibrationPoint(
                    context_length=context, batch_size=batch_size,
                    measured_allocated_gib=measured.torch_peak_allocated_gib,
                    measured_reserved_gib=measured.torch_peak_reserved_gib,
                    estimated_tensors_gib=estimated,
                    ratio_allocated=(measured.torch_peak_allocated_gib / estimated) if estimated else None,
                    ratio_reserved=(measured.torch_peak_reserved_gib / estimated) if estimated else None,
                    phase="prefill",
                )
            )
            del ids
            torch.cuda.empty_cache()

        prefill = [p for p in report.points if p.phase == "prefill"]
        allocated = [p.ratio_allocated for p in prefill if p.ratio_allocated]
        reserved = [p.ratio_reserved for p in prefill if p.ratio_reserved]
        if allocated:
            report.mean_ratio_allocated = sum(allocated) / len(allocated)
        if reserved:
            report.mean_ratio_reserved = sum(reserved) / len(reserved)
        report.notes.append(
            "the estimate's runtime_overhead term is zeroed here, so ratio_allocated "
            "compares modelled tensors against live tensors. ratio_reserved is the same "
            "estimate against allocator reserve and is therefore always the larger of "
            "the two; it is reported for deployment sizing, not as an estimator error."
        )
        if report.cuda_context_gib:
            report.notes.append(
                f"CUDA context measured at {report.cuda_context_gib:.2f} GiB before any "
                "model tensor existed. On a small probe model this alone can exceed the "
                "tensor footprint, which is why a single measured/estimated factor was "
                "meaningless."
            )
        del model
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - the failure is the result
        report.error = f"{type(exc).__name__}: {exc}"
    return report


@dataclass
class ComponentResidual:
    """One estimator term against the measurement it is supposed to predict."""

    component: str
    estimated_gib: float
    measured_gib: float

    @property
    def ratio(self) -> float | None:
        return self.measured_gib / self.estimated_gib if self.estimated_gib > 0 else None

    @property
    def error_gib(self) -> float:
        return self.measured_gib - self.estimated_gib

    @property
    def verdict(self) -> str:
        ratio = self.ratio
        if ratio is None:
            return "NOT MODELLED"
        if abs(ratio - 1.0) <= 0.15:
            return "OK"
        return "UNDERESTIMATED" if ratio > 1.0 else "OVERESTIMATED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "estimated_gib": self.estimated_gib,
            "measured_gib": self.measured_gib,
            "ratio": self.ratio,
            "error_gib": self.error_gib,
            "verdict": self.verdict,
        }


@dataclass
class TrainingCalibration:
    """Which *term* of the training estimate is wrong, and by how much."""

    experiment: str
    device: str | None = None
    residuals: list[ComponentResidual] = field(default_factory=list)
    error: str | None = None

    @property
    def worst(self) -> ComponentResidual | None:
        """The term contributing the most absolute error — where a fix pays most."""
        scored = [r for r in self.residuals if r.ratio is not None]
        return max(scored, key=lambda r: abs(r.error_gib), default=None)

    def corrections(self) -> list[str]:
        """Named, per-term changes. Never a global multiplier."""
        if self.error:
            return [self.error]
        lines: list[str] = []
        for residual in self.residuals:
            if residual.verdict in ("OK", "NOT MODELLED"):
                continue
            lines.append(
                f"{residual.component}: modelled {residual.estimated_gib:.3f} GiB, "
                f"measured {residual.measured_gib:.3f} GiB "
                f"({residual.ratio:.2f}x) — correct this term, not the total"
            )
        if not lines:
            lines.append("every modelled term is within 15% of measurement")
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "device": self.device,
            "residuals": [r.to_dict() for r in self.residuals],
            "worst_component": self.worst.component if self.worst else None,
            "corrections": self.corrections(),
            "error": self.error,
        }

    def render(self) -> str:
        lines = [f"training calibration: {self.experiment}"]
        if self.device:
            lines.append(f"  device: {self.device}")
        if self.error:
            return "\n".join(lines + [f"  ERROR: {self.error}"])
        lines.append(f"\n  {'component':<28}{'modelled':>10}{'measured':>10}{'ratio':>8}  verdict")
        for residual in self.residuals:
            ratio = f"{residual.ratio:.3f}" if residual.ratio is not None else "-"
            lines.append(
                f"  {residual.component:<28}{residual.estimated_gib:>10.3f}"
                f"{residual.measured_gib:>10.3f}{ratio:>8}  {residual.verdict}"
            )
        lines.append("\n  corrections:")
        lines += [f"    - {c}" for c in self.corrections()]
        return "\n".join(lines)


def calibrate_training_run(summary_path: str | Path) -> TrainingCalibration:
    """Compare a finished run's measured components against what was estimated.

    Reads the ``summary.json`` a training run writes, which carries both the analytical
    estimate and the memory probe's stage-by-stage measurement. Every term is compared
    **separately**: a total that happens to be right because two errors cancelled is
    still two errors, and a single measured/estimated number cannot tell you which.

    On a run with no CUDA measurement this returns an explanatory error rather than a
    table of zeros.
    """
    path = Path(summary_path)
    calibration = TrainingCalibration(experiment=str(path))
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        calibration.error = f"{type(exc).__name__}: {exc}"
        return calibration

    calibration.experiment = summary.get("experiment", str(path))
    memory = summary.get("memory") or {}
    estimate = summary.get("analytical_estimate") or {}
    calibration.device = memory.get("device_name")

    if not memory.get("cuda_available"):
        calibration.error = (
            "the run has no CUDA measurement, so there is nothing to calibrate against. "
            "Re-run on a GPU; a CPU run cannot validate a VRAM model."
        )
        return calibration
    if not estimate:
        calibration.error = "the run recorded no analytical estimate to compare against"
        return calibration

    measured = memory.get("components") or {}
    # Keys the estimator predicts, mapped onto the probe's measured components.
    pairs = (
        ("weights", estimate.get("base_weights_gib"), measured.get("weights_gib")),
        ("optimizer_state", estimate.get("optimizer_state_gib"), measured.get("optimizer_state_gib")),
        (
            "activations+logits",
            (estimate.get("activations_gib") or 0.0) + (estimate.get("logits_gib") or 0.0),
            measured.get("activations_gib"),
        ),
        (
            "peak_allocated",
            estimate.get("predicted_allocated_gib"),
            memory.get("peak_allocated_gib"),
        ),
        (
            "peak_reserved",
            estimate.get("predicted_reserved_gib") or estimate.get("total_gib"),
            memory.get("peak_reserved_gib"),
        ),
    )
    for name, estimated, measured_value in pairs:
        if estimated is None or measured_value is None:
            continue
        calibration.residuals.append(
            ComponentResidual(component=name, estimated_gib=float(estimated),
                              measured_gib=float(measured_value))
        )
    # Gradients are deliberately absent: the probe measures backward as a *net* change
    # (gradients allocated minus activations freed), which is not the gross gradient
    # size the estimator models. Comparing them would be the same category error that
    # produced the 2.85 factor.
    if not calibration.residuals:
        calibration.error = "no overlapping components between the estimate and the measurement"
    return calibration
