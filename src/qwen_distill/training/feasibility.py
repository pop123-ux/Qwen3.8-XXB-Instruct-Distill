"""Will this training run fit, and if not, what should change?

Runs before any weights are loaded, so a doomed configuration costs seconds rather than
an hour of rented GPU. The estimate reuses
:func:`qwen_distill.diagnostics.fit.estimate_training_memory` so there is one memory
model in the project, not two that can disagree.

The output deliberately names *actionable* changes in order of cost: enabling gradient
checkpointing and lowering batch size (recoverable via accumulation) cost nothing in
quality; shortening sequences and switching to QLoRA change what the experiment
measures; needing a bigger GPU is the last resort.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..architecture.params import format_params
from ..architecture.spec import HybridArchSpec
from ..diagnostics.devices import collect_devices, collect_system
from ..diagnostics.fit import TrainingFit, estimate_training_memory
from .config import ExperimentConfig


@dataclass
class FeasibilityReport:
    """Whether an experiment can run on the detected hardware."""

    experiment: str
    device_name: str
    available_gib: float
    detected_accelerator: bool
    fit: TrainingFit | None = None
    status: str = "UNKNOWN"
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "device_name": self.device_name,
            "available_gib": self.available_gib,
            "detected_accelerator": self.detected_accelerator,
            "status": self.status,
            "fit": self.fit.to_dict() if self.fit else None,
            "blockers": self.blockers,
            "warnings": self.warnings,
        }

    def render(self) -> str:
        rule = "=" * 60
        lines = [rule, "TRAINING FEASIBILITY CHECK", rule, ""]
        lines.append(f"  {'Experiment':<24}{self.experiment}")
        lines.append(f"  {'Device':<24}{self.device_name}")
        if self.fit:
            fit = self.fit
            lines += [
                f"  {'Model parameters':<24}{format_params(fit.total_parameters)}",
                f"  {'Trainable parameters':<24}{format_params(fit.trainable_parameters)} "
                f"({fit.trainable_parameters / max(fit.total_parameters, 1) * 100:.2f}%)",
                f"  {'Sequence length':<24}{fit.sequence_length}",
                f"  {'Batch size':<24}{fit.batch_size}",
                f"  {'Precision':<24}{fit.precision}",
                f"  {'Optimizer':<24}{fit.optimizer}",
                f"  {'Gradient checkpointing':<24}{'ON' if fit.gradient_checkpointing else 'OFF'}",
                "",
                f"  {'Base weights':<24}{fit.base_weights_gib:.2f} GiB",
                f"  {'Optimizer state':<24}{fit.optimizer_state_gib:.2f} GiB",
                f"  {'Activations':<24}{fit.activations_gib:.2f} GiB",
                f"  {'Runtime overhead':<24}{fit.overhead_gib:.2f} GiB",
                f"  {'Estimated VRAM':<24}{fit.total_gib:.2f} GiB",
                f"  {'Available VRAM':<24}{fit.available_gib:.2f} GiB",
                "",
            ]
        lines.append(f"  STATUS: {self.status}")
        if self.blockers:
            lines.append("  Reason:")
            lines += [f"    - {b}" for b in self.blockers]
        if self.fit and self.fit.suggestions:
            lines.append("  Suggested changes:")
            lines += [f"    - {s}" for s in self.fit.suggestions]
        if self.warnings:
            lines.append("  Warnings:")
            lines += [f"    ! {w}" for w in self.warnings]
        lines += ["", "  (analytical estimate; run scripts/hardware_info.py --calibrate",
                  "   to check the memory model against this machine)", rule]
        return "\n".join(lines)


def check_feasibility(
    config: ExperimentConfig,
    spec: HybridArchSpec | None,
    *,
    available_gib: float | None = None,
    device_name: str | None = None,
) -> FeasibilityReport:
    """Estimate whether ``config`` fits, detecting hardware unless it is supplied."""
    detected = False
    name = device_name or "unknown"

    if available_gib is None:
        system = collect_system()
        devices = [d for d in collect_devices(system) if d.vendor != "cpu"]
        if devices:
            detected = True
            primary = devices[0]
            name = device_name or primary.name
            available_gib = max(0.0, (primary.total_memory_gib or 0.0) - config.runtime.reserved_vram_gib)
        else:
            available_gib = 0.0
            name = device_name or "none (CPU only)"
    else:
        detected = True

    report = FeasibilityReport(
        experiment=config.name, device_name=name,
        available_gib=available_gib, detected_accelerator=detected,
    )

    if spec is None:
        report.status = "UNKNOWN"
        report.blockers.append(
            "the student architecture is defined by a pretrained checkpoint, so its "
            "parameter count cannot be estimated without loading it; run with weights "
            "available, or supply model.spec_path to estimate ahead of time"
        )
        return report

    if not detected or available_gib <= 0:
        # No GPU is not automatically a blocker. Level 0 of the development ladder is
        # CPU work, and a toy model trains on CPU perfectly well - slowly, but the
        # point of a prototype is to validate mechanics, not to be fast. So estimate
        # against system RAM and let size decide.
        system = collect_system()
        ram = system.total_ram_gib or 0.0
        cpu_budget = max(0.0, ram - 2.0)  # leave room for the OS
        cpu_fit = estimate_training_memory(
            spec, cpu_budget,
            strategy=config.training.strategy,
            optimizer=config.training.optimizer,
            sequence_length=config.data.max_sequence_length,
            batch_size=config.training.batch_size,
            gradient_checkpointing=config.training.gradient_checkpointing,
            label=config.name,
            precision=f"{config.training.precision} (CPU)",
            runtime_overhead_gib=0.5,   # no CUDA context on CPU
        )
        report.fit = cpu_fit
        report.available_gib = cpu_budget
        report.device_name = f"CPU ({ram:.1f} GiB system RAM)"
        if cpu_fit.feasible:
            report.status = "PLAUSIBLE (CPU — slow)"
            report.warnings.append(
                "no accelerator: this will run on CPU. Fine for a toy prototype "
                "(Level 0/1); unusably slow for anything larger."
            )
            if config.training.precision in ("bf16", "fp16"):
                report.warnings.append(
                    f"precision {config.training.precision} is poorly supported on CPU; "
                    "fp32 is usually faster and more stable there"
                )
        else:
            report.status = "NOT FEASIBLE"
            report.blockers.append(
                f"no accelerator, and the estimated {cpu_fit.total_gib:.2f} GiB exceeds "
                f"the {cpu_budget:.2f} GiB of usable system RAM"
            )
        return report

    fit = estimate_training_memory(
        spec, available_gib,
        strategy=config.training.strategy,
        optimizer=config.training.optimizer,
        sequence_length=config.data.max_sequence_length,
        batch_size=config.training.batch_size,
        gradient_checkpointing=config.training.gradient_checkpointing,
        label=config.name,
        precision=(
            f"4-bit base + {config.training.precision} compute"
            if config.training.strategy == "qlora" else config.training.precision
        ),
    )
    report.fit = fit
    report.status = {"PLAUSIBLE": "PLAUSIBLE", "TIGHT": "PLAUSIBLE (TIGHT)"}.get(
        fit.verdict, "NOT FEASIBLE"
    )
    if not fit.feasible:
        report.blockers.append(
            f"estimated peak VRAM {fit.total_gib:.2f} GiB exceeds available "
            f"{fit.available_gib:.2f} GiB"
        )
    if fit.verdict == "TIGHT":
        report.warnings.append(
            "less than 1 GiB of headroom: a longer sequence or a second process will OOM"
        )
    if config.training.strategy in ("lora", "qlora"):
        report.warnings.append(
            "LoRA/QLoRA trains an adapter over a frozen base. That is the right tool "
            "for prototyping and instruction tuning, but it does NOT produce a smaller "
            "student architecture - see docs/TRAINING_ON_LIMITED_HARDWARE.md"
        )
    if config.training.precision == "bf16":
        system = collect_system()
        devices = [d for d in collect_devices(system) if d.vendor != "cpu"]
        if devices and devices[0].supports_bf16 is False:
            report.status = "NOT FEASIBLE"
            report.blockers.append(
                f"{devices[0].name} does not support bf16 (compute capability "
                f"{devices[0].compute_capability}); use precision: fp16"
            )
    return report
