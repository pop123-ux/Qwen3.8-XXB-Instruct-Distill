"""Hardware and memory reporting, for reproducible deployment measurements.

``docs/DEPLOYMENT_PLAN.md`` requires that every published VRAM number carry the
context needed to reproduce it. This module collects that context, and measures peak
memory two ways because they differ and the difference matters: PyTorch's allocator
statistics report what the *tensors* need, while ``nvidia-smi`` reports what the
*process* holds, including the allocator's reserve and the CUDA context. A deployment
claim must use the second, larger number.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

GIB = 1024 ** 3


@dataclass
class HardwareReport:
    """Everything needed to reproduce a memory or throughput measurement."""

    platform: str
    python_version: str
    cuda_available: bool
    gpu_name: str | None = None
    gpu_count: int = 0
    total_vram_gib: float | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    versions: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return None


def collect_hardware() -> HardwareReport:
    """Describe the machine this measurement is running on."""
    from ..teacher.loader import collect_versions

    report = HardwareReport(
        platform=platform.platform(),
        python_version=platform.python_version(),
        cuda_available=False,
        versions=collect_versions(),
    )
    try:
        import torch

        report.cuda_available = torch.cuda.is_available()
        if report.cuda_available:
            report.gpu_count = torch.cuda.device_count()
            report.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            report.total_vram_gib = props.total_memory / GIB
            report.cuda_version = torch.version.cuda
        else:
            report.notes.append("CUDA not available; VRAM measurements cannot be taken here")
    except ImportError:
        report.notes.append("torch not installed; no GPU information available")

    report.driver_version = _nvidia_smi("driver_version")
    return report


@dataclass
class MemoryMeasurement:
    """Measured memory for one phase of inference."""

    phase: str
    torch_allocated_gib: float
    torch_reserved_gib: float
    torch_peak_allocated_gib: float
    torch_peak_reserved_gib: float
    nvidia_smi_used_gib: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reset_peak_memory() -> None:
    """Reset PyTorch's peak-memory counters, if CUDA is present."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except ImportError:
        pass


def measure_memory(phase: str) -> MemoryMeasurement:
    """Snapshot current and peak memory.

    On a machine with no CUDA this returns zeros rather than failing, so callers can
    run the same code path everywhere; the accompanying :class:`HardwareReport` records
    that no GPU was present, which is what stops a zero from being mistaken for a
    measurement.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return MemoryMeasurement(phase, 0.0, 0.0, 0.0, 0.0)
        torch.cuda.synchronize()
        used = _nvidia_smi("memory.used")
        used_gib = None
        if used:
            digits = "".join(c for c in used if c.isdigit())
            if digits:
                used_gib = int(digits) * (1024 ** 2) / GIB  # nvidia-smi reports MiB
        return MemoryMeasurement(
            phase=phase,
            torch_allocated_gib=torch.cuda.memory_allocated() / GIB,
            torch_reserved_gib=torch.cuda.memory_reserved() / GIB,
            torch_peak_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
            torch_peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB,
            nvidia_smi_used_gib=used_gib,
        )
    except ImportError:
        return MemoryMeasurement(phase, 0.0, 0.0, 0.0, 0.0)
