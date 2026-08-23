"""Accelerator detection that degrades gracefully instead of crashing.

The design rule here is that **absence of a GPU is a normal outcome, not an error**.
Most of this project's users are on a laptop, a free Colab T4, or a CPU-only CI runner,
and a diagnostics tool that raises on any of those is useless.

So every probe is defensive: a missing library, a missing driver, an unsupported query
on a particular vendor — all of them produce a recorded ``None`` and a note, never an
exception. A field we could not determine is reported as unknown rather than guessed,
because a fabricated capability is worse than an admitted gap when the user is about to
decide whether a 27B model fits.

AMD is handled by reporting what ROCm actually exposes rather than inventing
NVIDIA-shaped fields for it.
"""

from __future__ import annotations

import contextlib
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

GIB = 1024 ** 3


@dataclass
class DeviceInfo:
    """One detected accelerator. Every optional field means 'could not determine'."""

    index: int
    vendor: str                      # "nvidia" | "amd" | "apple" | "cpu"
    name: str
    backend: str                     # "cuda" | "rocm" | "mps" | "cpu"

    total_memory_gib: float | None = None
    allocated_memory_gib: float | None = None
    reserved_memory_gib: float | None = None
    free_memory_gib: float | None = None
    peak_allocated_gib: float | None = None
    peak_reserved_gib: float | None = None

    compute_capability: str | None = None
    multi_processor_count: int | None = None
    memory_bandwidth_gb_s: float | None = None
    driver_version: str | None = None

    supports_bf16: bool | None = None
    supports_fp16: bool | None = None
    supports_fp8: bool | None = None
    supports_int8: bool | None = None
    has_tensor_cores: bool | None = None

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SystemInfo:
    """The software side of the environment, which decides as much as the hardware."""

    os: str
    os_release: str
    machine: str
    python_version: str
    torch_version: str | None = None
    transformers_version: str | None = None
    cuda_runtime_version: str | None = None
    hip_version: str | None = None
    cuda_available: bool = False
    rocm_available: bool = False
    mps_available: bool = False
    cpu_count: int | None = None
    total_ram_gib: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(cmd: list[str], timeout: int = 15) -> str | None:
    """Run a command, returning stripped stdout or None on any failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip()


def _nvidia_smi_field(query: str, index: int = 0) -> str | None:
    out = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if not out:
        return None
    lines = out.splitlines()
    return lines[index].strip() if index < len(lines) else None


def _total_ram_gib() -> float | None:
    try:
        import os

        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB
    except (ValueError, OSError, AttributeError):
        pass
    # Windows fallback via ctypes.
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return status.ullTotalPhys / GIB
    except (ImportError, AttributeError, OSError):
        pass
    return None


def collect_system() -> SystemInfo:
    """Describe the OS and the installed ML stack."""
    import os

    info = SystemInfo(
        os=platform.system() or "unknown",
        os_release=platform.release() or "unknown",
        machine=platform.machine() or "unknown",
        python_version=platform.python_version(),
        cpu_count=os.cpu_count(),
        total_ram_gib=_total_ram_gib(),
    )

    try:
        import torch

        info.torch_version = torch.__version__
        info.cuda_available = bool(torch.cuda.is_available())
        info.cuda_runtime_version = getattr(torch.version, "cuda", None)
        info.hip_version = getattr(torch.version, "hip", None)
        # torch reports ROCm builds through torch.version.hip; CUDA stays None there.
        info.rocm_available = bool(info.hip_version) and info.cuda_available
        mps = getattr(torch.backends, "mps", None)
        info.mps_available = bool(mps and mps.is_available())
    except ImportError:
        info.notes.append("torch not installed: no accelerator detection possible")

    try:
        import transformers

        info.transformers_version = transformers.__version__
    except ImportError:
        info.notes.append("transformers not installed")

    return info


#: Compute capability -> (bf16, fp8, tensor cores). fp16 is available from 5.3 onward
#: on every card this project could plausibly target.
#: Ampere (8.0+) added bf16; Ada/Hopper (8.9/9.0+) added fp8.
def _nvidia_capabilities(major: int, minor: int) -> dict[str, bool]:
    return {
        "supports_fp16": (major, minor) >= (5, 3),
        "supports_bf16": major >= 8,
        "supports_fp8": (major, minor) >= (8, 9),
        # INT8 tensor-core paths from Turing (7.5); dp4a from Pascal (6.1).
        "supports_int8": (major, minor) >= (6, 1),
        "has_tensor_cores": major >= 7,
    }


def collect_devices(system: SystemInfo | None = None) -> list[DeviceInfo]:
    """Detect every accelerator torch can see. Returns [] when there is none."""
    system = system or collect_system()
    devices: list[DeviceInfo] = []

    try:
        import torch
    except ImportError:
        return devices

    if system.cuda_available:
        vendor = "amd" if system.hip_version else "nvidia"
        backend = "rocm" if system.hip_version else "cuda"
        driver = _nvidia_smi_field("driver_version") if vendor == "nvidia" else None

        for index in range(torch.cuda.device_count()):
            device = DeviceInfo(
                index=index,
                vendor=vendor,
                name=torch.cuda.get_device_name(index),
                backend=backend,
                driver_version=driver,
            )
            try:
                props = torch.cuda.get_device_properties(index)
                device.total_memory_gib = props.total_memory / GIB
                device.multi_processor_count = getattr(props, "multi_processor_count", None)
                major = getattr(props, "major", None)
                minor = getattr(props, "minor", None)
                if major is not None:
                    device.compute_capability = f"{major}.{minor}"
                    if vendor == "nvidia":
                        for key, value in _nvidia_capabilities(major, minor or 0).items():
                            setattr(device, key, value)
            except (RuntimeError, AssertionError) as exc:
                device.notes.append(f"device properties unavailable: {exc}")

            if vendor == "amd":
                # Report only what ROCm actually exposes; do not invent NVIDIA fields.
                device.notes.append(
                    "AMD/ROCm device: compute-capability-derived capability flags are "
                    "NVIDIA-specific and are left unknown. bf16/fp8 support depends on "
                    "the specific architecture and ROCm build."
                )
                try:
                    device.supports_fp16 = torch.cuda.is_bf16_supported() or True
                    device.supports_bf16 = bool(torch.cuda.is_bf16_supported())
                except (RuntimeError, AttributeError):
                    pass

            try:
                device.allocated_memory_gib = torch.cuda.memory_allocated(index) / GIB
                device.reserved_memory_gib = torch.cuda.memory_reserved(index) / GIB
                device.peak_allocated_gib = torch.cuda.max_memory_allocated(index) / GIB
                device.peak_reserved_gib = torch.cuda.max_memory_reserved(index) / GIB
                free, _total = torch.cuda.mem_get_info(index)
                device.free_memory_gib = free / GIB
            except (RuntimeError, AttributeError) as exc:
                device.notes.append(f"memory statistics partially unavailable: {exc}")

            if vendor == "nvidia":
                # nvidia-smi reports the bus width and clock we could derive bandwidth
                # from, but not bandwidth itself; only report it when it is unambiguous.
                total = _nvidia_smi_field("memory.total", index)
                if total and device.total_memory_gib is None:
                    with contextlib.suppress(ValueError):
                        device.total_memory_gib = float(total) * (1024 ** 2) / GIB
            devices.append(device)

    elif system.mps_available:
        devices.append(
            DeviceInfo(
                index=0, vendor="apple", name="Apple Silicon (MPS)", backend="mps",
                supports_fp16=True, supports_bf16=None,
                notes=["MPS: VRAM is shared with system RAM; no separate total is reported"],
            )
        )

    return devices


def cpu_device(system: SystemInfo | None = None) -> DeviceInfo:
    """A DeviceInfo describing the CPU, so CPU-only machines still get a row."""
    system = system or collect_system()
    return DeviceInfo(
        index=0,
        vendor="cpu",
        name=platform.processor() or platform.machine() or "CPU",
        backend="cpu",
        total_memory_gib=system.total_ram_gib,
        supports_fp16=False,   # possible but impractically slow for this project
        supports_bf16=False,
        notes=["no accelerator detected; system RAM shown in place of VRAM"],
    )
