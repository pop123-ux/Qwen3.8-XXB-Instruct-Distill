"""Shared utilities: hardware reporting, Hub diagnosis, offline enforcement."""

from .hardware import HardwareReport, MemoryMeasurement, collect_hardware, measure_memory
from .hub import HubAccessError, HubDiagnosis, diagnose_hub_error
from .offline import looks_local, offline_for, offline_mode

__all__ = [
    "HardwareReport",
    "MemoryMeasurement",
    "collect_hardware",
    "measure_memory",
    "HubDiagnosis",
    "HubAccessError",
    "diagnose_hub_error",
    "offline_mode",
    "offline_for",
    "looks_local",
]
