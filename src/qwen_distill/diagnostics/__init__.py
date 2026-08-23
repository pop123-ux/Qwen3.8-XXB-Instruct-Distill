"""Hardware diagnostics: what is this machine, and what can it actually do?"""

from .calibrate import (
    CalibrationReport,
    ComponentResidual,
    TrainingCalibration,
    calibrate,
    calibrate_training_run,
)
from .devices import DeviceInfo, SystemInfo, collect_devices, collect_system, cpu_device
from .fit import (
    InferenceFit,
    TrainingFit,
    analyse_inference_fit,
    estimate_training_memory,
    fit_matrix,
)
from .recommend import Recommendations, recommend
from .tiers import TIERS, Tier, classify, tier_for_devices

__all__ = [
    "SystemInfo", "DeviceInfo", "collect_system", "collect_devices", "cpu_device",
    "Tier", "TIERS", "classify", "tier_for_devices",
    "InferenceFit", "TrainingFit", "analyse_inference_fit", "estimate_training_memory",
    "fit_matrix",
    "Recommendations", "recommend",
    "CalibrationReport", "calibrate",
    "ComponentResidual", "TrainingCalibration", "calibrate_training_run",
]
