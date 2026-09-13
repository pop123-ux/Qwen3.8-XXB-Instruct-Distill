#!/usr/bin/env python3
"""Apply the frozen Run015/Run016 interpretation rule (PREREG.md section 5) to archived results."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments"


def val(run: str) -> float:
    return float(json.loads((EXP / run / "summary.json").read_text())["final_validation_loss"])


def main() -> dict:
    raw1 = val("run013_A3_behavioural_delta_raw_seed1")
    norm1 = val("run013_A3_behavioural_delta_normalised_seed1")
    raw2 = val("run015_A3_behavioural_delta_raw_seed2")
    norm2 = val("run016_A3_behavioural_delta_normalised_seed2")
    pure_raw1 = val("run008_raw_delta_seed1")
    pure_norm1 = val("run011_normalized_delta_seed1")

    c1, c2 = norm1 - raw1, norm2 - raw2
    spread = max(abs(raw2 - raw1), abs(norm2 - norm1))
    if c2 < 0 and abs(c2) > spread:
        cls = "REPLICATED_DIRECTION"
    elif c2 < 0:
        cls = "SAME_DIRECTION_WITHIN_SPREAD"
    elif c2 > 0:
        cls = "NOT_REPLICATED_ORDERING_CHANGED"
    else:
        cls = "NO_PRACTICALLY_CLEAR_EFFECT"

    result = {
        "composite_seed1": {"raw": raw1, "normalised": norm1, "contrast_norm_minus_raw": c1},
        "composite_seed2": {"raw": raw2, "normalised": norm2, "contrast_norm_minus_raw": c2},
        "seed_to_seed_spread_S": spread,
        "per_arm_seed_shift": {"raw": raw2 - raw1, "normalised": norm2 - norm1},
        "abs_c2_exceeds_S": abs(c2) > spread,
        "direction_consistent_across_seeds": (c1 < 0) == (c2 < 0),
        "mean_contrast_descriptive": statistics.mean([c1, c2]),
        "pure_hidden_kd_seed1_contrast_norm_minus_raw": pure_norm1 - pure_raw1,
        "pure_hidden_kd_seeds_available": 1,
        "classification": cls,
        "significance_claimed": False,
    }
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
