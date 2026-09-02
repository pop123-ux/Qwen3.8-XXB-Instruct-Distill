#!/usr/bin/env python3
"""Record the prefix-consistency probe's verdict in the ledger."""

from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.research.ledger import Ledger


def main() -> int:
    probe = json.loads(
        Path("runs/prefix_consistency_probe.json").read_text(encoding="utf-8"))
    Ledger(Path("experiments/ledger.jsonl")).measured(
        "teacher_verification",
        "Smoke-test check 9's prefix assertion fails on 4-bit rounding, not misalignment",
        {
            "revision": probe["revision"],
            "quantization": "4bit",
            "argmax_agreement": probe["argmax_agreement"],
            "argmax_disagreements": probe["argmax_disagreements"],
            "max_abs_logit_diff": probe["max_abs_logit_diff"],
            "logit_scale_max_abs": probe["logit_scale_max_abs"],
            "relative_max_diff": probe["relative_max_diff"],
            "max_total_variation": probe["max_total_variation"],
            "smoke_test_atol": 1e-2,
            "passes_at_atol_1e-2": probe["passes_at_atol_1e-2"],
            "passes_at_atol_5e-1": probe["passes_at_atol_5e-1"],
            "finding": (
                "positions ARE aligned: the argmax is identical at every shared position "
                "and the two distributions differ by a total variation of 0.022. The "
                "difference is 4-bit weight quantisation plus chunked linear-attention "
                "kernels choosing a different block decomposition for a 15-token and a "
                "7-token input."
            ),
            "recommendation": (
                "check 9's prefix tolerance should scale with quantisation rather than sit "
                "at a fixed atol=1e-2, which no 4-bit run can meet. NOT changed here: "
                "loosening a verification gate is a decision for whoever owns the gate, "
                "and the evidence for it belongs in the record first."
            ),
            "consequence_for_kd": (
                "none — the self-divergence check (teacher against its own signal) passed, "
                "which is the property KD actually depends on"
            ),
        },
        tags=["teacher", "smoke_test", "tolerance"],
    )
    print("recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
