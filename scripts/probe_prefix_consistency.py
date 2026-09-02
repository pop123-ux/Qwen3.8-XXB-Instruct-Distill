#!/usr/bin/env python3
"""Is the smoke test's prefix check failing on a real misalignment, or on 4-bit noise?

Check 9 of ``scripts/teacher_smoke_test.py`` asserts that logits for a prefix match the
same positions of the full sequence within ``atol=1e-2``. For a causal model that identity
holds in exact arithmetic, so a failure is worth separating into its two very different
causes:

* **a real misalignment** — position ``t`` is not the prediction for the token the student
  predicts at ``t``. Then the argmax moves, the distributions differ substantially, and
  every KD target is wrong;
* **numerical noise** — 4-bit weights and chunked linear-attention kernels that pick a
  different block decomposition for a 16-token and an 8-token input. Then the argmax is
  unchanged and the distributions differ by a rounding-scale amount.

This measures which one it is. It loads the teacher once and writes a JSON record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from qwen_distill.distillation.backends import TransformersTeacher
from qwen_distill.distillation.real_teacher import teacher_logits

REVISION = "dbdc473dea0d6a9763042881cc33d6058d1742d2"
LOCAL_PATH = "/workspace/models/qwen3.8-27b-dbdc473"
PROMPT = "The capital of France is Paris, and the capital of Germany is Berlin."


def main() -> int:
    backend = TransformersTeacher(
        model="Qwen/Qwen3.8-27B", revision=REVISION, local_path=LOCAL_PATH,
        quantization="4bit", strict_architecture=True,
    )
    loaded = backend.load()
    ids = loaded.tokenizer(PROMPT, return_tensors="pt").input_ids
    n = int(ids.shape[-1])
    half = n // 2

    full = teacher_logits(loaded, ids).float()
    prefix = teacher_logits(loaded, ids[:, :half]).float()
    shared_full, shared_prefix = full[:, :half], prefix

    diff = (shared_full - shared_prefix).abs()
    argmax_full = shared_full.argmax(-1)
    argmax_prefix = shared_prefix.argmax(-1)
    prob_full = shared_full.softmax(-1)
    prob_prefix = shared_prefix.softmax(-1)

    record = {
        "revision": REVISION,
        "quantization": "4bit",
        "n_tokens": n,
        "shared_positions": half,
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "logit_scale_max_abs": float(shared_full.abs().max()),
        "relative_max_diff": float(diff.max() / shared_full.abs().max()),
        "argmax_agreement": float((argmax_full == argmax_prefix).float().mean()),
        "argmax_disagreements": int((argmax_full != argmax_prefix).sum()),
        "max_abs_prob_diff": float((prob_full - prob_prefix).abs().max()),
        "max_total_variation": float(0.5 * (prob_full - prob_prefix).abs().sum(-1).max()),
        "smoke_test_tolerance_atol": 1e-2,
        "passes_at_atol_1e-2": bool(torch.allclose(shared_full, shared_prefix, atol=1e-2)),
        "passes_at_atol_1e-1": bool(torch.allclose(shared_full, shared_prefix, atol=1e-1)),
        "passes_at_atol_5e-1": bool(torch.allclose(shared_full, shared_prefix, atol=5e-1)),
    }
    record["verdict"] = (
        "numerical: the argmax is unchanged at every shared position, so positions are "
        "aligned and the difference is quantisation/kernel rounding"
        if record["argmax_disagreements"] == 0 else
        "MISALIGNMENT: the argmax moves at shared positions; KD targets would be wrong"
    )

    print(json.dumps(record, indent=2))
    out = Path("runs/prefix_consistency_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
