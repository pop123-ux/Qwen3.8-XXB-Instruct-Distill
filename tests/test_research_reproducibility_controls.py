from __future__ import annotations

import json
from pathlib import Path

from scripts.research_guard import _compare, fingerprint

ROOT = Path(__file__).resolve().parents[1]


def test_rq1_protocol_freezes_the_control_recipe() -> None:
    protocol = json.loads((ROOT / "research/protocols/RQ1_V1.json").read_text())
    t = protocol["training"]
    assert protocol["independent_variable"] == "objective"
    assert t["sequence_length"] == 1536
    assert t["max_tokens"] == 700000
    assert t["steps"] == 128
    assert t["batch_size"] == 1
    assert t["gradient_accumulation_steps"] == 1
    assert t["learning_rate"] == 0.0002
    assert t["optimizer"] == "adamw"
    assert t["scheduler"] == "cosine"
    assert t["warmup_steps"] == 10
    assert t["lora_dropout"] == 0.05
    assert t["seed"] == 0


def test_protocol_comparison_rejects_hidden_hyperparameter_drift() -> None:
    protocol = json.loads((ROOT / "research/protocols/RQ1_V1.json").read_text())
    locked = protocol["training"]
    resolved = dict(locked)
    resolved["learning_rate"] = 1e-4
    resolved["scheduler"] = "linear"
    errors = _compare(resolved, locked)
    assert any("learning_rate" in e for e in errors)
    assert any("scheduler" in e for e in errors)


def test_protocol_fingerprint_is_order_independent() -> None:
    left = {"b": 2, "a": {"z": 1, "y": 2}}
    right = {"a": {"y": 2, "z": 1}, "b": 2}
    assert fingerprint(left) == fingerprint(right)
