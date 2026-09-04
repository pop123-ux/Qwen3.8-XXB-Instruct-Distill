from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_preflight_module():
    path = ROOT / "scripts/lab_preflight.py"
    spec = importlib.util.spec_from_file_location("lab_preflight", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lab_plan_and_protocol_are_static_consistent() -> None:
    module = _load_preflight_module()
    report = module.inspect()
    assert report["ok"], report["errors"]
    assert report["protocol_id"] == "RQ1_OBJECTIVES_V2"
    assert report["plan_id"] == "RQ1_OBJECTIVE_LAB_V1"


def test_only_cpu_ready_arms_are_gpu_eligible() -> None:
    plan = json.loads((ROOT / "research/plans/RQ1_OBJECTIVE_LAB_V1.json").read_text())
    ready = {"existing_cpu_tested", "cpu_tested", "ready"}
    for arm in plan["arms"]:
        if arm["implementation_status"] in ready:
            assert arm["gpu_status"] != "blocked"
        else:
            assert arm["gpu_status"] == "blocked"


def test_fdd_cannot_be_mislabeled_as_residual_delta() -> None:
    plan = json.loads((ROOT / "research/plans/RQ1_OBJECTIVE_LAB_V1.json").read_text())
    arms = {a["arm"]: a for a in plan["arms"]}
    assert arms["B"]["space"] == "lm_head_prediction"
    assert {"output_kd", "trajectory_kl", "derivative_cosine"} <= set(arms["B"]["components"])
    assert "fdd" not in arms["D"]["id"].lower()


def test_matched_recipe_is_byte_for_byte_equal_between_plan_and_protocol() -> None:
    plan = json.loads((ROOT / "research/plans/RQ1_OBJECTIVE_LAB_V1.json").read_text())
    protocol = json.loads((ROOT / "research/protocols/RQ1_OBJECTIVES_V2.json").read_text())
    assert plan["matched_recipe"] == protocol["training"]


def test_new_protocol_uses_canonical_pytorch_allocator_variable() -> None:
    protocol = json.loads((ROOT / "research/protocols/RQ1_OBJECTIVES_V2.json").read_text())
    env = protocol["runtime_environment"]
    assert env["allocator_env_canonical"] == "PYTORCH_ALLOC_CONF"
    assert env["allocator_env_legacy_alias"] == "PYTORCH_CUDA_ALLOC_CONF"


def test_composite_arm_refuses_without_preregistered_weights() -> None:
    protocol = json.loads((ROOT / "research/protocols/RQ1_OBJECTIVES_V2.json").read_text())
    arm = protocol["arm_registry"]["F"]
    assert arm["composite_weights"] is None
    assert arm["status"].startswith("blocked")
