#!/usr/bin/env python3
"""Enforce a versioned research protocol before a controlled GPU run.

Two questions are deliberately separated:

1. Is the run scientifically comparable?  Teacher/student/data/training recipe, critical
   software, GPU family and allocator are quality locks.
2. Is its throughput directly comparable?  Host driver and immutable container digest are
   additional systems locks.

A driver mismatch must never silently invalidate a quality experiment, but it also must
never be hidden inside a throughput claim. Legacy RQ1_V1 protocols retain their stricter
all-or-nothing behavior; RQ1_OBJECTIVES_V2 uses the split policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def fingerprint(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _protocol(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _driver() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        return out[0].strip() if out else None
    except Exception:
        return None


def _allocator() -> tuple[str | None, str | None]:
    canonical = os.environ.get("PYTORCH_ALLOC_CONF")
    legacy = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if canonical and legacy and canonical != legacy:
        return None, "PYTORCH_ALLOC_CONF conflicts with legacy PYTORCH_CUDA_ALLOC_CONF"
    if canonical is not None:
        return canonical, "PYTORCH_ALLOC_CONF"
    if legacy is not None:
        return legacy, "PYTORCH_CUDA_ALLOC_CONF"
    return None, None


def current_environment() -> dict[str, object]:
    import torch
    import transformers

    gpu: dict[str, object] = {"available": False}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "compute_capability": f"{p.major}.{p.minor}",
            "total_vram_gib": p.total_memory / (1024**3),
        }
    allocator, allocator_source = _allocator()
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
        "nvidia_driver": _driver(),
        "platform": platform.platform(),
        "allocator": allocator,
        "allocator_source": allocator_source,
        "container_digest": os.environ.get("RESEARCH_CONTAINER_DIGEST"),
    }


def _compare(actual: dict, expected: dict, prefix: str = "") -> list[str]:
    errors: list[str] = []
    for key, exp in expected.items():
        if key not in actual:
            errors.append(f"missing locked field {prefix}{key}")
            continue
        act = actual[key]
        if isinstance(exp, dict):
            if not isinstance(act, dict):
                errors.append(f"{prefix}{key}: expected object")
            else:
                errors.extend(_compare(act, exp, prefix=f"{prefix}{key}."))
        elif act != exp:
            errors.append(f"{prefix}{key}: protocol={exp!r}, resolved={act!r}")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--objective", help="legacy RQ1_V1 independent variable")
    parser.add_argument("--arm", help="RQ1_OBJECTIVES_V2 arm (A-F)")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def _v2_checks(protocol: dict, resolved: dict, requested: str, env: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    allowed = protocol["allowed_arms"]
    if requested not in allowed:
        errors.append(f"arm {requested!r} is not allowed; expected one of {allowed}")
        return errors, warnings

    registry = protocol["arm_registry"][requested]
    ready_status = {"existing_cpu_tested", "cpu_tested", "ready"}
    if registry.get("status") not in ready_status:
        errors.append(
            f"arm {requested} is not CPU-ready: status={registry.get('status')!r}. "
            "Implement and test it before renting GPU time."
        )

    resolved_arm = resolved.get("arm")
    if resolved_arm != requested:
        errors.append(f"resolved arm={resolved_arm!r}, requested={requested!r}")

    actual_training = dict(resolved.get("training", {}))
    errors.extend(_compare(actual_training, protocol["training"], prefix="training."))

    errors.extend(_compare(env, protocol["software_quality_lock"], prefix="software."))
    quality = protocol["execution_quality_lock"]
    gpu = env["gpu"] if isinstance(env.get("gpu"), dict) else {}
    actual_quality = {
        "gpu_model": gpu.get("name"),
        "compute_capability": gpu.get("compute_capability"),
        "gpu_count": gpu.get("count"),
        "allocator": env.get("allocator"),
    }
    expected_quality = {k: quality[k] for k in actual_quality}
    errors.extend(_compare(actual_quality, expected_quality, prefix="execution."))

    systems = protocol["systems_comparability_lock"]
    if env.get("nvidia_driver") != systems.get("nvidia_driver"):
        warnings.append(
            f"throughput only: driver protocol={systems.get('nvidia_driver')!r}, "
            f"current={env.get('nvidia_driver')!r}"
        )
    expected_digest = systems.get("container_digest")
    current_digest = env.get("container_digest")
    if not current_digest:
        warnings.append("throughput only: RESEARCH_CONTAINER_DIGEST is not set")
    elif expected_digest is None:
        warnings.append(
            "throughput only: protocol has not pinned the built image digest yet; "
            "capture this image and pin it before making throughput claims"
        )
    elif current_digest != expected_digest:
        warnings.append(
            f"throughput only: container protocol={expected_digest!r}, current={current_digest!r}"
        )
    return errors, warnings


def _v1_checks(protocol: dict, resolved: dict, requested: str, env: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    if requested not in protocol["allowed_objectives"]:
        errors.append(f"objective {requested!r} is not allowed by protocol")
        return errors, []
    actual_training = dict(resolved.get("training", resolved))
    errors.extend(_compare(actual_training, {**protocol["training"], "objective": requested}))
    errors.extend(_compare(env, protocol["software_baseline"]))
    execution = protocol["execution"]
    gpu = env["gpu"] if isinstance(env.get("gpu"), dict) else {}
    for key, actual in {
        "gpu_model": gpu.get("name"),
        "compute_capability": gpu.get("compute_capability"),
        "gpu_count": gpu.get("count"),
    }.items():
        if actual != execution[key]:
            errors.append(f"execution {key}: protocol={execution[key]!r}, current={actual!r}")
    if env.get("nvidia_driver") != execution["baseline_driver"]:
        errors.append(
            f"NVIDIA driver: protocol={execution['baseline_driver']!r}, "
            f"current={env.get('nvidia_driver')!r}"
        )
    if env.get("allocator") != execution["cuda_allocator"]:
        errors.append(
            f"CUDA allocator: protocol={execution['cuda_allocator']!r}, "
            f"current={env.get('allocator')!r}"
        )
    if execution.get("container_digest_required"):
        current_digest = env.get("container_digest")
        expected_digest = execution.get("container_digest")
        if not current_digest:
            errors.append("container digest required: set RESEARCH_CONTAINER_DIGEST")
        elif expected_digest is None:
            errors.append("protocol container digest is not established yet")
        elif current_digest != expected_digest:
            errors.append(
                f"container digest: protocol={expected_digest!r}, current={current_digest!r}"
            )
    return errors, []


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("REFUSED: no child command supplied after --")
        return 2

    protocol = _protocol(args.protocol)
    resolved = json.loads(args.resolved_config.read_text(encoding="utf-8"))
    is_v2 = "allowed_arms" in protocol
    requested = args.arm if is_v2 else args.objective
    if not requested:
        flag = "--arm" if is_v2 else "--objective"
        print(f"REFUSED: {flag} is required for {protocol.get('protocol_id')}")
        return 2

    env = current_environment()
    if env.get("allocator_source") == "PYTORCH_ALLOC_CONF conflicts with legacy PYTORCH_CUDA_ALLOC_CONF":
        errors, warnings = ["allocator environment variables conflict"], []
    elif is_v2:
        errors, warnings = _v2_checks(protocol, resolved, requested, env)
    else:
        errors, warnings = _v1_checks(protocol, resolved, requested, env)

    manifest = {
        "manifest_schema": 2,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint(protocol),
        "resolved_config_fingerprint": fingerprint(resolved),
        "independent_variable": requested,
        "command": command,
        "environment": env,
        "environment_fingerprint": fingerprint(env),
        "quality_errors": errors,
        "systems_warnings": warnings,
        "quality_comparable": not errors,
        "throughput_comparable": not errors and not warnings,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if errors:
        print("REFUSED: research protocol mismatch")
        for error in errors:
            print(f"  - {error}")
        return 2

    print(f"Research protocol PASS: {protocol['protocol_id']} / {requested}")
    for warning in warnings:
        print(f"SYSTEMS WARNING: {warning}")
    print(f"protocol fingerprint: {manifest['protocol_fingerprint']}")
    print(f"resolved configuration fingerprint: {manifest['resolved_config_fingerprint']}")
    print(f"environment fingerprint: {manifest['environment_fingerprint']}")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
