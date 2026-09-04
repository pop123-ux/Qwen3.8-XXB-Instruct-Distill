#!/usr/bin/env python3
"""Enforce a versioned research protocol before a controlled run.

The guard consumes a fully resolved JSON configuration rather than inferring scientific
values from CLI defaults. It verifies critical software/GPU/runtime values and writes a
fingerprinted manifest before launching the child command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def _protocol(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
        "nvidia_driver": _driver(),
        "platform": platform.platform(),
        "allocator": __import__("os").environ.get("PYTORCH_CUDA_ALLOC_CONF"),
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
    parser.add_argument("--objective", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("REFUSED: no child command supplied after --")
        return 2

    protocol = _protocol(args.protocol)
    if args.objective not in protocol["allowed_objectives"]:
        print(f"REFUSED: objective {args.objective!r} is not allowed by {args.protocol}")
        return 2

    resolved = json.loads(args.resolved_config.read_text(encoding="utf-8"))
    actual_training = dict(resolved.get("training", resolved))
    errors = _compare(actual_training, {**protocol["training"], "objective": args.objective})

    env = current_environment()
    errors.extend(_compare(env, protocol["software_baseline"]))
    execution = protocol["execution"]
    gpu = env.get("gpu") or {}
    for key, env_key in (("gpu_model", "name"), ("compute_capability", "compute_capability"), ("gpu_count", "count")):
        if gpu.get(env_key) != execution[key]:
            errors.append(f"execution {key}: protocol={execution[key]!r}, current={gpu.get(env_key)!r}")
    if env.get("nvidia_driver") != execution["baseline_driver"]:
        errors.append(f"NVIDIA driver: protocol={execution['baseline_driver']!r}, current={env.get('nvidia_driver')!r}")
    if env.get("allocator") != execution["cuda_allocator"]:
        errors.append(f"CUDA allocator: protocol={execution['cuda_allocator']!r}, current={env.get('allocator')!r}")

    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint(protocol),
        "resolved_config_fingerprint": fingerprint(resolved),
        "objective": args.objective,
        "command": command,
        "environment": env,
        "environment_fingerprint": fingerprint(env),
        "quality_comparable": not errors,
        "throughput_comparable": not errors,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if errors:
        print("REFUSED: research protocol mismatch")
        for error in errors:
            print(f"  - {error}")
        return 2

    print(f"Research protocol PASS: {protocol['protocol_id']} / objective={args.objective}")
    print(f"protocol fingerprint: {manifest['protocol_fingerprint']}")
    print(f"resolved configuration fingerprint: {manifest['resolved_config_fingerprint']}")
    print(f"environment fingerprint: {manifest['environment_fingerprint']}")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
