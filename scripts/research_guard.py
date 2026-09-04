#!/usr/bin/env python3
"""Enforce a versioned research protocol before a controlled run.

A controlled run must provide a fully resolved JSON configuration. Raw CLI defaults are
never treated as scientific protocol values. The guard verifies the resolved values,
critical runtime versions, GPU/driver identity, and records fingerprints before launching
the child command.

The guard is intentionally independent of historical experiment artifacts: it only blocks
future runs whose declared protocol does not match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def _yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _version(module: str) -> str:
    mod = __import__(module)
    return getattr(mod, "__version__", "unknown")


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


def _gpu() -> dict[str, object]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        p = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "compute_capability": f"{p.major}.{p.minor}",
            "total_vram_gib": p.total_memory / (1024**3),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def fingerprint(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def current_environment() -> dict[str, object]:
    import torch
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": _version("transformers"),
        "cuda_runtime": torch.version.cuda,
        "gpu": _gpu(),
        "nvidia_driver": _driver(),
        "platform": platform.platform(),
        "allocator": __import__("os").environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


def _normalise_resolved(resolved: dict) -> dict:
    """Accept either a flat training object or the trainer's nested config shape."""
    if "training" in resolved:
        training = dict(resolved["training"])
    else:
        training = dict(resolved)
    aliases = {
        "max_sequence_length": "sequence_length",
        "max_steps": "steps",
    }
    return {aliases.get(k, k): v for k, v in training.items()}


def _compare_nested(actual: dict, expected: dict, prefix: str = "") -> list[str]:
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
                errors.extend(_compare_nested(act, exp, f"{prefix}{key}."))
        elif act != exp:
            errors.append(f"{prefix}{key}: protocol={exp!r}, resolved={act!r}")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--resolved-config", type=Path, required=True,
                   help="fully resolved JSON emitted by the launcher; no defaults are inferred")
    p.add_argument("--objective", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("REFUSED: no child command supplied after --")
        return 2

    protocol = _yaml(args.protocol)
    if args.objective not in set(protocol.get("allowed_objectives", [])):
        print(f"REFUSED: objective {args.objective!r} is not allowed by {args.protocol}")
        return 2

    resolved = json.loads(args.resolved_config.read_text(encoding="utf-8"))
    actual_training = _normalise_resolved(resolved)
    locked_training = dict(protocol["training"])
    locked_training["objective"] = args.objective
    errors = _compare_nested(actual_training, locked_training)

    env = current_environment()
    baseline = protocol["software_baseline"]
    for key in ("python", "pytorch", "transformers", "cuda_runtime"):
        if env.get(key) != baseline.get(key):
            errors.append(f"environment {key}: baseline={baseline.get(key)!r}, current={env.get(key)!r}")

    execution = protocol["execution"]
    gpu = env.get("gpu") or {}
    if gpu.get("name") != execution.get("gpu_model"):
        errors.append(f"GPU model: baseline={execution.get('gpu_model')!r}, current={gpu.get('name')!r}")
    if gpu.get("compute_capability") != execution.get("compute_capability"):
        errors.append(f"compute capability: baseline={execution.get('compute_capability')!r}, current={gpu.get('compute_capability')!r}")
    if gpu.get("count") != execution.get("gpu_count"):
        errors.append(f"GPU count: baseline={execution.get('gpu_count')!r}, current={gpu.get('count')!r}")
    if env.get("nvidia_driver") != execution.get("baseline_driver"):
        errors.append(f"NVIDIA driver: baseline={execution.get('baseline_driver')!r}, current={env.get('nvidia_driver')!r}")
    if env.get("allocator") != execution.get("cuda_allocator"):
        errors.append(f"CUDA allocator: baseline={execution.get('cuda_allocator')!r}, current={env.get('allocator')!r}")

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
        print(f"  manifest: {args.manifest}")
        return 2

    print(f"Research protocol PASS: {protocol['protocol_id']} / objective={args.objective}")
    print(f"protocol fingerprint: {manifest['protocol_fingerprint']}")
    print(f"resolved configuration fingerprint: {manifest['resolved_config_fingerprint']}")
    print(f"environment fingerprint: {manifest['environment_fingerprint']}")
    print("child command authorized")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
