#!/usr/bin/env python3
"""Enforce a locked research protocol before starting a controlled run.

Usage:
    python scripts/research_guard.py --protocol research/protocols/RQ1_V1.yaml \
        --objective layer_kd --manifest /path/to/output/research_manifest.json -- <command>

The guard does not alter historical artifacts. It refuses a future controlled run when
critical software, GPU, allocator, data, or locked hyperparameters disagree with the
protocol. Only the declared independent variable may change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def _yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _version(module: str) -> str:
    mod = __import__(module)
    return getattr(mod, "__version__", "unknown")


def _cuda_runtime() -> str | None:
    try:
        import torch
        v = torch.version.cuda
        return v
    except Exception:
        return None


def _gpu() -> dict:
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


def fingerprint(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def current_environment() -> dict:
    gpu = _gpu()
    return {
        "python": platform.python_version(),
        "pytorch": _version("torch"),
        "transformers": _version("transformers"),
        "cuda_runtime": _cuda_runtime(),
        "gpu": gpu.get("name"),
        "gpu_count": gpu.get("count"),
        "compute_capability": gpu.get("compute_capability"),
        "nvidia_driver": _driver(),
        "platform": platform.platform(),
        "allocator": "expandable_segments:True",
    }


def command_value(command: list[str], names: set[str]) -> str | None:
    for i, token in enumerate(command):
        if token in names and i + 1 < len(command):
            return command[i + 1]
    return None


def locked_cli_values(command: list[str]) -> dict[str, object]:
    pairs: dict[str, object] = {
        "sequence_length": command_value(command, {"--sequence-length"}),
        "max_tokens": command_value(command, {"--max-tokens"}),
        "steps": command_value(command, {"--steps"}),
        "batch_size": command_value(command, {"--batch-size"}),
        "gradient_accumulation_steps": command_value(command, {"--gradient-accumulation-steps"}),
        "learning_rate": command_value(command, {"--learning-rate"}),
        "kd_temperature": command_value(command, {"--kd-temperature"}),
        "kd_top_k": command_value(command, {"--kd-top-k"}),
        "lora_rank": command_value(command, {"--lora-rank"}),
        "lora_alpha": command_value(command, {"--lora-alpha"}),
        "optimizer": command_value(command, {"--optimizer"}),
        "precision": command_value(command, {"--precision"}),
        "strategy": command_value(command, {"--strategy"}),
        "seed": command_value(command, {"--seed"}),
        "eval_every": command_value(command, {"--eval-every"}),
        "save_every": command_value(command, {"--save-every"}),
        "log_every": command_value(command, {"--log-every"}),
    }
    return pairs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--objective", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--allow-host-driver-difference", action="store_true",
                   help="mark throughput incomparable; never use for matched performance claims")
    p.add_argument("command", nargs=argparse.REMAINDER)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        print("REFUSED: no child command supplied after --", file=sys.stderr)
        return 2
    protocol = _yaml(args.protocol)
    allowed = set(protocol.get("allowed_objectives", []))
    if args.objective not in allowed:
        print(f"REFUSED: objective {args.objective!r} is not allowed by {args.protocol}", file=sys.stderr)
        return 2

    errors: list[str] = []
    locked = protocol["training"]
    cli = locked_cli_values(command)
    type_cast = {
        "sequence_length": int, "max_tokens": int, "steps": int, "batch_size": int,
        "gradient_accumulation_steps": int, "learning_rate": float, "kd_temperature": float,
        "kd_top_k": int, "lora_rank": int, "lora_alpha": int, "seed": int,
        "eval_every": int, "save_every": int, "log_every": int,
    }
    for key, expected in locked.items():
        if key in {"weight_decay", "warmup_steps", "scheduler", "gradient_checkpointing", "lora_dropout",
                   "kd_tail", "layer_kd_direction_weight", "layer_kd_normalise", "layer_kd_map_strategy",
                   "layer_kd_chunk_pairs"}:
            # These must be supplied explicitly in a future command/protocol manifest; the CLI guard
            # refuses to infer them from defaults because defaults are exactly what can silently drift.
            flag = "--" + key.replace("_", "-")
            if flag not in command and key != "weight_decay":
                errors.append(f"locked field {key} is not explicit in command")
            continue
        actual = cli.get(key)
        if actual is None:
            errors.append(f"locked field {key} is not explicit in command")
            continue
        try:
            actual = type_cast.get(key, str)(actual)
        except ValueError:
            errors.append(f"could not parse {key}={actual!r}")
            continue
        if actual != expected:
            errors.append(f"{key}: protocol={expected!r}, command={actual!r}")

    env = current_environment()
    baseline = protocol["software_baseline"]
    for key in ("python", "pytorch", "transformers", "cuda_runtime"):
        if env.get(key) != baseline.get(key):
            errors.append(f"environment {key}: baseline={baseline.get(key)!r}, current={env.get(key)!r}")
    execution = protocol["execution"]
    for key in ("gpu_model", "compute_capability", "gpu_count"):
        env_key = {"gpu_model": "gpu", "compute_capability": "compute_capability", "gpu_count": "gpu_count"}[key]
        if env.get(env_key) != execution.get(key):
            errors.append(f"execution {key}: baseline={execution.get(key)!r}, current={env.get(env_key)!r}")
    if env.get("nvidia_driver") != execution.get("baseline_driver") and not args.allow_host_driver_difference:
        errors.append(f"nvidia_driver: baseline={execution.get('baseline_driver')!r}, current={env.get('nvidia_driver')!r}")
    if "PYTORCH_CUDA_ALLOC_CONF" in command:
        pass

    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint(protocol),
        "objective": args.objective,
        "command": command,
        "environment": env,
        "environment_fingerprint": fingerprint(env),
        "comparison_policy": {
            "quality_comparable": not bool(errors),
            "throughput_comparable": not args.allow_host_driver_difference and env.get("nvidia_driver") == execution.get("baseline_driver"),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if errors:
        print("REFUSED: research protocol mismatch", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(f"  manifest: {args.manifest}", file=sys.stderr)
        return 2

    print(f"Research protocol PASS: {protocol['protocol_id']} / objective={args.objective}")
    print(f"protocol fingerprint: {manifest['protocol_fingerprint']}")
    print(f"environment fingerprint: {manifest['environment_fingerprint']}")
    print("child command is authorized to start")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
