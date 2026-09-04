#!/usr/bin/env python3
"""Capture the complete runtime/package fingerprint of a research session.

This script is observational only. It never modifies an experiment. New controlled runs
store its output beside the run manifest so code, packages, container and host GPU can be
separated cleanly when assessing reproducibility and throughput comparability.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def _command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _git() -> dict[str, object]:
    return {
        "commit": _command("git", "rev-parse", "HEAD"),
        "branch": _command("git", "branch", "--show-current"),
        "dirty": bool(_command("git", "status", "--porcelain")),
    }


def _gpu() -> list[dict[str, object]]:
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        out: list[dict[str, object]] = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_vram_gib": round(p.total_memory / (1024**3), 6),
                "compute_capability": f"{p.major}.{p.minor}",
                "multi_processor_count": p.multi_processor_count,
            })
        return out
    except Exception:
        return []


def _packages() -> dict[str, str]:
    return dict(sorted(
        (d.metadata.get("Name") or "unknown", d.version)
        for d in importlib.metadata.distributions()
    ))


def _allocator() -> dict[str, object]:
    canonical = os.environ.get("PYTORCH_ALLOC_CONF")
    legacy = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    return {
        "value": canonical or legacy,
        "source_variable": (
            "PYTORCH_ALLOC_CONF" if canonical is not None
            else "PYTORCH_CUDA_ALLOC_CONF" if legacy is not None
            else None
        ),
        "canonical": canonical,
        "legacy_alias": legacy,
        "conflict": bool(canonical and legacy and canonical != legacy),
    }


def _container_identity() -> dict[str, object]:
    result: dict[str, object] = {
        "image_digest": os.environ.get("RESEARCH_CONTAINER_DIGEST"),
    }
    for path in (Path("/etc/machine-id"), Path("/run/.containerenv")):
        if path.exists():
            raw = path.read_bytes()
            result[path.as_posix()] = {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
    lock = Path("/opt/research-pip-freeze.txt")
    if lock.exists():
        raw = lock.read_bytes()
        result["build_pip_freeze_sha256"] = hashlib.sha256(raw).hexdigest()
        result["build_pip_freeze"] = lock.read_text(encoding="utf-8", errors="replace").splitlines()
    return result


def capture() -> dict[str, object]:
    import torch
    import transformers

    return {
        "environment_schema": 2,
        "git": _git(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "kernel": platform.release(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_driver": _command(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ),
        "gpu": _gpu(),
        "allocator": _allocator(),
        "container": _container_identity(),
        "packages": _packages(),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    payload = capture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
