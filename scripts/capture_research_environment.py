#!/usr/bin/env python3
"""Capture the full runtime/package fingerprint used by a research run.

The resulting JSON is intentionally additive: it describes the environment and does not
modify or rewrite any experiment output. Run it inside the controlled research container
before a new protocol family is opened.
"""

from __future__ import annotations

import argparse
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
    return {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}


def _container_identity() -> dict[str, object]:
    result: dict[str, object] = {}
    for path in (Path("/etc/machine-id"), Path("/run/.containerenv")):
        if path.exists():
            result[path.as_posix()] = path.read_text(encoding="utf-8", errors="replace")[:4096]
    result["container_image_digest"] = os.environ.get("RESEARCH_CONTAINER_DIGEST")
    return result


def capture() -> dict[str, object]:
    import torch
    import transformers

    return {
        "environment_schema": 1,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "kernel": platform.release(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_smi": _command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        "gpu": _gpu(),
        "container": _container_identity(),
        "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
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
