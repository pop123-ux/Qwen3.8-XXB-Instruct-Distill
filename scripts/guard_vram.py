#!/usr/bin/env python3
"""Run a command with a hard NVIDIA memory ceiling.

The guard polls ``nvidia-smi`` and terminates the child process if used GPU memory
reaches or exceeds ``--max-vram-gib``. This is an operational safety boundary for
research runs; the training summary remains the authoritative peak-allocation record.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def gpu_used_gib() -> float:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no GPU memory usage")
    return max(values) / 1024.0


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            return
        except ProcessLookupError:
            return
    process.terminate()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-vram-gib", type=float, default=45.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.max_vram_gib <= 0:
        parser.error("--max-vram-gib must be positive")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log_handle = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log.open("a", encoding="utf-8")

    try:
        process = subprocess.Popen(
            args.command,
            start_new_session=(os.name == "posix"),
            text=True,
        )
        while process.poll() is None:
            try:
                used = gpu_used_gib()
            except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
                message = f"VRAM guard probe failed: {exc}"
                print(message, file=sys.stderr)
                if log_handle:
                    log_handle.write(message + "\n")
                    log_handle.flush()
                terminate_process_tree(process)
                process.wait()
                return 2
            message = f"used_vram_gib={used:.3f} limit_gib={args.max_vram_gib:.3f}"
            print(message, flush=True)
            if log_handle:
                log_handle.write(message + "\n")
                log_handle.flush()
            if used >= args.max_vram_gib:
                message = f"VRAM CEILING BREACHED: {used:.3f} GiB >= {args.max_vram_gib:.3f} GiB"
                print(message, file=sys.stderr)
                if log_handle:
                    log_handle.write(message + "\n")
                    log_handle.flush()
                terminate_process_tree(process)
                process.wait()
                return 3
            time.sleep(args.interval)
        return process.returncode
    finally:
        if log_handle:
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
