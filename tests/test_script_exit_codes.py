"""Scripts must exit non-zero when verification fails.

A verification script that fails but exits 0 is worse than one that crashes: in a
pipeline or CI job it reports success for work that never happened. `inspect_teacher.py`
had exactly that bug against an unreachable Hub.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True, text=True, timeout=300, cwd=ROOT,
    )


def test_inspect_teacher_exits_nonzero_on_missing_local_path(tmp_path):
    result = run("inspect_teacher.py", "--path", str(tmp_path / "nope"), "--config-only")
    assert result.returncode != 0
    assert "Traceback" not in result.stderr


def test_verify_teacher_loader_exits_nonzero_on_missing_model(tmp_path):
    result = run("verify_teacher_loader.py", "--model", str(tmp_path / "nope"), "--config-only")
    assert result.returncode != 0


def test_benchmark_memory_exits_nonzero_without_cuda():
    """Must refuse rather than emit zeros that look like measurements."""
    result = run("benchmark_memory.py", "--model", "unused")
    try:
        import torch

        if torch.cuda.is_available():
            return  # this guard only applies on CPU-only hosts
    except ImportError:
        pass
    assert result.returncode != 0
    assert "Refusing to emit zeros" in result.stdout


def test_help_works_for_every_script():
    for script in sorted(SCRIPTS.glob("*.py")):
        if script.name.startswith("_"):
            continue
        result = run(script.name, "--help")
        assert result.returncode == 0, f"{script.name}: {result.stderr[:300]}"
