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


def test_validate_metadata_exits_2_when_directory_absent(tmp_path):
    result = run("validate_teacher_metadata.py", "--path", str(tmp_path / "nope"))
    assert result.returncode == 2
    assert "Metadata directory not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_inspect_chat_template_exits_2_when_directory_absent(tmp_path):
    result = run("inspect_chat_template.py", "--path", str(tmp_path / "nope"))
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_inspect_chat_template_exits_1_when_no_template(tmp_path):
    """A missing template must be reported, not worked around."""
    import json

    root = tmp_path / "meta"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_text", "architectures": []}), encoding="utf-8"
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "X"}), encoding="utf-8"
    )
    result = run("inspect_chat_template.py", "--path", str(root))
    assert result.returncode == 1
    assert "No chat template found" in result.stderr
    assert "UNKNOWN" in result.stderr


def test_validate_metadata_strict_flags_missing_required(tmp_path):
    import json

    root = tmp_path / "meta"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "x", "architectures": []}), encoding="utf-8"
    )
    result = run("validate_teacher_metadata.py", "--path", str(root), "--strict")
    assert result.returncode == 1
    assert "MISSING" in result.stdout
