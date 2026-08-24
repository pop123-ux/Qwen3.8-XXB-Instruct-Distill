"""Import helpers from ``scripts/`` so tests can exercise the real CLI functions.

The scripts directory is not an importable package, but the code in it is user-facing
and deserves the same test coverage as the library.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
# scripts/ import `_bootstrap` to put src/ on the path; make that importable here too.
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # @dataclass resolves annotations via sys.modules
    spec.loader.exec_module(module)
    return module


report_status = load("train_student").report_status
