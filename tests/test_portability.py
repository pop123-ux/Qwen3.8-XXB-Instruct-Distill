"""Guards against platform-portability bugs.

The motivating failure: on Windows, ``Path.read_text()`` and ``Path.write_text()``
default to the locale encoding (typically cp1252), not UTF-8. Qwen chat templates and
tokenizer files contain non-ASCII, so reading one on Windows without an explicit
encoding raises ``UnicodeDecodeError`` — and it fails precisely during the teacher
verification this repository exists to perform. Every text I/O call must therefore
name its encoding.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("src", "scripts", "tests")


def _python_files() -> list[Path]:
    """Every Python file to scan.

    This module is excluded: it contains the search patterns as string literals, and a
    naive scanner would match its own regexes.
    """
    files: list[Path] = []
    for directory in SOURCE_DIRS:
        files.extend(
            p for p in (ROOT / directory).rglob("*.py") if p.name != "test_portability.py"
        )
    return files


def _call_sites(source: str, pattern: str) -> list[tuple[int, str]]:
    """Yield (line_number, argument_text) for each call matching ``pattern``.

    Matches parentheses by balance rather than by line, so multi-line calls are
    handled correctly.
    """
    sites: list[tuple[int, str]] = []
    index = 0
    while True:
        match = re.search(pattern, source[index:])
        if not match:
            return sites
        open_paren = index + match.end() - 1
        depth, cursor = 0, open_paren
        while cursor < len(source):
            if source[cursor] == "(":
                depth += 1
            elif source[cursor] == ")":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        sites.append((source[:open_paren].count("\n") + 1, source[open_paren + 1 : cursor]))
        index = cursor + 1


def test_text_io_always_declares_utf8():
    """read_text/write_text must pass an explicit encoding."""
    offenders: list[str] = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        for line, args in _call_sites(source, r"\.(read_text|write_text)\("):
            if "encoding=" not in args:
                offenders.append(f"{path.relative_to(ROOT)}:{line}")
    assert not offenders, (
        "text I/O without an explicit encoding (breaks on Windows/cp1252):\n  "
        + "\n  ".join(offenders)
    )


def test_text_mode_open_declares_utf8():
    """open() in a text mode must pass an explicit encoding."""
    offenders: list[str] = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        for line, args in _call_sites(source, r"\.open\(|(?<![\w.])open\("):
            modes = re.findall(r"['\"]([rwax+bt]*)['\"]", args)
            if modes and "b" in modes[0]:
                continue  # binary mode takes no encoding
            if "encoding=" not in args:
                offenders.append(f"{path.relative_to(ROOT)}:{line}")
    assert not offenders, (
        "text-mode open() without an explicit encoding:\n  " + "\n  ".join(offenders)
    )


def test_no_hardcoded_platform_paths():
    """No absolute POSIX or Windows paths baked into source."""
    pattern = re.compile(r"""["'](/home/|/Users/|/tmp/|[A-Z]:\\\\)""")
    offenders: list[str] = []
    for path in _python_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:80]}")
    assert not offenders, "hard-coded absolute paths:\n  " + "\n  ".join(offenders)
