"""The pre-GPU acceptance gate.

The gate's value is that it *runs* the things it claims, so these tests check it cannot
pass vacuously: a broken architecture must make it fail, and its two most fragile checks
(substring matching on CLI flags, and the word "SOTA" appearing in a disclaimer) must not
produce false positives.
"""
from __future__ import annotations

import json

import pytest
from scripts_shim import load

pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def gate():
    return load("acceptance_gate")


@pytest.fixture(scope="module")
def results(gate, tmp_path_factory):
    path = tmp_path_factory.mktemp("gate") / "acceptance.json"
    code = gate.main(["--json", str(path)])
    return code, json.loads(path.read_text(encoding="utf-8"))


def test_the_gate_passes_and_reports_ready(results):
    code, data = results
    assert code == 0
    assert data["ready"] is True
    failed = [r for r in data["results"] if r["status"] == "FAIL"]
    assert failed == [], failed


def test_it_checks_everything_the_protocol_requires(results):
    _, data = results
    items = {r["item"] for r in data["results"]}
    for required in ("teacher identity", "pretrained teacher loader", "exact revision gate",
                     "no silent mock fallback", "exact student parameters",
                     "active parameters/token", "48-layer topology", "8 x 768 top-2 MoE",
                     "1 shared expert", "24 Q / 2 KV heads",
                     "262144 architectural context", "teacher -> student materialisation",
                     "64 -> 48 mapping", "4 -> 2 KV conversion", "FFN -> MoE initialisation",
                     "context specialisation", "16 GB accounting", "experiment provenance",
                     "research baselines", "plotting infrastructure", "README consistency",
                     "Further Questions section"):
        assert required in items, f"the gate does not check '{required}'"


def test_deferred_items_are_named_rather_than_claimed(results):
    """A feature that is not built must say so, not be quietly counted as passing."""
    _, data = results
    deferred = {r["item"] for r in data["results"] if r["status"] == "DEFERRED"}
    assert {"MTP", "DeltaNet state matching", "measured GPU memory",
            "benchmark results"} <= deferred


def test_the_parameter_check_is_exact_and_would_catch_a_drift(gate):
    """The gate must not pass on an approximately-right architecture."""
    assert gate.TOTAL == 13_008_505_728
    assert gate.ACTIVE == 9_611_119_488
    assert gate.TEACHER == "Qwen/Qwen3.8-27B"


def test_the_readme_check_allows_disclaiming_sota_but_not_claiming_it(gate, tmp_path,
                                                                     monkeypatch):
    """The false positive that made the gate wrong once: 'No SOTA claim is made' is the
    correct thing to write and must not be read as a claim."""
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    good = ("13,008,505,728 9,611,119,488 16 GB Qwen/Qwen3.8-27B DEMONSTRATED FUTURE WORK\n"
            "No SOTA claim is made.\n")
    (tmp_path / "README.md").write_text(good, encoding="utf-8")
    assert gate._readme_ok() is True
    (tmp_path / "README.md").write_text(good + "\nThis is SOTA on every benchmark.\n",
                                        encoding="utf-8")
    assert gate._readme_ok() is False


def test_the_readme_check_requires_the_exact_numbers(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    (tmp_path / "README.md").write_text("16 GB Qwen/Qwen3.8-27B DEMONSTRATED FUTURE WORK",
                                        encoding="utf-8")
    assert gate._readme_ok() is False


def test_a_raising_check_is_a_failing_check(gate, monkeypatch, capsys):
    """A gate that swallowed exceptions would report READY on a broken repository."""
    def boom():
        raise RuntimeError("exploded")

    monkeypatch.setattr(gate, "_checks", lambda: [("synthetic", "note", boom)])
    assert gate.main([]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_the_gate_fails_when_a_check_returns_false(gate, monkeypatch, capsys):
    monkeypatch.setattr(gate, "_checks", lambda: [("synthetic", "note", lambda: False)])
    assert gate.main([]) == 1
    assert "NOT READY" in capsys.readouterr().out
