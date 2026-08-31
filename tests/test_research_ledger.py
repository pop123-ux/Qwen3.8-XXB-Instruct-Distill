"""The experiment ledger: append-only, and provenance is not optional."""
from __future__ import annotations

import json

import pytest

from qwen_distill.research.ledger import (
    ESTIMATED,
    MEASURED,
    PROVENANCE,
    REPORTED,
    Entry,
    Ledger,
    environment,
)


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger.jsonl")


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def test_provenance_is_a_closed_set():
    assert PROVENANCE == (MEASURED, REPORTED, ESTIMATED)
    with pytest.raises(ValueError, match="no fourth option"):
        Entry(kind="note", title="x", provenance="probably_right")


def test_an_estimate_must_carry_its_method():
    """Without it an estimate is indistinguishable from a measurement in the record."""
    with pytest.raises(ValueError, match="method that produced it"):
        Entry(kind="memory_accounting", title="x", provenance=ESTIMATED)
    Entry(kind="memory_accounting", title="x", provenance=ESTIMATED, method="analytic model")


def test_a_third_party_number_must_carry_its_source():
    """The rule that stops unsourced competitor benchmarks entering the record."""
    with pytest.raises(ValueError, match="Unsourced competitor numbers"):
        Entry(kind="comparison", title="rival MMLU", provenance=REPORTED)
    Entry(kind="comparison", title="rival MMLU", provenance=REPORTED, source="model card")


def test_a_measured_entry_may_not_cite_an_external_source():
    """That combination reads as a measurement and is a citation."""
    with pytest.raises(ValueError, match="must not cite an external source"):
        Entry(kind="evaluation", title="x", provenance=MEASURED, source="someone's blog")


# ---------------------------------------------------------------------------
# append-only
# ---------------------------------------------------------------------------
def test_entries_append_and_survive_reopening(ledger, tmp_path):
    ledger.measured("evaluation", "first", {"score": 1})
    ledger.measured("evaluation", "second", {"score": 2})
    assert len(list(Ledger(tmp_path / "ledger.jsonl"))) == 2


def test_one_entry_per_line(ledger):
    ledger.measured("note", "multi\nline\ttitle", {"text": "a\nb"})
    assert len(ledger.path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_there_is_no_update_or_delete(ledger):
    assert not hasattr(ledger, "update")
    assert not hasattr(ledger, "delete")


def test_retraction_supersedes_without_erasing(ledger):
    entry = ledger.measured("evaluation", "wrong number", {"score": 99})
    ledger.retract(entry.id, "measured against the wrong checkpoint")
    assert len(list(ledger)) == 2, "the original line must still be on disk"
    assert len(ledger.entries()) == 1
    assert ledger.entries()[0]["kind"] == "retraction"
    assert entry.id in ledger.path.read_text(encoding="utf-8")
    assert ledger.entries(include_superseded=True)[0]["title"] == "wrong number"


def test_retracting_an_unknown_entry_is_refused(ledger):
    with pytest.raises(KeyError, match="unknown entry"):
        ledger.retract("deadbeef", "typo")


# ---------------------------------------------------------------------------
# identity, environment, querying
# ---------------------------------------------------------------------------
def test_ids_are_content_addressed_and_distinct(ledger):
    a = ledger.measured("evaluation", "run", {"score": 1})
    b = ledger.measured("evaluation", "run", {"score": 2})
    assert a.id != b.id
    assert len(a.id) == 16


def test_every_entry_records_its_environment(ledger):
    entry = ledger.measured("training_run", "A3 pilot", {"steps": 100}, arm="A3")
    assert entry.env["python"]
    assert set(entry.env) >= {"python", "platform", "commit", "branch", "dirty",
                              "torch", "transformers", "gpu"}
    assert entry.timestamp.endswith("+00:00")


def test_environment_never_raises_even_without_git_or_torch():
    env = environment()
    assert isinstance(env, dict) and "python" in env


def test_queries_filter_by_kind_arm_and_provenance(ledger):
    ledger.measured("evaluation", "a", {}, arm="A1")
    ledger.measured("evaluation", "b", {}, arm="A3")
    ledger.reported("comparison", "c", {}, source="card")
    assert len(ledger.entries(kind="evaluation")) == 2
    assert len(ledger.entries(arm="A3")) == 1
    assert len(ledger.entries(provenance=REPORTED)) == 1
    assert ledger.latest("evaluation")["title"] == "b"
    assert ledger.latest("evaluation", arm="A1")["title"] == "a"
    assert ledger.latest("training_run") is None


def test_get_returns_the_stored_row(ledger):
    entry = ledger.measured("note", "x", {"k": "v"})
    assert ledger.get(entry.id)["payload"] == {"k": "v"}
    assert ledger.get("nope") is None


def test_summary_counts_and_measures_the_measured_fraction(ledger):
    ledger.measured("evaluation", "a", {})
    ledger.reported("comparison", "b", {}, source="card")
    ledger.estimated("memory_accounting", "c", {}, method="analytic")
    summary = ledger.summary()
    assert summary["live_entries"] == 3
    assert summary["by_provenance"] == {ESTIMATED: 1, MEASURED: 1, REPORTED: 1}
    assert summary["measured_fraction"] == pytest.approx(1 / 3)


def test_render_is_readable(ledger):
    ledger.measured("evaluation", "a", {}, arm="A3")
    text = ledger.render()
    assert "live entries" in text and "A3" in text


def test_a_corrupt_line_names_the_file_and_line_number(ledger):
    ledger.measured("note", "good", {})
    with ledger.path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    with pytest.raises(ValueError, match=r"ledger\.jsonl:2"):
        list(ledger)


def test_an_empty_ledger_reads_as_empty(tmp_path):
    empty = Ledger(tmp_path / "missing.jsonl")
    assert list(empty) == []
    assert empty.summary()["entries"] == 0


def test_rows_are_plain_json(ledger):
    ledger.measured("architecture_audit", "audit", {"params": 22_072_134_528})
    row = json.loads(ledger.path.read_text(encoding="utf-8").strip())
    assert row["payload"]["params"] == 22_072_134_528
