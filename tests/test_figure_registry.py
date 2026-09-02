"""The figure registry must describe the figure system, not advertise it.

A registry is only worth consulting if its claims are checked. These tests check the three
ways it could drift: a status that does not match what the builder does, a declared source
that does not exist, and a generated `REGISTRY.md` that is out of date with `registry.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "plots"
for _path in (PLOTS, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import registry  # noqa: E402

KNOWN_RESEARCH_QUESTIONS = {"RQ1", "RQ2", "RQ3", "RQ4"}


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------
def test_the_register_is_structurally_coherent():
    assert registry.check_integrity() == []


def test_figure_ids_are_unique_and_contiguous():
    ids = [spec.id for spec in registry.FIGURES]
    assert len(ids) == len(set(ids))
    numbers = sorted(int(figure_id[1:]) for figure_id in ids)
    assert numbers == list(range(1, len(numbers) + 1)), \
        "figure numbering has a gap or a duplicate"


def test_every_figure_declares_one_question_and_its_research_question():
    for spec in registry.FIGURES:
        assert spec.question.endswith("?"), f"{spec.id}: the question is not a question"
        assert set(spec.research_questions) <= KNOWN_RESEARCH_QUESTIONS, \
            f"{spec.id}: unknown research question {spec.research_questions}"


def test_every_figure_names_its_source_metric_fields():
    """Without the field names, "what produced this point?" needs a code read."""
    for spec in registry.FIGURES:
        assert spec.metrics, f"{spec.id} declares no source metric fields"
        assert spec.sources, f"{spec.id} declares no sources"


def test_a_real_figures_declared_artifact_paths_exist():
    """A source path that does not exist means the registry is describing a plan."""
    for spec in registry.with_status(registry.REAL):
        for source in spec.sources:
            if source.startswith("qwen_distill") or "*" in source:
                continue  # a module path or a glob, checked by the render tests instead
            assert (ROOT / source).exists(), \
                f"{spec.id} declares {source}, which does not exist"


def test_every_real_figure_names_the_experiments_it_reads():
    for spec in registry.with_status(registry.REAL):
        reads_a_run = any(source.startswith("experiments/") for source in spec.sources)
        assert bool(spec.experiments) == reads_a_run, \
            f"{spec.id}: experiment list and artifact paths disagree"


# ---------------------------------------------------------------------------
# the status claim is checked against behaviour
# ---------------------------------------------------------------------------
def test_declared_status_matches_what_the_builders_actually_do(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("transformers")
    import common
    import make_figures

    monkeypatch.setattr(common.Profile, "directory",
                        property(lambda self: tmp_path / self.name))
    specs = list(registry.FIGURES)
    results = make_figures.run(specs, [common.README], quiet=True)
    problems = make_figures.reconcile(specs, results)
    assert problems == [], "\n".join(problems)


def test_the_generated_registry_document_is_current():
    """`REGISTRY.md` is generated; a stale one is a registry that says the wrong thing."""
    current = (PLOTS / "REGISTRY.md").read_text(encoding="utf-8")
    assert current == registry.render_markdown(), \
        "run `python plots/make_figures.py --write-registry`"


def test_the_registry_document_marks_which_figures_are_backed_by_real_data():
    document = (PLOTS / "REGISTRY.md").read_text(encoding="utf-8")
    for spec in registry.with_status(registry.REAL):
        assert spec.id in document
    assert "**real**" in document
    assert "unavailable | the experiment has not happened" in document


def test_output_names_are_stable_and_profile_scoped():
    for spec in registry.FIGURES:
        outputs = spec.outputs()
        assert any(name.startswith("paper/") for name in outputs)
        assert any(name.startswith("readme/") for name in outputs)
        assert all(spec.stem in name for name in outputs)
        assert sum(name.endswith(".json") for name in outputs) == len(spec.profiles), \
            f"{spec.id}: every profile must write a provenance sidecar"
