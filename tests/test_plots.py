"""The plotting system's one rule: a figure never invents its numbers.

These tests do not check that figures look good. They check the properties that protect the
research record — that a script with no data fails loudly instead of drawing a plausible
curve, that trajectories come from the per-step record rather than a summary, that the
declared output profiles are one implementation rather than two, and that regenerating a
figure produces the same bytes.

matplotlib is an optional extra (``pip install -e ".[plots]"``), so the drawing tests skip
when it is absent. The rules checkable by reading the source are checked unconditionally.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "plots"
FIGURE_MODULES = sorted((PLOTS / "figures").glob("*.py"))

# At import time, not in a fixture: the parametrised cases below are built during
# collection and need the register available then.
for _path in (PLOTS, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import registry as _registry  # noqa: E402

REAL_FIGURES = [spec.id for spec in _registry.FIGURES if spec.status == _registry.REAL]
UNAVAILABLE_FIGURES = [spec.id for spec in _registry.FIGURES
                       if spec.status == _registry.UNAVAILABLE]


def _module(name: str):
    import importlib

    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------
def test_the_plotting_package_has_its_documented_parts():
    for required in ("README.md", "REGISTRY.md", "common.py", "data.py", "registry.py",
                     "make_figures.py"):
        assert (PLOTS / required).exists(), f"plots/{required} is missing"
    assert (PLOTS / "figures" / "__init__.py").exists()


def test_every_registered_builder_exists_and_is_callable():
    registry = _module("registry")
    for spec in registry.FIGURES:
        module_name, function_name = spec.builder.split(":")
        module = _module(module_name)
        assert callable(getattr(module, function_name, None)), \
            f"{spec.id}: {spec.builder} is not callable"


def test_no_figure_module_hard_codes_an_experimental_value():
    """The failure this guards: a figure carrying a number nobody measured.

    Loss values, agreement fractions and memory peaks must arrive through ``data.py``. The
    literals allowed in a builder are layout (positions, alphas, widths) and thresholds
    that are *declared* as thresholds — those are experimental design, not results.
    """
    banned = ("mmlu", "gpqa", "ifeval", "livecodebench",
              "7.190", "1.410", "5.033", "0.3375", "40.58", "43.126", "196608",
              "13008505728", "26895998464")
    for script in [*FIGURE_MODULES, PLOTS / "common.py", PLOTS / "data.py"]:
        source = script.read_text(encoding="utf-8").lower()
        for literal in banned:
            assert literal.lower() not in source, \
                f"{script.name} hard-codes the experimental value {literal!r}"


def test_every_builder_saves_with_provenance():
    """A figure that does not say where its numbers came from cannot be audited later."""
    for script in FIGURE_MODULES:
        source = script.read_text(encoding="utf-8")
        if "save(" in source:
            assert "provenance=" in source, f"{script.name} saves without provenance"


def test_the_sixteen_gb_limit_is_the_measured_usable_capacity_not_the_nominal_one():
    """Drawing the boundary at 16.0 would show configurations fitting that do not fit."""
    common = _module("common")
    assert common.VRAM_LIMIT_GIB == 13.56
    assert common.VRAM_LIMIT_GIB < common.NOMINAL_VRAM_GIB


def test_the_schematic_stamp_is_visible_and_says_so():
    import inspect

    common = _module("common")
    assert "SCHEMATIC" in inspect.getsource(common.schematic)
    assert "fabricated result" in inspect.getdoc(common.schematic)


# ---------------------------------------------------------------------------
# style policy: the code and the documentation must agree
# ---------------------------------------------------------------------------
def test_the_gridline_policy_in_the_code_matches_the_documented_one():
    """The defect this test exists for: the README said "no gridlines fighting the data"
    while ``style()`` switched on a full grid. Whichever policy is chosen, both must state
    it — so the test reads the actual rcParams rather than the source text."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    common = _module("common")
    common.style(common.PAPER)
    assert plt.rcParams["axes.grid"] is True
    assert plt.rcParams["axes.grid.axis"] == "y", "the documented policy is horizontal only"
    assert plt.rcParams["axes.axisbelow"] is True, "gridlines must sit beneath the data"
    assert plt.rcParams["grid.alpha"] <= 0.35

    readme = (PLOTS / "README.md").read_text(encoding="utf-8")
    assert "Gridlines: horizontal only" in readme
    assert "no gridlines fighting the data" not in readme


# ---------------------------------------------------------------------------
# output profiles
# ---------------------------------------------------------------------------
def test_the_two_profiles_differ_in_output_not_in_drawing_code():
    common = _module("common")
    paper, readme = common.PAPER, common.README
    assert "pdf" in paper.formats, "paper output should be vector where practical"
    assert readme.formats == ("png",)
    assert paper.font_size != readme.font_size
    assert paper.directory != readme.directory
    # One implementation: no builder may import or branch on a profile *name*.
    for script in FIGURE_MODULES:
        source = script.read_text(encoding="utf-8")
        assert 'profile.name ==' not in source, \
            f"{script.name} branches on the profile name instead of its parameters"


def test_a_truth_label_is_never_dropped_by_the_smaller_profile():
    """README output carries less annotation, but never less honesty."""
    memory = (PLOTS / "figures" / "memory.py").read_text(encoding="utf-8")
    analytical = memory.index('"ANALYTICAL')
    preceding = memory[max(0, analytical - 400):analytical]
    assert "if profile.annotate" not in preceding.split("ax.text")[-1]


# ---------------------------------------------------------------------------
# the trajectory comes from the per-step record
# ---------------------------------------------------------------------------
def test_run002_trajectory_is_the_full_per_step_record(tmp_path):
    data = _module("data")
    run = data.load_run("run002_logit_kd")
    assert run.n_logged_steps == 128, "a 128-step run must be drawn as 128 points"
    xs, ys = run.series("loss")
    assert len(xs) == len(ys) == 128
    assert run.tokens_seen == 128 * run.sequence_length


def test_validation_uses_only_actual_observations():
    data = _module("data")
    run = data.load_run("run002_logit_kd")
    xs, ys = run.validation_series()
    steps = [r["step"] for r in run.validations]
    assert len(ys) == len(steps) == 4, "only the run's four evaluations may be plotted"
    assert steps == sorted(steps)


def test_the_record_wins_over_a_summary_that_disagrees_with_it():
    """``kd_run_001``'s summary.json is the one-step smoke; metrics.jsonl is the 50-step
    pilot. A trajectory read from the summary would be wrong by a factor of 50."""
    data = _module("data")
    run = data.load_run("kd_run_001")
    assert run.summary["steps"] == 1
    assert run.last_step == 50
    assert run.n_logged_steps > 1


def test_a_missing_metric_raises_rather_than_drawing_nothing():
    data = _module("data")
    run = data.load_run("run002_logit_kd")
    with pytest.raises(SystemExit) as exit_info:
        run.series("hidden_state_similarity")
    assert exit_info.value.code == 2


def test_missing_data_raises_rather_than_returning_a_plausible_figure():
    common = _module("common")
    with pytest.raises(SystemExit) as exit_info:
        common.require(Path("/nonexistent/artifact.json"), "a result", "run something")
    assert exit_info.value.code == 2


# ---------------------------------------------------------------------------
# controlled comparison: matching is derived, not asserted
# ---------------------------------------------------------------------------
def test_run001_is_excluded_from_the_run002_comparison_with_a_stated_reason():
    """A13: Run 001 is mechanism validation and must never join a performance curve."""
    data = _module("data")
    armset = data.matched_arms("run002_logit_kd")
    assert "kd_run_001" not in {run.experiment_id for _, run in armset.ordered()}
    reasons = dict(armset.excluded)
    assert "kd_run_001" in reasons
    assert "protocol differs" in reasons["kd_run_001"]


def test_only_the_logit_kd_arm_is_matched_today():
    data = _module("data")
    armset = data.matched_arms("run002_logit_kd")
    assert set(armset.arms) == {"logit_kd"}
    assert armset.arms["logit_kd"].experiment_id == "run002_logit_kd"


def test_a_second_arm_at_the_matched_protocol_is_admitted(tmp_path, monkeypatch):
    """The comparison figures must populate themselves when Run 003 lands, without anyone
    editing a list of run names."""
    data = _module("data")
    source = ROOT / "experiments" / "run002_logit_kd"
    fixture = tmp_path / "experiments"
    for name in ("run002_logit_kd", "run003_layer_kd"):
        directory = fixture / name
        directory.mkdir(parents=True)
        (directory / "metrics.jsonl").write_text(
            (source / "metrics.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        if name.startswith("run003"):
            summary["objective"] = "layer_kd"
            summary["config"]["training"]["objective"] = "layer_kd"
        summary["experiment"] = name
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    monkeypatch.setattr(data, "EXPERIMENTS", fixture)
    armset = data.matched_arms("run002_logit_kd")
    assert set(armset.arms) == {"logit_kd", "layer_kd"}
    assert [objective for objective, _ in armset.ordered()] == ["logit_kd", "layer_kd"]


def test_a_run_at_a_different_sequence_length_is_refused_as_an_arm(tmp_path, monkeypatch):
    data = _module("data")
    source = ROOT / "experiments" / "run002_logit_kd"
    fixture = tmp_path / "experiments"
    for name, sequence_length in (("run002_logit_kd", 1536), ("run003_layer_kd", 1024)):
        directory = fixture / name
        directory.mkdir(parents=True)
        (directory / "metrics.jsonl").write_text(
            (source / "metrics.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        summary["config"]["data"]["max_sequence_length"] = sequence_length
        if name.startswith("run003"):
            summary["objective"] = "layer_kd"
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    monkeypatch.setattr(data, "EXPERIMENTS", fixture)
    armset = data.matched_arms("run002_logit_kd")
    assert set(armset.arms) == {"logit_kd"}
    assert "max_sequence_length" in dict(armset.excluded)["run003_layer_kd"]


def test_the_comparison_thresholds_are_declared_in_source_before_any_arm_is_read():
    comparison = _module("figures.comparison")
    assert isinstance(comparison.VALIDATION_LOSS_THRESHOLD, float)
    assert isinstance(comparison.AGREEMENT_THRESHOLD, float)
    source = (PLOTS / "figures" / "comparison.py").read_text(encoding="utf-8")
    # Declared above the first builder, so no builder can compute one from its own data.
    assert source.index("VALIDATION_LOSS_THRESHOLD =") < source.index("def matched_")


# ---------------------------------------------------------------------------
# they render, and they render the same way twice
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("figure_id", REAL_FIGURES)
def test_every_real_figure_renders_in_both_profiles(figure_id, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("transformers")
    common = _module("common")
    make = _module("make_figures")
    registry = _module("registry")
    monkeypatch.setattr(common, "OUTPUTS", tmp_path)
    monkeypatch.setattr(common.PAPER.__class__, "directory",
                        property(lambda self: tmp_path / self.name))
    spec = registry.get(figure_id)
    for profile in (common.PAPER, common.README):
        outcome = make.build_one(spec, profile)
        assert outcome["outcome"] == make.RENDERED, \
            f"{figure_id} [{profile.name}]: {outcome.get('detail')}"
    written = list(tmp_path.rglob("*"))
    assert any(p.suffix == ".pdf" for p in written)
    assert any(p.suffix == ".png" for p in written)
    assert any(p.suffix == ".json" for p in written), "no provenance sidecar"


def test_figure_generation_is_deterministic(tmp_path, monkeypatch):
    """Two runs must produce identical bytes, or a regenerated figure set is an unreadable
    diff and 'the figure changed' stops meaning 'the data changed'."""
    pytest.importorskip("matplotlib")
    common = _module("common")
    make = _module("make_figures")
    registry = _module("registry")
    spec = registry.get("F03")
    digests = []
    for attempt in ("first", "second"):
        directory = tmp_path / attempt
        monkeypatch.setattr(common.PAPER.__class__, "directory",
                            property(lambda self, d=directory: d / self.name))
        assert make.build_one(spec, common.README)["outcome"] == make.RENDERED
        png = next((directory / "readme").glob("*.png"))
        digests.append(png.read_bytes())
    assert digests[0] == digests[1], "figure output is not byte-reproducible"


def test_the_provenance_sidecar_answers_what_produced_this_point(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    common = _module("common")
    make = _module("make_figures")
    registry = _module("registry")
    monkeypatch.setattr(common.PAPER.__class__, "directory",
                        property(lambda self: tmp_path / self.name))
    assert make.build_one(registry.get("F03"), common.README)["outcome"] == make.RENDERED
    sidecar = json.loads(
        next((tmp_path / "readme").glob("*.json")).read_text(encoding="utf-8"))
    assert sidecar["figure_id"] == "F03"
    assert sidecar["experiments"] == ["run002_logit_kd"]
    assert any("metrics.jsonl" in s for s in sidecar["sources"])
    assert "loss" in sidecar["metrics"]
    assert sidecar["value_kind"] == "measured"
    assert len(sidecar["data_commit"]) == 40
    assert sidecar["extra"]["n_points"] == 128


@pytest.mark.parametrize("figure_id", UNAVAILABLE_FIGURES)
def test_every_unavailable_figure_refuses_and_names_what_would_produce_it(
        figure_id, capsys, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    common = _module("common")
    make = _module("make_figures")
    registry = _module("registry")
    monkeypatch.setattr(common.PAPER.__class__, "directory",
                        property(lambda self: tmp_path / self.name))
    outcome = make.build_one(registry.get(figure_id), common.README)
    assert outcome["outcome"] == make.MISSING, f"{figure_id}: {outcome}"
    assert "produce it with" in capsys.readouterr().err
    assert not list(tmp_path.rglob("*.png")), f"{figure_id} wrote a figure anyway"
