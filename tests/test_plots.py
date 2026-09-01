"""The plotting system's one rule: a figure never invents its numbers.

These tests do not check that figures look good. They check the property that matters for
a research repository — that a script with no data fails loudly instead of drawing a
plausible curve, and that anything drawn without measurements is stamped as a schematic.

matplotlib is an optional extra (``pip install -e ".[plots]"``), so the drawing tests skip
when it is absent. The rules that can be checked by reading the source are checked
unconditionally, because those are the ones that protect the record.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "plots"
SCRIPTS = sorted(PLOTS.glob("plot_*.py"))


def _load(name: str):
    import importlib.util

    sys.path.insert(0, str(PLOTS))
    spec = importlib.util.spec_from_file_location(name, PLOTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------
def test_the_plotting_package_exists_with_a_readme_and_outputs():
    assert (PLOTS / "README.md").exists()
    assert (PLOTS / "common.py").exists()
    assert (PLOTS / "outputs" / "paper").is_dir()
    assert (PLOTS / "outputs" / "readme").is_dir()


def test_every_required_figure_has_a_script():
    names = {p.stem for p in SCRIPTS}
    for required in ("plot_architecture", "plot_memory", "plot_behavior_alignment",
                     "plot_distillation", "plot_context_specialization"):
        assert required in names


def test_no_script_hard_codes_a_benchmark_number():
    """The failure this guards: a figure carrying a number nobody measured. Benchmark-shaped
    literals (0.0-1.0 accuracies, MMLU-style scores) must come from artifacts."""
    for script in SCRIPTS:
        source = script.read_text(encoding="utf-8")
        for banned in ("mmlu", "gpqa", "ifeval", "livecodebench", "82.5", "81.7", "91.5"):
            assert banned.lower() not in source.lower(), f"{script.name} names {banned}"


def test_every_script_records_provenance_on_its_figures():
    """A figure that does not say where its numbers came from cannot be audited later."""
    for script in SCRIPTS:
        source = script.read_text(encoding="utf-8")
        if "save(" in source:
            assert "source=" in source, f"{script.name} saves without provenance"


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------
def test_missing_data_raises_rather_than_returning_a_plausible_figure():
    common = _load("common")
    with pytest.raises(SystemExit) as exit_info:
        common.require(Path("/nonexistent/artifact.json"), "a result", "run something")
    assert exit_info.value.code == 2


def test_the_sixteen_gb_limit_is_the_measured_usable_capacity_not_the_nominal_one():
    """Drawing the boundary at 16.0 would make figures show configurations fitting that do
    not fit on real hardware."""
    common = _load("common")
    assert common.VRAM_LIMIT_GIB == 13.56
    assert common.VRAM_LIMIT_GIB < common.NOMINAL_VRAM_GIB


def test_the_schematic_stamp_is_visible_and_says_so():
    common = _load("common")
    import inspect

    source = inspect.getsource(common.schematic)
    assert "SCHEMATIC" in source
    assert "fabricated result" in inspect.getdoc(common.schematic)


def test_student_facts_are_computed_not_typed():
    """The architecture figure must not drift from the architecture."""
    pytest.importorskip("transformers")
    common = _load("common")
    facts = common.student_facts()
    assert facts["audit"]["exact_parameter_count"] == 13_008_505_728
    assert facts["audit"]["active_parameters_per_token"] == 9_611_119_488


# ---------------------------------------------------------------------------
# they run
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["plot_architecture", "plot_memory",
                                  "plot_behavior_alignment", "plot_context_specialization"])
def test_the_data_backed_and_schematic_figures_render(name, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("transformers")
    module = _load(name)
    common = _load("common")
    monkeypatch.setattr(common, "PAPER", tmp_path)
    monkeypatch.setattr(module, "save", common.save)
    monkeypatch.setattr(common, "OUTPUTS", tmp_path)
    assert module.main() == 0
    assert list(tmp_path.glob("*.png")), "no figure was written"


def test_the_distillation_figure_refuses_to_draw_without_runs(capsys):
    """No arm has been trained, so there is nothing to plot. Exiting 2 and naming the
    missing artifact is the correct behaviour; a drawn curve would be a fabrication."""
    pytest.importorskip("matplotlib")
    module = _load("plot_distillation")
    assert module.main() == 2
    assert "no training runs" in capsys.readouterr().err
