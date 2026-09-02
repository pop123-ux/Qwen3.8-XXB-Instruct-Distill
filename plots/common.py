"""Style, output profiles, provenance and the one rule, shared by every figure.

The rule, which is the reason this module exists rather than seven copies of matplotlib
defaults:

    **A figure never invents its numbers.** Every figure here either reads a real artifact
    this repository produced — a run's ``metrics.jsonl``, a summary, an audit computed at
    plot time — or it is explicitly a schematic and is stamped as one. There is no third
    case. :class:`MissingData` exists so the absence of a result is a loud failure rather
    than a plausible-looking curve.

One implementation, two profiles
--------------------------------
``paper`` and ``readme`` are the same drawing code with different output parameters
(:class:`Profile`), not two code paths. A figure asks for ``profile.annotate`` when it has
an annotation worth suppressing at README density, and otherwise never branches.

Gridline policy
---------------
Horizontal only, beneath the data, 30% alpha. Reading a value off a loss curve needs a
horizontal reference; vertical rules mostly restate the x ticks and cross every series.
Figures with a log-scaled x-axis may ask for both axes through :func:`grid`. This is the
deliberate policy the documentation states, and :func:`style` is where it is enforced —
the two are not allowed to disagree.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PLOTS = Path(__file__).resolve().parent
OUTPUTS = PLOTS / "outputs"

#: The deployment constraint drawn on every deployment-memory figure. Usable capacity on a
#: real 16 GB card, not the nominal number — see qwen_distill/research/memory.py.
VRAM_LIMIT_GIB = 13.56
NOMINAL_VRAM_GIB = 16.0

#: A restrained sequence: dark neutral first, so a single-series figure is greyscale-safe.
COLOURS = ("#1a1a1a", "#c1553b", "#3a6ea5", "#6b8f3a", "#8b5fa8", "#a08020")
MARKERS = ("o", "s", "^", "D", "v", "P")

#: Value classes a figure may carry. A figure mixing two says so rather than picking one.
MEASURED = "measured"        #: read off hardware or off a training run
ANALYTICAL = "analytical"    #: computed from a model, never observed
AUDITED = "audited"          #: a counted property of the frozen specification
DESIGN = "design"            #: a declared experimental design, not an observation
MIXED = "measured+analytical"
VALUE_KINDS = (MEASURED, ANALYTICAL, AUDITED, DESIGN, MIXED)


# ---------------------------------------------------------------------------
# missing data
# ---------------------------------------------------------------------------
class MissingData(SystemExit):
    """Raised when a figure has no real artifact to draw. Exits 2, naming what to run."""

    def __init__(self, what: str, how: str) -> None:
        super().__init__(2)
        self.what, self.how = what, how
        print(f"  no data for {what}.\n  produce it with:\n    {how}", file=sys.stderr)


def require(path: Path, what: str, how: str) -> dict[str, Any]:
    """Load a JSON artifact or fail loudly, naming the command that would create it."""
    if not Path(path).exists():
        raise MissingData(what, how)
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# output profiles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Profile:
    """Output parameters for one audience. The drawing code is identical across profiles."""

    name: str
    #: Width and height in inches for a single-panel figure. Panelled figures scale it.
    base_size: tuple[float, float]
    dpi: int
    #: Raster first; ``pdf`` is added for paper output, where vector matters.
    formats: tuple[str, ...]
    font_size: float
    #: Whether the figure draws its secondary annotations. README output stays sparser.
    annotate: bool
    #: Whether the provenance footnote is rendered on the figure itself. It is always
    #: written to the sidecar regardless.
    footnote: bool

    @property
    def directory(self) -> Path:
        return OUTPUTS / self.name


#: Paper output is vector: a PDF drops straight into LaTeX, scales without resampling and
#: costs ~25 kB, where the same figure as a 300 dpi raster costs ~250 kB and is worse. The
#: README profile supplies the raster GitHub needs.
PAPER = Profile(
    name="paper", base_size=(5.6, 3.5), dpi=300, formats=("pdf",),
    font_size=8.0, annotate=True, footnote=True,
)
README = Profile(
    name="readme", base_size=(7.2, 4.0), dpi=110, formats=("png",),
    font_size=10.0, annotate=False, footnote=True,
)
PROFILES: dict[str, Profile] = {p.name: p for p in (PAPER, README)}


def style(profile: Profile) -> None:
    """Apply the house style for a profile. Called by every figure before it draws."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.figsize": profile.base_size,
        "figure.dpi": 110,
        "savefig.dpi": profile.dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "font.size": profile.font_size,
        "font.family": "sans-serif",
        "axes.titlesize": profile.font_size + 1.0,
        "axes.labelsize": profile.font_size,
        "axes.titlelocation": "left",
        "axes.titlepad": 6.0,
        "xtick.labelsize": profile.font_size - 0.5,
        "ytick.labelsize": profile.font_size - 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        # Gridline policy: horizontal only, beneath the data. See the module docstring.
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.30,
        "grid.linewidth": 0.5,
        "grid.color": "#9a9a9a",
        "legend.frameon": False,
        "legend.fontsize": profile.font_size - 0.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 3.5,
        # Deterministic output: no timestamps, no hash-order-dependent ids.
        "svg.hashsalt": "qwen-distill",
        "pdf.compression": 6,
    })


def grid(ax, axis: str = "y") -> None:
    """Opt one axes into a different grid axis than the house default."""
    ax.grid(True, axis=axis, alpha=0.30, linewidth=0.5, color="#9a9a9a")
    ax.set_axisbelow(True)


def figure(profile: Profile, *, ncols: int = 1, nrows: int = 1,
           width: float = 1.0, height: float = 1.0, **kwargs: Any):
    """A figure sized from the profile, scaled by panel count.

    Panels widen the canvas sublinearly: two panels are wider than one but not twice as
    wide, because the axis furniture is shared.
    """
    import matplotlib.pyplot as plt

    base_w, base_h = profile.base_size
    w = base_w * width * (1.0 + 0.62 * (ncols - 1))
    h = base_h * height * (1.0 + 0.78 * (nrows - 1))
    return plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def repo_commit() -> str | None:
    """The current HEAD, for figures computed at plot time rather than read from a run."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git absent
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


@dataclass
class Provenance:
    """What produced every point on a figure.

    A researcher should be able to answer "what produced this point?" from the sidecar
    without searching the repository, which is why the metric *field names* are recorded
    and not only the file paths.
    """

    figure_id: str
    #: Experiment ids whose artifacts this figure reads. Empty for computed-at-plot-time.
    experiments: tuple[str, ...] = ()
    #: Repository-relative artifact paths, or a module path for computed figures.
    sources: tuple[str, ...] = ()
    #: The metric field names actually plotted.
    metrics: tuple[str, ...] = ()
    #: MEASURED / ANALYTICAL / DESIGN / MIXED.
    value_kind: str = MEASURED
    #: The commit that produced the *data*. Differs from the plotting commit and is the
    #: one that matters for reproducing a number.
    data_commit: str | None = None
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def footnote(self) -> str:
        """One restrained line for the figure itself. The sidecar carries the rest."""
        bits = [self.figure_id]
        if self.experiments:
            bits.append(", ".join(self.experiments))
        if self.sources:
            head = self.sources[0]
            bits.append(head if len(self.sources) == 1 else f"{head} (+{len(self.sources) - 1})")
        if self.metrics:
            # The footnote is a pointer, not the manifest: the sidecar carries every field.
            shown = list(self.metrics[:4])
            extra = len(self.metrics) - len(shown)
            bits.append("fields: " + ", ".join(shown) + (f" (+{extra})" if extra else ""))
        bits.append(self.value_kind)
        commit = self.data_commit or repo_commit()
        if commit:
            bits.append(f"git {commit[:7]}")
        return "  ·  ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "experiments": list(self.experiments),
            "sources": list(self.sources),
            "metrics": list(self.metrics),
            "value_kind": self.value_kind,
            "data_commit": self.data_commit,
            "plotting_commit": repo_commit(),
            "note": self.note,
            **({"extra": self.extra} if self.extra else {}),
        }


def schematic(ax, note: str = "SCHEMATIC — no measured data") -> None:
    """Stamp a figure that shows a shape rather than a result.

    A schematic that is not labelled becomes a fabricated result the moment it is pasted
    into a document, so the label is part of drawing one.
    """
    ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center", va="center",
            fontsize=15, color="#c1553b", alpha=0.18, rotation=16, zorder=0,
            fontweight="bold")


# ---------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------
def display_path(path: Path) -> str:
    """Repository-relative when it can be, absolute otherwise.

    Output can be redirected outside the tree (a test's tmp_path), and a figure must not
    fail to save because its own log line could not be shortened.
    """
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _png_metadata() -> dict[str, Any]:
    # matplotlib stamps "Software: Matplotlib version ..." into every PNG by default,
    # which makes byte-identical regeneration depend on the installed version. Pinning the
    # tag is what lets test_deterministic_figure_generation compare bytes.
    return {"Software": "qwen-distill/plots"}


def _pdf_metadata() -> dict[str, Any]:
    # CreationDate=None omits the timestamp; without it no two PDF runs are identical.
    return {"Creator": "qwen-distill/plots", "Producer": "matplotlib",
            "CreationDate": None}


def save(fig, slug: str, *, profile: Profile, provenance: Provenance,
         quiet: bool = False) -> list[Path]:
    """Write a figure in every format the profile asks for, plus its provenance sidecar.

    Returns the paths written, figure formats first, sidecar last.
    """
    if profile.footnote:
        fig.text(0.004, 0.004, provenance.footnote(), fontsize=max(5.0, profile.font_size - 3.2),
                 color="#767676", ha="left", va="bottom")
    directory = profile.directory
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{provenance.figure_id}_{slug}"
    written: list[Path] = []
    for fmt in profile.formats:
        path = directory / f"{stem}.{fmt}"
        metadata = _pdf_metadata() if fmt == "pdf" else _png_metadata()
        fig.savefig(path, format=fmt, metadata=metadata)
        written.append(path)
    sidecar = directory / f"{stem}.json"
    sidecar.write_text(json.dumps(provenance.to_dict(), indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    written.append(sidecar)
    if not quiet:
        for path in written[:-1]:
            print(f"  wrote {display_path(path)}")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# architecture facts, computed rather than typed in
# ---------------------------------------------------------------------------
def student_facts() -> dict[str, Any]:
    """The canonical student's audited numbers, computed at plot time."""
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from qwen_distill.architecture.moe_student import FROZEN_STUDENT, audit

    return {"spec": FROZEN_STUDENT, "audit": audit(FROZEN_STUDENT)}


def teacher_facts() -> dict[str, Any]:
    """The frozen teacher's numbers, computed at plot time from the pinned preset."""
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from qwen_distill.architecture.params import count_parameters
    from qwen_distill.architecture.presets import get_spec

    spec = get_spec("teacher")
    return {"spec": spec, "total": count_parameters(spec).total}


os.environ.setdefault("SOURCE_DATE_EPOCH", "1000000000")
