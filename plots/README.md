# Figures

A research-paper figure system: one scientific question per figure, one primary dependent
variable per figure, and a registry that says which figures are backed by real data.

```bash
python plots/make_figures.py                  # every figure, paper + README profiles
python plots/make_figures.py --status real    # only the data-backed ones
python plots/make_figures.py F03 F04 --profile paper
python plots/make_figures.py --list           # the register as a table
python plots/make_figures.py --write-registry # regenerate REGISTRY.md
```

[**REGISTRY.md**](REGISTRY.md) is the index: figure id, scientific question, research
question, source experiments, source artifact paths, source metric fields, plotting script,
outputs, and status. It is generated from `registry.py`; do not edit it by hand.

## The one rule

**A figure never invents its numbers.**

Every figure either reads a real artifact this repository produced, or it is explicitly a
schematic and stamped as one. There is no third case. A figure with no data behind it exits
2 and prints the command that would produce it — it never falls back to plausible values,
never estimates a curve that is not declared analytical, and never silently substitutes a
different metric for a missing one. A plausible-looking curve is indistinguishable from a
result once it is pasted into a document.

Status vocabulary, checked against behaviour by `tests/test_figure_registry.py`:

| status | meaning |
| --- | --- |
| **real** | every series is backed by an artifact this repository produced |
| partial | some declared series exist and some do not; gaps are stated, never filled |
| schematic | a conceptual diagram, stamped as one — never a result |
| unavailable | the experiment has not happened; the builder exits 2 |

Today: **12 real**, 15 unavailable, 0 schematic. See `REGISTRY.md` for the breakdown.

## Layout

```
plots/
    common.py        style, output profiles, provenance, MissingData
    data.py          normalised access to run artifacts; matched-arm selection
    registry.py      the figure register and its integrity checks
    make_figures.py  the driver
    figures/         one module per theme, one function per figure
    outputs/
        paper/       .pdf (vector) + .json provenance sidecar
        readme/      .png + .json provenance sidecar
        manifest.json
```

The data pipeline is deliberately one-directional:

```
raw run artifacts  ->  data.py  ->  figure-specific selection  ->  matplotlib  ->  outputs
```

A figure never parses a file itself. Trajectories come from the per-step `metrics.jsonl` —
a 128-step run is drawn as 128 points, never reduced to the summary's first/final pair. The
experiment ledger stays the provenance and index layer.

## Output profiles

One implementation, two profiles, no duplicated drawing code.

| | `paper` | `readme` |
| --- | --- | --- |
| format | PDF (vector) | PNG |
| canvas | 5.6 x 3.5 in, 300 dpi metadata | 7.2 x 4.0 in, 110 dpi |
| type size | 8 pt | 10 pt |
| secondary annotation | on | off |
| provenance footnote | on | on |

A figure branches on `profile.annotate` only for *secondary* annotation. Truth-in-labelling —
"ANALYTICAL", "DESIGN", "not a capability claim", "TRAINING memory" — is never dropped by a
smaller profile.

## Provenance

Every figure carries a one-line footnote (figure id, experiments, source path, metric
fields, value class, git SHA of the data) and writes a JSON sidecar next to itself with the
complete record: every experiment id, every source path, every metric field, the commit
that produced the data, the commit that drew the figure, and the figure's own key numbers.

A researcher must be able to answer *"what produced this point?"* from the sidecar without
searching the repository.

Value classes: `measured` (off hardware or a run), `analytical` (computed from a model,
never observed), `audited` (a counted property of the frozen specification), `design` (a
declared experimental design), `measured+analytical` (a figure that carries both and keeps
them in separate panels).

## Style

Deliberately plain, optimised for scientific readability rather than impact.

- **Gridlines: horizontal only**, beneath the data, 30 % alpha. Reading a value off a loss
  curve needs a horizontal reference; vertical rules mostly restate the x ticks and cross
  every series. Figures with a log-scaled x-axis opt into both axes through `grid(ax,
  "both")`. `common.style()` enforces this, so the code and this paragraph cannot drift
  apart.
- Dark neutral first, so a single-series figure survives greyscale print. Six colours, no
  more.
- No 3D, no decorative graphics, no dual y-axes where a crossing could be misread.
- Units on every axis. Reference lines named on a right-hand tick rather than by a label
  dropped onto the data.
- Reference-line values, thresholds and smoothing windows are stated on the figure and
  recorded in the sidecar.

## Scientific labelling rules these figures follow

- Run 001 is a **mechanism-validation** experiment. It is never drawn on the same axes as
  Run 002; `data.matched_arms` excludes it automatically because its protocol differs, and
  prints which fields differ.
- Training memory is never presented as a deployment result.
- Analytical memory is never presented as a measurement, and the two never share an axes.
- Teacher top-1 agreement is an imitation diagnostic on the training batches, not accuracy.
- Comparison thresholds are declared in source before the arms are compared, and written
  into the sidecar of every figure that uses them.
