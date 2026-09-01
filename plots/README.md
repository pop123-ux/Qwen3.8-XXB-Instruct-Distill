# Figures

Reproducible figure generation. Every script writes to `outputs/paper/` (publication) or
`outputs/readme/` (documentation), and every figure carries a provenance footnote saying
where its numbers came from.

```bash
python plots/plot_architecture.py          # teacher vs student, compression, sparsity
python plots/plot_memory.py                # 16 GB accounting, context x VRAM, Pareto
python plots/plot_behavior_alignment.py    # behaviour-matching metrics
python plots/plot_distillation.py          # recovery curves per objective
python plots/plot_context_specialization.py # context-performance curves
```

## The one rule

**A figure never invents its numbers.**

Each script either reads a real artifact this repository produced, or it draws an
explicitly stamped schematic. There is no third case. A script with no data available
exits 2 and prints the command that would produce it — it does not fall back to plausible
values, because a plausible-looking curve is indistinguishable from a result once it is
pasted into a document.

| figure | data today | source |
|---|---|---|
| architecture | **real** | `audit()` — computed at plot time |
| memory, context × VRAM | **real** | `research/memory.py` — analytical, labelled as such |
| Pareto | axes + 16 GB boundary only | needs benchmark scores |
| behaviour alignment | **schematic** | needs a materialised student and a real teacher |
| distillation recovery | **schematic** | needs training runs |
| context specialisation | **schematic** | needs evaluation at each length |
| MoE routing | **real** at init | `measure_router_balance` |

Schematics show the *shape of the question*, not an answer, and are stamped diagonally so
they cannot be mistaken for one. They are replaced as experiments produce artifacts.

## Style

Plain on purpose: no gridlines fighting the data, no colour where a shape will do, dark
neutral first so single-series figures survive greyscale print. Optimised for "the result
is immediately understandable", not for looking impressive. Aesthetics get another pass
once there is real data worth presenting.
