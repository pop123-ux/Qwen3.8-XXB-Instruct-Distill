"""Research components: what is being claimed, how it is measured, and what would refute it.

Four modules, each answering one question the project has to answer honestly.

``ablations``   which comparisons are being run, and what observation refutes each arm
``baselines``   the historical dense candidate, retained as the control it is
``context``     context regimes, length curricula, and the context-performance curve
``ledger``      the append-only record, where an estimate can never pass as a measurement
``memory``      end-to-end VRAM against the 16 GB ceiling, fully GPU-resident
"""
from .ablations import ARMS, arms, control, matrix
from .baselines import baselines, comparison, dense_h5120_l40
from .context import (
    CONTEXT_REGIMES,
    CURRICULA,
    ContextCurve,
    ContextPoint,
    compare_curves,
    curriculum,
    regime_for,
)
from .ledger import ESTIMATED, MEASURED, REPORTED, Entry, Ledger
from .memory import RuntimeConfig, account, build_table, frontier, headline, render_table

__all__ = [
    "ARMS", "CONTEXT_REGIMES", "CURRICULA", "ESTIMATED", "MEASURED", "REPORTED",
    "ContextCurve", "ContextPoint", "Entry", "Ledger", "RuntimeConfig",
    "account", "arms", "baselines", "build_table", "compare_curves", "comparison",
    "control", "curriculum", "dense_h5120_l40", "frontier", "headline", "matrix",
    "regime_for", "render_table",
]
