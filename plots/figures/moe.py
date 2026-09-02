"""MoE routing figures.

All three need a per-step routing record that no run writes today. The student's forward
can produce it — ``output_router_logits=True`` gives the gate distribution, from which
per-expert token counts, routing entropy and dead-expert counts all follow — but the
trainer does not request it and ``metrics.jsonl`` therefore has no routing fields. The
refusals name that gap precisely rather than pointing at "MoE metrics" in general.

Routing statistics at initialisation are computable without a run
(``moe_init.measure_router_balance``), but initialisation routing is not the quantity any
of these figures asks about: all three are questions about how routing *changes* during
training. Substituting the init measurement would answer a different question under the
same title.
"""
from __future__ import annotations

from common import MissingData, Profile

_HOW = ("record routing statistics per step: pass output_router_logits=True in the "
        "trainer's forward and log {field} to metrics.jsonl")


def _refuse(what: str, field: str) -> None:
    raise MissingData(what, _HOW.format(field=field))


def expert_utilisation(profile: Profile) -> list:
    """F17 — per-expert token share and load imbalance."""
    _refuse("MoE expert utilisation over training",
            "expert_token_counts (length num_experts) and load_imbalance")


def routing_entropy(profile: Profile) -> list:
    """F18 — entropy of the router's distribution over training."""
    _refuse("MoE routing entropy over training",
            "routing_entropy (mean over tokens, nats)")


def dead_experts(profile: Profile) -> list:
    """F19 — experts receiving no or almost no tokens, over training."""
    _refuse("dead and near-dead expert counts over training",
            "dead_experts and near_dead_experts (counts per step)")
