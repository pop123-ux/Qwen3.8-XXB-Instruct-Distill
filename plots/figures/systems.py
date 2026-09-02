"""Deployment and systems figures — the 16 GB frontier, throughput, latency, trade-off.

Every one of these needs an inference benchmark that has not been run. The memory axis is
available analytically (F09, F22), but a frontier with no quality axis is a frame rather
than a figure, and a throughput curve substituted from *training* throughput would be a
different quantity wearing the same label. Both are refused rather than approximated.
"""
from __future__ import annotations

from common import ROOT, MissingData, Profile

DEPLOYMENT_DIR = ROOT / "experiments" / "deployment"

_BENCHMARK = (
    "run an inference benchmark on the quantised, adapter-merged student and write "
    f"{DEPLOYMENT_DIR.relative_to(ROOT)}/<config>.json carrying "
    "{fields}"
)


def _refuse(what: str, fields: str) -> None:
    raise MissingData(what, _BENCHMARK.format(fields=fields))


def deployment_frontier_16gb(profile: Profile) -> list:
    """F24 — quality against peak VRAM, with the 16 GB boundary marked."""
    _refuse("the 16 GB deployment frontier (no quality measurement exists)",
            "peak_vram_gib and a quality score from a named, reproducible benchmark")


def throughput_vs_context(profile: Profile) -> list:
    """F25 — generation throughput against deployment context length."""
    _refuse("inference throughput against context length",
            "context_length and tokens_per_second. Training throughput is recorded in "
            "experiments/*/summary.json but measures a different thing — teacher forward, "
            "QLoRA backward and an optimizer step — and is not a substitute")


def latency_vs_context(profile: Profile) -> list:
    """F26 — time-to-first-token and per-token latency against context length."""
    _refuse("inference latency against context length",
            "context_length, time_to_first_token_ms and per_token_latency_ms")


def quality_memory_throughput(profile: Profile) -> list:
    """F27 — the quality/memory/throughput trade-off, as a 2D scatter with sized markers."""
    _refuse("the quality/memory/throughput trade-off",
            "quality, peak_vram_gib and tokens_per_second for each candidate configuration")
