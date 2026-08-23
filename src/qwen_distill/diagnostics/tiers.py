"""Coarse hardware capability tiers.

Tiers exist to answer *"which experiments are plausible here?"*, not to recommend a
model size directly. Two machines in the same tier can still differ in what they can
run — bandwidth, bf16 support and driver all matter — so a tier narrows the search and
the concrete fit analysis in :mod:`qwen_distill.diagnostics.fit` decides.

Bands are deliberately named after the cards people actually have rather than round
powers of two, and the gaps between published bands (8-12, 16-20, 24-32 GB) resolve
downward: a 10 GB card gets Tier 1's expectations, not Tier 2's.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    """One capability band."""

    level: int
    name: str
    min_vram_gib: float
    max_vram_gib: float | None
    examples: str
    summary: str

    def contains(self, vram_gib: float | None) -> bool:
        if vram_gib is None:
            return self.level == 0
        if vram_gib < self.min_vram_gib:
            return False
        return self.max_vram_gib is None or vram_gib < self.max_vram_gib


#: Ordered low to high. ``max_vram_gib`` is exclusive; the last tier is open-ended.
TIERS: tuple[Tier, ...] = (
    Tier(0, "CPU only", 0.0, 0.01, "no accelerator",
         "correctness work only: unit tests, analysis, metadata verification"),
    Tier(1, "<=8 GB VRAM", 0.01, 12.0, "RX 480 8GB, GTX 1070, RTX 3060 8GB",
         "small quantised inference and toy-scale prototypes"),
    Tier(2, "12-16 GB VRAM", 12.0, 20.0, "Tesla T4, RTX 3060 12GB, RTX 4060 Ti 16GB",
         "the project's deployment target; small-student training and evaluation"),
    Tier(3, "20-24 GB VRAM", 20.0, 30.0, "RTX 3090, RTX 4090, RTX A5000, L4",
         "quantised teacher inference and mid-size student experiments"),
    Tier(4, "32 GB VRAM", 30.0, 38.0, "RTX 5090, V100 32GB",
         "headroom above 24 GB for longer contexts and larger batches"),
    Tier(5, "40-48 GB VRAM", 38.0, 60.0, "A40, RTX A6000, L40S, A100 40GB",
         "comfortable teacher baseline work; serious student training"),
    Tier(6, "64-80 GB+ VRAM", 60.0, None, "A100 80GB, H100, MI300X",
         "unquantised teacher inference and large-scale training"),
)


def classify(vram_gib: float | None) -> Tier:
    """Return the tier a device falls into. ``None`` VRAM means CPU-only."""
    if vram_gib is None or vram_gib < 0.01:
        return TIERS[0]
    for tier in TIERS:
        if tier.contains(vram_gib):
            return tier
    return TIERS[-1]


def tier_for_devices(devices: list) -> Tier:
    """Classify a machine by its largest single accelerator.

    Deliberately *not* the sum: without model/pipeline parallelism, a model must fit
    one device, and pretending two 8 GB cards are a 16 GB card would mislead exactly
    the users this tool exists to help.
    """
    # CPU rows carry system RAM in total_memory_gib so the display has something to
    # show; counting it as VRAM would classify a CPU-only laptop as a 16 GB GPU.
    memories = [
        d.total_memory_gib for d in devices
        if d.total_memory_gib and getattr(d, "vendor", None) != "cpu"
    ]
    if not memories:
        return TIERS[0]
    return classify(max(memories))
