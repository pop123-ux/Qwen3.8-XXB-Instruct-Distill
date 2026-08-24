"""Stage-by-stage GPU memory instrumentation for training runs.

The previous T4 calibration reported a measured/estimated ratio of ~2.85, which is not
usable as a correction factor: it compared *modelled tensor memory* against *allocator
reserve*, which are different quantities. Multiplying future estimates by 2.85 would
bake that confusion into the estimator permanently.

The fix is to measure the components separately. Snapshotting at each stage of the first
training step lets each term be attributed by **difference**:

    weights          = after_model_creation   - baseline
    optimizer state  = after_optimizer_step   - after_backward   (lazily allocated)
    activations      = after_forward          - after_model_creation
    gradients        = after_backward         - after_forward
    allocator reserve= reserved               - allocated
    runtime overhead = baseline (CUDA context, before any tensor)

`torch.cuda.memory_allocated` counts live tensors; `memory_reserved` counts what the
caching allocator holds from the driver. **Deployment claims must use reserved**, since
that is what the process actually occupies — but attribution must use allocated, since
reserve does not decompose.

Note that AdamW allocates its state lazily, on the first `step()`, so an optimizer-state
snapshot taken before that would read zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

GIB = 1024 ** 3

#: The stages worth snapshotting, in execution order.
#:
#: ``after_model_creation`` and ``after_model_to_device`` are separate because building
#: on CPU and moving to CUDA are separate allocations, and only the second shows on the
#: card. ``after_input_allocation`` and ``after_loss`` isolate two terms that the failed
#: Level-2 run made relevant: the batch itself, and the three copies of the logits the
#: loss path holds.
STAGES = (
    "baseline",
    "after_model_creation",
    "after_model_to_device",
    "after_optimizer_creation",
    "after_input_allocation",
    "after_forward",
    "after_loss",
    "after_backward",
    "after_optimizer_step",
    "peak_training",
)


@dataclass
class MemorySnapshot:
    """CUDA memory at one instant, in GiB."""

    stage: str
    allocated_gib: float
    reserved_gib: float
    max_allocated_gib: float
    max_reserved_gib: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryProfile:
    """Snapshots plus the components derived from them."""

    cuda_available: bool
    device_name: str | None = None
    total_vram_gib: float | None = None
    snapshots: list[MemorySnapshot] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def snapshot(self, stage: str) -> MemorySnapshot | None:
        return next((s for s in self.snapshots if s.stage == stage), None)

    @property
    def peak_allocated_gib(self) -> float | None:
        if not self.snapshots:
            return None
        return max(s.max_allocated_gib for s in self.snapshots)

    @property
    def peak_reserved_gib(self) -> float | None:
        if not self.snapshots:
            return None
        return max(s.max_reserved_gib for s in self.snapshots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cuda_available": self.cuda_available,
            "device_name": self.device_name,
            "total_vram_gib": self.total_vram_gib,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "components": self.components,
            "peak_allocated_gib": self.peak_allocated_gib,
            "peak_reserved_gib": self.peak_reserved_gib,
            "notes": self.notes,
        }


def _cuda_ready() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def new_profile() -> MemoryProfile:
    """Start a profile, recording the device it will describe."""
    profile = MemoryProfile(cuda_available=_cuda_ready())
    if not profile.cuda_available:
        profile.notes.append(
            "no CUDA device: memory instrumentation is unavailable. Figures are omitted "
            "rather than reported as zero, which would be indistinguishable from a real "
            "measurement of zero."
        )
        return profile
    import torch

    profile.device_name = torch.cuda.get_device_name(0)
    profile.total_vram_gib = torch.cuda.get_device_properties(0).total_memory / GIB
    return profile


def reset_peak() -> None:
    """Reset peak counters so a later snapshot measures a bounded window."""
    if not _cuda_ready():
        return
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def take(profile: MemoryProfile, stage: str) -> MemorySnapshot | None:
    """Record a snapshot, synchronising first so async work is counted."""
    if not profile.cuda_available:
        return None
    import torch

    torch.cuda.synchronize()
    snapshot = MemorySnapshot(
        stage=stage,
        allocated_gib=torch.cuda.memory_allocated() / GIB,
        reserved_gib=torch.cuda.memory_reserved() / GIB,
        max_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
        max_reserved_gib=torch.cuda.max_memory_reserved() / GIB,
    )
    profile.snapshots.append(snapshot)
    return snapshot


def derive_components(profile: MemoryProfile) -> dict[str, float]:
    """Attribute memory to weights / gradients / optimizer / activations by difference.

    Each term is a difference between consecutive stages, so it is measured rather than
    modelled. Missing stages simply omit their term instead of producing a wrong number.
    """
    if not profile.cuda_available:
        return {}

    def allocated(stage: str) -> float | None:
        snapshot = profile.snapshot(stage)
        return snapshot.allocated_gib if snapshot else None

    baseline = allocated("baseline")
    # Weights only appear on the card once the model is moved there, so prefer that
    # stage when it exists; a model built directly on CUDA has only the creation stage.
    model = allocated("after_model_to_device") or allocated("after_model_creation")
    inputs = allocated("after_input_allocation")
    forward = allocated("after_forward")
    loss = allocated("after_loss")
    backward = allocated("after_backward")
    stepped = allocated("after_optimizer_step")

    components: dict[str, float] = {}
    if baseline is not None:
        components["cuda_context_and_baseline_gib"] = baseline
    if model is not None and baseline is not None:
        components["weights_gib"] = max(0.0, model - baseline)
    if inputs is not None and model is not None:
        components["input_batch_gib"] = max(0.0, inputs - model)
    if forward is not None:
        # Measure activations from the last stage before the forward pass, so the input
        # batch is not counted twice.
        before_forward = inputs if inputs is not None else model
        if before_forward is not None:
            components["activations_gib"] = max(0.0, forward - before_forward)
    if loss is not None and forward is not None:
        components["loss_path_gib"] = max(0.0, loss - forward)
    reference_for_backward = loss if loss is not None else forward
    if backward is not None and reference_for_backward is not None:
        forward = reference_for_backward
    if backward is not None and forward is not None:
        # The backward pass frees activations as it consumes them and allocates
        # gradients, so this difference is a net figure, not gross gradient size.
        components["gradients_net_of_freed_activations_gib"] = backward - forward
    if stepped is not None and backward is not None:
        # AdamW allocates its moments lazily on the first step().
        components["optimizer_state_gib"] = max(0.0, stepped - backward)

    peak_allocated = profile.peak_allocated_gib
    peak_reserved = profile.peak_reserved_gib
    if peak_allocated is not None and peak_reserved is not None:
        components["peak_allocated_gib"] = peak_allocated
        components["peak_reserved_gib"] = peak_reserved
        components["allocator_reserve_overhead_gib"] = max(0.0, peak_reserved - peak_allocated)

    profile.components = components
    return components


def compare_with_estimate(
    profile: MemoryProfile, estimated_total_gib: float
) -> dict[str, Any]:
    """Compare measurement against the analytical estimate, both ways.

    Two ratios, because they answer different questions and conflating them is exactly
    what produced the unusable 2.85 figure:

    * ``ratio_allocated`` — do our *tensor* formulas match the tensors actually
      allocated? This is the one that should drive corrections to the estimator.
    * ``ratio_reserved`` — does the estimate predict the memory the process actually
      occupies? This is the one a deployment claim must satisfy.
    """
    if not profile.cuda_available or estimated_total_gib <= 0:
        return {"available": False}

    peak_allocated = profile.peak_allocated_gib or 0.0
    peak_reserved = profile.peak_reserved_gib or 0.0
    return {
        "available": True,
        "estimated_total_gib": estimated_total_gib,
        "measured_peak_allocated_gib": peak_allocated,
        "measured_peak_reserved_gib": peak_reserved,
        "ratio_allocated": peak_allocated / estimated_total_gib,
        "ratio_reserved": peak_reserved / estimated_total_gib,
        "interpretation": (
            "ratio_allocated compares modelled tensors against allocated tensors and is "
            "what should drive estimator corrections; ratio_reserved includes allocator "
            "reserve and CUDA context and is what a deployment claim must satisfy. "
            "Do not apply either as a blind multiplier."
        ),
    }


@dataclass
class OOMRecord:
    """An out-of-memory failure, recorded as data rather than lost to a traceback.

    A run that OOMs is an experimental result: it establishes an upper bound on what the
    configuration costs, which is exactly what the estimator needs. Throwing that away
    and retrying with a smaller batch discards the measurement.
    """

    phase: str
    device_name: str | None = None
    total_vram_gib: float | None = None
    allocated_at_failure_gib: float | None = None
    reserved_at_failure_gib: float | None = None
    free_at_failure_gib: float | None = None
    estimated_total_gib: float | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    @property
    def lower_bound_gib(self) -> float | None:
        """What the configuration provably needs *more* than.

        The allocation failed, so the true requirement exceeds what was held at the
        moment of failure. That is a real bound, and it is what makes an OOM useful.
        """
        return self.allocated_at_failure_gib

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lower_bound_gib"] = self.lower_bound_gib
        if self.estimated_total_gib and self.allocated_at_failure_gib:
            data["estimate_underestimated_by_at_least"] = (
                self.allocated_at_failure_gib / self.estimated_total_gib
            )
        return data

    def render(self) -> str:
        lines = [
            "OUT OF MEMORY — recorded as an experimental result",
            f"  phase                : {self.phase}",
            f"  device               : {self.device_name or 'unknown'}",
        ]
        if self.total_vram_gib:
            lines.append(f"  total VRAM           : {self.total_vram_gib:.2f} GiB")
        if self.allocated_at_failure_gib is not None:
            lines.append(f"  allocated at failure : {self.allocated_at_failure_gib:.2f} GiB")
        if self.reserved_at_failure_gib is not None:
            lines.append(f"  reserved at failure  : {self.reserved_at_failure_gib:.2f} GiB")
        if self.estimated_total_gib:
            lines.append(f"  estimated total      : {self.estimated_total_gib:.2f} GiB")
            if self.allocated_at_failure_gib:
                ratio = self.allocated_at_failure_gib / self.estimated_total_gib
                lines.append(
                    f"  the estimate was low by AT LEAST {ratio:.2f}x — the allocation "
                    "failed, so the true requirement is higher still"
                )
        return "\n".join(lines)


def is_oom(exc: BaseException) -> bool:
    """Whether an exception is a CUDA out-of-memory failure.

    Matches on the message as well as the type: some allocation failures surface as a
    plain ``RuntimeError`` depending on where in the allocator they are raised.
    """
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def record_oom(
    profile: MemoryProfile,
    phase: str,
    exc: BaseException,
    *,
    estimated_total_gib: float | None = None,
    configuration: dict[str, Any] | None = None,
) -> OOMRecord:
    """Capture the memory state at the moment an allocation failed."""
    record = OOMRecord(
        phase=phase,
        device_name=profile.device_name,
        total_vram_gib=profile.total_vram_gib,
        estimated_total_gib=estimated_total_gib,
        configuration=configuration or {},
        message=str(exc)[:2000],
    )
    if not profile.cuda_available:
        return record
    try:
        import torch

        record.allocated_at_failure_gib = torch.cuda.memory_allocated() / GIB
        record.reserved_at_failure_gib = torch.cuda.memory_reserved() / GIB
        free, _total = torch.cuda.mem_get_info()
        record.free_at_failure_gib = free / GIB
    except Exception:  # noqa: BLE001 - never fail while reporting a failure
        pass
    return record
