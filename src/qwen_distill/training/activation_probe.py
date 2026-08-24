"""Attribute retained-activation memory to the modules that create it.

`memory_probe` measures CUDA memory at stage boundaries, which answers *how much* but
not *what*. When the Level-2 T4 run OOMed at 24.8 GiB against a 4.53 GiB estimate, the
gap was 66% inside the Gated DeltaNet mixers — and no stage-boundary measurement could
have said so.

This works by hooking `torch.autograd.graph.saved_tensors_hooks`, which fires for every
tensor the autograd graph retains for the backward pass, and attributing each to the
module executing at the time. That set *is* the activation memory: it is what a
no-checkpointing run holds at peak.

**No GPU is required.** Tensor shapes and dtypes do not depend on the device, so the
attribution measured on CPU is the one a GPU would allocate. That matters: it makes this
a pre-flight check rather than a post-mortem.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

from ..architecture.spec import HybridArchSpec

MIB = 1024 ** 2
GIB = 1024 ** 3


@dataclass
class ScopeUsage:
    """Retained activation bytes attributed to one kind of module."""

    scope: str
    bytes_retained: int
    tensor_count: int

    @property
    def mib(self) -> float:
        return self.bytes_retained / MIB

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "bytes_retained": self.bytes_retained,
            "mib": self.mib,
            "tensor_count": self.tensor_count,
        }


@dataclass
class ActivationProfile:
    """What a forward pass retains, and where it came from."""

    batch_size: int
    sequence_length: int
    scopes: list[ScopeUsage] = field(default_factory=list)
    gradient_checkpointing: bool = False
    error: str | None = None

    @property
    def total_bytes(self) -> int:
        return sum(s.bytes_retained for s in self.scopes)

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB

    def scope(self, name: str) -> ScopeUsage | None:
        return next((s for s in self.scopes if s.scope == name), None)

    def dominant(self) -> ScopeUsage | None:
        return max(self.scopes, key=lambda s: s.bytes_retained, default=None)

    def bytes_per_token(self, scope: str) -> float:
        usage = self.scope(scope)
        tokens = self.batch_size * self.sequence_length
        return usage.bytes_retained / tokens if usage and tokens else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "gradient_checkpointing": self.gradient_checkpointing,
            "total_bytes": self.total_bytes,
            "total_gib": self.total_gib,
            "scopes": [s.to_dict() for s in self.scopes],
            "dominant_scope": self.dominant().scope if self.dominant() else None,
            "error": self.error,
        }

    def render(self) -> str:
        lines = [
            f"retained activations at batch={self.batch_size} seq={self.sequence_length}"
            + (" (gradient checkpointing ON)" if self.gradient_checkpointing else ""),
            "",
            f"  {'scope':<26}{'MiB':>10}{'tensors':>10}{'B/token':>12}",
        ]
        for usage in sorted(self.scopes, key=lambda s: -s.bytes_retained):
            per_token = self.bytes_per_token(usage.scope)
            lines.append(
                f"  {usage.scope:<26}{usage.mib:>10.1f}{usage.tensor_count:>10}{per_token:>12,.0f}"
            )
        lines.append(f"  {'TOTAL':<26}{self.total_bytes / MIB:>10.1f}"
                     f"{sum(s.tensor_count for s in self.scopes):>10}")
        if self.error:
            lines.append(f"\n  ERROR: {self.error}")
        return "\n".join(lines)


def probe_activations(
    spec: HybridArchSpec,
    *,
    batch_size: int = 1,
    sequence_length: int = 1024,
    gradient_checkpointing: bool = False,
    device: str = "cpu",
) -> ActivationProfile:
    """Run one forward pass and attribute every retained tensor to its module."""
    profile = ActivationProfile(
        batch_size=batch_size, sequence_length=sequence_length,
        gradient_checkpointing=gradient_checkpointing,
    )
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        fields = {k: v for k, v in spec.to_hf_text_config().items() if k != "model_type"}
        model = AutoModelForCausalLM.from_config(AutoConfig.for_model("qwen3_5_text", **fields))
        model = model.to(device)
        model.train()
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        scope_stack = ["embedding+norms+logits"]
        totals: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        seen: set[tuple[int, tuple[int, ...], Any]] = set()

        def register(module, name: str) -> None:
            def pre(_m, _a):
                scope_stack.append(name)

            def post(_m, _a, _o):
                scope_stack.pop()  # must return None, or it replaces the module output

            module.register_forward_pre_hook(pre)
            module.register_forward_hook(post)

        for layer in model.model.layers:
            is_full = hasattr(layer, "self_attn")
            kind = "full_attention" if is_full else "deltanet"
            register(getattr(layer, "self_attn", None) or layer.linear_attn, f"{kind}.mixer")
            register(layer.mlp, f"{kind}.mlp")

        class Attribute(torch.autograd.graph.saved_tensors_hooks):
            def __init__(self) -> None:
                super().__init__(self.pack, lambda x: x)

            def pack(self, tensor):
                # Deduplicate: one storage saved by several ops is one allocation.
                key = (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
                if key not in seen:
                    seen.add(key)
                    entry = totals[scope_stack[-1]]
                    entry[0] += tensor.numel() * tensor.element_size()
                    entry[1] += 1
                return tensor

        ids = torch.randint(0, spec.vocab_size, (batch_size, sequence_length), device=device)
        with Attribute():
            model(input_ids=ids, labels=ids)

        profile.scopes = [
            ScopeUsage(scope=scope, bytes_retained=data[0], tensor_count=data[1])
            for scope, data in totals.items()
        ]
    except Exception as exc:  # noqa: BLE001 - the failure is the result
        profile.error = f"{type(exc).__name__}: {exc}"
    return profile


def scaling_study(
    spec: HybridArchSpec,
    *,
    batch_sizes: tuple[int, ...] = (1, 2),
    sequence_length: int = 1024,
    gradient_checkpointing: bool = False,
) -> dict[str, Any]:
    """Fit retained activations against batch size, then extrapolate.

    Two points give the intercept and slope of a term that is linear in batch, which is
    how a batch that will not fit can be ruled out without ever allocating it — the
    check the failed Level-2 run needed and did not have.
    """
    profiles = [
        probe_activations(spec, batch_size=b, sequence_length=sequence_length,
                          gradient_checkpointing=gradient_checkpointing)
        for b in batch_sizes
    ]
    failed = [p for p in profiles if p.error]
    if failed or len(profiles) < 2:
        return {"available": False, "error": failed[0].error if failed else "need >= 2 points"}

    (b0, t0), (b1, t1) = ((p.batch_size, p.total_bytes) for p in profiles[:2])
    slope = (t1 - t0) / (b1 - b0)
    intercept = t0 - slope * b0
    return {
        "available": True,
        "sequence_length": sequence_length,
        "gradient_checkpointing": gradient_checkpointing,
        "measured": [p.to_dict() for p in profiles],
        "model": {"intercept_bytes": intercept, "bytes_per_batch": slope},
        "extrapolated_gib": {
            str(b): (intercept + slope * b) / GIB for b in (1, 2, 4, 8, 16)
        },
    }
