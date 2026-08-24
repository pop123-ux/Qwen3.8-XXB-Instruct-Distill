"""What may change between a checkpoint and the run resuming from it.

Not every configuration difference is equal. Three kinds:

**Fatal** — the checkpoint cannot be loaded onto this run at all. A different
architecture means the weights do not fit; a different sequence length or batch size
means the saved data position addresses sequences that no longer exist. Loading anyway
would either crash somewhere confusing or, worse, silently train on the wrong data.

**Extends the schedule** — ``max_steps`` is the intended *total* training length, so
raising it is a normal thing to want: train 20 steps, look at the curve, continue to 40.
But OneCycleLR's learning rate at step *t* is a function of ``total_steps``, so its saved
state cannot simply be replayed against a longer horizon. The schedule is rebuilt for the
new length and fast-forwarded to the restored step. That is a real change to the LR curve
and it is reported, never silent.

**Cosmetic** — a renamed run, a different log interval. Carry on.

The distinction is the whole point. Rejecting everything blocks a legitimate workflow;
accepting everything corrupts runs quietly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Config paths that must match exactly. A difference here means the checkpoint's
#: tensors or its saved data position do not describe this run.
FATAL_KEYS: tuple[tuple[str, ...], ...] = (
    ("model", "architecture"),
    ("model", "pretrained"),
    ("model", "spec_path"),
    ("data", "max_sequence_length"),
    ("training", "batch_size"),
    ("training", "strategy"),
)

#: Differences that change what is being optimised. Allowed, because a resumed run may
#: legitimately want them, but never applied silently.
NOTABLE_KEYS: tuple[tuple[str, ...], ...] = (
    ("training", "learning_rate"),
    ("training", "weight_decay"),
    ("training", "optimizer"),
    ("training", "precision"),
    ("training", "gradient_accumulation_steps"),
    ("training", "gradient_checkpointing"),
    ("data", "text_path"),
    ("data", "procedural_bytes"),
    ("data", "shuffle_seed"),
)

#: Why each fatal key matters, so the error explains rather than just refuses.
FATAL_REASONS: dict[tuple[str, ...], str] = {
    ("model", "architecture"): "the checkpoint's tensors have different shapes",
    ("model", "pretrained"): "the checkpoint was initialised from a different base model",
    ("model", "spec_path"): "the checkpoint was built from a different architecture spec",
    ("data", "max_sequence_length"): (
        "the corpus is chunked by sequence length, so the saved data position points at "
        "sequences that no longer exist"
    ),
    ("training", "batch_size"): (
        "the saved data position is a batch index, which does not carry across a "
        "different batch size"
    ),
    ("training", "strategy"): "a different training strategy holds different parameters",
}


def _get(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


@dataclass
class ResumeCompatibility:
    """Whether a checkpoint's config and the current one can be reconciled."""

    saved_max_steps: int | None = None
    requested_max_steps: int | None = None
    fatal: list[str] = field(default_factory=list)
    notable: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.fatal

    @property
    def extends_schedule(self) -> bool:
        """Whether the LR schedule has to be rebuilt for a new horizon."""
        return (
            self.saved_max_steps is not None
            and self.requested_max_steps is not None
            and self.saved_max_steps != self.requested_max_steps
        )

    def render(self) -> str:
        lines: list[str] = []
        if self.fatal:
            # A fatal mismatch ends the resume, so notes about the LR schedule would be
            # noise on top of the reason it stopped.
            lines.append("  This checkpoint cannot be resumed onto the current config:")
            lines += [f"    - {item}" for item in self.fatal]
            lines.append(
                "  Weights, the optimizer or the saved data position would not match. "
                "Start a new run directory instead."
            )
            return "\n".join(lines)
        if self.extends_schedule:
            direction = (
                "extended" if (self.requested_max_steps or 0) > (self.saved_max_steps or 0)
                else "shortened"
            )
            lines.append(
                f"    training target {direction}: {self.saved_max_steps} -> "
                f"{self.requested_max_steps} steps"
            )
            lines.append(
                "    the one-cycle LR schedule is rebuilt for the new total and "
                "fast-forwarded;"
            )
            lines.append(
                "    remaining steps follow the new curve, not the original one"
            )
        if self.notable:
            lines.append("    changed since the checkpoint (allowed, applied as given):")
            lines += [f"      - {item}" for item in self.notable]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "saved_max_steps": self.saved_max_steps,
            "requested_max_steps": self.requested_max_steps,
            "extends_schedule": self.extends_schedule,
            "fatal": self.fatal,
            "notable": self.notable,
        }


def check_resume_compatibility(
    saved_config: dict[str, Any] | None, current_config: dict[str, Any]
) -> ResumeCompatibility:
    """Classify the differences between a checkpoint's config and this run's."""
    result = ResumeCompatibility(
        requested_max_steps=_get(current_config, ("training", "max_steps"))
    )
    if not saved_config:
        # No saved config to compare against: an older checkpoint, or one written by
        # hand. Proceed rather than refuse, but the schedule cannot be reconciled.
        return result

    result.saved_max_steps = _get(saved_config, ("training", "max_steps"))

    for path in FATAL_KEYS:
        before, after = _get(saved_config, path), _get(current_config, path)
        if before != after:
            reason = FATAL_REASONS.get(path, "this value must match")
            result.fatal.append(f"{'.'.join(path)}: {before!r} -> {after!r} — {reason}")

    for path in NOTABLE_KEYS:
        before, after = _get(saved_config, path), _get(current_config, path)
        if before != after:
            result.notable.append(f"{'.'.join(path)}: {before!r} -> {after!r}")

    return result


def rebuild_schedule(scheduler: Any, optimizer: Any, *, total_steps: int, completed: int,
                     max_lr: float, pct_start: float = 0.1) -> Any:
    """Return a OneCycleLR for ``total_steps``, advanced to ``completed`` steps.

    OneCycleLR computes its LR from ``last_epoch`` against ``total_steps``, so extending
    a run means a new schedule object rather than a patched old one. Fast-forwarding by
    stepping is exact and cheap — it touches learning rates, not parameters — and it
    leaves the optimizer's own state untouched.
    """
    import warnings

    import torch

    fresh = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=max(total_steps, 1), pct_start=pct_start,
    )
    with warnings.catch_warnings():
        # PyTorch warns when scheduler.step() precedes optimizer.step(), because that
        # usually means a mis-ordered training loop. Here it is a deliberate
        # fast-forward before training resumes: no parameter is touched, only the
        # schedule's position.
        warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
        for _ in range(min(completed, total_steps)):
            fresh.step()
    return fresh
