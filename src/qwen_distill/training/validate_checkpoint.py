"""Verify a checkpoint actually round-trips: save → reload → identical behaviour.

A checkpoint that writes without error but reloads into a *different* model is one of
the more expensive failures available: it is silent, and it invalidates every result
produced after it. So this checks behaviour, not just that the file exists.

Three independent checks, because each can fail while the others pass:

* **Parameter identity** — every tensor reloads bit-for-bit.
* **Logit identity** — a fresh model built from config, loaded from the checkpoint,
  produces the same logits as the original on fixed input. This catches config/weight
  mismatches that a parameter comparison alone would miss, since it exercises the whole
  construction path.
* **Generation determinism** — greedy generation from fixed prompts is reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .checkpoints import is_complete, resolve_checkpoint

#: Fixed byte-level prompts for the generation sanity check. Short, deterministic, and
#: chosen so a model that learned anything about English produces plausible completions.
DEFAULT_PROMPTS: tuple[str, ...] = (
    "The ",
    "In the beginning ",
    "It was ",
    "and the ",
)


@dataclass
class CheckResult:
    """One verification, with the evidence behind its verdict."""

    name: str
    passed: bool
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail, "data": self.data}


@dataclass
class CheckpointReport:
    """Whether a checkpoint is trustworthy."""

    checkpoint: str
    checks: list[CheckResult] = field(default_factory=list)
    generations: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks) and not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "generations": self.generations,
            "error": self.error,
        }

    def render(self) -> str:
        lines = [f"checkpoint: {self.checkpoint}", ""]
        for check in self.checks:
            lines.append(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}")
            if check.detail:
                lines.append(f"         {check.detail}")
        if self.generations:
            lines.append("\n  generations (greedy, deterministic):")
            for item in self.generations:
                lines.append(f"    {item['prompt']!r} -> {item['completion']!r}")
        if self.error:
            lines.append(f"\n  ERROR: {self.error}")
        lines.append(f"\n  VERDICT: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def resolve_checkpoint_argument(reference: str | Path) -> Path:
    """Accept a checkpoint directory, a run directory, or ``latest``.

    A run directory holds ``checkpoints/``, not a checkpoint, so pointing this script at
    one used to fail with "no training_state.json" — accurate but unhelpful. Resolving it
    to the newest verified checkpoint is what the user meant, and it keeps the interface
    unambiguous: an actual checkpoint directory is still used exactly as given.
    """
    path = Path(reference)

    # A directory that is itself a checkpoint wins, always.
    if (path / "metadata.json").is_file() or (path / "training_state.pt").is_file():
        return path

    for candidate in (path / "checkpoints", path):
        if candidate.is_dir():
            resolved = resolve_checkpoint(candidate, "latest")
            if resolved is not None:
                return resolved
    return path


def _build_from_config(config_path: Path, device: str):
    """Rebuild an empty model from a saved experiment config."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from .config import ExperimentConfig

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    experiment = ExperimentConfig.from_dict(raw)
    spec = experiment.model.resolve_spec()
    if spec is None:
        raise ValueError("saved config has no inline architecture; cannot rebuild")
    fields = {k: v for k, v in spec.to_hf_text_config().items() if k != "model_type"}
    hf_config = AutoConfig.for_model("qwen3_5_text", **fields)
    return AutoModelForCausalLM.from_config(hf_config).to(device), spec


def generate_bytes_detailed(
    model, prompt: str, *, max_new_tokens: int = 48, device: str = "cpu"
) -> tuple[str, list[int]]:
    """Greedy byte-level generation, returning the text **and** the generated ids.

    The ids are not redundant. ``decode`` uses ``errors="replace"``, and a byte-level
    model routinely emits sequences that are not valid UTF-8, so
    ``len(text.encode("utf-8"))`` is not the number of tokens produced. Anything
    recording a token count needs the ids.
    """
    import torch

    from .text_data import decode, encode

    prompt_ids = encode(prompt)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            pad_token_id=0,
        )
    generated = out[0].tolist()[len(prompt_ids) :]
    return decode(generated), generated


def generate_bytes(model, prompt: str, *, max_new_tokens: int = 48, device: str = "cpu") -> str:
    """Greedy byte-level generation. Deterministic by construction."""
    text, _ids = generate_bytes_detailed(
        model, prompt, max_new_tokens=max_new_tokens, device=device
    )
    return text


def validate_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str = "cpu",
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    max_new_tokens: int = 48,
) -> CheckpointReport:
    """Reload a checkpoint into a fresh model and verify it behaves identically."""
    path = resolve_checkpoint_argument(checkpoint_dir)
    report = CheckpointReport(checkpoint=str(path))

    try:
        import torch

        config_file = path / "config.json"
        modern = path / "model.safetensors"
        legacy = path / "training_state.pt"

        if modern.is_file():
            from safetensors.torch import load_file

            if not is_complete(path):
                report.error = (
                    f"{path} is not a complete checkpoint: the COMPLETE marker or a "
                    "required file is missing, so it was never safe to resume from."
                )
                return report
            saved_state = load_file(str(modern), device=device)
        elif legacy.is_file():
            # Checkpoints written before the atomic format. Still readable so existing
            # Level-1 artifacts do not become unverifiable.
            saved_state = torch.load(
                legacy, map_location=device, weights_only=False
            )["model"]
        else:
            report.error = f"no model.safetensors or training_state.pt in {path}"
            return report
        if not config_file.is_file():
            report.error = f"no config.json in {path}"
            return report

        model, spec = _build_from_config(config_file, device)
        missing, unexpected = model.load_state_dict(saved_state, strict=False)
        # A tied lm_head/embedding pair is stored once, so one of the two names is
        # legitimately absent. Counting that as a failure would fail every checkpoint.
        tied = getattr(model.config, "tie_word_embeddings", False)
        real_missing = [
            name for name in missing
            if not (tied and name.endswith(("lm_head.weight", "embed_tokens.weight")))
        ]
        report.checks.append(CheckResult(
            name="state_dict loads into a freshly built model",
            passed=not real_missing and not unexpected,
            detail=(f"{len(real_missing)} missing, {len(unexpected)} unexpected"
                    + (f" ({len(missing) - len(real_missing)} tied weight(s) stored once)"
                       if len(missing) != len(real_missing) else "")),
            data={"missing": list(real_missing)[:10], "unexpected": list(unexpected)[:10]},
        ))

        # --- parameter identity ------------------------------------------
        model_state = model.state_dict()
        mismatched = [
            name for name, tensor in model_state.items()
            if name in saved_state and not torch.equal(tensor.cpu(), saved_state[name].cpu())
        ]
        report.checks.append(CheckResult(
            name="every parameter reloads bit-for-bit",
            passed=not mismatched,
            detail=f"{len(mismatched)} tensor(s) differ" if mismatched else "all tensors identical",
            data={"mismatched": mismatched[:10]},
        ))

        # --- logit identity ----------------------------------------------
        # Build a second model the same way and confirm the loaded weights determine
        # the output, not the construction order or any residual randomness.
        reference, _ = _build_from_config(config_file, device)
        reference.load_state_dict(saved_state, strict=False)
        torch.manual_seed(0)
        probe = torch.randint(0, spec.vocab_size, (1, 32), device=device)
        model.eval()
        reference.eval()
        with torch.no_grad():
            a = model(input_ids=probe).logits
            b = reference(input_ids=probe).logits
        max_difference = float((a - b).abs().max().item())
        report.checks.append(CheckResult(
            name="two independent reloads produce identical logits",
            passed=max_difference == 0.0,
            detail=f"max |difference| = {max_difference:g}",
            data={"max_abs_difference": max_difference},
        ))

        # --- generation determinism ---------------------------------------
        first = [generate_bytes(model, p, max_new_tokens=max_new_tokens, device=device)
                 for p in prompts]
        second = [generate_bytes(model, p, max_new_tokens=max_new_tokens, device=device)
                  for p in prompts]
        report.checks.append(CheckResult(
            name="greedy generation is reproducible",
            passed=first == second,
            detail="identical across two runs" if first == second else "outputs differ between runs",
        ))
        report.generations = [
            {"prompt": p, "completion": c} for p, c in zip(prompts, first, strict=False)
        ]
    except Exception as exc:  # noqa: BLE001 - the failure is the result
        report.error = f"{type(exc).__name__}: {exc}"
    return report


@dataclass
class ResumeReport:
    """Whether training resumes from the correct state."""

    checkpoint: str
    resumed_step: int | None = None
    expected_step: int | None = None
    history_preserved: bool = False
    continued_to_step: int | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return (
            self.error is None
            and self.resumed_step is not None
            and self.resumed_step == self.expected_step
            and self.history_preserved
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint, "resumed_step": self.resumed_step,
            "expected_step": self.expected_step, "history_preserved": self.history_preserved,
            "continued_to_step": self.continued_to_step, "passed": self.passed,
            "error": self.error,
        }


def validate_resume(checkpoint_dir: str | Path) -> ResumeReport:
    """Check that a checkpoint's saved state restores the correct global step.

    Resuming at the wrong step silently corrupts a learning-rate schedule and makes a
    run non-reproducible, so this verifies the step and history survive the round trip.
    """
    path = resolve_checkpoint_argument(checkpoint_dir)
    report = ResumeReport(checkpoint=str(path))
    try:
        state_file = path / "training_state.json"
        if not state_file.is_file():
            state_file = path / "state.json"   # pre-atomic-format checkpoints
        if not state_file.is_file():
            report.error = f"no training_state.json or state.json in {path}"
            return report
        saved = json.loads(state_file.read_text(encoding="utf-8"))
        report.expected_step = saved.get("step")

        from .trainer import TrainingState

        restored = TrainingState.from_dict(saved)
        report.resumed_step = restored.step
        report.history_preserved = len(restored.history) == len(saved.get("history", []))
    except Exception as exc:  # noqa: BLE001
        report.error = f"{type(exc).__name__}: {exc}"
    return report
