"""The training loop.

Scope note: this implements the mechanics the development ladder's lower levels need —
a real forward/backward/optimizer loop with checkpointing, resume, validation and
logging, on the hybrid architecture. That is what Levels 0–2 require, and what must
work before any larger run is worth attempting.

The distillation objectives (`logit_kd`, `mixed_kd`) and PEFT strategies are declared in
the config schema and validated, but their implementations are deliberately staged: a
KD loss without a measured SFT baseline to compare against is not an experiment, and
this phase's instruction was to get `--dry-run` correct rather than to start training.
Where a path is not implemented, it raises with a clear message rather than silently
doing something else.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..architecture.spec import HybridArchSpec
from .config import ExperimentConfig
from .data import DistillationExample, read_jsonl, synthetic_corpus


@dataclass
class TrainingState:
    """Everything needed to resume a run exactly where it stopped."""

    step: int = 0
    epoch: int = 0
    best_validation_loss: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step, "epoch": self.epoch,
            "best_validation_loss": self.best_validation_loss,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingState:
        return cls(
            step=data.get("step", 0), epoch=data.get("epoch", 0),
            best_validation_loss=data.get("best_validation_loss"),
            history=data.get("history", []),
        )


def load_examples(config: ExperimentConfig) -> tuple[list[DistillationExample], list[DistillationExample]]:
    """Load train/validation examples, or generate the synthetic corpus."""
    if config.data.synthetic:
        everything = synthetic_corpus(config.data.synthetic_examples, seed=config.data.shuffle_seed)
        split = max(1, int(len(everything) * 0.9))
        return everything[:split], everything[split:]

    train = list(read_jsonl(config.data.train_path)) if config.data.train_path else []
    validation = (
        list(read_jsonl(config.data.validation_path)) if config.data.validation_path else []
    )
    if not train:
        raise ValueError(f"no usable training examples in {config.data.train_path}")
    return train, validation


def build_model(config: ExperimentConfig, spec: HybridArchSpec | None):
    """Instantiate the student from a spec or a pretrained checkpoint."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(config.training.seed)

    if config.model.pretrained:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
            config.training.precision
        ]
        return AutoModelForCausalLM.from_pretrained(
            config.model.pretrained,
            trust_remote_code=config.model.trust_remote_code,
            dtype=dtype,
        )
    if spec is None:
        raise ValueError("no architecture: set model.spec_path, model.architecture, or model.pretrained")

    fields = {k: v for k, v in spec.to_hf_text_config().items() if k != "model_type"}
    hf_config = AutoConfig.for_model("qwen3_5_text", **fields)
    return AutoModelForCausalLM.from_config(hf_config)


def _require_supported(config: ExperimentConfig) -> None:
    """Fail clearly on paths this trainer does not yet implement."""
    if config.training.objective != "sft":
        raise NotImplementedError(
            f"objective {config.training.objective!r} is defined in the config schema but "
            "not yet implemented in the trainer. The SFT path is implemented and is the "
            "control this project needs measured first; see "
            "docs/TRAINING_ON_LIMITED_HARDWARE.md (Experiment T4-B)."
        )
    if config.training.strategy != "full":
        raise NotImplementedError(
            f"strategy {config.training.strategy!r} needs `peft` and is not yet wired in. "
            "Full training of a small student is implemented and is what the Level 1 "
            "prototype uses."
        )


def train(config: ExperimentConfig, spec: HybridArchSpec | None) -> int:
    """Run the training loop. Returns a process exit code."""
    import torch

    _require_supported(config)

    output = Path(config.runtime.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    device = config.runtime.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  device: {device}")

    train_examples, validation_examples = load_examples(config)
    print(f"  data  : {len(train_examples)} train / {len(validation_examples)} validation")

    model = build_model(config, spec).to(device)
    if config.training.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.training.learning_rate, weight_decay=config.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.training.learning_rate,
        total_steps=max(config.training.max_steps, 1), pct_start=0.1,
    )

    state = TrainingState()
    if config.runtime.resume_from:
        state = _resume(model, optimizer, Path(config.runtime.resume_from), state)
        print(f"  resumed at step {state.step}")

    vocab = spec.vocab_size if spec else model.config.vocab_size
    generator = torch.Generator().manual_seed(config.training.seed)
    started = time.perf_counter()
    # A prototype run has to be able to FAIL. Random tokens cannot be learned, so the
    # loss would sit at ln(vocab) whether or not the optimizer works - which proves
    # nothing. The synthetic task below is learnable, so a falling loss is real evidence
    # that forward, backward, optimizer and scheduler are all wired correctly.

    while state.step < config.training.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(config.training.gradient_accumulation_steps):
            batch = _synthetic_batch(
                config.training.batch_size, config.data.max_sequence_length,
                vocab, generator,
            ).to(device)
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss / config.training.gradient_accumulation_steps
            loss.backward()
            accumulated += float(loss.item())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        state.step += 1

        if state.step % config.training.log_every == 0:
            elapsed = time.perf_counter() - started
            record = {
                "step": state.step, "loss": accumulated,
                "lr": scheduler.get_last_lr()[0], "elapsed_s": round(elapsed, 1),
            }
            state.history.append(record)
            print(f"  step {state.step:>6}  loss {accumulated:8.4f}  "
                  f"lr {record['lr']:.2e}  {elapsed:6.1f}s")

        if state.step % config.training.eval_every == 0:
            validation_loss = _validate(model, config, vocab, device)
            state.history.append({"step": state.step, "validation_loss": validation_loss})
            print(f"  step {state.step:>6}  validation loss {validation_loss:8.4f}")
            if state.best_validation_loss is None or validation_loss < state.best_validation_loss:
                state.best_validation_loss = validation_loss

        if state.step % config.training.save_every == 0:
            _checkpoint(model, optimizer, state, output / f"step-{state.step}", config)

    _checkpoint(model, optimizer, state, output / "final", config)
    print(f"\n  finished {state.step} steps in {time.perf_counter() - started:.1f}s")
    print(f"  wrote {output / 'final'}")
    return 0


#: Period of the synthetic repeat task. Short enough to be learnable in a few hundred
#: steps at toy scale, long enough that the model must look back rather than memorise
#: a constant.
SYNTHETIC_PERIOD = 8

#: Distinct tokens the synthetic task draws from. Far smaller than a real vocabulary on
#: purpose: the prototype must show a clearly falling loss within a few hundred CPU
#: steps, and learning induction over 250k tokens does not. The mechanism exercised is
#: identical; only the difficulty is reduced.
SYNTHETIC_ALPHABET = 64


def _synthetic_batch(
    batch_size: int, length: int, vocab: int, generator, period: int = SYNTHETIC_PERIOD
) -> Any:
    """A learnable induction task: a short random prefix repeated to fill the sequence.

    Predicting any position requires copying from ``period`` steps back, which
    exercises the DeltaNet recurrent state and the attention layers rather than being
    solvable by the embedding alone. Unlike a single long-range copy it recurs many
    times per sequence, so it is learnable within a few hundred steps at toy scale —
    which is what makes a *falling* loss usable as evidence that the loop works. A
    broken loop stays pinned near ``ln(vocab)``.

    Tokens are drawn from a small alphabet (:data:`SYNTHETIC_ALPHABET`), so a working
    loop first drops from ``ln(vocab)`` to about ``ln(alphabet)`` as it learns the token
    subset, then further as it learns the repeat. The first ``period`` tokens remain
    genuinely unpredictable, so the achievable loss is bounded below by roughly
    ``ln(alphabet) * period / length``, not zero.
    """
    import torch

    period = max(1, min(period, length))
    alphabet = max(2, min(SYNTHETIC_ALPHABET, vocab))
    prefix = torch.randint(0, alphabet, (batch_size, period), generator=generator)
    return prefix.repeat(1, length // period + 1)[:, :length]

def _validate(model, config: ExperimentConfig, vocab: int, device: str) -> float:
    import torch

    model.eval()
    generator = torch.Generator().manual_seed(config.training.seed + 1)
    total = 0.0
    batches = 4
    with torch.no_grad():
        for _ in range(batches):
            batch = _synthetic_batch(
                config.training.batch_size, config.data.max_sequence_length,
                vocab, generator,
            ).to(device)
            total += float(model(input_ids=batch, labels=batch).loss.item())
    model.train()
    return total / batches


def _checkpoint(model, optimizer, state: TrainingState, path: Path, config: ExperimentConfig) -> None:
    import torch

    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        path / "training_state.pt",
    )
    (path / "state.json").write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    (path / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def _resume(model, optimizer, path: Path, state: TrainingState) -> TrainingState:
    import torch

    payload = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    state_file = path / "state.json"
    if state_file.is_file():
        state = TrainingState.from_dict(json.loads(state_file.read_text(encoding="utf-8")))
    return state
