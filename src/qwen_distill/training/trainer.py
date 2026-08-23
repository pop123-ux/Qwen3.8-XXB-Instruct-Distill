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

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..architecture.spec import HybridArchSpec
from .config import ExperimentConfig
from .data import DistillationExample, read_jsonl, synthetic_corpus
from .memory_probe import (
    compare_with_estimate,
    derive_components,
    new_profile,
    reset_peak,
    take,
)
from .text_data import BYTE_VOCAB_SIZE, bits_per_byte, iterate_batches, prepare_corpus


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


def resolve_precision(precision: str, device: str) -> tuple[str, str | None]:
    """Decide what precision will *actually* be used, and say so.

    Building a model with ``AutoModelForCausalLM.from_config`` ignores the configured
    precision entirely — it always produces fp32 — so a run declaring ``fp16`` silently
    trained in fp32 at roughly twice the modelled weight/gradient/optimizer memory. This
    resolves the request against the device and returns ``(effective, note)`` so the
    discrepancy is reported rather than discovered from an OOM.

    The mechanism is **autocast**, not fp16 weights: AdamW on fp16 parameters
    underflows, so mixed precision keeps fp32 master weights and casts only the
    forward/backward compute. That is what a T4 wants, and what GradScaler exists for.
    """
    if precision == "fp32":
        return "fp32", None
    if device == "cpu":
        return "fp32", (
            f"precision {precision} requested but the device is CPU, where autocast "
            "covers few operations and is usually slower; training in fp32"
        )
    if precision == "bf16":
        import torch

        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            return "fp16", (
                f"{torch.cuda.get_device_name(0)} does not support bf16; "
                "falling back to fp16 autocast with gradient scaling"
            )
    return precision, None


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

    profile = new_profile()
    reset_peak()
    take(profile, "baseline")

    # --- data ---------------------------------------------------------
    corpus_stats = None
    text_mode = config.data.mode == "text"
    if text_mode:
        train_sequences, validation_sequences, corpus_stats = prepare_corpus(
            text_path=config.data.text_path,
            sequence_length=config.data.max_sequence_length,
            procedural_bytes=config.data.procedural_bytes,
            validation_fraction=config.data.validation_fraction,
            seed=config.data.shuffle_seed,
            max_bytes=config.data.max_corpus_bytes,
        )
        print(f"  corpus: {corpus_stats.source}")
        print(f"          {corpus_stats.n_bytes:,} bytes, {corpus_stats.n_sequences} sequences "
              f"of {corpus_stats.sequence_length} "
              f"({corpus_stats.n_train} train / {corpus_stats.n_validation} validation)")
        print(f"          sha256 {corpus_stats.sha256[:16]}")
        batches = iterate_batches(
            train_sequences, config.training.batch_size, seed=config.training.seed
        )
    else:
        train_examples, validation_examples = load_examples(config)
        print(f"  data  : {len(train_examples)} train / {len(validation_examples)} validation")

    model = build_model(config, spec).to(device)
    if config.training.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    take(profile, "after_model_creation")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.training.learning_rate, weight_decay=config.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.training.learning_rate,
        total_steps=max(config.training.max_steps, 1), pct_start=0.1,
    )
    take(profile, "after_optimizer_creation")

    # Mixed precision, applied to the compute rather than to the weights. Under
    # autocast the master weights and gradients stay fp32 while matmuls run in
    # fp16/bf16; GradScaler then keeps fp16 gradients from underflowing to zero.
    precision, precision_note = resolve_precision(config.training.precision, device)
    if precision_note:
        print(f"  note  : {precision_note}")
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    use_amp = autocast_dtype is not None
    scaler = torch.amp.GradScaler(device, enabled=precision == "fp16")
    print(f"  precision: {precision}" + (" (autocast)" if use_amp else ""))

    def autocast():
        return (
            torch.autocast(device_type=device, dtype=autocast_dtype)
            if use_amp else contextlib.nullcontext()
        )

    state = TrainingState()
    if config.runtime.resume_from:
        state = _resume(model, optimizer, Path(config.runtime.resume_from), state)
        print(f"  resumed at step {state.step}")

    vocab = BYTE_VOCAB_SIZE if text_mode else (spec.vocab_size if spec else model.config.vocab_size)
    generator = torch.Generator().manual_seed(config.training.seed)
    metrics_path = output / "metrics.jsonl"
    metrics_handle = metrics_path.open("a", encoding="utf-8")
    started = time.perf_counter()
    tokens_seen = 0
    first_step = True

    try:
        while state.step < config.training.max_steps:
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for _ in range(config.training.gradient_accumulation_steps):
                if text_mode:
                    batch = torch.tensor(next(batches), dtype=torch.long, device=device)
                else:
                    batch = _synthetic_batch(
                        config.training.batch_size, config.data.max_sequence_length,
                        vocab, generator,
                    ).to(device)
                with autocast():
                    outputs = model(input_ids=batch, labels=batch)
                    loss = outputs.loss / config.training.gradient_accumulation_steps
                if first_step:
                    # Activations are live between forward and backward; snapshot here
                    # or the backward pass will have already freed them.
                    take(profile, "after_forward")
                scaler.scale(loss).backward()
                accumulated += float(loss.item())
                tokens_seen += batch.numel()
                if first_step:
                    take(profile, "after_backward")
                    first_step = False
            # Gradients must be unscaled before clipping, or the norm is computed
            # against the loss scale rather than the true gradient.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            state.step += 1
            if state.step == 1:
                take(profile, "after_optimizer_step")

            if state.step % config.training.log_every == 0:
                elapsed = time.perf_counter() - started
                record = {
                    "step": state.step, "loss": accumulated,
                    "lr": scheduler.get_last_lr()[0], "elapsed_s": round(elapsed, 1),
                    "tokens_seen": tokens_seen,
                    "tokens_per_second": round(tokens_seen / elapsed, 1) if elapsed else 0.0,
                }
                if text_mode:
                    record["bits_per_byte"] = round(bits_per_byte(accumulated), 4)
                state.history.append(record)
                metrics_handle.write(json.dumps(record) + "\n")
                metrics_handle.flush()
                extra = f"  bpb {record['bits_per_byte']:.3f}" if text_mode else ""
                print(f"  step {state.step:>6}  loss {accumulated:8.4f}{extra}  "
                      f"{record['tokens_per_second']:>8.0f} tok/s  {elapsed:6.1f}s")

            if state.step % config.training.eval_every == 0:
                validation_loss = (
                    _validate_text(model, validation_sequences, config, device)
                    if text_mode
                    else _validate(model, config, vocab, device)
                )
                record = {"step": state.step, "validation_loss": validation_loss}
                if text_mode:
                    record["validation_bits_per_byte"] = round(bits_per_byte(validation_loss), 4)
                state.history.append(record)
                metrics_handle.write(json.dumps(record) + "\n")
                metrics_handle.flush()
                extra = f"  bpb {record['validation_bits_per_byte']:.3f}" if text_mode else ""
                print(f"  step {state.step:>6}  validation {validation_loss:8.4f}{extra}")
                if state.best_validation_loss is None or validation_loss < state.best_validation_loss:
                    state.best_validation_loss = validation_loss

            if state.step % config.training.save_every == 0:
                _checkpoint(model, optimizer, state, output / f"step-{state.step}", config)
    finally:
        metrics_handle.close()

    take(profile, "peak_training")
    derive_components(profile)

    _checkpoint(model, optimizer, state, output / "final", config)
    elapsed = time.perf_counter() - started
    print(f"\n  finished {state.step} steps in {elapsed:.1f}s "
          f"({tokens_seen / elapsed:,.0f} tokens/s)")

    _write_summary(
        output, config, spec, state, profile, corpus_stats,
        elapsed=elapsed, tokens_seen=tokens_seen, device=device, text_mode=text_mode,
        effective_precision=precision, precision_note=precision_note,
    )
    print(f"  wrote {output / 'final'} and summary.json")
    return 0


#: Period of the synthetic repeat task (Level-1 mechanism test only).
SYNTHETIC_PERIOD = 8

#: Distinct tokens the synthetic task draws from. Far smaller than a real vocabulary on
#: purpose: the Level-1 prototype must show a clearly falling loss within a few hundred
#: steps. Level 2 uses real text instead (see qwen_distill.training.text_data).
SYNTHETIC_ALPHABET = 64


def _validate_text(model, sequences: list[list[int]], config: ExperimentConfig, device: str) -> float:
    """Mean cross-entropy over the held-out byte sequences.

    Evaluates every validation sequence rather than sampling, so the number is
    deterministic and two runs are directly comparable.
    """
    import torch

    model.eval()
    total, batches = 0.0, 0
    batch_size = max(1, config.training.batch_size)
    # Validate under the same precision as training, or the two losses are not
    # comparable and the validation curve measures the dtype change as much as learning.
    precision, _ = resolve_precision(config.training.precision, device)
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    autocast = (
        torch.autocast(device_type=device, dtype=autocast_dtype)
        if autocast_dtype else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast:
        for start in range(0, len(sequences) - batch_size + 1, batch_size):
            batch = torch.tensor(
                sequences[start : start + batch_size], dtype=torch.long, device=device
            )
            total += float(model(input_ids=batch, labels=batch).loss.item())
            batches += 1
    model.train()
    return total / max(batches, 1)


def _write_summary(
    output: Path,
    config: ExperimentConfig,
    spec: HybridArchSpec | None,
    state: TrainingState,
    profile,
    corpus_stats,
    *,
    elapsed: float,
    tokens_seen: int,
    device: str,
    text_mode: bool,
    effective_precision: str,
    precision_note: str | None,
) -> None:
    """Write the full artifact set: summary, hardware, git commit, resolved config.

    Everything needed to reproduce or audit the run, in one directory, so a result is
    never separated from the conditions that produced it.
    """
    import subprocess

    from ..architecture.params import count_parameters
    from ..diagnostics.devices import collect_devices, collect_system
    from ..diagnostics.fit import estimate_training_memory

    commit = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
        commit = result.stdout.strip() or None if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.SubprocessError):
        commit = None
    (output / "git_commit.txt").write_text((commit or "unknown") + "\n", encoding="utf-8")

    system = collect_system()
    devices = collect_devices(system)
    (output / "hardware.json").write_text(
        json.dumps(
            {"system": system.to_dict(), "devices": [d.to_dict() for d in devices]},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    losses = [h["loss"] for h in state.history if "loss" in h]
    validations = [h["validation_loss"] for h in state.history if "validation_loss" in h]

    estimate = None
    comparison = {"available": False}
    if spec is not None:
        fit = estimate_training_memory(
            spec, profile.total_vram_gib or 0.0,
            strategy=config.training.strategy, optimizer=config.training.optimizer,
            sequence_length=config.data.max_sequence_length,
            batch_size=config.training.batch_size,
            gradient_checkpointing=config.training.gradient_checkpointing,
            precision=effective_precision,
        )
        estimate = fit.to_dict()
        comparison = compare_with_estimate(profile, fit.total_gib)

    summary = {
        "experiment": config.name,
        "level": config.level,
        "git_commit": commit,
        "device": device,
        # The requested precision and the one actually used can differ (fp16 on CPU,
        # bf16 on Turing). Record both, so a memory or speed figure can be read against
        # what really ran rather than against what the config asked for.
        "requested_precision": config.training.precision,
        "effective_precision": effective_precision,
        "precision_note": precision_note,
        "objective": "byte-level causal LM" if text_mode else config.training.objective,
        "parameters": count_parameters(spec).as_dict() if spec else None,
        "steps": state.step,
        "runtime_s": round(elapsed, 2),
        "tokens_seen": tokens_seen,
        "tokens_per_second": round(tokens_seen / elapsed, 1) if elapsed else None,
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "best_validation_loss": state.best_validation_loss,
        "first_validation_loss": validations[0] if validations else None,
        "final_validation_loss": validations[-1] if validations else None,
        "corpus": corpus_stats.to_dict() if corpus_stats else None,
        "memory": profile.to_dict(),
        "analytical_estimate": estimate,
        "estimate_vs_measured": comparison,
        "config": config.to_dict(),
    }
    if text_mode and state.best_validation_loss is not None:
        summary["best_validation_bits_per_byte"] = round(
            bits_per_byte(state.best_validation_loss), 4
        )
        summary["uniform_baseline_bits_per_byte"] = 8.0
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


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
