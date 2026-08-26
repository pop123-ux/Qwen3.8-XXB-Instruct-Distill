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
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..architecture.spec import HybridArchSpec
from .checkpoints import (
    CheckpointMetadata,
    capture_rng_state,
    cleanup_incomplete,
    config_sha256,
    is_complete,
    load_checkpoint,
    resolve_checkpoint,
    restore_rng_state,
    save_checkpoint,
    step_dirname,
)
from .config import ExperimentConfig
from .data import DistillationExample, read_jsonl, synthetic_corpus
from .memory_probe import (
    OOMRecord,
    compare_with_estimate,
    derive_components,
    is_oom,
    new_profile,
    record_oom,
    reset_peak,
    take,
)
from .progress import ProgressWriter
from .resume_compat import (
    check_resume_compatibility,
    make_schedule,
    rebuild_schedule,
)
from .text_data import (
    BYTE_VOCAB_SIZE,
    ResumableBatchSampler,
    bits_per_byte,
    prepare_corpus,
)
from .throughput import ThroughputTracker


@dataclass
class TrainingState:
    """Everything needed to resume a run exactly where it stopped."""

    step: int = 0
    epoch: int = 0
    best_validation_loss: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    #: Cumulative tokens, so throughput and data coverage survive a resume. Recomputing
    #: it from the step count would be wrong the moment batch size ever changed.
    tokens_seen: int = 0
    #: Where the batch stream was, from ResumableBatchSampler.state_dict(). Without it a
    #: resumed run silently rewinds to epoch 0 and re-trains on data it has already seen.
    data_state: dict[str, Any] = field(default_factory=dict)
    #: Wall-clock from previous segments, so a run resumed three times still reports the
    #: total time it actually took.
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step, "epoch": self.epoch,
            "best_validation_loss": self.best_validation_loss,
            "history": self.history,
            "tokens_seen": self.tokens_seen,
            "data_state": self.data_state,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingState:
        return cls(
            step=data.get("step", 0), epoch=data.get("epoch", 0),
            best_validation_loss=data.get("best_validation_loss"),
            history=data.get("history", []),
            tokens_seen=data.get("tokens_seen", 0),
            data_state=data.get("data_state") or {},
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
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
        batches = ResumableBatchSampler(
            train_sequences, config.training.batch_size, seed=config.training.seed
        )
    else:
        train_examples, validation_examples = load_examples(config)
        print(f"  data  : {len(train_examples)} train / {len(validation_examples)} validation")

    model = build_model(config, spec)
    take(profile, "after_model_creation")
    model = model.to(device)
    if config.training.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        # Non-reentrant checkpointing composes correctly with autocast and does not
        # require the inputs to have requires_grad, unlike the legacy reentrant path.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("  gradient checkpointing: ON")
    model.train()
    take(profile, "after_model_to_device")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.training.learning_rate, weight_decay=config.training.weight_decay
    )
    scheduler = make_schedule(
        optimizer, total_steps=config.training.max_steps,
        max_lr=config.training.learning_rate,
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

    # --- persistence ----------------------------------------------------
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    # A previous process killed mid-write leaves a staging directory behind. It is
    # ignored by discovery, but removing it keeps the run directory honest.
    for removed in cleanup_incomplete(checkpoint_root):
        print(f"  removed incomplete checkpoint from a previous run: {removed}")

    config_digest = config_sha256(config.to_dict())
    progress = ProgressWriter(output, git_commit=_git_commit(), config_sha256=config_digest)

    state = TrainingState()
    if config.runtime.resume_from:
        resolved = resolve_checkpoint(checkpoint_root, config.runtime.resume_from)
        if resolved is None:
            print(
                f"\n  ERROR: no complete checkpoint matches {config.runtime.resume_from!r} "
                f"under {checkpoint_root}.",
                file=sys.stderr,
            )
            print("  Resuming from a partial write would silently produce a different "
                  "run, so this stops instead.", file=sys.stderr)
            return 2
        # max_steps is the intended TOTAL training length, so raising it is a normal
        # thing to want: train 20, look at the curve, continue to 40. Differences that
        # would make the weights or the saved data position meaningless are still fatal.
        compatibility = check_resume_compatibility(
            _checkpoint_config(resolved), config.to_dict()
        )
        if not compatibility.ok:
            print(f"\n  ERROR: cannot resume from {resolved}.", file=sys.stderr)
            print(compatibility.render(), file=sys.stderr)
            return 2

        loaded = load_checkpoint(
            resolved, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, map_location=device,
        )
        state = TrainingState.from_dict(loaded["training_state"])

        if compatibility.extends_schedule:
            # OneCycleLR's LR at step t is a function of total_steps, so a longer run
            # needs a new schedule rather than the restored one replayed further.
            scheduler = rebuild_schedule(
                scheduler, optimizer,
                total_steps=config.training.max_steps, completed=state.step,
                max_lr=config.training.learning_rate,
            )
        rng_restored = restore_rng_state(loaded["rng_state"])
        if text_mode and state.data_state:
            batches.load_state_dict(state.data_state)
        print("\n  RESUMED")
        print(f"    resume requested   : {config.runtime.resume_from}")
        print(f"    resolved checkpoint: {resolved}")
        print(f"    restored step      : {state.step} of {config.training.max_steps}")
        print(f"    restored           : {', '.join(loaded['restored']) or 'nothing'}")
        print(f"    RNG restored       : {', '.join(rng_restored) or 'none'}")
        if text_mode:
            print(f"    data position      : epoch {batches.epoch}, batch {batches.index}")
        print(f"    tokens seen        : {state.tokens_seen:,}")
        if compatibility.extends_schedule or compatibility.notable:
            print(compatibility.render())
        # The compatibility check above itemises every difference it knows about. The
        # digest catches anything it does not classify — a renamed run, a changed eval
        # interval — which is worth stating once without dressing it up as a problem.
        saved_digest = (loaded.get("metadata") or {}).get("config_sha256")
        if (
            saved_digest and saved_digest != config_digest
            and not compatibility.extends_schedule and not compatibility.notable
        ):
            print("    note               : the config differs from the checkpoint's in "
                  "ways that do not affect resuming")

    print("\n  CHECKPOINTING")
    print(f"    full checkpoint : every {config.training.save_every} steps -> {checkpoint_root}")
    print(f"    progress record : every {config.training.resolved_progress_every} steps "
          f"-> {progress.metrics_path.name} + progress/latest.json")
    print(f"    resume          : "
          f"{config.runtime.resume_from or 'no (starting from step 0)'}")
    persistence = None
    if config.training.persistent_backup:
        from .persist import preflight

        persistence = preflight(config.training.persistent_backup)
        print(persistence.render())
        if not persistence.usable:
            print(
                "\n  ERROR: persistence was requested but cannot work, so this run would "
                "train\n  with no durable copy while reporting that it has one. Fix the "
                "destination\n  or remove training.persistent_backup to run local-only.",
                file=sys.stderr,
            )
            return 2
    else:
        print("    persistent copy : off (local only)")
        print("                      NOTHING SURVIVES THIS RUNTIME. Set "
              "training.persistent_backup")
        print("                      to a mounted Drive path to keep checkpoints.")

    vocab = BYTE_VOCAB_SIZE if text_mode else (spec.vocab_size if spec else model.config.vocab_size)
    generator = torch.Generator().manual_seed(config.training.seed)
    # Computed before the loop so an OOM can be compared against it in place. The
    # estimate is the thing on trial when a run fails.
    estimated_total_gib = None
    if spec is not None and profile.total_vram_gib:
        from ..diagnostics.fit import estimate_training_memory

        estimated_total_gib = estimate_training_memory(
            spec, profile.total_vram_gib,
            strategy=config.training.strategy, optimizer=config.training.optimizer,
            sequence_length=config.data.max_sequence_length,
            batch_size=config.training.batch_size,
            gradient_checkpointing=config.training.gradient_checkpointing,
            precision=precision,
        ).total_gib

    def write_checkpoint(step: int, *, reason: str) -> Path | None:
        """Persist everything needed to resume, atomically. Returns the directory."""
        state.data_state = batches.state_dict() if text_mode else {}
        state.elapsed_seconds = resumed_elapsed + (time.perf_counter() - started)
        last_loss = next(
            (h["loss"] for h in reversed(state.history) if "loss" in h), None
        )
        metadata = CheckpointMetadata(
            step=step,
            parameter_count=_parameter_count(spec),
            architecture_sha256=config_sha256(spec.to_dict()) if spec else None,
            config_sha256=config_digest,
            precision=precision,
            optimizer=config.training.optimizer,
            sequence_length=config.data.max_sequence_length,
            batch_size=config.training.batch_size,
            effective_batch_size=config.training.effective_batch_size,
            gradient_checkpointing=config.training.gradient_checkpointing,
            tokens_seen=state.tokens_seen,
            training_loss=last_loss,
            validation_loss=state.best_validation_loss,
            validation_bits_per_byte=(
                bits_per_byte(state.best_validation_loss)
                if text_mode and state.best_validation_loss is not None else None
            ),
        )
        try:
            path = save_checkpoint(
                checkpoint_root, step,
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                training_state=state.to_dict(), config=config.to_dict(),
                rng_state=capture_rng_state(), metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - a failed write must not kill the run
            print(f"  ! checkpoint at step {step} FAILED to write: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            print("    The previous checkpoint is untouched and still resumable.",
                  file=sys.stderr)
            return None
        print(f"  checkpoint: {path.name} ({reason})")
        if persistence is not None:
            _persist_checkpoint(config, path, checkpoint_root, output, persisted)
        return path

    #: Which checkpoints reached persistent storage, and which failed. A run must never
    #: end implying that everything is safe when a copy failed at step 800.
    persisted: dict[str, list[str]] = {"ok": [], "failed": []}

    resumed_elapsed = state.elapsed_seconds
    started = time.perf_counter()
    # Tokens restored from a checkpoint were generated by an EARLIER session. Dividing
    # them by this session's clock is what made a resumed 2,090 tok/s run report 139,256.
    throughput = ThroughputTracker.from_state(
        tokens_seen=state.tokens_seen, elapsed_seconds=state.elapsed_seconds
    )
    first_step = True
    #: The phase currently executing, so an OOM can say *where* it ran out rather than
    #: only that it did. The first Level-2 attempt died in the forward pass and the
    #: traceback named SDPA, which was the next allocation rather than the cause.
    phase = ["before first step"]
    oom: OOMRecord | None = None
    #: Set by a signal handler. The loop finishes the step in flight and then stops
    #: cleanly, so the checkpoint written on the way out is a real one. A hard kill
    #: (SIGKILL, a Colab runtime teardown) cannot be caught at all — that case is
    #: covered by atomic writes, not by this.
    stopping = {"requested": False, "signal": None}

    def request_stop(signum, _frame):
        stopping["requested"] = True
        stopping["signal"] = signal.Signals(signum).name
        print(f"\n  received {stopping['signal']}: finishing this step, then saving "
              "a checkpoint and stopping.")

    previous_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        # Not the main thread, or a platform without the signal: training still runs,
        # it just cannot stop gracefully. Atomic writes cover that case anyway.
        with contextlib.suppress(ValueError, OSError):
            previous_handlers[sig] = signal.signal(sig, request_stop)

    try:
        while state.step < config.training.max_steps and not stopping["requested"]:
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
                phase[0] = "input allocation"
                if first_step:
                    take(profile, "after_input_allocation")
                phase[0] = "forward pass"
                with autocast():
                    outputs = model(input_ids=batch, labels=batch)
                    if first_step:
                        # Activations are live between forward and backward; snapshot
                        # here or the backward pass will have already freed them.
                        take(profile, "after_forward")
                    loss = outputs.loss / config.training.gradient_accumulation_steps
                if first_step:
                    # The loss path holds the logits three times over, so it gets its
                    # own stage rather than being folded into the forward pass.
                    take(profile, "after_loss")
                phase[0] = "backward pass"
                scaler.scale(loss).backward()
                accumulated += float(loss.item())
                throughput.add_tokens(batch.numel())
                state.tokens_seen += batch.numel()
                if first_step:
                    take(profile, "after_backward")
                    first_step = False
            # Gradients must be unscaled before clipping, or the norm is computed
            # against the loss scale rather than the true gradient.
            phase[0] = "optimizer step"
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            state.step += 1
            if state.step == 1:
                take(profile, "after_optimizer_step")

            if state.step % config.training.log_every == 0:
                rates = throughput.rates(time.perf_counter() - started)
                elapsed = rates["elapsed_s"]
                record = {
                    "step": state.step, "loss": accumulated,
                    "lr": scheduler.get_last_lr()[0], "elapsed_s": elapsed,
                    "tokens_seen": rates["total_tokens"],
                    # Three scopes, reported separately. `tokens_per_second` is run-wide
                    # and is the only one comparable to a headline figure; the interval
                    # rate is the one that reacts to a slowdown as it happens.
                    "tokens_per_second": rates["tokens_per_second"],
                    "interval_tokens_per_second": rates["interval_tokens_per_second"],
                    "session_tokens_per_second": rates["session_tokens_per_second"],
                    "epoch": batches.epoch if text_mode else state.epoch,
                }
                if text_mode:
                    record["bits_per_byte"] = round(bits_per_byte(accumulated), 4)
                state.history.append(record)
                extra = f"  bpb {record['bits_per_byte']:.3f}" if text_mode else ""
                print(f"  step {state.step:>6}  loss {accumulated:8.4f}{extra}  "
                      f"{record['interval_tokens_per_second']:>8.0f} tok/s"
                      f"  (run {record['tokens_per_second']:.0f})  {elapsed:6.1f}s")

            # Progress is written on its own interval, independently of checkpoints:
            # it costs kilobytes, so it can be frequent, and it is what survives when
            # the process dies between checkpoints.
            if state.step % config.training.resolved_progress_every == 0:
                latest = next(
                    (h for h in reversed(state.history) if h.get("step") == state.step),
                    {"step": state.step, "loss": accumulated},
                )
                progress.write({**latest, "best_validation_loss": state.best_validation_loss})

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
                progress.write({**record, "step": state.step}, status="validated")
                extra = f"  bpb {record['validation_bits_per_byte']:.3f}" if text_mode else ""
                print(f"  step {state.step:>6}  validation {validation_loss:8.4f}{extra}")
                if state.best_validation_loss is None or validation_loss < state.best_validation_loss:
                    state.best_validation_loss = validation_loss

            if state.step % config.training.save_every == 0:
                phase[0] = "checkpoint"
                write_checkpoint(state.step, reason="periodic")
    except Exception as exc:  # noqa: BLE001 - an OOM is a result, not just a crash
        if not is_oom(exc):
            raise
        # Record the memory state *before* anything is freed, then let the summary be
        # written: a run that OOMs establishes a lower bound on what the configuration
        # costs, and discarding that throws away the most expensive measurement here.
        oom = record_oom(
            profile, phase[0], exc,
            estimated_total_gib=estimated_total_gib,
            configuration={
                "parameters": _parameter_count(spec),
                "sequence_length": config.data.max_sequence_length,
                "batch_size": config.training.batch_size,
                "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
                "precision": precision,
                "optimizer": config.training.optimizer,
                "gradient_checkpointing": config.training.gradient_checkpointing,
                "steps_completed": state.step,
            },
        )
        print(f"\n{oom.render()}\n")
    finally:
        for sig, handler in previous_handlers.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, handler)

    take(profile, "peak_training")
    derive_components(profile)

    final_checkpoint = None
    if oom is None and state.step > 0:
        # Unconditional, so a run whose last step does not land on save_every — a
        # 20-step smoke test against save_every=500, or an interrupted run — still
        # leaves something resumable. Previously that produced no recoverable artifact.
        phase[0] = "final checkpoint"
        existing = checkpoint_root / step_dirname(state.step)
        if is_complete(existing):
            final_checkpoint = existing
        else:
            reason = "interrupted" if stopping["requested"] else "final step"
            final_checkpoint = write_checkpoint(state.step, reason=reason)
    session_seconds = time.perf_counter() - started
    throughput_summary = throughput.summary(session_seconds)
    elapsed = resumed_elapsed + session_seconds
    if oom is None:
        verb = "stopped at" if stopping["requested"] else "finished"
        print(f"\n  {verb} step {state.step} of {config.training.max_steps} "
              f"in {elapsed:.1f}s ({state.tokens_seen:,} tokens seen)")
        if stopping["requested"]:
            print(f"  interrupted by {stopping['signal']}. Resume with:\n"
                  f"    python scripts/train_student.py --config <config> --resume latest")

    _write_summary(
        output, config, spec, state, profile, corpus_stats,
        elapsed=elapsed, tokens_seen=throughput.total_tokens, device=device,
        text_mode=text_mode, throughput=throughput_summary,
        effective_precision=precision, precision_note=precision_note, oom=oom,
        persisted=persisted, persistent_destination=config.training.persistent_backup,
    )
    if oom is not None:
        print(f"  wrote {output / 'summary.json'} recording the OOM.")
        print("  This is usable data: run scripts/hardware_info.py --calibrate-run on it.")
        return 1
    # The run must never end implying everything is stored when a copy failed midway.
    if config.training.persistent_backup:
        if persisted["failed"]:
            print(f"\n  ! {len(persisted['failed'])} checkpoint(s) FAILED to persist: "
                  f"{', '.join(persisted['failed'])}", file=sys.stderr)
            print("    Those exist locally only. Copy them before this runtime ends:",
                  file=sys.stderr)
            print(f"      python scripts/backup_colab_to_drive.py --source {output} \\\n"
                  f"          --destination {config.training.persistent_backup} "
                  "--checkpoints-only", file=sys.stderr)
        elif persisted["ok"]:
            print(f"  persisted {len(persisted['ok'])} checkpoint(s) to "
                  f"{config.training.persistent_backup}")

    if final_checkpoint is not None:
        print(f"  wrote {final_checkpoint} and summary.json")
    elif state.step == 0:
        print(f"  wrote {output / 'summary.json'} (no steps ran, so no checkpoint)")
    else:
        # Never report success for a checkpoint that failed to write: the run looks
        # complete and is not recoverable, which is the exact failure being fixed here.
        print(f"  ! no checkpoint could be written at step {state.step}.", file=sys.stderr)
        return 1
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


def _parameter_count(spec: HybridArchSpec | None) -> int | None:
    if spec is None:
        return None
    from ..architecture.params import count_parameters

    return count_parameters(spec).total


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
    throughput: dict[str, Any] | None = None,
    effective_precision: str,
    precision_note: str | None,
    oom: OOMRecord | None = None,
    persisted: dict[str, list[str]] | None = None,
    persistent_destination: str | None = None,
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
        # Run-wide and session figures kept apart, because a resumed run's session rate
        # is not what the experiment cost and reporting one as the other overstated a
        # 2,090 tok/s run as 139,256.
        "throughput": throughput,
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "best_validation_loss": state.best_validation_loss,
        "first_validation_loss": validations[0] if validations else None,
        "final_validation_loss": validations[-1] if validations else None,
        "corpus": corpus_stats.to_dict() if corpus_stats else None,
        # A run either completed or ran out of memory. Both are results; only one of
        # them is a success, and the artifact must say which without being read closely.
        "outcome": "OOM" if oom else "completed",
        "oom": oom.to_dict() if oom else None,
        # Which checkpoints are actually durable. Reading this months later must not
        # require trusting that persistence was on and worked.
        "persistence": {
            "enabled": bool(persistent_destination),
            "destination": persistent_destination,
            "persisted": (persisted or {}).get("ok", []),
            "failed": (persisted or {}).get("failed", []),
            "all_checkpoints_persisted": (
                bool(persistent_destination) and not (persisted or {}).get("failed")
            ),
        },
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


def _checkpoint_config(checkpoint: Path) -> dict[str, Any] | None:
    """The config a checkpoint was written under, for compatibility checking."""
    config_file = Path(checkpoint) / "config.json"
    if not config_file.is_file():
        return None
    try:
        return json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _git_commit() -> str | None:
    """The commit this run is executing, recorded with every progress line."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _persist_checkpoint(
    config: ExperimentConfig,
    checkpoint: Path,
    root: Path,
    run_directory: Path,
    persisted: dict[str, list[str]],
) -> None:
    """Copy a completed checkpoint to persistent storage and record whether it worked.

    Training continues through a failure — the local checkpoint is intact, and losing
    the copy is recoverable while losing the run is not — but the failure is recorded so
    the run cannot finish implying that everything is safely stored.
    """
    destination = config.training.persistent_backup
    if not destination:
        return
    try:
        from .persist import persist_checkpoint, persist_run_metadata

        target = persist_checkpoint(checkpoint, destination, checkpoint_root=root)
        persist_run_metadata(run_directory, destination)
        persisted["ok"].append(checkpoint.name)
        print(f"    persisted -> {target}")
    except Exception as exc:  # noqa: BLE001 - a backup failure must not end the run
        persisted["failed"].append(checkpoint.name)
        print(f"\n  ! PERSISTENT COPY FAILED for {checkpoint.name}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("    This checkpoint exists LOCALLY ONLY and will not survive the runtime.",
              file=sys.stderr)
        print("    The persistent pointer still names the last checkpoint that did copy.",
              file=sys.stderr)
        print("    Training continues; fix the destination and re-run the backup script.\n",
              file=sys.stderr)
