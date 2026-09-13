#!/usr/bin/env python3
"""Run014 gradient calibration: estimate the fixed multiplier, once, before training.

For each preregistered TRAINING sequence, from the identical seed-1 QLoRA initialisation, this
measures the global L2 norm of the trainable (LoRA) gradients under the RAW residual-delta
objective and under the NORMALISED residual-delta objective, and records their ratio. The Run014
multiplier is the median of the four ratios.

What it deliberately does not do:
  * build an optimiser or a scheduler -- no parameter is ever updated;
  * read validation data;
  * adapt, sweep, or recalibrate.

Raw and normalised measurements of one sequence share one model instance whose trainable state is
digested before and after every measurement, and share one RNG seed so both see identical LoRA
dropout masks. The only thing that differs between them is ``normalise``.

The student, teacher, corpus and layer mapping are built through the same functions the trainer
uses, from the configuration Run011 ran with, so the measured gradients are the ones Run011's
first step would have produced. Refuses to run from a dirty worktree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import torch

import kd_run
import run008_raw_delta_seed1 as run008
from qwen_distill.distillation.backends import TransformersTeacher
from qwen_distill.distillation.behavioral import behavioral_loss_chunked
from qwen_distill.research import gradient_calibration as gc
from qwen_distill.training import trainer as trainer_module
from qwen_distill.training.tokenized_data import prepare_tokenized_corpus

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "run014_normalized_delta_gradmatched_seed1"
EXPERIMENT_DIR = REPO / "experiments" / EXPERIMENT_ID
PREREG = EXPERIMENT_DIR / "PREREG.md"
SAMPLE_FREEZE = EXPERIMENT_DIR / "sample_freeze.json"
RUN_DIR = Path("/workspace/runs") / EXPERIMENT_ID
MANIFEST = RUN_DIR / "calibration_manifest.json"
NO_NORMALISE = "--layer-kd-no-normalise"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run011_argv() -> list[str]:
    """Run011's argv: Run008's minus exactly the no-normalise flag (as Run011 was built)."""
    base = run008.command()
    assert NO_NORMALISE in base
    return [tok for tok in base if tok != NO_NORMALISE]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="testing only; a real calibration refuses a dirty worktree")
    args = ap.parse_args(argv)

    status = git("status", "--porcelain")
    if status and not args.allow_dirty:
        print(f"REFUSED: worktree is dirty; calibration must run from a committed design:\n{status}",
              file=sys.stderr)
        return 2
    if MANIFEST.exists():
        print(f"REFUSED: {MANIFEST} already exists; Run014 is calibrated exactly once.",
              file=sys.stderr)
        return 2
    for required in (PREREG, SAMPLE_FREEZE):
        if not required.is_file():
            print(f"REFUSED: missing {required}", file=sys.stderr)
            return 2

    freeze = json.loads(SAMPLE_FREEZE.read_text())
    frozen_hashes = {int(k): v for k, v in freeze["token_hashes"].items()}
    if tuple(freeze["calibration_indices"]) != gc.CALIBRATION_INDICES:
        print("REFUSED: sample_freeze indices differ from gradient_calibration.CALIBRATION_INDICES",
              file=sys.stderr)
        return 2

    parsed = kd_run.parse_args(run011_argv())
    config = kd_run.build_config(parsed, Path(parsed.pretrained))
    t = config.training
    if not (t.objective == "layer_kd" and t.layer_kd_normalise is True and t.seed == 1
            and t.layer_kd_chunk_pairs == 4 and t.layer_kd_direction_weight == 1.0):
        print("REFUSED: derived configuration is not Run011's", file=sys.stderr)
        return 2
    device = "cuda"

    # -- corpus: identical call to the trainer's, then the frozen samples ------------------
    train_seq, val_seq, stats = prepare_tokenized_corpus(
        text_path=config.data.text_path, tokenizer_path=config.data.tokenizer_path,
        sequence_length=config.data.max_sequence_length,
        validation_fraction=config.data.validation_fraction,
        document_separator=config.data.document_separator,
        max_documents=config.data.max_documents, max_tokens=config.data.max_tokens,
        max_bytes=config.data.max_corpus_bytes,
        expected_vocab_size=config.data.expected_vocab_size,
        teacher_model=config.teacher.get("model"), teacher_revision=config.teacher.get("revision"),
        trust_remote_code=config.model.trust_remote_code,
    )
    if stats.sha256 != gc.PACKED_CORPUS_SHA256:
        print(f"REFUSED: packed corpus sha {stats.sha256} != {gc.PACKED_CORPUS_SHA256}",
              file=sys.stderr)
        return 2
    samples = gc.select_calibration_samples(train_seq)
    gc.check_frozen_hashes(samples, frozen_hashes)
    print(f"  corpus ok: {stats.sha256}; {len(train_seq)} train / {len(val_seq)} validation")
    for s in samples:
        print(f"  sample {s['index']:>3}  {s['sha256']}")

    # -- teacher: as kd_run builds it ------------------------------------------------------
    backend = TransformersTeacher(model=parsed.teacher_model, revision=parsed.revision,
                                  local_path=str(parsed.teacher), quantization=parsed.quantization,
                                  strict_architecture=True)
    backend.load()
    teacher = backend.signal_provider(top_k=parsed.kd_top_k, temperature=parsed.kd_temperature,
                                      capture_hidden_states=True)

    # -- student: as the trainer builds it -------------------------------------------------
    model = trainer_module.build_model(config, None)
    if t.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    trainable = gc.trainable_parameters(model)
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  trainable parameters: {n_trainable:,}")

    initial_digest = gc.trainable_state_digest(model)
    measurements = []
    started = time.time()
    for sample in samples:
        batch = torch.tensor([sample["tokens"]], device=device)
        seed = gc.calibration_seed(t.seed, sample["index"])
        norms = {}
        teacher_fingerprints = {}
        for mode, normalise in (("raw", False), ("normalised", True)):
            if gc.trainable_state_digest(model) != initial_digest:
                raise gc.CalibrationError("trainable parameters changed before a measurement")
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                signal = teacher.signal_for(batch)
            teacher_fingerprints[mode] = float(signal.hidden_states[-1].double().sum())
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Mirrors trainer.train(): forward AND the chunked hidden loss run inside autocast,
            # and the mapping comes from the depths this forward actually returned.
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(input_ids=batch, output_hidden_states=True)
                mapping = trainer_module._layer_mapping(
                    len(outputs.hidden_states) - 1, len(signal.hidden_states) - 1,
                    t.layer_kd_map_strategy).mapping
                held = behavioral_loss_chunked(
                    outputs.hidden_states, signal.hidden_states, mapping, mode="delta",
                    direction_weight=t.layer_kd_direction_weight, normalise=normalise,
                    chunk_pairs=t.layer_kd_chunk_pairs, loss_scale=1.0,
                )
            # ...and propagation into the student happens outside it, as in the backward phase.
            held.backward()
            norms[mode] = gc.trainable_grad_norm(model)
            norms[f"{mode}_objective"] = float(held.output.total)
            del outputs, held, signal
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        if teacher_fingerprints["raw"] != teacher_fingerprints["normalised"]:
            raise gc.CalibrationError(f"teacher signal not deterministic for sample {sample['index']}")
        measurements.append({
            "index": sample["index"], "sha256": sample["sha256"], "rng_seed": seed,
            "raw_grad_norm": norms["raw"], "normalised_grad_norm": norms["normalised"],
            "raw_objective": norms["raw_objective"],
            "normalised_objective": norms["normalised_objective"],
            "raw_exceeds_clip_1.0": norms["raw"] > 1.0,
            "normalised_exceeds_clip_1.0": norms["normalised"] > 1.0,
            "teacher_hidden_fingerprint": teacher_fingerprints["raw"],
        })
        print(f"  sample {sample['index']:>3}: raw {norms['raw']:.6e}  "
              f"normalised {norms['normalised']:.6e}")

    final_digest = gc.trainable_state_digest(model)
    mutated = final_digest != initial_digest
    if mutated:
        raise gc.CalibrationError("trainable parameters changed during calibration")
    ratios, multiplier = gc.median_multiplier(
        [m["raw_grad_norm"] for m in measurements],
        [m["normalised_grad_norm"] for m in measurements])
    for m, r in zip(measurements, ratios, strict=True):
        m["ratio"] = r

    import bitsandbytes
    import peft
    import transformers

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "method_version": gc.METHOD_VERSION,
        "calibration_indices": list(gc.CALIBRATION_INDICES),
        "token_hashes": {str(s["index"]): s["sha256"] for s in samples},
        "token_byte_format": gc.TOKEN_BYTE_FORMAT,
        "packed_corpus_sha256": stats.sha256,
        "teacher_model": parsed.teacher_model,
        "teacher_revision": parsed.revision,
        "student_id": "qwen38_19b_h5120_l48_moe",
        "student_path": str(parsed.pretrained),
        "seed": t.seed,
        "lora_rank": t.lora_rank, "lora_alpha": t.lora_alpha, "lora_dropout": t.lora_dropout,
        "trainable_parameters": n_trainable,
        "preregistration_sha256": sha256_file(PREREG),
        "sample_freeze_sha256": sha256_file(SAMPLE_FREEZE),
        "git_commit": git("rev-parse", "HEAD"),
        "git_worktree_clean": not status,
        "config_argv_run011": run011_argv(),
        "config_argv_sha256": hashlib.sha256("\0".join(run011_argv()).encode()).hexdigest(),
        "gradient_norm_definition": ("global L2 norm of .grad over parameters with "
                                     "requires_grad=True (LoRA only), pre-clip, loss_scale=1.0"),
        "multiplier_estimator": "statistics.median of raw_grad_norm/normalised_grad_norm over 4 samples "
                                "(even count: mean of the 2nd and 3rd sorted ratios)",
        "samples": measurements,
        "ratios": ratios,
        "multiplier": multiplier,
        "initial_trainable_digest": initial_digest,
        "final_trainable_digest": final_digest,
        "parameters_mutated": mutated,
        "optimizer_constructed": False,
        "scheduler_constructed": False,
        "validation_data_used": False,
        "clip_grad_norm_threshold_in_training": 1.0,
        "elapsed_seconds": round(time.time() - started, 2),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
            "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\n  ratios     : {[f'{r:.6f}' for r in ratios]}")
    print(f"  multiplier : {multiplier!r}")
    print(f"  manifest   : {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
