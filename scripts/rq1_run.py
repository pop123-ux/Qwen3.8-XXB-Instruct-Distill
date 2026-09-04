#!/usr/bin/env python3
"""Execute one CPU-approved RQ1 arm through the validated generic KD trainer.

This file is intentionally *not* a second training loop. It adapts the already validated
``layer_kd`` branch by patching only the research loss seam, exactly as the historical
Run-004 launcher did, while making the selected arm and every scientific setting come from
``RQ1_OBJECTIVES_V2.json`` rather than from mutable CLI defaults.

Use ``scripts/rq1_launch.py`` for normal execution; it places the research guard in front of
this process. Direct invocation is kept for CI/dry-run diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

PROTOCOL_PATH = REPO / "research/protocols/RQ1_OBJECTIVES_V2.json"
READY = {"existing_cpu_tested", "cpu_tested", "ready"}


def _load_protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=list("ABCDEF"), required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--text-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _verify_registered_data(protocol: dict, teacher: Path, text_path: Path) -> dict[str, Any]:
    """Fast byte-level identity checks before any model weight is loaded."""
    problems = []
    if not text_path.is_file():
        problems.append(f"missing corpus: {text_path}")
    if not (teacher / "config.json").is_file():
        problems.append(f"missing teacher checkpoint/config: {teacher}")
    if problems:
        raise RuntimeError("; ".join(problems))

    data = protocol["data"]
    corpus_sha = _sha256(text_path)
    if corpus_sha != data["corpus_sha256"]:
        raise RuntimeError(
            f"corpus SHA mismatch: protocol={data['corpus_sha256']} actual={corpus_sha}"
        )
    tok = {}
    for name, expected in data["tokenizer_sha256"].items():
        path = teacher / name
        if not path.is_file():
            raise RuntimeError(f"missing registered tokenizer file: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"tokenizer {name} SHA mismatch: protocol={expected} actual={actual}")
        tok[name] = actual
    return {"corpus_sha256": corpus_sha, "tokenizer_sha256": tok}


def _trainer_args(args: argparse.Namespace, protocol: dict) -> list[str]:
    t = protocol["training"]
    command = [
        "--teacher", str(args.teacher),
        "--revision", protocol["teacher"]["revision"],
        "--teacher-model", protocol["teacher"]["model"],
        "--quantization", protocol["teacher"]["quantization"],
        "--student", "canonical",
        "--pretrained", str(args.pretrained),
        "--text-path", str(args.text_path),
        "--sequence-length", str(t["sequence_length"]),
        "--max-tokens", str(t["max_tokens"]),
        "--steps", str(t["steps"]),
        "--batch-size", str(t["batch_size"]),
        "--gradient-accumulation-steps", str(t["gradient_accumulation_steps"]),
        "--learning-rate", str(t["learning_rate"]),
        "--objective", "layer_kd",
        "--layer-kd-direction-weight", str(t["layer_kd_direction_weight"]),
        "--layer-kd-chunk-pairs", str(t["layer_kd_chunk_pairs"]),
        "--kd-temperature", str(t["kd_temperature"]),
        "--kd-top-k", str(t["kd_top_k"]),
        "--strategy", t["strategy"],
        "--optimizer", t["optimizer"],
        "--lora-rank", str(t["lora_rank"]),
        "--lora-alpha", str(t["lora_alpha"]),
        "--precision", t["precision"],
        "--seed", str(t["seed"]),
        "--log-every", str(t["log_every"]),
        "--eval-every", str(t["eval_every"]),
        "--save-every", str(t["save_every"]),
        "--output", str(args.output),
        "--name", f"rq1_v2_arm_{args.arm.lower()}",
    ]
    if not t["layer_kd_normalise"]:
        command.append("--layer-kd-no-normalise")
    return command


def _patch_for_arm(arm: str, arm_spec: dict, output: Path):
    """Patch only the generic trainer's layer-loss seam; return a restoration callable."""
    trainer = import_module("qwen_distill.training.trainer")
    rq1 = import_module("qwen_distill.distillation.rq1_objectives")

    original_chunked = trainer.behavioral_loss_chunked
    original_loss = trainer.behavioral_loss
    original_definition = trainer._layer_kd_definition
    original_build_model = trainer.build_model
    original_train = trainer.train
    live: dict[str, Any] = {}
    metrics_path = output / "rq1_objective_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def captured_build_model(*a, **kw):
        model = original_build_model(*a, **kw)
        live["student_model"] = model
        return model

    def captured_train(*a, **kw):
        teacher = kw.get("teacher")
        if teacher is None and len(a) >= 3:
            teacher = a[2]
        live["teacher_provider"] = teacher
        return original_train(*a, **kw)

    def _record(result):
        out = result.output
        components = getattr(out, "components", None) or {}
        payload = {
            "call": sum(1 for _ in metrics_path.open("r", encoding="utf-8")) + 1
            if metrics_path.exists() else 1,
            "arm": arm,
            "total": float(out.total),
            "magnitude": float(out.magnitude),
            "direction": float(out.direction),
            "mode": getattr(out, "mode", None),
            "components": components,
        }
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
        return result

    def chunked(student_hidden, teacher_hidden, mapping, **kwargs):
        if arm == "A":
            return _record(original_chunked(student_hidden, teacher_hidden, mapping, **kwargs))
        if arm == "C":
            kwargs["mode"] = "delta"
            return _record(original_chunked(student_hidden, teacher_hidden, mapping, **kwargs))

        common = {
            "direction_weight": kwargs.get("direction_weight", 1.0),
            "normalise": kwargs.get("normalise", True),
            "chunk_pairs": kwargs.get("chunk_pairs", 4),
            "loss_scale": kwargs.get("loss_scale", 1.0),
            "backward": kwargs.get("backward"),
        }
        params = arm_spec["objective_parameters"]
        if arm == "D":
            return _record(rq1.anchored_transition_loss_chunked(
                student_hidden, teacher_hidden, mapping,
                transition="adjacent",
                pointwise_weight=params["pointwise_weight"],
                transition_weight=params["adjacent_transition_weight"],
                **common,
            ))
        if arm == "E":
            return _record(rq1.anchored_transition_loss_chunked(
                student_hidden, teacher_hidden, mapping,
                transition="span",
                pointwise_weight=params["pointwise_weight"],
                transition_weight=params["span_transition_weight"],
                **common,
            ))
        if arm == "B":
            student_model = live.get("student_model")
            teacher_provider = live.get("teacher_provider")
            teacher_model = getattr(teacher_provider, "model", None)
            if student_model is None or teacher_model is None:
                raise RuntimeError("FDD could not resolve the live student/teacher LM heads")
            student_head = student_model.get_output_embeddings()
            teacher_head = teacher_model.get_output_embeddings()
            return _record(rq1.fdd_prediction_dynamics_chunked(
                student_hidden, teacher_hidden, student_head, teacher_head,
                sampled_layers=params["sampled_intermediate_layers"],
                alpha=params["alpha_trajectory"],
                beta=params["beta_derivative"],
                output_kd_weight=params["output_kd_weight"],
                trajectory_temperature=params["trajectory_temperature"],
                output_temperature=params["output_kd_temperature"],
                token_chunk=params["token_chunk"],
                loss_scale=kwargs.get("loss_scale", 1.0),
                backward=kwargs.get("backward"),
            ))
        raise RuntimeError(f"arm {arm} has no registered implementation")

    def unchunked(*a, **kw):
        if arm == "A":
            return original_loss(*a, **kw)
        if arm == "C":
            kw["mode"] = "delta"
            return original_loss(*a, **kw)
        raise RuntimeError(
            f"arm {arm} requires the registered chunked objective path; refusing fallback"
        )

    def definition(config, mapping):
        base = original_definition(config, mapping)
        base.update({
            "rq1_protocol": "RQ1_OBJECTIVES_V2",
            "rq1_arm": arm,
            "rq1_objective_id": arm_spec["id"],
            "rq1_objective_parameters": arm_spec.get("objective_parameters"),
        })
        if arm == "C":
            base.update({
                "objective": "topology_span_delta",
                "mode": "delta",
                "teacher_representation": "h_t[b]-h_t[a] over the complete assigned teacher span",
                "student_representation": "h_s[l+1]-h_s[l]",
                "span_semantics": "all teacher layers tile exactly once across student spans",
            })
        elif arm == "D":
            base.update({
                "objective": "pointwise_plus_adjacent_residual_delta",
                "mode": "pointwise+adjacent_residual",
                "prior_art_label": "internal abstraction ablation; NOT FDD",
            })
        elif arm == "E":
            base.update({
                "objective": "pointwise_plus_span_delta",
                "mode": "pointwise+topology_span",
            })
        elif arm == "B":
            base.update({
                "objective": "fdd_prediction_dynamics",
                "mode": "LM-head prediction-space trajectory + derivative + output KD",
                "mapping_note": "FDD uses its own uniformly sampled depth schedule; the generic type-preserving map is retained only because the validated layer_kd trainer constructs it before entering the loss seam.",
                "prior_art": "Gong et al., ACL 2025, 2025.acl-long.1125",
            })
        return base

    trainer.behavioral_loss_chunked = chunked
    trainer.behavioral_loss = unchunked
    trainer._layer_kd_definition = definition
    trainer.build_model = captured_build_model
    trainer.train = captured_train

    def restore():
        trainer.behavioral_loss_chunked = original_chunked
        trainer.behavioral_loss = original_loss
        trainer._layer_kd_definition = original_definition
        trainer.build_model = original_build_model
        trainer.train = original_train
    return trainer, restore


def _patch_config_builder(kd_run, protocol: dict, arm: str, output: Path):
    """Force hidden defaults to their registered explicit values and record resolution."""
    original = kd_run.build_config
    t = protocol["training"]
    arm_spec = protocol["arm_registry"][arm]

    def build(args, pretrained):
        config = original(args, pretrained)
        # Fields not exposed by kd_run's CLI are still scientific protocol values. Set them
        # explicitly here and then validate the complete resolved recipe.
        config.training.weight_decay = t["weight_decay"]
        config.training.warmup_steps = t["warmup_steps"]
        config.training.scheduler = t["scheduler"]
        config.training.gradient_checkpointing = t["gradient_checkpointing"]
        config.training.lora_dropout = t["lora_dropout"]
        config.training.kd_tail = t["kd_tail"]
        config.training.layer_kd_map_strategy = t["layer_kd_map_strategy"]
        config.training.progress_every = t["log_every"]
        config.objective = {
            **config.objective,
            "rq1_protocol": protocol["protocol_id"],
            "rq1_arm": arm,
            "rq1_objective_id": arm_spec["id"],
            "rq1_objective_parameters": arm_spec.get("objective_parameters"),
        }
        actual = {
            "sequence_length": config.data.max_sequence_length,
            "max_tokens": config.data.max_tokens,
            "steps": config.training.max_steps,
            "batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "strategy": config.training.strategy,
            "optimizer": config.training.optimizer,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "warmup_steps": config.training.warmup_steps,
            "scheduler": config.training.scheduler,
            "precision": config.training.precision,
            "gradient_checkpointing": config.training.gradient_checkpointing,
            "lora_rank": config.training.lora_rank,
            "lora_alpha": config.training.lora_alpha,
            "lora_dropout": config.training.lora_dropout,
            "seed": config.training.seed,
            "kd_temperature": config.training.kd_temperature,
            "kd_top_k": config.training.kd_top_k,
            "kd_tail": config.training.kd_tail,
            "layer_kd_direction_weight": config.training.layer_kd_direction_weight,
            "layer_kd_normalise": config.training.layer_kd_normalise,
            "layer_kd_map_strategy": config.training.layer_kd_map_strategy,
            "layer_kd_chunk_pairs": config.training.layer_kd_chunk_pairs,
            "eval_every": config.training.eval_every,
            "save_every": config.training.save_every,
            "log_every": config.training.log_every,
        }
        if actual != t:
            differing = {k: {"protocol": t.get(k), "actual": actual.get(k)} for k in t if t.get(k) != actual.get(k)}
            raise RuntimeError(f"resolved trainer config drifted from protocol: {differing}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "rq1_resolved_training.json").write_text(
            json.dumps({"arm": arm, "training": actual, "objective": config.objective}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return config

    kd_run.build_config = build
    return original


def _write_sidecar(output: Path, arm: str, protocol: dict, data_identity: dict, command: list[str]):
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "protocol_id": protocol["protocol_id"],
        "arm": arm,
        "arm_definition": protocol["arm_registry"][arm],
        "teacher": protocol["teacher"],
        "student": protocol["student"],
        "training": protocol["training"],
        "data_identity": data_identity,
        "git_sha": _git_sha(),
        "delegated_kd_run_args": command,
        "generic_trainer_objective": "layer_kd",
        "note": "The generic layer_kd trainer is the validated execution engine; this sidecar is authoritative about the RQ1 loss installed at its research loss seam.",
    }
    (output / "rq1_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = _load_protocol()
    spec = protocol["arm_registry"][args.arm]
    if spec["status"] not in READY:
        print(f"REFUSED: arm {args.arm} is not CPU-ready: {spec['status']}", file=sys.stderr)
        return 2
    if args.arm == "F":
        print("REFUSED: arm F weights are intentionally not preregistered yet", file=sys.stderr)
        return 2

    if not args.pretrained.is_dir():
        print(f"REFUSED: no canonical student directory at {args.pretrained}", file=sys.stderr)
        return 2
    try:
        identity = _verify_registered_data(protocol, args.teacher, args.text_path)
    except RuntimeError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    command = _trainer_args(args, protocol)
    _write_sidecar(args.output, args.arm, protocol, identity, command)

    trainer, restore = _patch_for_arm(args.arm, spec, args.output)
    kd_run = import_module("kd_run")
    original_builder = _patch_config_builder(kd_run, protocol, args.arm, args.output)
    # kd_run imported `train` after trainer.train was patched, so its local callable also
    # captures the live teacher provider required by FDD.
    try:
        return kd_run.main(command + (["--dry-run"] if args.dry_run else []))
    finally:
        kd_run.build_config = original_builder
        restore()


if __name__ == "__main__":
    raise SystemExit(main())
