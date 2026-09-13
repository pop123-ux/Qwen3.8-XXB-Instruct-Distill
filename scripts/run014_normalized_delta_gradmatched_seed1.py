#!/usr/bin/env python3
"""Run014: normalised residual-delta KD with a preregistered, gradient-matched multiplier.

Identical to Run011 except that the normalised residual-delta hidden objective's GRADIENT is
multiplied by one fixed scalar, read from the calibration manifest produced by
``scripts/run014_calibrate.py`` before any training. The multiplier rides in
``behavioral_loss_chunked``'s ``loss_scale`` (see ``gradient_calibration.scaled_hidden_loss``),
so it is applied exactly once and never to a detached logging value.

Consequence for the logs: the trainer's reported ``loss`` / ``layer_kd_loss`` is built from
detached floats and is therefore the UNWEIGHTED normalised residual-delta loss, directly
comparable to Run011's. The objective actually optimised is ``multiplier x`` that value.

The argv is Run011's (Run008's minus ``--layer-kd-no-normalise``) with only the output directory
and run name changed, and is checked against the archived Run011 arm manifest.

Refuses to run (fail closed) on: a dirty worktree; HEAD differing from the commit the calibration
ran at; a missing or incompatible calibration manifest; calibration sample hashes differing from
the preregistered freeze; a packed corpus SHA mismatch; a wrong teacher revision; a seed or
configuration that is not Run011's; or a recorded multiplier that does not equal the median
recomputed from the recorded gradient norms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

import run008_raw_delta_seed1 as run008
from qwen_distill.research import gradient_calibration as gc

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "run014_normalized_delta_gradmatched_seed1"
EXPERIMENT_DIR = REPO / "experiments" / EXPERIMENT_ID
PREREG = EXPERIMENT_DIR / "PREREG.md"
SAMPLE_FREEZE = EXPERIMENT_DIR / "sample_freeze.json"
RUN011_MANIFEST = REPO / "experiments" / "run011_normalized_delta_seed1" / "arm_manifest.json"
OUTPUT = Path("/workspace/runs") / EXPERIMENT_ID
MANIFEST = OUTPUT / "calibration_manifest.json"

NO_NORMALISE = "--layer-kd-no-normalise"
ARTIFACT_FLAGS = ("--output", "--name")
TEACHER_MODEL = "Qwen/Qwen3.8-27B"
TEACHER_REVISION = "dbdc473dea0d6a9763042881cc33d6058d1742d2"
STUDENT_ID = "qwen38_19b_h5120_l48_moe"
TRAINABLE_PARAMETERS = 23_003_136
EXPECTED_STEPS = 128


def command(output: Path = OUTPUT, name: str = EXPERIMENT_ID) -> list[str]:
    """Run011's argv with only the artifact paths redirected."""
    argv = [tok for tok in run008.command() if tok != NO_NORMALISE]
    argv[argv.index("--output") + 1] = str(output)
    argv[argv.index("--name") + 1] = name
    return argv


def _strip_artifacts(argv: list[str]) -> list[str]:
    out, skip = [], False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok in ARTIFACT_FLAGS:
            skip = True
            continue
        out.append(tok)
    return out


def _flag(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def expected_provenance(*, git_commit: str, prereg_sha256: str,
                        freeze: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "method_version": gc.METHOD_VERSION,
        "calibration_indices": list(gc.CALIBRATION_INDICES),
        "token_hashes": {str(k): v for k, v in freeze["token_hashes"].items()},
        "packed_corpus_sha256": gc.PACKED_CORPUS_SHA256,
        "teacher_model": TEACHER_MODEL,
        "teacher_revision": TEACHER_REVISION,
        "student_id": STUDENT_ID,
        "seed": 1,
        "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "preregistration_sha256": prereg_sha256,
        "git_commit": git_commit,
    }


def preflight(*, worktree_status: str, head: str, manifest: dict[str, Any] | None,
              prereg_sha256: str, freeze: dict[str, Any], packed_sha256: str,
              argv: list[str], run011_argv: list[str]) -> float:
    """Every fail-closed check Run014 requires. Returns the verified multiplier or raises."""
    if worktree_status.strip():
        raise gc.CalibrationError(f"worktree is dirty:\n{worktree_status}")
    if manifest is None:
        raise gc.CalibrationError(f"calibration manifest missing: {MANIFEST}")
    if tuple(freeze.get("calibration_indices", ())) != gc.CALIBRATION_INDICES:
        raise gc.CalibrationError("sample freeze indices differ from the preregistered indices")

    if NO_NORMALISE in argv:
        raise gc.CalibrationError("Run014 must use the NORMALISED objective; no-normalise flag present")
    if _flag(argv, "--revision") != TEACHER_REVISION:
        raise gc.CalibrationError(f"teacher revision {_flag(argv, '--revision')!r} != {TEACHER_REVISION}")
    if _flag(argv, "--seed") != "1":
        raise gc.CalibrationError(f"seed {_flag(argv, '--seed')!r} != '1'")
    if _flag(argv, "--steps") != str(EXPECTED_STEPS):
        raise gc.CalibrationError(f"steps {_flag(argv, '--steps')!r} != {EXPECTED_STEPS}")
    if _flag(argv, "--layer-kd-chunk-pairs") != "4":
        raise gc.CalibrationError("chunk pairs must be 4: the multiplier rides in the chunked loss_scale")
    if _strip_artifacts(argv) != _strip_artifacts(run011_argv):
        raise gc.CalibrationError("argv differs from Run011 beyond --output/--name")

    if packed_sha256 != gc.PACKED_CORPUS_SHA256:
        raise gc.CalibrationError(f"packed corpus sha {packed_sha256} != {gc.PACKED_CORPUS_SHA256}")
    if manifest.get("git_commit") != head:
        raise gc.CalibrationError(
            f"HEAD {head} != calibration commit {manifest.get('git_commit')}: Run014 must train at "
            "the exact commit it was calibrated from")
    expected = expected_provenance(git_commit=head, prereg_sha256=prereg_sha256, freeze=freeze)
    return gc.validate_manifest(manifest, expected)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _packed_corpus_sha256(argv: list[str]) -> str:
    import kd_run
    from qwen_distill.training.tokenized_data import prepare_tokenized_corpus

    parsed = kd_run.parse_args(argv)
    config = kd_run.build_config(parsed, Path(parsed.pretrained))
    d = config.data
    _, _, stats = prepare_tokenized_corpus(
        text_path=d.text_path, tokenizer_path=d.tokenizer_path,
        sequence_length=d.max_sequence_length, validation_fraction=d.validation_fraction,
        document_separator=d.document_separator, max_documents=d.max_documents,
        max_tokens=d.max_tokens, max_bytes=d.max_corpus_bytes,
        expected_vocab_size=d.expected_vocab_size, teacher_model=config.teacher.get("model"),
        teacher_revision=config.teacher.get("revision"),
        trust_remote_code=config.model.trust_remote_code,
    )
    return stats.sha256


def main(cli: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="run every fail-closed check, then exit without training")
    args = ap.parse_args(cli)

    argv = command()
    freeze = json.loads(SAMPLE_FREEZE.read_text())
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else None
    try:
        multiplier = preflight(
            worktree_status=_git("status", "--porcelain"), head=_git("rev-parse", "HEAD"),
            manifest=manifest, prereg_sha256=hashlib.sha256(PREREG.read_bytes()).hexdigest(),
            freeze=freeze, packed_sha256=_packed_corpus_sha256(argv), argv=argv,
            run011_argv=json.loads(RUN011_MANIFEST.read_text())["command"],
        )
    except gc.CalibrationError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(f"  preflight PASS; frozen multiplier = {multiplier!r}")
    if args.dry_run:
        print("  dry run: every fail-closed check passed; nothing was trained.")
        return 0
    if (OUTPUT / "summary.json").exists():
        print(f"REFUSED: completed evidence exists at {OUTPUT / 'summary.json'}", file=sys.stderr)
        return 2

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "arm_manifest.json").write_text(json.dumps({
        "experiment": EXPERIMENT_ID,
        "purpose": ("gradient-scale-matched control: does normalised residual-delta KD remain much "
                    "worse than raw once its initial trainable gradient scale is matched?"),
        "reference_runs": {"run011_normalized_delta_seed1": 10.8220,
                           "run008_raw_delta_seed1": 7.41131,
                           "run009_pointwise_seed1": 8.73926},
        "only_intended_scientific_change_vs_run011": (
            "normalised residual-delta hidden-objective gradient multiplied by the frozen "
            "calibration multiplier, via behavioral_loss_chunked loss_scale"),
        "multiplier": multiplier,
        "calibration_manifest": str(MANIFEST),
        "calibration_git_commit": manifest["git_commit"],
        "preregistration_sha256": manifest["preregistration_sha256"],
        "logged_loss_semantics": ("summary loss / layer_kd_loss are the UNWEIGHTED normalised "
                                  "residual-delta loss (comparable to Run011); optimised objective "
                                  "= multiplier x that value"),
        "objective": "behavioral_kd", "behavioral_mode": "delta", "normalise": True,
        "seed": 1, "teacher_revision": TEACHER_REVISION,
        "command": argv,
    }, indent=2) + "\n", encoding="utf-8")

    from kd_run import main as kd_main
    from run004_behavioral_kd import patch_trainer_for_delta, restore_trainer
    from qwen_distill.training import trainer as trainer_module

    patched = patch_trainer_for_delta()
    try:
        with gc.scaled_hidden_loss(trainer_module, multiplier) as calls:
            rc = kd_main(argv)
    finally:
        restore_trainer(patched)
    print(f"  scaled hidden-loss invocations: {calls['n']} (expected {EXPECTED_STEPS})")
    if rc == 0 and calls["n"] != EXPECTED_STEPS:
        print(f"WARNING: expected {EXPECTED_STEPS} scaled invocations, saw {calls['n']}",
              file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
