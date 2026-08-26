#!/usr/bin/env python3
"""Is this checkpoint real? Structural integrity first, then behaviour.

A Level-2R run left three directories on Drive carrying ``COMPLETE``, ``metadata.json``
and every small file — and no ``model.safetensors``, no ``optimizer.pt``. ``latest.json``
said ``"complete": true``. This is the command that answers, in one line, whether that is
what you are looking at.

Two independent layers, because they fail differently:

**Integrity** — is every artifact present, non-empty, of a plausible size, matching its
recorded SHA-256, and does it deserialize? Runs without a GPU and without building the
model. This is what catches deletion, truncation and partial copies.

**Behaviour** — does the checkpoint reload into a fresh model bit-identically, produce
identical logits, generate reproducibly, and resume to the right step? Needs torch and
transformers. This is what catches a checkpoint that is intact and wrong.

Three things can be validated:

``a checkpoint directory``    one checkpoint
``a run directory``           every checkpoint under ``checkpoints/``, plus the pointer
``--persistent <path>``       the copy on Drive, which is the one that matters after a
                              runtime dies

Verification levels, in cost order::

    --level structure   existence, non-empty, plausible size          (no large reads)
    --level manifest    + SHA-256 against checkpoint_manifest.json    (reads everything)
    --level load        + deserialize weights and optimizer state     (default)

Examples::

    python scripts/validate_checkpoint.py experiments/runs/t4_level2r_100m_real_english
    python scripts/validate_checkpoint.py <checkpoint> --level manifest
    python scripts/validate_checkpoint.py /content/drive/MyDrive/run --persistent
    python scripts/validate_checkpoint.py <checkpoint> --behaviour   # reload + generate

Exit codes: ``0`` everything valid, ``1`` something is invalid, ``2`` could not run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.training.checkpoint_validation import (
    LEVELS,
    LOAD,
    STRUCTURE,
    checkpoint_directories,
    resolve_latest,
    validate_checkpoint_dir,
)

RULE = "=" * 78


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=Path, nargs="?",
                        help="a checkpoint directory, a run directory, or a persistent "
                             "run directory with --persistent")
    parser.add_argument("--checkpoint", dest="checkpoint_flag", type=Path,
                        help=argparse.SUPPRESS)   # kept so existing invocations work
    parser.add_argument("--persistent", action="store_true",
                        help="treat the path as a persistent (Drive) run directory")
    parser.add_argument("--level", choices=LEVELS, default=LOAD,
                        help="how thoroughly to check (default: load)")
    parser.add_argument("--inference-only", action="store_true",
                        help="do not require optimizer state; a checkpoint without it is "
                             "loadable but cannot continue training")
    parser.add_argument("--expect-parameters", type=int,
                        help="parameter count this checkpoint must hold, to catch weights "
                             "belonging to a different architecture")
    parser.add_argument("--behaviour", "--behavior", action="store_true",
                        dest="behaviour",
                        help="also reload into a fresh model and check logit identity, "
                             "generation determinism and resume (needs torch)")
    parser.add_argument("--device", default="cpu", help="device to reload onto (default cpu)")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--prompts", nargs="+")
    parser.add_argument("--json", type=Path, help="write the full report here")
    return parser


def _is_checkpoint_dir(path: Path) -> bool:
    return (path / "metadata.json").is_file() or (path / "training_state.pt").is_file()


def _checkpoint_root(path: Path, *, persistent: bool) -> Path | None:
    """Where the checkpoints live, given a checkpoint / run / persistent-run path."""
    if persistent or (path / "checkpoints").is_dir():
        return path / "checkpoints" if (path / "checkpoints").is_dir() else None
    if checkpoint_directories(path):
        return path
    return None


def _validate_one(path: Path, args: argparse.Namespace) -> int:
    validation = validate_checkpoint_dir(
        path, level=args.level,
        require_resumable=not args.inference_only,
        expected_parameter_count=args.expect_parameters,
    )
    print(validation.render())
    if args.json:
        _write_json(args.json, {"checkpoint": validation.to_dict()})
    return 0 if validation.valid else 1


def _validate_root(root: Path, args: argparse.Namespace, *, label: str) -> int:
    resolution = resolve_latest(
        root, level=args.level,
        require_resumable=not args.inference_only,
        expected_parameter_count=args.expect_parameters,
    )
    print(resolution.render_inventory(label=label))
    invalid = [v for v in resolution.checked if not v.valid]
    if invalid:
        print()
        print("-" * 78)
        print("WHY EACH INVALID CHECKPOINT FAILED")
        print("-" * 78)
        for validation in invalid:
            print()
            print(validation.render())
    if resolution.pointer_path and not resolution.pointer_valid:
        print()
        print("-" * 78)
        print(resolution.render())
    if args.json:
        _write_json(args.json, {"root": resolution.to_dict(),
                                "checkpoints": [v.to_dict() for v in resolution.checked]})
    if not resolution.checked:
        print("\n  no checkpoints here at all")
        return 1
    return 0 if not invalid else 1


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")


def _run_behaviour_checks(path: Path, args: argparse.Namespace) -> int:
    """Reload into a fresh model and check it behaves identically. Needs the stack."""
    from qwen_distill.training.validate_checkpoint import (
        DEFAULT_PROMPTS,
        validate_checkpoint,
        validate_resume,
    )

    print()
    print(RULE)
    print("BEHAVIOUR — reload, logit identity, generation determinism, resume")
    print(RULE)
    report = validate_checkpoint(
        path, device=args.device,
        prompts=tuple(args.prompts or DEFAULT_PROMPTS),
        max_new_tokens=args.max_new_tokens,
    )
    print(report.render())
    resume = validate_resume(path)
    print(f"\n  resume: step {resume.resumed_step} (expected {resume.expected_step}), "
          f"history preserved: {resume.history_preserved} -> "
          f"{'PASS' if resume.passed else 'FAIL'}")
    if resume.error:
        print(f"  ERROR: {resume.error}")
    return 0 if (report.passed and resume.passed) else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    path = args.path or args.checkpoint_flag
    if path is None:
        parser.error("a checkpoint, run, or persistent run directory is required")
    if not path.is_dir():
        print(f"not a directory: {path}", file=sys.stderr)
        return 2

    print(f"{RULE}\nCHECKPOINT VALIDATION  (level: {args.level})\n{RULE}\n")

    root = _checkpoint_root(path, persistent=args.persistent)
    if root is not None:
        label = "persistent checkpoints found" if args.persistent else "checkpoints found"
        status = _validate_root(root, args, label=label)
    elif _is_checkpoint_dir(path):
        status = _validate_one(path, args)
    else:
        print(f"{path} is neither a checkpoint directory nor a run directory "
              f"(no metadata.json, no checkpoints/)", file=sys.stderr)
        return 2

    if args.behaviour:
        target = path
        if root is not None:
            resolution = resolve_latest(root, level=STRUCTURE)
            if not resolution.resolved:
                print("\n  no valid checkpoint to run behaviour checks against",
                      file=sys.stderr)
                return 1
            target = Path(resolution.resolved)
        status = max(status, _run_behaviour_checks(target, args))

    return status


if __name__ == "__main__":
    sys.exit(main())
