#!/usr/bin/env python3
"""Download the exact pinned Qwen3.8-27B checkpoint. That is all it does.

It does not load the model, run inference, materialise the student, generate teacher data
or train. Keeping it to one job is deliberate: this script answers "were the files for the
requested revision fetched?", and it is **not** evidence that the weights are structurally
correct or that they load as the intended teacher. That remains the job of
``load_verified_teacher()`` and ``scripts/teacher_smoke_test.py``, whose missing-weight gate
is the authoritative protection against a freshly-initialised model masquerading as the
teacher.

The revision check is not reimplemented here. It reuses ``TeacherLoadPlan.validate()``, so
this script and the loader can never drift into accepting different things.

Workflow::

    python scripts/download_teacher.py --revision <SHA> --output /data/models/qwen3.8-27b
    python scripts/teacher_smoke_test.py --local-path /data/models/qwen3.8-27b \\
        --revision <SHA> --quantization 4bit --json runs/teacher_smoke.json
    python scripts/distill_pilot.py --teacher /data/models/qwen3.8-27b --revision <SHA>

Authentication comes from the Hugging Face CLI or ``HF_TOKEN`` in the environment. No token
is read from or written to a file in this repository.

Exit codes: ``0`` complete, ``1`` the download finished but the checkpoint is incomplete,
``2`` the request was refused before any network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.distillation.real_teacher import (
    DEFAULT_TEACHER_MODEL,
    TeacherLoadPlan,
)

MANIFEST_NAME = "teacher_download_manifest.json"

#: What ``load_verified_teacher`` needs to exist before it can even try. Weight shards are
#: checked separately because their naming is a packaging detail that upstream can change.
REQUIRED_FILES: tuple[str, ...] = ("config.json",)

#: Any one of these satisfies the tokenizer requirement. ``AutoTokenizer`` accepts several
#: layouts and pinning one filename would reject a valid checkpoint.
TOKENIZER_ANY: tuple[str, ...] = (
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json",
)

#: Weight files, by glob rather than by exact name: sharded and single-file checkpoints are
#: both valid and the shard count is not ours to predict.
WEIGHT_GLOBS: tuple[str, ...] = (
    "*.safetensors", "*.bin", "*.safetensors.index.json", "*.bin.index.json",
)

#: Hashing 54 GB to verify a download would take longer than the download. Files at or under
#: this size get a checksum; larger ones are recorded by size only, and the manifest says so.
CHECKSUM_MAX_BYTES = 64 * 1024 * 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--revision", required=True,
                        help="exact commit SHA. Branch names, tags and 'main' are refused.")
    parser.add_argument("--model", default=DEFAULT_TEACHER_MODEL, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=Path("/data/models/qwen3.8-27b"),
                        help="destination directory. Defaults outside the repository on "
                             "purpose: a 54 GB checkpoint must never land in the git tree.")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the request and the destination, download nothing")
    parser.add_argument("--verify-only", action="store_true",
                        help="check an existing directory for completeness and rewrite the "
                             "manifest; download nothing")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="parallel file downloads")
    return parser.parse_args(argv)


def inside_repository(path: Path) -> bool:
    """True when ``path`` would land inside this checkout.

    A 54 GB checkpoint written into ``src/`` or ``tests/`` would be catastrophic for the
    working tree, so the destination is checked rather than trusted.
    """
    repo = Path(__file__).resolve().parent.parent
    try:
        Path(path).resolve().relative_to(repo)
    except ValueError:
        return False
    return True


def inventory(directory: Path) -> list[dict[str, object]]:
    """Every regular file under ``directory``, with size and — where cheap — a checksum."""
    files: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        size = path.stat().st_size
        entry: dict[str, object] = {
            "path": str(path.relative_to(directory)), "bytes": size, "sha256": None,
        }
        if size <= CHECKSUM_MAX_BYTES:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            entry["sha256"] = digest.hexdigest()
        files.append(entry)
    return files


def completeness(directory: Path) -> list[str]:
    """Reasons the directory is not a usable teacher checkpoint. Empty means it looks whole.

    This is a *presence* check. It cannot tell a truncated shard from a good one, and it is
    not meant to: the smoke test's missing-weight gate is what proves the weights load.
    """
    problems: list[str] = []
    if not directory.is_dir():
        return [f"{directory} is not a directory"]
    for name in REQUIRED_FILES:
        if not (directory / name).exists():
            problems.append(f"missing {name}")
    if not any((directory / name).exists() for name in TOKENIZER_ANY):
        problems.append(
            "no tokenizer files: AutoTokenizer needs one of " + ", ".join(TOKENIZER_ANY)
            + ". The distillation path needs the teacher's own tokenizer and the vendored "
              "metadata cannot substitute."
        )
    weights = [p for pattern in WEIGHT_GLOBS for p in directory.glob(pattern)]
    if not weights:
        problems.append("no weight files (*.safetensors / *.bin) and no shard index")
    else:
        index = [p for p in weights if p.name.endswith("index.json")]
        if index:
            try:
                weight_map = json.loads(index[0].read_text(encoding="utf-8"))["weight_map"]
            except (OSError, ValueError, KeyError) as exc:
                problems.append(f"{index[0].name} is unreadable: {exc}")
            else:
                absent = sorted({s for s in weight_map.values()
                                 if not (directory / s).exists()})
                if absent:
                    problems.append(
                        f"{len(absent)} shard(s) named by {index[0].name} are missing, "
                        f"starting with {absent[0]}"
                    )
    empty = [p.name for pattern in WEIGHT_GLOBS for p in directory.glob(pattern)
             if p.stat().st_size == 0]
    if empty:
        problems.append(f"zero-length weight files: {', '.join(sorted(empty)[:5])}")
    return problems


def write_manifest(directory: Path, model: str, revision: str) -> Path:
    """A small record of what was fetched. Not a copy of the checkpoint."""
    try:
        import huggingface_hub

        hub_version = huggingface_hub.__version__
    except ImportError:
        hub_version = None
    files = inventory(directory)
    manifest = {
        "model": model,
        "revision": revision,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "local_path": str(directory.resolve()),
        "n_files": len(files),
        "total_bytes": sum(int(f["bytes"]) for f in files),
        "huggingface_hub_version": hub_version,
        "checksum_policy": (
            f"sha256 for files up to {CHECKSUM_MAX_BYTES} bytes; larger files are recorded "
            "by size only, because hashing the full checkpoint costs more than fetching it"
        ),
        "files": files,
        "verifies": "that the requested revision's files are present and non-empty",
        "does_not_verify": (
            "that the weights load as the intended teacher. Run "
            "scripts/teacher_smoke_test.py, whose missing-weight gate is the authoritative "
            "check: transformers returns a freshly-initialised model rather than raising "
            "when a checkpoint's keys do not match."
        ),
    }
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # One validator, shared with the loader, so the two can never drift apart.
    plan = TeacherLoadPlan(model=args.model, revision=args.revision)
    problems = plan.validate()
    if problems:
        print("  refused:\n    - " + "\n    - ".join(problems), file=sys.stderr)
        return 2

    destination = args.output
    if inside_repository(destination):
        print(f"  refused: {destination} is inside the repository. A 54 GB checkpoint must "
              "not be written into\n  the git tree — pass --output somewhere outside it.",
              file=sys.stderr)
        return 2

    print(f"  model    : {args.model}")
    print(f"  revision : {args.revision}")
    print(f"  output   : {destination}")

    if args.verify_only or args.dry_run:
        if args.dry_run and not args.verify_only:
            print("\n  dry run: the request is valid; nothing was downloaded.")
            return 0
        issues = completeness(destination)
        if issues:
            print("\n  INCOMPLETE:\n    - " + "\n    - ".join(issues), file=sys.stderr)
            return 1
        print(f"\n  complete. manifest: {write_manifest(destination, args.model, args.revision)}")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  huggingface_hub is not installed (pip install huggingface_hub)",
              file=sys.stderr)
        return 2

    destination.mkdir(parents=True, exist_ok=True)
    print("\n  downloading (resumable; re-run to continue an interrupted fetch) ...")
    try:
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            local_dir=str(destination),
            max_workers=args.max_workers,
        )
    except Exception as exc:  # noqa: BLE001 - the failure is the result
        print(f"\n  download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    issues = completeness(destination)
    if issues:
        print("\n  the download finished but the checkpoint is incomplete:\n    - "
              + "\n    - ".join(issues), file=sys.stderr)
        return 1

    manifest = write_manifest(destination, args.model, args.revision)
    print(f"\n  complete. manifest: {manifest}")
    print("\n  This proves the files arrived. It does NOT prove they load as the teacher.")
    print("  Next:\n"
          f"    python scripts/teacher_smoke_test.py --local-path {destination} \\\n"
          f"        --revision {args.revision} --quantization 4bit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
