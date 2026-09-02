#!/usr/bin/env python3
"""Create, checksum, archive and verify the Run 001 experiment record.

The record is what survives the Pod. This script is the only thing that writes it, so
there is one place that decides what a complete record is::

    # before the run: capture everything knowable up front
    python scripts/run_record.py init --config configs/experiments/<run001>.yaml \
        --teacher /workspace/models/qwen3.8-27b-dbdc473 \
        --corpus /workspace/corpora/gutenberg/corpus_manifest.json \
        --command "python scripts/train_student.py --config ..."

    # while or after the run
    python scripts/run_record.py checksums              # SHA-256 every artefact
    python scripts/run_record.py archive                # copy the text record into Git
    python scripts/run_record.py terminate --status failed --reason "CUDA OOM at step 1200"

    # before destroying the Pod
    python scripts/run_record.py verify                 # the pre-termination checklist

Exit codes: ``0`` the action succeeded (and, for ``verify``, every item passed); ``1``
verification failed; ``2`` the request could not be set up.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.training.run_record import (
    ARCHIVE_ROOT,
    DEFAULT_RUN_ROOT,
    RUN_ID,
    archive_to_repository,
    build_manifest,
    initialise_run,
    record_termination,
    verify_record,
    write_checksums,
)

RULE = "=" * 78
REPOSITORY = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT,
                        help=f"the run directory (default: {DEFAULT_RUN_ROOT})")
    sub = parser.add_subparsers(dest="action", required=True)

    init = sub.add_parser("init", help="create the run directory and write the manifest")
    init.add_argument("--config", type=Path, help="experiment YAML the run will use")
    init.add_argument("--teacher", type=Path, help="teacher checkpoint directory")
    init.add_argument("--tokenizer", type=Path,
                      help="tokenizer directory (defaults to --teacher)")
    init.add_argument("--corpus", type=Path, help="corpus_manifest.json")
    init.add_argument("--command", help="the exact command that will launch the run")
    init.add_argument("--note", action="append", default=[],
                      help="free-text note to record in the manifest; repeatable")

    checksums = sub.add_parser("checksums", help="SHA-256 every artefact")
    checksums.add_argument("--max-bytes", type=int, default=None,
                           help="skip files larger than this (fast pass mid-run)")
    checksums.add_argument("--external", action="append", default=[],
                           metavar="RELPATH=LOCATION",
                           help="record where a large artefact was backed up; repeatable")

    archive = sub.add_parser("archive", help="copy the text record into the repository")
    archive.add_argument("--into", type=Path, default=REPOSITORY / ARCHIVE_ROOT)

    terminate = sub.add_parser("terminate", help="record how the run ended")
    terminate.add_argument("--status", required=True,
                           choices=("completed", "failed", "interrupted", "oom"))
    terminate.add_argument("--reason", required=True)
    terminate.add_argument("--exit-code", type=int)
    terminate.add_argument("--last-step", type=int)

    sub.add_parser("verify", help="run the pre-termination checklist")
    return parser.parse_args(argv)


def _load_config(path: Path | None) -> tuple[dict | None, str | None]:
    if path is None:
        return None, None
    if not path.is_file():
        print(f"  ERROR: no config at {path}")
        raise SystemExit(2)
    from qwen_distill.training.config import ExperimentConfig

    return ExperimentConfig.load(path).to_dict(), str(path)


def do_init(args: argparse.Namespace) -> int:
    config, config_path = _load_config(args.config)
    tokenizer = args.tokenizer or args.teacher
    command = args.command
    if command is None and config_path:
        command = f"python scripts/train_student.py --config {shlex.quote(config_path)}"
    notes = list(args.note)
    if config is None:
        notes.append(
            "No experiment config was supplied at init time, so the training and KD "
            "sections of this manifest are unpopulated. Re-run `run_record.py init` "
            "with --config before the run starts; it is idempotent."
        )
    manifest = build_manifest(
        repository=REPOSITORY, config=config, config_path=Path(config_path) if config_path else None,
        teacher_directory=args.teacher, tokenizer_path=tokenizer,
        corpus_manifest=args.corpus, command=command, notes=notes,
    )
    written = initialise_run(args.root, manifest, config=config, command=command)
    print(f"{RULE}\nRUN RECORD: {RUN_ID}\n{RULE}\n")
    print(f"  root : {args.root}")
    for path in written:
        print(f"    wrote {path.relative_to(args.root)}")
    git = manifest["git"]
    print(f"\n  commit  : {git['commit']}  ({'DIRTY' if git['dirty'] else 'clean'})")
    print(f"  branch  : {git['branch']}")
    print(f"  teacher : {manifest['teacher'].get('model')} @ "
          f"{manifest['teacher'].get('revision') or 'UNPINNED'}")
    print(f"  corpus  : {manifest['dataset'].get('name') or 'unset'}")
    if manifest["teacher"].get("revision") is None:
        print("\n  WARNING: the teacher revision is not recorded. This run would not be "
              "reproducible from the Hub.")
    return 0


def do_checksums(args: argparse.Namespace) -> int:
    external = {}
    for item in args.external:
        if "=" not in item:
            print(f"  ERROR: --external expects RELPATH=LOCATION, got {item!r}")
            return 2
        key, value = item.split("=", 1)
        external[key] = value
    target = write_checksums(args.root, external_locations=external, max_bytes=args.max_bytes)
    entries = sum(1 for line in target.read_text(encoding="utf-8").splitlines()
                  if line and not line.startswith("#"))
    print(f"  wrote {target} ({entries} artefacts)")
    return 0


def do_archive(args: argparse.Namespace) -> int:
    index = archive_to_repository(args.root, args.into)
    print(f"  archived into {args.into}")
    for entry in index["copied"]:
        print(f"    copied     {entry['name']:<28} {entry['bytes']:>10,} bytes")
    for entry in index["referenced_not_copied"]:
        print(f"    referenced {entry['name']:<28} {entry['bytes']:>10,} bytes  "
              f"({entry['reason']})")
    return 0


def do_terminate(args: argparse.Namespace) -> int:
    target = record_termination(
        args.root, status=args.status, reason=args.reason,
        exit_code=args.exit_code, last_step=args.last_step,
    )
    print(f"  wrote {target}: {args.status} — {args.reason}")
    return 0


def do_verify(args: argparse.Namespace) -> int:
    report = verify_record(args.root)
    print(f"{RULE}\nPRE-TERMINATION VERIFICATION: {RUN_ID}\n{RULE}\n")
    for item in report["items"]:
        mark = "[x]" if item["ok"] else "[ ]"
        print(f"  {mark} {item['item']:<34} {item['detail']}")
    print(f"\n  {report['status']}\n")
    (Path(args.root) / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if report["verified"] else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return {
        "init": do_init, "checksums": do_checksums, "archive": do_archive,
        "terminate": do_terminate, "verify": do_verify,
    }[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
