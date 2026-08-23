#!/usr/bin/env python3
"""Validate a locally supplied metadata directory against Phase 1's requirements.

Reports FOUND / MISSING / OPTIONAL / UNKNOWN for every expected file and field.
Crucially, a field is FOUND only when its **value was parsed and read** — never merely
because the file containing it exists. And a field the metadata cannot settle at all
(anything needing weights or a runtime experiment) is UNKNOWN, not MISSING: collapsing
those two would overstate what has been verified.

Runs entirely offline. Never contacts the Hub.

Example::

    python scripts/validate_teacher_metadata.py --path vendor/qwen38-metadata
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.teacher.metadata import (
    blocking_gaps,
    build_verified_spec,
    hash_metadata_files,
    implementation_disagreements,
    load_metadata,
    summarise_counts,
    validate_metadata,
)
from qwen_distill.utils.offline import offline_mode

STATUS_ORDER = ("FOUND", "MISSING", "OPTIONAL", "UNKNOWN")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--path", type=Path, default=Path("vendor/qwen38-metadata"),
        help="directory holding the supplied metadata files",
    )
    parser.add_argument("--json", type=Path, help="write the structured report here")
    parser.add_argument(
        "--strict", action="store_true",
        help="exit non-zero if any required file or field is MISSING",
    )
    parser.add_argument(
        "--save-verified", type=Path,
        help="write the canonical verified teacher specification here "
             "(generated from the metadata, stamped with file hashes)",
    )
    parser.add_argument("--source", default="Qwen/Qwen3.8-27B", help="upstream model id")
    parser.add_argument("--revision", help="upstream commit SHA, if known")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.path.is_dir():
        print(f"Metadata directory not found: {args.path}\n", file=sys.stderr)
        print("Create it and place the upstream files inside:", file=sys.stderr)
        print(f"  mkdir -p {args.path}", file=sys.stderr)
        print("  # config.json, tokenizer.json, tokenizer_config.json,", file=sys.stderr)
        print("  # generation_config.json, special_tokens_map.json", file=sys.stderr)
        print("\nSee vendor/README.md for how to obtain them.", file=sys.stderr)
        return 2

    with offline_mode():
        metadata = load_metadata(args.path)
        files, fields = validate_metadata(metadata)

    print(f"metadata directory : {args.path}")
    print("network access     : disabled (offline mode enforced)\n")

    print("--- files ---")
    width = max(len(f.name) for f in files)
    for report in files:
        size = f"{report.size_bytes:,} B" if report.size_bytes is not None else ""
        line = f"  {report.name:<{width}}  {report.status:<9}{size:>12}"
        if report.parse_error:
            line += f"   ! {report.parse_error}"
        elif report.status == "MISSING":
            line += f"   blocks: {report.blocks}"
        print(line)

    print("\n--- fields ---")
    width = max(len(f.name) for f in fields)
    for report in fields:
        line = f"  {report.name:<{width}}  {report.status:<9} {report.display_value()}"
        print(line)
        if report.status in ("MISSING", "UNKNOWN") and report.note:
            print(f"  {'':<{width}}  {'':<9} -> {report.note}")

    counts = summarise_counts(files, fields)
    print("\n--- summary ---")
    print("  " + "   ".join(f"{status}: {counts[status]}" for status in STATUS_ORDER))

    gaps = blocking_gaps(files, fields)
    if gaps:
        print("\n--- blocking gaps ---")
        for gap in gaps:
            print(f"  ! {gap}")
    else:
        print("\n  No blocking gaps: every required file and field was read.")

    print("\n  Note: UNKNOWN entries are not gaps in the supplied files. They are facts")
    print("  that metadata cannot establish at all (they need the weights or a runtime")
    print("  experiment) and must stay UNKNOWN in docs/VERIFICATION.md.")

    digests = hash_metadata_files(args.path)
    if digests:
        print("\n--- file hashes (SHA-256) ---")
        for name, digest in digests.items():
            print(f"  {name:<26}{digest}")

    disagreements = implementation_disagreements(metadata)
    if disagreements:
        print("\n--- config keys the installed transformers does not read ---")
        for line in disagreements:
            print(f"  - {line}")
        print("  These are not errors. They matter because a key that looks like it")
        print("  controls behaviour may not, and a future release could start honouring it.")

    if args.save_verified:
        spec = build_verified_spec(metadata, source=args.source, revision=args.revision)
        args.save_verified.parent.mkdir(parents=True, exist_ok=True)
        args.save_verified.write_text(
            json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nwrote verified teacher specification: {args.save_verified}")
        print(f"  parameters (text tower): {spec['parameters']['total']:,}")
        if spec["provenance"]["revision"] is None:
            print("  WARNING: revision unpinned - see provenance.revision_note")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "path": str(args.path),
                    "offline": True,
                    "files": [vars(f) | {"status": f.status} for f in files],
                    "fields": [
                        {"name": f.name, "status": f.status, "value": f.value,
                         "source": f.source, "note": f.note}
                        for f in fields
                    ],
                    "counts": counts,
                    "blocking_gaps": gaps,
                    "parse_errors": metadata.errors,
                },
                indent=2, ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    if args.strict and gaps:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
