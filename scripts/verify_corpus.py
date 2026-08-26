#!/usr/bin/env python3
"""Check a prepared corpus against its own manifest. Offline, always.

**No network access. Ever.** Every check recomputes from the files on disk. This is not
an incidental property — the Gutenberg catalogue is unreachable from the environment this
project's tooling was built in, and working around that restriction is out of scope. So
the corpus has to be checkable without it.

Two failures this catches:

**A corpus that is not the corpus you think it is.** ``prepare_level2r_dataset.py`` names
works by numeric Gutenberg id, and those ids could not be verified against the live
catalogue. A wrong id silently downloads a different book. The manifest records the title
each file *declares about itself*, and ``--level2r`` compares that against what was
actually requested. A wrong id shows up as a mismatch, with no network involved.

**Contamination.** A document-level split makes verbatim train/validation overlap
impossible — unless the same work appears under two ids, or an anthology reprints a
held-out text. Then the validation score measures memorisation and looks like learning.

Also checked: SHA-256 of both splits against the manifest, byte totals, UTF-8 validity,
NFC normalisation, line endings, leftover Gutenberg boilerplate, split proportions, and
how many steps the corpus is worth at Level 2's measured throughput.

Exit codes: ``0`` clean, ``1`` errors found, ``2`` could not run.

Examples::

    python scripts/verify_corpus.py data/level2r
    python scripts/verify_corpus.py data/level2r --level2r        # check book identity
    python scripts/verify_corpus.py data/level2r --expect titles.json --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.training.corpus_verify import verify_corpus


def _level2r_expected_titles() -> dict[str, str]:
    """The id -> title table the Level-2R preparation script asked for.

    Imported from the script rather than copied, so the two cannot drift. If they were
    duplicated, an edit to the corpus would leave this checking the old list and
    reporting a clean bill of health for the wrong books.
    """
    from prepare_level2r_dataset import TRAIN_IDS, VALIDATION_IDS

    return {identifier: title for identifier, title in (*TRAIN_IDS, *VALIDATION_IDS)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path,
                        help="corpus directory (train.txt, validation.txt, corpus_manifest.json)")
    parser.add_argument("--level2r", action="store_true",
                        help="check document identity against the Level-2R id table")
    parser.add_argument("--expect", type=Path,
                        help='JSON mapping identifier -> requested title, e.g. {"1342": "Pride and Prejudice"}')
    parser.add_argument("--tokens-per-step", type=int, default=16_384,
                        help="bytes consumed per optimizer step (default: Level 2's 16384)")
    parser.add_argument("--overlap-samples", type=int, default=64,
                        help="validation passages sampled for the contamination check")
    parser.add_argument("--json", type=Path, help="write the full report here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.directory.is_dir():
        # An invocation error, not a corpus error: exit 2 so a caller can tell "you
        # pointed me at nothing" from "this corpus is bad".
        print(f"not a directory: {args.directory}", file=sys.stderr)
        return 2

    expected: dict[str, str] = {}
    if args.level2r:
        try:
            expected.update(_level2r_expected_titles())
        except ImportError as exc:
            print(f"could not import the Level-2R id table: {exc}", file=sys.stderr)
            return 2
    if args.expect:
        try:
            loaded = json.loads(args.expect.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"could not read {args.expect}: {exc}", file=sys.stderr)
            return 2
        if not isinstance(loaded, dict):
            print(f"{args.expect} must be a JSON object mapping id -> title", file=sys.stderr)
            return 2
        expected.update({str(k): str(v) for k, v in loaded.items()})

    report = verify_corpus(
        args.directory,
        expected_titles=expected or None,
        tokens_per_step=args.tokens_per_step,
        overlap_samples=args.overlap_samples,
    )
    print(report.render())

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
