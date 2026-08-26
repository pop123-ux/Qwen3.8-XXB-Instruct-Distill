#!/usr/bin/env python3
"""Build the Level 2R corpus: real public-domain English, deterministically.

Level 2 trained on procedural text and saturated by step 400, then generated
`"and and and"`. Both facts follow from the corpus: it had word frequencies and no
syntax. Level 2R asks whether the architecture can learn real language, so it needs real
language — enough of it that the model cannot exhaust it, and split so that validation
measures generalisation rather than continuation.

**The corpus is never committed.** This script reconstructs it from catalogue ids, and
records a manifest with every hash so a result can be tied to the exact bytes that
produced it.

**The split is at document level.** Whole books go to train or validation and never both.
Level 2 held out a contiguous tail of one text, which measures how well a model continues
a passage it has been reading; holding out whole works measures generalisation to prose
it has never seen. The validation set is written once and fixed for the experiment.

Examples::

    # download and build (needs network — run this in Colab)
    python scripts/prepare_level2r_dataset.py --output data/level2r

    # build from files already on disk, no network at all
    python scripts/prepare_level2r_dataset.py --output data/level2r --from-local downloads/

    # see the plan without fetching anything
    python scripts/prepare_level2r_dataset.py --output data/level2r --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.training.corpus import (
    PREPARATION_VERSION,
    Document,
    assign_splits,
    build_corpus,
    check_overlap,
    estimate_run,
    extract_metadata,
    load_documents_from_directory,
    normalise,
    strip_gutenberg_boilerplate,
)

RULE = "=" * 72

#: Project Gutenberg plain-text endpoint. The catalogue ids below are the identifiers;
#: the manifest records the title each file actually declares, so a wrong id is visible
#: rather than assumed away.
GUTENBERG_URL = "https://www.gutenberg.org/ebooks/{id}.txt.utf-8"
GUTENBERG_MIRROR = "https://gutenberg.pglaf.org/{a}/{b}/{id}/{id}.txt"

PUBLIC_DOMAIN_BASIS = (
    "Project Gutenberg distributes these works as public domain in the United States; "
    "all are pre-1929 publications whose copyright has expired."
)

#: Curated English prose, chosen for length and variety of author, period and register.
#: Titles are what we expect; the script records what each file actually says.
TRAIN_IDS: tuple[tuple[str, str], ...] = (
    ("1342", "Pride and Prejudice — Austen"),
    ("158", "Emma — Austen"),
    ("161", "Sense and Sensibility — Austen"),
    ("84", "Frankenstein — Shelley"),
    ("1661", "The Adventures of Sherlock Holmes — Doyle"),
    ("2097", "The Sign of the Four — Doyle"),
    ("2701", "Moby Dick — Melville"),
    ("98", "A Tale of Two Cities — Dickens"),
    ("1400", "Great Expectations — Dickens"),
    ("46", "A Christmas Carol — Dickens"),
    ("730", "Oliver Twist — Dickens"),
    ("766", "David Copperfield — Dickens"),
    ("174", "The Picture of Dorian Gray — Wilde"),
    ("76", "Adventures of Huckleberry Finn — Twain"),
    ("74", "The Adventures of Tom Sawyer — Twain"),
    ("11", "Alice's Adventures in Wonderland — Carroll"),
    ("12", "Through the Looking-Glass — Carroll"),
    ("345", "Dracula — Stoker"),
    ("1260", "Jane Eyre — Brontë"),
    ("768", "Wuthering Heights — Brontë"),
    ("145", "Middlemarch — Eliot"),
    ("120", "Treasure Island — Stevenson"),
    ("43", "The Strange Case of Dr Jekyll and Mr Hyde — Stevenson"),
    ("35", "The Time Machine — Wells"),
    ("36", "The War of the Worlds — Wells"),
    ("5230", "The Invisible Man — Wells"),
    ("16", "Peter Pan — Barrie"),
    ("205", "Walden — Thoreau"),
    ("219", "Heart of Darkness — Conrad"),
    ("2591", "Grimms' Fairy Tales"),
    ("1232", "The Prince — Machiavelli"),
    ("2600", "War and Peace — Tolstoy"),
    ("1399", "Anna Karenina — Tolstoy"),
    ("2554", "Crime and Punishment — Dostoyevsky"),
    ("28054", "The Brothers Karamazov — Dostoyevsky"),
    ("1184", "The Count of Monte Cristo — Dumas"),
    ("135", "Les Misérables — Hugo"),
    ("6130", "The Iliad — Homer"),
    ("1727", "The Odyssey — Homer"),
    ("2814", "Dubliners — Joyce"),
    ("863", "The Mysterious Affair at Styles — Christie"),
    ("541", "The Age of Innocence — Wharton"),
    ("64317", "The Great Gatsby — Fitzgerald"),
    ("1080", "A Modest Proposal — Swift"),
    ("829", "Gulliver's Travels — Swift"),
    ("113", "The Secret Garden — Burnett"),
    ("271", "Black Beauty — Sewell"),
    ("209", "The Turn of the Screw — James"),
    ("432", "The Ambassadors — James"),
    ("1013", "Dracula's Guest — Stoker"),
)

#: Held out for the entire experiment. Different authors from the training set wherever
#: possible, so validation BPB measures English rather than an author's habits. Named
#: explicitly rather than taken as a fraction: a fraction shifts when the list is edited.
VALIDATION_IDS: tuple[tuple[str, str], ...] = (
    ("1250", "Anthem — Rand"),
    ("2542", "A Doll's House — Ibsen"),
    ("4300", "Ulysses — Joyce"),
    ("103", "Around the World in Eighty Days — Verne"),
    ("164", "Twenty Thousand Leagues under the Sea — Verne"),
    ("55", "The Wonderful Wizard of Oz — Baum"),
    ("1257", "The Three Musketeers — Dumas"),
    ("844", "The Importance of Being Earnest — Wilde"),
)

SPLIT_RULE = (
    "Document-level. Whole works are assigned to train or validation by explicit "
    "catalogue id and never split across the two, so validation measures generalisation "
    "to unseen prose rather than continuation of a passage already being read. The "
    "assignment is fixed in this script and written once into train.txt/validation.txt, "
    "so no seed, fraction or ordering can change it between sessions."
)

CONTAMINATION_NOTES = (
    "Source is Project Gutenberg literary prose only. No benchmark dataset, question "
    "bank, exam set or evaluation suite was included, deliberately.",
    "Gutenberg licence headers and footers are removed: they are identical across every "
    "book and would otherwise be the most predictable text in the corpus.",
    "Validation works are by authors held out of training wherever possible, so a low "
    "validation BPB cannot be explained by having learned one author's style.",
    "These are famous works that a large pretrained model would likely have memorised. "
    "That is irrelevant here — this model is trained from scratch on this corpus alone — "
    "but it would matter if these texts were ever reused to evaluate a distilled student "
    "whose teacher saw them.",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True, help="corpus directory")
    parser.add_argument("--cache", type=Path, help="where to keep downloads "
                        "(default: <output>/.downloads)")
    parser.add_argument("--from-local", type=Path,
                        help="build from .txt files already on disk; no network is used")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without fetching or writing")
    parser.add_argument("--limit", type=int, help="use at most this many training works")
    parser.add_argument("--min-mb", type=float, default=40.0,
                        help="warn if the corpus comes out smaller than this")
    parser.add_argument("--timeout", type=int, default=60, help="per-request timeout")
    parser.add_argument("--retries", type=int, default=3)
    return parser


def fetch(identifier: str, cache: Path, *, timeout: int, retries: int) -> str | None:
    """Download one work, caching it so a re-run costs nothing.

    Returns ``None`` on failure rather than raising: one unreachable id should not
    discard the rest of the corpus, and the manifest records which ones were missed.
    """
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"{identifier}.txt"
    if cached.is_file() and cached.stat().st_size > 1024:
        return cached.read_text(encoding="utf-8", errors="replace")

    urls = [GUTENBERG_URL.format(id=identifier)]
    if len(identifier) > 1:
        urls.append(GUTENBERG_MIRROR.format(
            a=identifier[0], b=identifier[1] if len(identifier) > 2 else identifier[0],
            id=identifier,
        ))

    for attempt in range(retries):
        for url in urls:
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "qwen-distill-level2r/1.0"}
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = response.read().decode("utf-8", errors="replace")
                if len(payload) < 1024:
                    continue
                cached.write_text(payload, encoding="utf-8")
                return payload
            except (urllib.error.URLError, OSError, TimeoutError):
                continue
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return None


def collect(args: argparse.Namespace) -> tuple[list[Document], list[str]]:
    """Gather documents, from local files or by download. Returns (documents, failures)."""
    if args.from_local:
        # Filenames are the catalogue ids, so the same split assignment applies.
        documents = load_documents_from_directory(
            args.from_local, source_template=GUTENBERG_URL,
            public_domain_basis=PUBLIC_DOMAIN_BASIS,
        )
        return documents, []

    cache = args.cache or (args.output / ".downloads")
    wanted = list(TRAIN_IDS)
    if args.limit:
        wanted = wanted[: args.limit]
    wanted += list(VALIDATION_IDS)

    documents: list[Document] = []
    failures: list[str] = []
    for position, (identifier, expected) in enumerate(wanted, 1):
        raw = fetch(identifier, cache, timeout=args.timeout, retries=args.retries)
        if raw is None:
            failures.append(f"{identifier} ({expected})")
            print(f"  [{position:>3}/{len(wanted)}] {identifier:>6}  FAILED  {expected}",
                  file=sys.stderr)
            continue
        title, author = extract_metadata(raw)
        text = normalise(strip_gutenberg_boilerplate(raw))
        documents.append(Document(
            identifier=identifier, text=text, title=title, author=author,
            source_url=GUTENBERG_URL.format(id=identifier),
            public_domain_basis=PUBLIC_DOMAIN_BASIS,
        ))
        print(f"  [{position:>3}/{len(wanted)}] {identifier:>6}  "
              f"{len(text) / 1e6:>5.2f} MB  {title or expected}")
    return documents, failures


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(f"{RULE}\nLEVEL 2R CORPUS PREPARATION  (v{PREPARATION_VERSION})\n{RULE}\n")
    train_count = min(args.limit, len(TRAIN_IDS)) if args.limit else len(TRAIN_IDS)
    print(f"  training works   : {train_count}")
    print(f"  validation works : {len(VALIDATION_IDS)}  (held out for the whole experiment)")
    print(f"  output           : {args.output}")
    print(f"  source           : {'local files' if args.from_local else 'Project Gutenberg'}")

    if args.dry_run:
        print("\n  validation set (fixed):")
        for identifier, label in VALIDATION_IDS:
            print(f"    {identifier:>6}  {label}")
        print(f"\n  split rule: {SPLIT_RULE}")
        print("\n  DRY RUN: nothing was downloaded or written.")
        return 0

    print()
    documents, failures = collect(args)
    if not documents:
        print("\n  no documents were obtained — nothing to build.", file=sys.stderr)
        if not args.from_local:
            print("  If the network is restricted here, download elsewhere and use "
                  "--from-local.", file=sys.stderr)
        return 2

    validation_ids = tuple(i for i, _ in VALIDATION_IDS)
    documents = assign_splits(documents, validation_ids=validation_ids)
    if not any(d.split == "validation" for d in documents):
        print("\n  none of the validation works could be obtained. Refusing to build a "
              "corpus\n  with no held-out set: there would be nothing to measure "
              "generalisation with.", file=sys.stderr)
        return 2

    train_text, validation_text, manifest = build_corpus(
        documents, name="level2r_public_domain_english",
        split_rule=SPLIT_RULE, contamination_notes=CONTAMINATION_NOTES,
    )
    if failures:
        manifest.warnings.append(
            f"{len(failures)} work(s) could not be fetched and are absent from this "
            f"corpus: {', '.join(failures[:10])}"
        )

    overlap = check_overlap(train_text, validation_text)
    if not overlap["ok"]:
        manifest.warnings.append(
            f"{overlap['overlaps']} of {overlap['checked']} sampled validation passages "
            "also appear in the training text — a work is probably duplicated under two ids"
        )

    size_mb = manifest.total_bytes / 1e6
    if size_mb < args.min_mb:
        manifest.warnings.append(
            f"corpus is {size_mb:.1f} MB, below the {args.min_mb:.0f} MB target. Level 2 "
            "exhausted an 8 MB corpus in ~400 steps; a small corpus here risks repeating "
            "that saturation rather than answering the question."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train.txt").write_text(train_text, encoding="utf-8")
    (args.output / "validation.txt").write_text(validation_text, encoding="utf-8")
    manifest_path = args.output / "corpus_manifest.json"
    payload = manifest.to_dict()
    payload["overlap_check"] = overlap
    payload["run_estimate"] = estimate_run(manifest.train_bytes)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print()
    print(manifest.render())
    estimate = payload["run_estimate"]
    print(f"\n  one epoch over the training text is {estimate['steps_per_epoch']:,.0f} steps")
    print(f"  at the measured {estimate['measured_tokens_per_second']:,.0f} tok/s, "
          f"1000 steps is {estimate['hours_per_1000_steps']:.2f} h")
    print(f"\n  wrote {args.output / 'train.txt'}")
    print(f"        {args.output / 'validation.txt'}")
    print(f"        {manifest_path}")
    print("\n  The corpus is NOT tracked by git. Keep it on Drive alongside checkpoints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
