"""Check a prepared corpus against its own manifest. No network, ever.

Two failures this exists to catch, both of which have already nearly happened here:

**A corpus that is not the corpus you think it is.** The Level-2R preparation script names
Gutenberg works by numeric id, and those ids could not be checked against the live
catalogue from this environment — the proxy blocks it, and working around that is out of
scope. So a wrong id silently produces a different book. The manifest records the title
each downloaded file *declares about itself*, and this compares that against what was
asked for. A mismatch is visible without any network.

**Contamination.** A document-level split makes verbatim overlap between train and
validation impossible — unless the same work appears twice under different ids, or an
anthology reprints a held-out text. Then the validation BPB is measuring memorisation and
looks like learning.

Everything here recomputes from the files. The manifest is the claim; the bytes are the
evidence. A check that reads the manifest's own numbers back to you verifies nothing.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .corpus import (
    GUTENBERG_END,
    GUTENBERG_START,
    check_overlap,
    estimate_run,
    sha256_text,
)

#: Strings that should not survive preparation. Their presence means boilerplate stripping
#: did not fire — usually because the file's header did not match the expected form, which
#: also means the metadata extracted from it is suspect.
BOILERPLATE_MARKERS: tuple[str, ...] = (
    "PROJECT GUTENBERG EBOOK",
    "START OF THE PROJECT GUTENBERG",
    "END OF THE PROJECT GUTENBERG",
    "www.gutenberg.org",
    "gutenberg.net",
)

#: Below this, a "corpus" is a test fixture. Level 2 exhausted 8 MB in ~400 steps.
MIN_USEFUL_BYTES = 1_000_000

#: A validation split much smaller than this measures noise; much larger wastes training
#: data. Not a hard rule — reported, not enforced.
VALIDATION_FRACTION_RANGE = (0.02, 0.20)


@dataclass
class Finding:
    """One problem, at a severity that says whether it blocks a run."""

    level: str            # ERROR | WARNING | NOTE
    check: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "check": self.check, "message": self.message}


@dataclass
class CorpusVerification:
    """What the files actually contain, next to what the manifest claims."""

    directory: str
    manifest_present: bool = False
    findings: list[Finding] = field(default_factory=list)
    measured: dict[str, Any] = field(default_factory=dict)
    claimed: dict[str, Any] = field(default_factory=dict)
    run_estimate: dict[str, Any] = field(default_factory=dict)
    documents_checked: int = 0

    def add(self, level: str, check: str, message: str) -> None:
        self.findings.append(Finding(level=level, check=check, message=message))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "ERROR"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "WARNING"]

    @property
    def passed(self) -> bool:
        """No errors. Warnings do not block a run — they are things to read first."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": self.directory,
            "passed": self.passed,
            "manifest_present": self.manifest_present,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "documents_checked": self.documents_checked,
            "measured": self.measured,
            "claimed": self.claimed,
            "run_estimate": self.run_estimate,
            "findings": [f.to_dict() for f in self.findings],
            "network": "none — every check is local",
        }

    def render(self) -> str:
        rule = "=" * 78
        lines = [rule, "CORPUS VERIFICATION", rule, "", f"  directory : {self.directory}"]
        if not self.manifest_present:
            lines.append("  manifest  : MISSING")
        for key, label in (
            ("name", "corpus"), ("split_rule", "split rule"),
        ):
            if self.claimed.get(key):
                lines.append(f"  {label:<10}: {self.claimed[key]}")

        if self.measured:
            lines += ["", "-" * 78, "MEASURED FROM THE FILES", "-" * 78]
            for split in ("train", "validation"):
                size = self.measured.get(f"{split}_bytes")
                digest = self.measured.get(f"{split}_sha256", "")
                if size is None:
                    continue
                claimed = self.claimed.get(f"{split}_sha256", "")
                match = "matches manifest" if digest == claimed else "DOES NOT MATCH MANIFEST"
                lines.append(
                    f"  {split:<11} {size / 1e6:>8.2f} MB  sha256 {digest[:16]}  ({match})"
                )
            fraction = self.measured.get("validation_fraction")
            if fraction is not None:
                lines.append(f"  validation is {fraction:.1%} of the corpus")
            lines.append(f"  documents checked: {self.documents_checked}")

        if self.run_estimate:
            lines += [
                "", "-" * 78, "WHAT THIS CORPUS BUYS", "-" * 78,
                f"  {self.run_estimate['steps_per_epoch']:,.0f} steps per epoch at "
                f"{self.run_estimate['tokens_per_step']:,} bytes/step",
                f"  {self.run_estimate['hours_per_1000_steps']:.2f} h per 1000 steps "
                f"({self.run_estimate['basis']})",
            ]

        for level, title in (("ERROR", "ERRORS — do not train on this"),
                             ("WARNING", "WARNINGS — read before training"),
                             ("NOTE", "NOTES")):
            entries = [f for f in self.findings if f.level == level]
            if not entries:
                continue
            lines += ["", "-" * 78, title, "-" * 78]
            for finding in entries:
                lines.append(f"  [{finding.check}] {finding.message}")

        verdict = "PASS" if self.passed else "FAIL"
        lines += [
            "", "-" * 78,
            f"  VERDICT: {verdict}  ({len(self.errors)} errors, {len(self.warnings)} warnings)",
            "  Every check is local. Nothing here confirms a document is the work its id",
            "  names — only that the file's own declared title matches what was requested.",
            rule,
        ]
        return "\n".join(lines)


def _read(path: Path) -> str | None:
    """Read the file as bytes and decode explicitly. Never ``read_text``.

    ``read_text`` opens in universal-newline mode, which silently translates ``\r\n``
    to ``\n``. For a verifier whose whole claim is "the bytes are the evidence" that is
    fatal twice over: a CRLF file would hash identically to its LF twin, so the digest
    check would pass on byte-different files, and the carriage-return check could never
    fire at all because no ``\r`` would survive the read.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def verify_corpus(
    directory: str | Path,
    *,
    expected_titles: dict[str, str] | None = None,
    tokens_per_step: int = 16_384,
    overlap_samples: int = 64,
) -> CorpusVerification:
    """Recompute everything the manifest claims, from the files themselves.

    ``expected_titles`` maps document identifier to the title that document was *asked
    for*. Supplying it is what turns "the manifest says this is Ulysses" into "the file
    says it is Ulysses too". Without it that check is skipped and the report says so.
    """
    directory = Path(directory)
    result = CorpusVerification(directory=str(directory))

    if not directory.is_dir():
        result.add("ERROR", "directory", f"{directory} is not a directory")
        return result

    train_path = directory / "train.txt"
    validation_path = directory / "validation.txt"
    manifest_path = directory / "corpus_manifest.json"

    # --- the manifest -------------------------------------------------------------
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
                result.manifest_present = True
        except (OSError, json.JSONDecodeError) as exc:
            result.add("ERROR", "manifest", f"corpus_manifest.json is unreadable: {exc}")
    else:
        result.add(
            "ERROR", "manifest",
            "no corpus_manifest.json — without it there is no record of what these bytes "
            "are, and a result trained on them cannot be reproduced",
        )
    result.claimed = {
        key: manifest.get(key)
        for key in ("name", "split_rule", "train_sha256", "validation_sha256",
                    "train_bytes", "validation_bytes", "n_documents",
                    "n_train_documents", "n_validation_documents", "preparation_version")
        if manifest.get(key) is not None
    }

    # --- the files ----------------------------------------------------------------
    texts: dict[str, str] = {}
    for split, path in (("train", train_path), ("validation", validation_path)):
        if not path.is_file():
            result.add("ERROR", "files", f"{path.name} is missing")
            continue
        text = _read(path)
        if text is None:
            result.add(
                "ERROR", "encoding",
                f"{path.name} is not valid UTF-8. Byte-level training reads it as bytes "
                f"either way, but the preparation should have normalised it, so this "
                f"means something upstream went wrong",
            )
            continue
        texts[split] = text
        result.measured[f"{split}_bytes"] = len(text.encode("utf-8"))
        result.measured[f"{split}_sha256"] = sha256_text(text)

    if not texts:
        return result

    # --- digests: the claim against the evidence ----------------------------------
    for split in ("train", "validation"):
        measured = result.measured.get(f"{split}_sha256")
        claimed = manifest.get(f"{split}_sha256")
        if measured is None or not claimed:
            continue
        if measured != claimed:
            result.add(
                "ERROR", "digest",
                f"{split}.txt hashes to {measured[:16]} but the manifest claims "
                f"{claimed[:16]} — the file has changed since it was prepared, so every "
                f"result recorded against that digest is about different bytes",
            )

    for split in ("train", "validation"):
        measured = result.measured.get(f"{split}_bytes")
        claimed = manifest.get(f"{split}_bytes")
        if measured is not None and claimed and measured != claimed:
            result.add(
                "WARNING", "size",
                f"{split}.txt is {measured:,} bytes; the manifest says {claimed:,}",
            )

    # --- both splits must be non-empty --------------------------------------------
    for split, text in texts.items():
        if not text.strip():
            result.add("ERROR", "empty", f"{split}.txt is empty or whitespace only")

    total_bytes = sum(result.measured.get(f"{s}_bytes", 0) for s in ("train", "validation"))
    if total_bytes:
        validation_fraction = result.measured.get("validation_bytes", 0) / total_bytes
        result.measured["total_bytes"] = total_bytes
        result.measured["validation_fraction"] = round(validation_fraction, 4)
        low, high = VALIDATION_FRACTION_RANGE
        if validation_fraction < low:
            result.add(
                "WARNING", "split",
                f"validation is only {validation_fraction:.1%} of the corpus — a "
                f"validation set this small measures noise as much as generalisation",
            )
        elif validation_fraction > high:
            result.add(
                "WARNING", "split",
                f"validation is {validation_fraction:.1%} of the corpus — that is "
                f"training data not being used",
            )
    if total_bytes and total_bytes < MIN_USEFUL_BYTES:
        result.add(
            "WARNING", "size",
            f"the corpus is {total_bytes / 1e6:.2f} MB. Level 2 exhausted 8 MB in about "
            f"400 steps and then flat-lined for 1600 more; expect the same shape",
        )

    # --- contamination -------------------------------------------------------------
    if "train" in texts and "validation" in texts:
        overlap = check_overlap(
            texts["train"], texts["validation"], samples=overlap_samples
        )
        result.measured["overlap"] = overlap
        if not overlap["ok"]:
            result.add(
                "ERROR", "contamination",
                f"{overlap['overlaps']} of {overlap['checked']} sampled validation "
                f"passages appear verbatim in the training text. A document-level split "
                f"makes this impossible unless the same work is present twice — the "
                f"validation score would be measuring memorisation",
            )
        elif overlap["checked"] == 0:
            result.add(
                "NOTE", "contamination",
                "validation text was too short to sample for overlap; the check did not "
                "run, which is not the same as passing",
            )

    return _verify_documents(result, manifest, texts, expected_titles, tokens_per_step)


def _verify_documents(
    result: CorpusVerification,
    manifest: dict[str, Any],
    texts: dict[str, str],
    expected_titles: dict[str, str] | None,
    tokens_per_step: int,
) -> CorpusVerification:
    """Check the manifest's document list against the text it claims to describe."""
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        result.add(
            "WARNING", "documents",
            "the manifest lists no documents, so the split cannot be checked at document "
            "level and no title can be compared against what was requested",
        )
        documents = []
    result.documents_checked = len(documents)

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    digests: dict[str, list[str]] = {}
    identifiers: dict[str, int] = {}

    for entry in documents:
        if not isinstance(entry, dict):
            continue
        split = str(entry.get("split", "train"))
        by_split.setdefault(split, []).append(entry)
        identifier = str(entry.get("identifier", ""))
        identifiers[identifier] = identifiers.get(identifier, 0) + 1
        digest = str(entry.get("sha256", ""))
        if digest:
            digests.setdefault(digest, []).append(identifier)

    # --- the same work under two ids is the contamination that survives a split ------
    for digest, ids in digests.items():
        if len(ids) > 1:
            splits = {
                str(e.get("split")) for e in documents
                if isinstance(e, dict) and str(e.get("sha256", "")) == digest
            }
            level = "ERROR" if len(splits) > 1 else "WARNING"
            result.add(
                level, "duplicates",
                f"documents {', '.join(sorted(ids))} are byte-identical"
                + (
                    " and land on opposite sides of the split — the validation score "
                    "would be measuring memorisation"
                    if len(splits) > 1
                    else " (same split, so it is duplicated training data, not contamination)"
                ),
            )
    for identifier, count in identifiers.items():
        if count > 1:
            result.add(
                "WARNING", "duplicates",
                f"identifier {identifier!r} appears {count} times in the manifest",
            )

    for split in ("train", "validation"):
        entries = by_split.get(split, [])
        claimed = manifest.get(f"n_{split}_documents")
        if claimed is not None and len(entries) != claimed:
            result.add(
                "WARNING", "documents",
                f"the manifest lists {len(entries)} {split} documents but claims "
                f"{claimed}",
            )
        if documents and not entries:
            result.add(
                "ERROR", "documents",
                f"no documents are assigned to the {split} split",
            )

    # --- byte totals: the documents must add up to the file --------------------------
    for split in ("train", "validation"):
        entries = by_split.get(split, [])
        if not entries:
            continue
        summed = sum(int(e.get("n_bytes", 0) or 0) for e in entries)
        measured = result.measured.get(f"{split}_bytes")
        if measured is None or not summed:
            continue
        # Documents are joined with a separator, so the file is slightly larger than the
        # sum of its parts. A shortfall, or a large excess, is a real discrepancy.
        separator_allowance = 8 * max(0, len(entries) - 1) + 16
        if summed > measured or measured - summed > separator_allowance:
            result.add(
                "WARNING", "documents",
                f"{split} documents sum to {summed:,} bytes but {split}.txt is "
                f"{measured:,} — a difference of {measured - summed:+,}, more than "
                f"joining {len(entries)} documents accounts for",
            )

    # --- is this the book you asked for? ---------------------------------------------
    if expected_titles:
        for entry in documents:
            if not isinstance(entry, dict):
                continue
            identifier = str(entry.get("identifier", ""))
            wanted = expected_titles.get(identifier)
            if not wanted:
                continue
            declared = entry.get("title")
            if not declared:
                result.add(
                    "WARNING", "identity",
                    f"document {identifier} declares no title, so it cannot be checked "
                    f"against the requested {wanted!r}",
                )
            elif not _titles_agree(str(declared), wanted):
                result.add(
                    "ERROR", "identity",
                    f"document {identifier} was requested as {wanted!r} but the file "
                    f"declares itself {str(declared)!r} — this is a wrong id, and the "
                    f"corpus is not what the manifest says it is",
                )
    else:
        result.add(
            "NOTE", "identity",
            "no expected titles supplied, so document identity was not checked. The "
            "manifest's titles come from the files themselves and agree with themselves "
            "by construction; pass --expect to compare them against what was requested",
        )

    # --- boilerplate ------------------------------------------------------------------
    for split, text in texts.items():
        upper = text.upper()
        found = [marker for marker in BOILERPLATE_MARKERS if marker.upper() in upper]
        if found:
            result.add(
                "WARNING", "boilerplate",
                f"{split}.txt still contains {', '.join(repr(m) for m in found)} — "
                f"boilerplate stripping did not fire on at least one document, which "
                f"also means the metadata read from its header is unreliable",
            )
        if GUTENBERG_START.search(text) or GUTENBERG_END.search(text):
            result.add(
                "ERROR", "boilerplate",
                f"{split}.txt contains an unstripped Project Gutenberg start/end marker",
            )

    # --- normalisation ----------------------------------------------------------------
    for split, text in texts.items():
        if "\r" in text:
            result.add(
                "WARNING", "normalisation",
                f"{split}.txt contains carriage returns; preparation should have "
                f"normalised line endings",
            )
        if text != unicodedata.normalize("NFC", text):
            result.add(
                "WARNING", "normalisation",
                f"{split}.txt is not NFC-normalised, so visually identical text may be "
                f"different bytes — and the model sees bytes",
            )

    total = result.measured.get("total_bytes")
    if total:
        result.run_estimate = estimate_run(total, tokens_per_step=tokens_per_step)

    return result


def _titles_agree(declared: str, wanted: str) -> bool:
    """Loose title comparison.

    Gutenberg headers carry subtitles, editions and translator credits that a short
    reference title will not have — ``"Ulysses"`` against
    ``"Ulysses: A Novel"``. Requiring an exact match would report a false mismatch on
    almost every book, and a check that always fires is a check nobody reads. Substring
    containment either way, on a normalised form, is the useful test.
    """
    def _normalise(value: str) -> str:
        folded = unicodedata.normalize("NFKD", value).casefold()
        return " ".join("".join(c if c.isalnum() else " " for c in folded).split())

    left, right = _normalise(declared), _normalise(wanted)
    if not left or not right:
        return False
    return left in right or right in left
