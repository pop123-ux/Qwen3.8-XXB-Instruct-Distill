"""Building a real English corpus from public-domain texts, deterministically.

Level 2 trained on procedurally generated text: words drawn independently from a fixed
Zipfian distribution. It reached 1.270 BPB and generated `"and and and"`, because those
are the same fact — the corpus had word frequencies and no syntax, and the optimal model
for it predicts common words forever. Level 2R exists to ask the question that corpus
could not: **can this architecture learn real language structure?**

Two design decisions carry the scientific weight.

**The split is at document level, not byte level.** Level 2 held out a contiguous tail of
one concatenated text, which measures how well the model continues a passage it has been
reading. Here whole books go to train or to validation and never both, so validation BPB
measures generalisation to prose the model has never seen — different author, different
subject, different century. It is a harder and far more meaningful target.

**The split is fixed in the files, not computed at train time.** The preparation script
writes `train.txt` and `validation.txt` once. No seed, no fraction, no ordering
assumption can drift between Colab sessions, because there is nothing left to recompute.

Nothing here downloads anything. Acquisition is the script's job; this module is the
deterministic transformation from raw texts to a corpus, so it is fully testable offline.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Bump when normalisation changes in a way that alters the produced bytes. A corpus
#: prepared under a different version is a different corpus and its hash will say so.
PREPARATION_VERSION = "1.0"

#: Project Gutenberg wraps every text in licence boilerplate. It is highly repetitive,
#: identical across books, and would teach the model to predict legal notices — the
#: clearest contamination available in this corpus, so it is removed rather than trimmed.
GUTENBERG_START = re.compile(
    r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
GUTENBERG_END = re.compile(
    r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$",
    re.IGNORECASE | re.MULTILINE,
)

#: Metadata lines Gutenberg leaves above the body even after the START marker.
_HEADER_FIELDS = re.compile(
    r"^(Title|Author|Release [Dd]ate|Language|Credits|Produced by|Character set encoding|"
    r"Editor|Translator|Illustrator|Posting Date|Most recently updated)\s*:",
    re.IGNORECASE,
)

_TITLE_LINE = re.compile(r"^Title:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_AUTHOR_LINE = re.compile(r"^Author:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

#: Separator between concatenated documents. A blank line pair is what a paragraph break
#: already looks like, so the model sees a document boundary as prose, not as a token it
#: could learn to exploit.
DOCUMENT_SEPARATOR = "\n\n\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the licence header and footer, leaving the work itself.

    Falls back to returning the text unchanged when the markers are absent, because a
    non-Gutenberg source is a legitimate input and silently truncating it would be worse
    than leaving boilerplate in.
    """
    start = GUTENBERG_START.search(text)
    if start is not None:
        text = text[start.end():]
    end = GUTENBERG_END.search(text)
    if end is not None:
        text = text[: end.start()]

    # A handful of metadata lines can survive the START marker. Drop them only while
    # they appear at the very top, so a "Title:" inside the prose is untouched.
    lines = text.split("\n")
    index = 0
    while index < len(lines) and index < 40:
        stripped = lines[index].strip()
        if not stripped or _HEADER_FIELDS.match(stripped):
            index += 1
            continue
        break
    return "\n".join(lines[index:])


def normalise(text: str) -> str:
    """Deterministic normalisation, documented because it changes the bytes modelled.

    * **NFC Unicode** — the same character can be one code point or two; at byte level
      those are different sequences, and modelling both wastes capacity on an encoding
      artefact rather than on language.
    * **LF line endings** — otherwise the same book hashes differently depending on which
      machine downloaded it, and the corpus stops being reproducible.
    * **At most two consecutive blank lines** — Gutenberg spacing is irregular and long
      blank runs are trivially predictable filler that would flatter the BPB number.
    * **Trailing whitespace removed** per line, for the same reason.

    Deliberately NOT done: lowercasing, punctuation stripping, sentence splitting. The
    experiment is whether the model learns English as written.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def extract_metadata(raw_text: str) -> tuple[str | None, str | None]:
    """Title and author as the file itself declares them.

    Read from the downloaded text rather than trusted from a hard-coded table, so a
    wrong catalogue id shows up in the manifest instead of being papered over.
    """
    title = _TITLE_LINE.search(raw_text)
    author = _AUTHOR_LINE.search(raw_text)
    return (title.group(1).strip() if title else None,
            author.group(1).strip() if author else None)


@dataclass
class Document:
    """One public-domain work, after normalisation."""

    identifier: str
    text: str
    title: str | None = None
    author: str | None = None
    source_url: str | None = None
    public_domain_basis: str = ""
    split: str = "train"

    @property
    def n_bytes(self) -> int:
        return len(self.text.encode("utf-8"))

    @property
    def sha256(self) -> str:
        return sha256_text(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier, "title": self.title, "author": self.author,
            "source_url": self.source_url,
            "public_domain_basis": self.public_domain_basis,
            "split": self.split, "n_bytes": self.n_bytes, "sha256": self.sha256,
        }


@dataclass
class CorpusManifest:
    """Everything needed to reconstruct and audit the corpus."""

    name: str
    preparation_version: str = PREPARATION_VERSION
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    documents: list[dict[str, Any]] = field(default_factory=list)
    n_documents: int = 0
    n_train_documents: int = 0
    n_validation_documents: int = 0
    total_bytes: int = 0
    train_bytes: int = 0
    validation_bytes: int = 0
    train_sha256: str = ""
    validation_sha256: str = ""
    split_rule: str = ""
    normalisation: list[str] = field(default_factory=list)
    contamination_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [
            f"corpus: {self.name}  (preparation v{self.preparation_version})",
            f"  documents  : {self.n_documents} "
            f"({self.n_train_documents} train / {self.n_validation_documents} validation)",
            f"  total      : {self.total_bytes / 1e6:.1f} MB",
            f"  train      : {self.train_bytes / 1e6:.1f} MB  sha256 {self.train_sha256[:16]}",
            f"  validation : {self.validation_bytes / 1e6:.1f} MB  "
            f"sha256 {self.validation_sha256[:16]}",
            f"  split rule : {self.split_rule}",
        ]
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


def assign_splits(
    documents: list[Document], *, validation_ids: tuple[str, ...]
) -> list[Document]:
    """Assign whole documents to train or validation, by explicit id.

    Explicit rather than fractional. A fraction depends on ordering and on how many
    documents happen to be present, so it can silently change when the list is edited;
    a named set cannot. The validation books are then fixed for the life of the
    experiment, which is what makes BPB comparable across sessions and across runs.
    """
    wanted = set(validation_ids)
    for document in documents:
        document.split = "validation" if document.identifier in wanted else "train"
    return documents


def build_corpus(
    documents: list[Document],
    *,
    name: str,
    split_rule: str,
    contamination_notes: tuple[str, ...] = (),
) -> tuple[str, str, CorpusManifest]:
    """Concatenate into ``(train_text, validation_text, manifest)``.

    Ordering is by identifier, not by input order or filesystem order, so two
    preparations of the same document set produce byte-identical corpora and therefore
    the same hash.
    """
    ordered = sorted(documents, key=lambda d: d.identifier)
    train_docs = [d for d in ordered if d.split == "train"]
    validation_docs = [d for d in ordered if d.split == "validation"]

    if not train_docs:
        raise ValueError("no training documents: every document was assigned to validation")
    if not validation_docs:
        raise ValueError(
            "no validation documents. A held-out set is not optional here — without it "
            "there is no measurement of generalisation, which is the entire question "
            "Level 2R asks."
        )

    train_text = DOCUMENT_SEPARATOR.join(d.text for d in train_docs)
    validation_text = DOCUMENT_SEPARATOR.join(d.text for d in validation_docs)

    manifest = CorpusManifest(
        name=name,
        documents=[d.to_dict() for d in ordered],
        n_documents=len(ordered),
        n_train_documents=len(train_docs),
        n_validation_documents=len(validation_docs),
        total_bytes=len(train_text.encode("utf-8")) + len(validation_text.encode("utf-8")),
        train_bytes=len(train_text.encode("utf-8")),
        validation_bytes=len(validation_text.encode("utf-8")),
        train_sha256=sha256_text(train_text),
        validation_sha256=sha256_text(validation_text),
        split_rule=split_rule,
        normalisation=[
            "Project Gutenberg licence header and footer removed",
            "residual metadata lines (Title/Author/Release Date/...) dropped from the top",
            "Unicode NFC",
            "CRLF and CR converted to LF",
            "trailing whitespace stripped per line",
            "runs of 3+ blank lines collapsed to 2",
            "documents ordered by identifier, joined by a blank-line separator",
            "NOT done: lowercasing, punctuation stripping, sentence splitting",
        ],
        contamination_notes=list(contamination_notes),
    )
    return train_text, validation_text, manifest


def check_overlap(train_text: str, validation_text: str, *, window: int = 512,
                  samples: int = 64) -> dict[str, Any]:
    """Sample validation passages and check none appear verbatim in training text.

    A document-level split should make this impossible, so a hit means something is
    genuinely wrong — the same work included twice under different ids, or an anthology
    reprinting a held-out text. Cheap enough to run every preparation.
    """
    if len(validation_text) < window or not train_text:
        return {"checked": 0, "overlaps": 0, "ok": True}

    step = max(1, (len(validation_text) - window) // max(1, samples))
    hits = 0
    checked = 0
    for start in range(0, len(validation_text) - window, step):
        passage = validation_text[start: start + window]
        checked += 1
        if passage in train_text:
            hits += 1
        if checked >= samples:
            break
    return {"checked": checked, "overlaps": hits, "ok": hits == 0}


def estimate_run(corpus_bytes: int, *, tokens_per_step: int = 16_384,
                 tokens_per_second: float = 2089.5) -> dict[str, Any]:
    """How long a corpus lasts, in the units the experiment is actually run in.

    Byte-level, so one token is one byte and the corpus size is directly the number of
    tokens per epoch. Level 2 exhausted an 8 MB corpus in roughly 400 steps and then
    flat-lined for 1600 more; knowing where that point falls is how Level 2R avoids
    repeating it.
    """
    steps_per_epoch = corpus_bytes / tokens_per_step if tokens_per_step else 0
    return {
        "corpus_bytes": corpus_bytes,
        "tokens_per_step": tokens_per_step,
        "steps_per_epoch": round(steps_per_epoch, 1),
        "seconds_per_1000_steps": round(1000 * tokens_per_step / tokens_per_second, 1),
        "hours_per_1000_steps": round(1000 * tokens_per_step / tokens_per_second / 3600, 2),
        "measured_tokens_per_second": tokens_per_second,
        "basis": "Level 2 measured 2,089 tok/s at seq 1024, batch 4, checkpointing ON",
    }


def load_documents_from_directory(
    directory: str | Path, *, source_template: str | None = None,
    public_domain_basis: str = "",
) -> list[Document]:
    """Load already-downloaded `.txt` files, normalising each.

    Lets a corpus be prepared from local files — a mirror, a manual download, a test
    fixture — without any network access.
    """
    base = Path(directory)
    if not base.is_dir():
        raise FileNotFoundError(f"corpus source directory not found: {base}")
    documents: list[Document] = []
    for path in sorted(base.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        title, author = extract_metadata(raw)
        identifier = path.stem
        documents.append(Document(
            identifier=identifier,
            text=normalise(strip_gutenberg_boilerplate(raw)),
            title=title, author=author,
            source_url=source_template.format(id=identifier) if source_template else None,
            public_domain_basis=public_domain_basis,
        ))
    if not documents:
        raise ValueError(f"no .txt files found in {base}")
    return documents
