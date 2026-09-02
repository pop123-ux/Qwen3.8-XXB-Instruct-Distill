"""Tokenizer-backed corpora: real text through the teacher's own tokenizer.

:mod:`.text_data` encodes text as raw UTF-8 bytes, vocabulary 256. That was the right
choice for Levels 1-2R — no tokenizer to download, any file works, and the loss reads
directly as bits per byte. It is also why the canonical student cannot train on it: the
frozen target ``qwen38_19b_h5120_l48_moe`` has a **248,320-entry embedding**, and a byte
stream only ever indexes the first 256 rows of it. Every other row would receive no
gradient, and the run would be training a 248k-vocabulary model on a 256-symbol language.
``scripts/chain_selftest.py`` refuses exactly this, by design.

This module closes that gap and nothing else. The flow is::

    local .txt corpus -> teacher tokenizer -> Qwen ids -> EOS-packed chunks -> batches

Three properties are load-bearing.

**The tokenizer is loaded from files, never from the Hub and never with the model.**
``AutoTokenizer.from_pretrained(path, local_files_only=True)`` reads ``tokenizer.json``
and friends out of the teacher checkpoint directory. A 27B checkpoint is ~54 GB of
weights sitting beside a few megabytes of tokenizer; this touches only the latter, so
corpus preparation runs on a laptop with no GPU and no network.

**The vocabulary is read off the tokenizer, not asserted.** :func:`prepare_tokenized_corpus`
takes an optional ``expected_vocab_size`` and *fails* when the tokenizer disagrees with it.
It never resizes an embedding to make the two match: a silent resize is how a student ends
up with rows that were never trained and a checkpoint nobody can explain.

**Packing is deterministic.** Documents are concatenated in file order with an explicit
EOS between them and chunked at a fixed stride, so the same corpus, tokenizer and sequence
length always produce the same token stream — the precondition for comparing two runs.

The byte-level path in :mod:`.text_data` is untouched and remains what every historical
experiment used. This is an additional source, selected explicitly by
``DataConfig.tokenized_text``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..distillation.provenance import sha256_file
from ..distillation.real_teacher import TokenizerFacts, describe_tokenizer
from .text_data import CorpusStats

#: Bump when packing changes in a way that alters the produced token stream. A corpus
#: packed under a different version is a different corpus, and the recorded value says so.
PACKING_VERSION = "1.0"

#: How documents are split out of a corpus file, and what each policy is for.
#:
#: ``blank_line``  paragraphs/documents separated by one or more blank lines. The default,
#:                 because it is what a concatenated plain-text corpus looks like.
#: ``line``        one document per non-empty line. For JSONL-adjacent or one-example-per-
#:                 line files.
#: ``file``        the whole file is a single document. For one continuous text.
DOCUMENT_SEPARATORS = ("blank_line", "line", "file")

#: Embeddings are padded up to a hardware alignment boundary, so a tokenizer smaller than
#: the embedding is normal rather than wrong: Qwen3.8-27B declares vocab_size 248,320
#: (= 256 x 970) against a tokenizer holding 248,077, leaving 243 rows no id reaches.
#: A gap at or above this bound is not padding — it is the wrong tokenizer, and stays fatal.
VOCAB_ALIGNMENT = 256

#: Files the tokenizer identity is hashed over, in this order. Absent files hash to None,
#: which is recorded rather than treated as a match.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


class TokenizerLoadError(RuntimeError):
    """The tokenizer could not be loaded from the given path."""


class CorpusError(ValueError):
    """The corpus cannot produce usable training data, and why."""


@dataclass
class TokenizerProvenance:
    """Which tokenizer produced a token stream, precisely enough to reproduce it.

    Separate from :class:`~qwen_distill.distillation.provenance.TeacherIdentity` because a
    corpus can be tokenised without loading a teacher at all — which is the whole point of
    this module — while still needing to record which teacher's tokenizer it used.
    """

    source: str
    tokenizer_class: str
    vocab_size: int
    eos_token: str | None
    eos_token_id: int | None
    #: SHA-256 of each tokenizer file present, so an edited or substituted tokenizer is
    #: detectable. Cheap: these are megabytes, not the corpus.
    file_sha256: dict[str, str | None] = field(default_factory=dict)
    teacher_model: str | None = None
    teacher_revision: str | None = None

    @property
    def is_pinned(self) -> bool:
        """Whether the teacher this tokenizer came from is pinned to a revision."""
        return self.teacher_revision is not None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "source": self.source,
            "tokenizer_class": self.tokenizer_class,
            "vocab_size": self.vocab_size,
            "eos_token": self.eos_token,
            "eos_token_id": self.eos_token_id,
            "file_sha256": dict(self.file_sha256),
            "teacher_model": self.teacher_model,
            "teacher_revision": self.teacher_revision,
            "is_pinned": self.is_pinned,
        }
        if not self.is_pinned:
            data["pinning_note"] = (
                "teacher_revision is unset, so the tokenizer is identified by its file "
                "hashes rather than by a checkpoint pin"
            )
        return data


@dataclass
class TokenizedCorpusStats(CorpusStats):
    """:class:`~qwen_distill.training.text_data.CorpusStats` plus what tokenised it.

    Inherits the byte-level fields so the trainer, the summary writer and every existing
    consumer of ``corpus_stats`` keep working unchanged. ``n_bytes`` still counts the raw
    UTF-8 bytes of the source text, which is what makes a tokenised corpus comparable in
    size to a byte-level one; ``n_tokens`` is the new figure that actually describes the
    training stream.
    """

    n_tokens: int = 0
    n_documents: int = 0
    #: Tokens dropped from the tail because they could not fill a whole sequence. Recorded
    #: rather than hidden: a large number here means the sequence length does not suit the
    #: corpus.
    n_tokens_dropped: int = 0
    document_separator: str = "blank_line"
    packing_version: str = PACKING_VERSION
    eos_token_id: int | None = None
    tokenizer: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_byte(self) -> float:
        """Compression the tokenizer achieved. ~0.25 for BPE English, 1.0 for byte-level."""
        return self.n_tokens / self.n_bytes if self.n_bytes else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["tokens_per_byte"] = round(self.tokens_per_byte, 4)
        return data


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
def load_tokenizer(path: str | Path, *, trust_remote_code: bool = False) -> Any:
    """Load a tokenizer from a local directory. No network, no model weights.

    ``path`` is a teacher checkpoint directory (the GPU workflow's
    ``/data/models/qwen3.8-27b``) or any directory holding tokenizer files. Only the
    tokenizer files are read — the 27B weights beside them are never opened.

    ``local_files_only=True`` is not a hint. Without it, a path that does not resolve
    locally is silently treated as a Hub repo id and downloaded, which turns a typo into a
    network fetch of somebody else's tokenizer.
    """
    directory = Path(path)
    if not directory.exists():
        raise TokenizerLoadError(
            f"tokenizer path does not exist: {directory}\n"
            "  Point it at a local teacher checkpoint directory (the GPU workflow uses "
            "/data/models/qwen3.8-27b) or at any directory holding tokenizer.json."
        )
    if directory.is_dir() and not any((directory / name).exists() for name in TOKENIZER_FILES):
        raise TokenizerLoadError(
            f"no tokenizer files in {directory}\n"
            f"  Expected one of: {', '.join(TOKENIZER_FILES)}.\n"
            "  vendor/qwen38-metadata carries tokenizer_config.json but no tokenizer.json, "
            "so it cannot serve as a tokenizer source; use the downloaded checkpoint."
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise TokenizerLoadError(
            "the tokenized_text data path needs `transformers` "
            "(pip install -r requirements/training.txt)"
        ) from exc

    try:
        return AutoTokenizer.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=trust_remote_code
        )
    except Exception as exc:  # noqa: BLE001 - surfaced with the path that failed
        raise TokenizerLoadError(
            f"could not load a tokenizer from {directory}: {type(exc).__name__}: {exc}"
        ) from exc


def tokenizer_provenance(
    tokenizer: Any,
    source: str | Path,
    *,
    teacher_model: str | None = None,
    teacher_revision: str | None = None,
) -> TokenizerProvenance:
    """Record which tokenizer this is, reading its behaviour rather than assuming it.

    Reuses :func:`~qwen_distill.distillation.real_teacher.describe_tokenizer`, so the facts
    recorded here are the same facts the teacher loader records — one description of a
    tokenizer in this repository, not two that can drift apart.
    """
    facts: TokenizerFacts = describe_tokenizer(tokenizer)
    directory = Path(source)
    hashes: dict[str, str | None] = {}
    if directory.is_dir():
        for name in TOKENIZER_FILES:
            candidate = directory / name
            if candidate.exists():
                hashes[name] = sha256_file(candidate)
    return TokenizerProvenance(
        source=str(source),
        tokenizer_class=facts.tokenizer_class,
        vocab_size=facts.vocab_size,
        eos_token=facts.eos_token,
        eos_token_id=facts.eos_token_id,
        file_sha256=hashes,
        teacher_model=teacher_model,
        teacher_revision=teacher_revision,
    )


def resolve_eos_id(tokenizer: Any) -> int:
    """The id that separates documents, or a clear failure.

    Packing without a document boundary teaches the model that one document flows into the
    next, so a missing EOS is fatal rather than something to work around silently. No token
    is ever *added* to the tokenizer to manufacture one: that would change the vocabulary
    the student was built for.
    """
    for attribute in ("eos_token_id", "sep_token_id", "pad_token_id"):
        candidate = getattr(tokenizer, attribute, None)
        if isinstance(candidate, int) and candidate >= 0:
            return candidate
    raise CorpusError(
        "the tokenizer defines no eos_token_id, so documents cannot be separated. "
        "Packing without a boundary would train the model to run one document into the "
        "next. Use a tokenizer that declares an EOS rather than adding one here — adding "
        "a token changes the vocabulary the student was built for."
    )


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
def read_documents(
    path: str | Path,
    *,
    separator: str = "blank_line",
    max_documents: int | None = None,
    max_bytes: int | None = None,
) -> list[str]:
    """Split a UTF-8 text file into documents, in file order.

    Order is the file's order and is never shuffled here: the packed stream must be
    reproducible from the file alone.
    """
    if separator not in DOCUMENT_SEPARATORS:
        raise CorpusError(
            f"unknown document_separator {separator!r}; known: {', '.join(DOCUMENT_SEPARATORS)}"
        )
    source = Path(path)
    if not source.is_file():
        raise CorpusError(f"text corpus not found: {source}")

    text = source.read_text(encoding="utf-8", errors="replace")
    if max_bytes:
        text = text[:max_bytes]

    if separator == "file":
        documents = [text]
    elif separator == "line":
        documents = text.split("\n")
    else:
        documents = text.split("\n\n")

    documents = [d.strip() for d in documents]
    documents = [d for d in documents if d]
    if max_documents is not None:
        documents = documents[:max_documents]
    if not documents:
        raise CorpusError(
            f"{source} contains no non-empty documents under separator={separator!r}. "
            "An empty corpus produces an empty dataset, which would train on nothing "
            "while reporting success."
        )
    return documents


def tokenize_documents(
    documents: list[str],
    tokenizer: Any,
    *,
    eos_id: int,
    max_tokens: int | None = None,
) -> list[int]:
    """Concatenate documents into one id stream with an explicit EOS after each.

    ``add_special_tokens=False`` is deliberate. A tokenizer that prepends a BOS would put
    one in the *middle* of every packed sequence, at a position where the model is being
    asked to predict ordinary text — the boundary token is EOS, placed here explicitly,
    and nothing else is inserted.
    """
    stream: list[int] = []
    for document in documents:
        ids = tokenizer(document, add_special_tokens=False)["input_ids"]
        stream.extend(int(i) for i in ids)
        stream.append(eos_id)
        if max_tokens is not None and len(stream) >= max_tokens:
            return stream[:max_tokens]
    return stream


def pack_sequences(tokens: list[int], sequence_length: int) -> tuple[list[list[int]], int]:
    """Chunk a token stream into non-overlapping sequences of exactly ``sequence_length``.

    Returns ``(sequences, dropped)``. The trailing partial chunk is dropped rather than
    padded, because the trainer's contract is a rectangular batch with no attention mask
    and no ``-100`` label masking: a padded tail would be trained on as if it were text.
    The dropped count is returned so it can be recorded instead of vanishing.
    """
    if sequence_length < 2:
        raise CorpusError(
            f"sequence_length must be at least 2 to form a next-token pair, got "
            f"{sequence_length}"
        )
    usable = (len(tokens) // sequence_length) * sequence_length
    sequences = [
        tokens[start : start + sequence_length] for start in range(0, usable, sequence_length)
    ]
    return sequences, len(tokens) - usable


def validate_token_ids(sequences: list[list[int]], vocab_size: int) -> None:
    """Every id must index a row that exists, or the embedding lookup is undefined.

    Out-of-range ids do not always raise: on CPU they index garbage, and on CUDA they
    produce a device-side assert thousands of steps in. Checking here makes the failure
    immediate and legible.
    """
    for index, sequence in enumerate(sequences):
        for token in sequence:
            if not 0 <= token < vocab_size:
                raise CorpusError(
                    f"token id {token} at sequence {index} is outside the tokenizer's "
                    f"vocabulary of {vocab_size}. The embedding has no such row, so this "
                    "would be an undefined lookup rather than a training signal."
                )


def prepare_tokenized_corpus(
    *,
    text_path: str | Path,
    tokenizer_path: str | Path,
    sequence_length: int = 1024,
    validation_fraction: float = 0.05,
    document_separator: str = "blank_line",
    max_documents: int | None = None,
    max_tokens: int | None = None,
    max_bytes: int | None = None,
    expected_vocab_size: int | None = None,
    teacher_model: str | None = None,
    teacher_revision: str | None = None,
    trust_remote_code: bool = False,
    tokenizer: Any = None,
) -> tuple[list[list[int]], list[list[int]], TokenizedCorpusStats]:
    """Build deterministic train/validation splits of tokenizer-encoded sequences.

    The return shape is deliberately identical to
    :func:`~qwen_distill.training.text_data.prepare_corpus`: ``list[list[int]]`` of
    sequences exactly ``sequence_length`` long, which is what
    :class:`~qwen_distill.training.text_data.ResumableBatchSampler` and the trainer's
    ``model(input_ids=batch, labels=batch)`` already consume. No new batch contract is
    introduced — the causal shift stays where it already lives, inside the model.

    The split is a **contiguous tail**, matching the byte-level path: held-out sequences
    come from the end of the packed stream, so validation text never appears in training
    text.

    ``tokenizer`` accepts an already-loaded tokenizer, which lets a caller that has one
    (a resident teacher, a test fixture) avoid loading it twice. ``tokenizer_path`` is
    still required, because it is what gets recorded as the corpus's provenance.
    """
    loaded = tokenizer if tokenizer is not None else load_tokenizer(
        tokenizer_path, trust_remote_code=trust_remote_code
    )
    provenance = tokenizer_provenance(
        loaded, tokenizer_path, teacher_model=teacher_model, teacher_revision=teacher_revision
    )
    vocab_size = provenance.vocab_size

    # Checked before any tokenisation, so a mismatch costs a second rather than a corpus.
    # Loud on purpose: the alternative — resizing the student's embedding to fit — leaves
    # rows that were never trained and a checkpoint whose vocabulary nobody can account for.
    #
    # The direction matters, and only one of the two is dangerous. A tokenizer that can
    # emit an id at or past the student's embedding produces an out-of-range index and a
    # run that is wrong from its first batch, so that stays fatal. A tokenizer *smaller*
    # than the embedding is the normal case for this teacher and not an error: Qwen3.8-27B
    # declares vocab_size 248,320 = 128 x 1,940 while its tokenizer holds 248,077, because
    # the embedding is padded to a multiple of 128 for kernel alignment. Demanding exact
    # equality made this check unsatisfiable against the real teacher and blocked every KD
    # run; the padding is reported instead of being tolerated silently.
    if expected_vocab_size is not None and vocab_size != expected_vocab_size:
        padding = expected_vocab_size - vocab_size
        if not 0 < padding < VOCAB_ALIGNMENT:
            raise CorpusError(
                f"the tokenizer at {tokenizer_path} has a vocabulary of {vocab_size}, but "
                f"the student expects {expected_vocab_size}.\n"
                + ("  It can emit ids the student cannot index, so the run would be wrong "
                   "from its first batch.\n" if padding < 0 else
                   f"  The embedding is {padding} rows larger, which is too much to be "
                   f"alignment padding (that is bounded by {VOCAB_ALIGNMENT}); this is a "
                   "different tokenizer, not a padded one.\n")
                + "  These must match. Do not resize the student's embedding to close the "
                  "gap: that leaves rows the run never trains and a checkpoint whose "
                  "vocabulary cannot be accounted for.\n"
                  "  Check that tokenizer_path points at the teacher checkpoint this "
                  "student was derived from."
            )
        print(
            f"  tokenizer vocabulary {vocab_size:,} against a student embedding of "
            f"{expected_vocab_size:,}: {padding} alignment padding row(s) that no token id "
            "reaches. Carried, not trimmed — trimming would change the student's geometry."
        )

    eos_id = resolve_eos_id(loaded)
    documents = read_documents(
        text_path,
        separator=document_separator,
        max_documents=max_documents,
        max_bytes=max_bytes,
    )
    text_bytes = sum(len(d.encode("utf-8")) for d in documents)

    tokens = tokenize_documents(documents, loaded, eos_id=eos_id, max_tokens=max_tokens)
    sequences, dropped = pack_sequences(tokens, sequence_length)
    if not sequences:
        raise CorpusError(
            f"{text_path} tokenised to {len(tokens)} tokens, which is fewer than one "
            f"sequence of {sequence_length}. Use a larger corpus or a shorter "
            "sequence_length."
        )
    validate_token_ids(sequences, vocab_size)

    n_validation = max(1, int(len(sequences) * validation_fraction))
    if n_validation >= len(sequences):
        raise CorpusError(
            f"{len(sequences)} sequence(s) cannot be split into train and validation at "
            f"validation_fraction={validation_fraction}. Without a held-out set there is "
            "nothing measuring generalisation."
        )
    train = sequences[:-n_validation]
    validation = sequences[-n_validation:]

    # Hashed over the *token stream*, not the source text: two corpora that read the same
    # but tokenise differently are different training data, and the digest must say so.
    digest = hashlib.sha256()
    digest.update(f"{PACKING_VERSION}|{sequence_length}|{eos_id}|".encode())
    for token in tokens:
        digest.update(token.to_bytes(4, "little"))

    stats = TokenizedCorpusStats(
        source=f"{text_path} tokenised by {provenance.tokenizer_class} at {tokenizer_path}",
        n_bytes=text_bytes,
        n_sequences=len(sequences),
        sequence_length=sequence_length,
        sha256=digest.hexdigest(),
        n_train=len(train),
        n_validation=len(validation),
        n_tokens=len(tokens),
        n_documents=len(documents),
        n_tokens_dropped=dropped,
        document_separator=document_separator,
        eos_token_id=eos_id,
        tokenizer=provenance.to_dict(),
    )
    return train, validation, stats
