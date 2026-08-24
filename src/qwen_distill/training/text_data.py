"""Byte-level text corpora for Level-2 language training.

Level 1 used a synthetic induction task, chosen because it is *learnable* — a falling
loss there proves the optimizer works. It proves nothing about language. Level 2 needs
real text, and that raises a supply problem: the training environment may have no
network, and the project must stay reproducible and free of licensing questions.

**Byte-level tokenization solves all three at once.** Vocabulary is exactly 256, there
is no tokenizer to download or version, any text file works unchanged, and the loss is
directly interpretable as **bits per byte** — a standard, comparable metric that does
not depend on a tokenizer's compression rate. It also puts nearly all parameters in the
layers rather than in a 248k embedding table, which makes a ~100M run a much better
test of the *architecture* than of the embedding.

The trade-off is honest: byte-level sequences cover ~4x less text per token than BPE, so
a 1024-byte window is roughly a 250-token window. That is fine for Level 2, whose
question is whether the architecture learns language structure at all.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

#: Byte-level vocabulary. Every possible byte value, nothing else.
BYTE_VOCAB_SIZE = 256

#: Words for the procedural corpus, with a deliberately Zipfian frequency profile:
#: a few very common function words and a long tail of content words, which is what
#: gives the corpus learnable statistical structure rather than uniform noise.
_FUNCTION_WORDS = (
    "the", "of", "and", "to", "a", "in", "is", "it", "that", "was", "for", "on",
    "with", "as", "by", "at", "from", "but", "not", "are", "this", "be", "have",
)
_CONTENT_WORDS = (
    "model", "memory", "system", "value", "signal", "state", "layer", "context",
    "pattern", "sequence", "structure", "process", "network", "function", "method",
    "result", "measure", "problem", "question", "answer", "reason", "number",
    "language", "machine", "learning", "attention", "gradient", "parameter",
    "training", "example", "distance", "history", "picture", "morning", "island",
    "garden", "silence", "window", "letter", "distance", "shadow", "current",
)


@dataclass
class CorpusStats:
    """What a corpus actually contains, recorded alongside every experiment."""

    source: str
    n_bytes: int
    n_sequences: int
    sequence_length: int
    sha256: str
    n_train: int
    n_validation: int

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def generate_procedural_text(n_bytes: int = 2_000_000, seed: int = 0) -> str:
    """Deterministically generate English-like prose with no download required.

    Not natural language, and not presented as such. It has sentence boundaries,
    punctuation, capitalisation and a Zipfian word distribution, so a model must learn
    real structure — word boundaries, common-word statistics, sentence shape — rather
    than a single repeating pattern. That is a genuine step up from Level 1's induction
    task while remaining fully reproducible offline.

    Prefer a real text file (:func:`load_text_file`) whenever the environment allows it.
    """
    rng = random.Random(seed)
    # Zipf-like: function words dominate, content words form the tail.
    weighted = list(_FUNCTION_WORDS) * 6 + list(_CONTENT_WORDS)
    parts: list[str] = []
    size = 0
    while size < n_bytes:
        sentence_length = rng.randint(6, 22)
        words = [rng.choice(weighted) for _ in range(sentence_length)]
        words[0] = words[0].capitalize()
        # Occasional commas give the model sub-sentence structure to learn.
        if sentence_length > 12 and rng.random() < 0.6:
            comma = rng.randint(3, sentence_length - 3)
            words[comma] = words[comma] + ","
        sentence = " ".join(words) + rng.choice([".", ".", ".", "!", "?"])
        parts.append(sentence)
        size += len(sentence) + 1
        if rng.random() < 0.05:
            parts.append("\n\n")
            size += 2
    return " ".join(parts)[:n_bytes]


def load_text_file(path: str | Path, *, max_bytes: int | None = None) -> str:
    """Read a UTF-8 text file, optionally truncated.

    Any plain-text file works: a public-domain book, documentation, source code. The
    experiment records the file's SHA-256 so a result is tied to the exact bytes it was
    produced from.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"text corpus not found: {source}")
    text = source.read_text(encoding="utf-8", errors="replace")
    return text[:max_bytes] if max_bytes else text


def encode(text: str) -> list[int]:
    """UTF-8 bytes as token ids. Lossless and tokenizer-free."""
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    """Inverse of :func:`encode`; invalid sequences degrade rather than raise."""
    return bytes(t % BYTE_VOCAB_SIZE for t in tokens).decode("utf-8", errors="replace")


def build_sequences(
    text: str, sequence_length: int, *, stride: int | None = None
) -> list[list[int]]:
    """Chunk text into fixed-length byte sequences.

    Deterministic and non-overlapping by default, so two runs see identical data in an
    identical order — a precondition for comparing two training configurations.
    """
    tokens = encode(text)
    stride = stride or sequence_length
    return [
        tokens[i : i + sequence_length]
        for i in range(0, len(tokens) - sequence_length + 1, stride)
    ]


def prepare_corpus(
    *,
    text_path: str | Path | None = None,
    sequence_length: int = 1024,
    procedural_bytes: int = 2_000_000,
    validation_fraction: float = 0.05,
    seed: int = 0,
    max_bytes: int | None = None,
) -> tuple[list[list[int]], list[list[int]], CorpusStats]:
    """Build deterministic train/validation splits of byte sequences.

    The split is a **contiguous tail**, not a random sample: with overlapping or
    shuffled chunks, validation text can appear inside training text and the validation
    loss stops measuring generalisation. Holding out the end of the corpus keeps the two
    disjoint.
    """
    if text_path:
        text = load_text_file(text_path, max_bytes=max_bytes)
        source = str(text_path)
    else:
        text = generate_procedural_text(procedural_bytes, seed=seed)
        source = f"procedural(seed={seed}, bytes={procedural_bytes})"

    sequences = build_sequences(text, sequence_length)
    if not sequences:
        raise ValueError(
            f"corpus of {len(text)} bytes is too small for sequence_length={sequence_length}"
        )

    n_validation = max(1, int(len(sequences) * validation_fraction))
    train = sequences[:-n_validation]
    validation = sequences[-n_validation:]
    if not train:
        raise ValueError("corpus yields no training sequences after the validation split")

    stats = CorpusStats(
        source=source,
        n_bytes=len(text.encode("utf-8")),
        n_sequences=len(sequences),
        sequence_length=sequence_length,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        n_train=len(train),
        n_validation=len(validation),
    )
    return train, validation, stats


def iterate_batches(
    sequences: list[list[int]], batch_size: int, *, seed: int = 0, shuffle: bool = True
) -> Iterator[list[list[int]]]:
    """Yield batches forever, reshuffling each epoch with a deterministic seed."""
    rng = random.Random(seed)
    order = list(range(len(sequences)))
    while True:
        if shuffle:
            rng.shuffle(order)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            yield [sequences[i] for i in order[start : start + batch_size]]


def bits_per_byte(cross_entropy_nats: float) -> float:
    """Convert a natural-log cross-entropy into bits per byte.

    The interpretable form for byte-level modelling: 8.0 is a model that has learned
    nothing (uniform over 256 bytes), and good byte-level models on English land near
    1.0-1.5.
    """
    import math

    return cross_entropy_nats / math.log(2)


class ResumableBatchSampler:
    """A batch stream that can say where it is, and be put back there.

    :func:`iterate_batches` is a plain generator: its position lives in the interpreter
    and dies with the process. Resuming from a checkpoint then silently restarts the
    data at epoch 0, so the resumed run re-sees sequences it has already trained on and
    the "continued" run is not the run it claims to be.

    This tracks ``epoch`` and ``index`` explicitly and restores them by *replaying* the
    per-epoch shuffles. Replay is exact — ``random.Random(seed)`` is deterministic — and
    cheap, because it permutes an index list rather than touching the data.
    """

    def __init__(
        self,
        sequences: list[list[int]],
        batch_size: int,
        *,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.sequences = sequences
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.index = 0
        self._rng = random.Random(seed)
        self._order = list(range(len(sequences)))
        self._start_epoch()

    @property
    def batches_per_epoch(self) -> int:
        return max(0, (len(self.sequences) - self.batch_size) // self.batch_size + 1)

    def _start_epoch(self) -> None:
        if self.shuffle:
            self._rng.shuffle(self._order)

    def __iter__(self) -> Iterator[list[list[int]]]:
        return self

    def __next__(self) -> list[list[int]]:
        if self.batches_per_epoch == 0:
            raise ValueError(
                f"{len(self.sequences)} sequences cannot fill a batch of {self.batch_size}"
            )
        if self.index >= self.batches_per_epoch:
            self.epoch += 1
            self.index = 0
            self._start_epoch()
        start = self.index * self.batch_size
        batch = [self.sequences[i] for i in self._order[start : start + self.batch_size]]
        self.index += 1
        return batch

    def state_dict(self) -> dict[str, object]:
        """Where the stream is, in terms that survive a process restart."""
        return {
            "epoch": self.epoch,
            "index": self.index,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "batch_size": self.batch_size,
            "n_sequences": len(self.sequences),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore the position by replaying the shuffles that produced it.

        Refuses a state that describes a different corpus or batch size rather than
        resuming onto data it does not match, which would corrupt the run quietly.
        """
        n_sequences = state.get("n_sequences")
        if n_sequences is not None and n_sequences != len(self.sequences):
            raise ValueError(
                f"checkpoint was taken over {n_sequences} sequences but this corpus has "
                f"{len(self.sequences)}; the data changed, so the position is meaningless"
            )
        batch_size = state.get("batch_size")
        if batch_size is not None and batch_size != self.batch_size:
            raise ValueError(
                f"checkpoint used batch_size={batch_size} but this run uses "
                f"{self.batch_size}; batch indices do not carry across"
            )
        self.seed = int(state.get("seed", self.seed))
        self.shuffle = bool(state.get("shuffle", self.shuffle))
        self.epoch = int(state.get("epoch", 0))
        self.index = int(state.get("index", 0))

        # Replay: one shuffle per epoch reached, from a freshly seeded generator.
        self._rng = random.Random(self.seed)
        self._order = list(range(len(self.sequences)))
        for _ in range(self.epoch + 1):
            self._start_epoch()
