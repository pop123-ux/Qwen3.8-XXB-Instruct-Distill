"""Tests for byte-level corpus preparation.

Two properties matter more than the rest, because both fail *silently* and both
invalidate every number the experiment produces:

* **Determinism** — two runs must see identical data in an identical order, or a
  comparison between two configurations measures the data shuffle as much as the change.
* **Disjoint splits** — if validation text also appears in training text, the validation
  loss stops measuring generalisation and starts measuring memorisation.

No torch, no transformers, no network: this module is pure Python by design.
"""

from __future__ import annotations

import math

import pytest

from qwen_distill.training.text_data import (
    BYTE_VOCAB_SIZE,
    bits_per_byte,
    build_sequences,
    decode,
    encode,
    generate_procedural_text,
    iterate_batches,
    load_text_file,
    prepare_corpus,
)


# --- encoding -------------------------------------------------------------
def test_vocabulary_is_exactly_the_byte_range():
    assert BYTE_VOCAB_SIZE == 256


def test_encode_decode_round_trips_including_non_ascii():
    """Byte-level tokenisation is lossless; that is most of why it was chosen."""
    for text in ["hello", "", "The quick brown fox.", "café — naïve — 日本語", "\n\t\x00end"]:
        assert decode(encode(text)) == text


def test_every_token_is_a_valid_byte():
    tokens = encode("Mixed ASCII and — em dashes — plus 日本語")
    assert all(0 <= t < BYTE_VOCAB_SIZE for t in tokens)


def test_non_ascii_costs_more_than_one_token_per_character():
    """A real property of UTF-8 bytes, and the reason byte windows are ~4x shorter."""
    assert len(encode("日本語")) == 9
    assert len(encode("abc")) == 3


# --- procedural corpus ----------------------------------------------------
def test_procedural_text_is_deterministic_for_a_seed():
    assert generate_procedural_text(20_000, seed=7) == generate_procedural_text(20_000, seed=7)


def test_different_seeds_give_different_text():
    assert generate_procedural_text(20_000, seed=0) != generate_procedural_text(20_000, seed=1)


def test_procedural_text_respects_the_requested_size():
    text = generate_procedural_text(5_000, seed=0)
    assert len(text) == 5_000


def test_procedural_text_has_learnable_structure_not_noise():
    """A Zipfian word distribution is the point: uniform noise would teach nothing."""
    text = generate_procedural_text(100_000, seed=0)
    words = [w.strip(".,!?").lower() for w in text.split()]
    assert "the" in words
    # Function words are weighted 6x against the content tail, so a function word must
    # be several times more common than a mid-tail content word. Without that frequency
    # structure the corpus is noise and a falling loss would mean nothing.
    assert words.count("the") > 3 * words.count("gradient")
    assert text.count(".") > 100, "sentence boundaries must exist"


def test_procedural_text_entropy_is_well_below_uniform():
    """If per-byte entropy were near 8 bits, the corpus would be unlearnable noise."""
    text = generate_procedural_text(100_000, seed=0)
    tokens = encode(text)
    counts: dict[int, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    entropy = -sum(
        (c / len(tokens)) * math.log2(c / len(tokens)) for c in counts.values()
    )
    assert entropy < 5.0, f"unigram entropy {entropy:.2f} bits/byte is too close to noise"


# --- file loading ---------------------------------------------------------
def test_load_text_file_reads_utf8_regardless_of_platform_locale(tmp_path):
    """Explicit UTF-8: the default locale encoding is cp1252 on Windows."""
    path = tmp_path / "corpus.txt"
    path.write_text("naïve café — 日本語", encoding="utf-8")
    assert load_text_file(path) == "naïve café — 日本語"


def test_load_text_file_truncates_to_max_bytes(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    assert load_text_file(path, max_bytes=4) == "abcd"


def test_load_text_file_raises_on_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_text_file(tmp_path / "absent.txt")


# --- sequencing -----------------------------------------------------------
def test_sequences_are_fixed_length_and_non_overlapping():
    sequences = build_sequences("x" * 100, sequence_length=10)
    assert len(sequences) == 10
    assert all(len(s) == 10 for s in sequences)
    flattened = [t for s in sequences for t in s]
    assert flattened == encode("x" * 100)


def test_a_short_text_yields_no_sequences_rather_than_a_padded_one():
    assert build_sequences("short", sequence_length=64) == []


def test_stride_controls_overlap():
    sequences = build_sequences("abcdefghij", sequence_length=4, stride=2)
    assert [decode(s) for s in sequences] == ["abcd", "cdef", "efgh", "ghij"]


# --- splits ---------------------------------------------------------------
def test_train_and_validation_sequences_are_disjoint():
    """The whole reason the split is a contiguous tail rather than a random sample."""
    train, validation, _ = prepare_corpus(sequence_length=64, procedural_bytes=40_000)
    train_set = {tuple(s) for s in train}
    assert train_set
    assert all(tuple(s) not in train_set for s in validation)


def test_prepare_corpus_is_deterministic():
    a_train, a_val, a_stats = prepare_corpus(sequence_length=64, procedural_bytes=40_000, seed=3)
    b_train, b_val, b_stats = prepare_corpus(sequence_length=64, procedural_bytes=40_000, seed=3)
    assert a_train == b_train
    assert a_val == b_val
    assert a_stats.sha256 == b_stats.sha256


def test_stats_record_the_exact_bytes_the_result_came_from(tmp_path):
    """A result is only reproducible if it is tied to a specific corpus digest."""
    import hashlib

    path = tmp_path / "corpus.txt"
    text = "The model reads bytes. " * 500
    path.write_text(text, encoding="utf-8")

    _, _, stats = prepare_corpus(text_path=path, sequence_length=64)

    assert stats.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert stats.source == str(path)
    assert stats.n_bytes == len(text.encode("utf-8"))
    assert stats.n_train + stats.n_validation == stats.n_sequences


def test_validation_split_is_never_empty():
    _, validation, stats = prepare_corpus(
        sequence_length=64, procedural_bytes=10_000, validation_fraction=0.0001
    )
    assert len(validation) >= 1
    assert stats.n_validation >= 1


def test_a_corpus_too_small_for_the_window_fails_loudly():
    """Better to stop than to train on one sequence and report a meaningless loss."""
    with pytest.raises(ValueError, match="too small"):
        prepare_corpus(sequence_length=4096, procedural_bytes=200)


def test_a_corpus_with_no_room_for_training_fails_loudly():
    with pytest.raises(ValueError, match="no training sequences"):
        prepare_corpus(sequence_length=64, procedural_bytes=200, validation_fraction=1.0)


# --- batching -------------------------------------------------------------
def test_batches_have_the_requested_shape():
    train, _, _ = prepare_corpus(sequence_length=32, procedural_bytes=20_000)
    batch = next(iterate_batches(train, batch_size=4))
    assert len(batch) == 4
    assert all(len(s) == 32 for s in batch)


def test_batch_order_is_reproducible_for_a_seed():
    train, _, _ = prepare_corpus(sequence_length=32, procedural_bytes=20_000)
    a = iterate_batches(train, 4, seed=11)
    b = iterate_batches(train, 4, seed=11)
    # Several batches, so an epoch boundary and its reshuffle are covered too.
    assert [next(a) for _ in range(20)] == [next(b) for _ in range(20)]

    different = iterate_batches(train, 4, seed=12)
    assert [next(iterate_batches(train, 4, seed=11)) for _ in range(1)] != [
        next(different) for _ in range(1)
    ]


def test_shuffling_off_preserves_corpus_order():
    train, _, _ = prepare_corpus(sequence_length=32, procedural_bytes=20_000)
    batch = next(iterate_batches(train, 4, shuffle=False))
    assert batch == train[:4]


def test_batches_are_yielded_indefinitely():
    """Training runs for a step count, not an epoch count; the stream must not end."""
    train, _, _ = prepare_corpus(sequence_length=32, procedural_bytes=8_000)
    stream = iterate_batches(train, batch_size=4, seed=0)
    produced = [next(stream) for _ in range(len(train) + 10)]
    assert len(produced) == len(train) + 10


# --- metric ---------------------------------------------------------------
def test_uniform_over_256_bytes_is_exactly_8_bits_per_byte():
    """The anchor for every bpb number this project reports."""
    assert bits_per_byte(math.log(256)) == pytest.approx(8.0)


def test_a_perfect_model_is_zero_bits_per_byte():
    assert bits_per_byte(0.0) == 0.0


def test_bits_per_byte_is_cross_entropy_in_base_two():
    assert bits_per_byte(1.0) == pytest.approx(1 / math.log(2))
