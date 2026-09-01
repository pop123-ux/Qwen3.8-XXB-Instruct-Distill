"""Tests for the tokenizer-backed corpus path.

The blocker this path closes is a vocabulary mismatch: the corpus pipeline emitted
byte-level ids (vocab 256) while the canonical student ``qwen38_19b_h5120_l48_moe`` has a
248,320-row embedding. Two failure modes matter more than the rest, because both are
silent:

* **an out-of-range id**, which on CPU indexes garbage and on CUDA asserts thousands of
  steps into a paid run;
* **a tokenizer that is not the student's**, which trains a 248k-vocabulary model against
  someone else's token ids and produces a checkpoint nobody can account for.

Both are asserted here as *refusals*, not as warnings.

No network and no Qwen download: the fixture is the repository's own tiny checkpoint,
built offline by :mod:`qwen_distill.testing`. It needs `transformers` for
``AutoTokenizer`` but **not** `torch` — nothing here builds a model.
"""

from __future__ import annotations

import pytest
from conftest import HAS_STACK, HAS_TRANSFORMERS

from qwen_distill.training.config import DataConfig, ExperimentConfig, ModelConfig
from qwen_distill.training.text_data import BYTE_VOCAB_SIZE

pytestmark = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="requires transformers for AutoTokenizer"
)

TINY_VOCAB = 512


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory):
    """A local directory holding tokenizer files and nothing that needs loading.

    ``with_weights=False`` is the point: the tokenized path must work against a checkpoint
    directory without ever opening model weights, which is what makes corpus preparation
    possible on a laptop while the teacher is 54 GB.
    """
    from qwen_distill.testing import write_tiny_checkpoint

    path = tmp_path_factory.mktemp("tok") / "checkpoint"
    return write_tiny_checkpoint(path, with_weights=False)


@pytest.fixture(scope="module")
def corpus_file(tmp_path_factory):
    """Forty short blank-line-separated documents."""
    path = tmp_path_factory.mktemp("corpus") / "corpus.txt"
    path.write_text(
        "\n\n".join(
            f"Document number {i} concerns language models, memory and structure."
            for i in range(40)
        ),
        encoding="utf-8",
    )
    return path


# --- tokenizer loading ----------------------------------------------------
def test_tokenizer_loads_from_a_local_path(tokenizer_dir):
    from qwen_distill.training.tokenized_data import load_tokenizer

    tokenizer = load_tokenizer(tokenizer_dir)
    assert tokenizer is not None


def test_a_missing_tokenizer_path_fails_with_the_path_in_the_message(tmp_path):
    from qwen_distill.training.tokenized_data import TokenizerLoadError, load_tokenizer

    with pytest.raises(TokenizerLoadError, match="does not exist"):
        load_tokenizer(tmp_path / "absent")


def test_a_directory_without_tokenizer_files_is_refused_not_downloaded(tmp_path):
    """The failure mode this guards: a path that does not resolve locally is otherwise
    treated as a Hub repo id and fetched over the network."""
    from qwen_distill.training.tokenized_data import TokenizerLoadError, load_tokenizer

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TokenizerLoadError, match="no tokenizer files"):
        load_tokenizer(empty)


def test_vocabulary_size_is_read_off_the_tokenizer(tokenizer_dir):
    """Reported, never hardcoded — 248320 is a fact about a checkpoint, not a constant."""
    from qwen_distill.training.tokenized_data import load_tokenizer, tokenizer_provenance

    tokenizer = load_tokenizer(tokenizer_dir)
    provenance = tokenizer_provenance(tokenizer, tokenizer_dir)
    assert provenance.vocab_size == TINY_VOCAB
    assert provenance.vocab_size == len(tokenizer)
    assert provenance.vocab_size != BYTE_VOCAB_SIZE


def test_simple_text_tokenizes_to_ids_inside_the_vocabulary(tokenizer_dir):
    from qwen_distill.training.tokenized_data import load_tokenizer

    tokenizer = load_tokenizer(tokenizer_dir)
    ids = tokenizer("hello world", add_special_tokens=False)["input_ids"]
    assert ids
    assert all(0 <= int(i) < TINY_VOCAB for i in ids)


# --- EOS ------------------------------------------------------------------
def test_eos_is_resolved_from_the_tokenizer(tokenizer_dir):
    from qwen_distill.training.tokenized_data import load_tokenizer, resolve_eos_id

    tokenizer = load_tokenizer(tokenizer_dir)
    assert resolve_eos_id(tokenizer) == tokenizer.eos_token_id


def test_a_tokenizer_with_no_eos_is_refused_rather_than_given_one():
    """Adding a token would change the vocabulary the student was built for."""
    from qwen_distill.training.tokenized_data import CorpusError, resolve_eos_id

    class NoEos:
        eos_token_id = None
        sep_token_id = None
        pad_token_id = None

    with pytest.raises(CorpusError, match="no eos_token_id"):
        resolve_eos_id(NoEos())


def test_every_document_is_followed_by_exactly_one_eos(tokenizer_dir):
    from qwen_distill.training.tokenized_data import load_tokenizer, tokenize_documents

    tokenizer = load_tokenizer(tokenizer_dir)
    eos = tokenizer.eos_token_id
    documents = ["first document", "second document", "third"]
    stream = tokenize_documents(documents, tokenizer, eos_id=eos)
    assert stream.count(eos) == len(documents)
    assert stream[-1] == eos


def test_packing_preserves_the_eos_between_documents(tokenizer_dir, corpus_file):
    """EOS must survive chunking, or the model learns documents run into each other."""
    from qwen_distill.training.tokenized_data import prepare_tokenized_corpus

    train, _, stats = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir, sequence_length=32
    )
    assert stats.eos_token_id is not None
    assert any(stats.eos_token_id in sequence for sequence in train)


# --- packing --------------------------------------------------------------
def test_packing_produces_sequences_of_exactly_the_requested_length(tokenizer_dir, corpus_file):
    from qwen_distill.training.tokenized_data import prepare_tokenized_corpus

    for length in (16, 32, 64):
        train, validation, stats = prepare_tokenized_corpus(
            text_path=corpus_file, tokenizer_path=tokenizer_dir, sequence_length=length
        )
        assert {len(s) for s in train} == {length}
        assert {len(s) for s in validation} == {length}
        assert stats.sequence_length == length


def test_the_dropped_tail_is_counted_not_padded():
    """Padding would be trained on as text: the trainer uses no mask and no -100 label."""
    from qwen_distill.training.tokenized_data import pack_sequences

    sequences, dropped = pack_sequences(list(range(70)), 32)
    assert len(sequences) == 2
    assert dropped == 6
    assert all(len(s) == 32 for s in sequences)


def test_a_sequence_length_below_two_is_refused():
    from qwen_distill.training.tokenized_data import CorpusError, pack_sequences

    with pytest.raises(CorpusError, match="at least 2"):
        pack_sequences([1, 2, 3], 1)


def test_train_and_validation_are_disjoint_and_validation_is_the_tail(tokenizer_dir, corpus_file):
    from qwen_distill.training.tokenized_data import prepare_tokenized_corpus

    train, validation, stats = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir,
        sequence_length=32, validation_fraction=0.2,
    )
    assert len(train) + len(validation) == stats.n_sequences
    assert stats.n_train == len(train)
    assert stats.n_validation == len(validation)
    # a contiguous tail, so held-out text never appears in training text
    assert [tuple(s) for s in validation] == [
        tuple(s) for s in (train + validation)[len(train):]
    ]


# --- the trainer's batch contract ----------------------------------------
def test_sequences_match_what_the_trainer_consumes(tokenizer_dir, corpus_file):
    """The trainer builds one (B, L) tensor and passes it as *both* input and label; the
    causal shift lives inside the model. So the data layer owes it rectangular integer
    sequences and nothing else — no separate label tensor, no padding, no mask."""
    from qwen_distill.training.text_data import ResumableBatchSampler
    from qwen_distill.training.tokenized_data import prepare_tokenized_corpus

    train, _, _ = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir, sequence_length=32
    )
    sampler = ResumableBatchSampler(train, batch_size=4, seed=0)
    batch = next(iter(sampler))
    assert len(batch) == 4
    assert all(len(s) == 32 for s in batch)
    assert all(isinstance(t, int) for s in batch for t in s)


def test_input_and_label_alignment_is_the_models_shift(tokenizer_dir, corpus_file):
    """Assert the alignment the trainer actually relies on: for a sequence used as both
    input and label, position i predicts position i+1."""
    from qwen_distill.training.tokenized_data import prepare_tokenized_corpus

    train, _, _ = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir, sequence_length=16
    )
    sequence = train[0]
    inputs, labels = sequence[:-1], sequence[1:]
    assert len(inputs) == len(labels) == 15
    assert labels[0] == sequence[1]
    assert inputs[-1] == sequence[-2]


# --- vocabulary safety ----------------------------------------------------
def test_all_token_ids_are_inside_the_vocabulary(tokenizer_dir, corpus_file):
    from qwen_distill.training.tokenized_data import prepare_tokenized_corpus

    train, validation, stats = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir, sequence_length=32
    )
    vocab = stats.tokenizer["vocab_size"]
    for sequence in train + validation:
        assert all(0 <= token < vocab for token in sequence)


def test_an_out_of_range_id_is_refused():
    from qwen_distill.training.tokenized_data import CorpusError, validate_token_ids

    validate_token_ids([[0, 1, 511]], 512)
    with pytest.raises(CorpusError, match="outside the tokenizer's vocabulary"):
        validate_token_ids([[0, 1, 512]], 512)
    with pytest.raises(CorpusError, match="outside the tokenizer's vocabulary"):
        validate_token_ids([[-1]], 512)


def test_a_vocabulary_mismatch_fails_loudly_and_does_not_resize(tokenizer_dir, corpus_file):
    """The canonical student's 248,320 rows against a different tokenizer is the blocker
    this whole path exists for. It must refuse, not adapt."""
    from qwen_distill.training.tokenized_data import CorpusError, prepare_tokenized_corpus

    with pytest.raises(CorpusError) as excinfo:
        prepare_tokenized_corpus(
            text_path=corpus_file, tokenizer_path=tokenizer_dir,
            sequence_length=32, expected_vocab_size=248_320,
        )
    message = str(excinfo.value)
    assert "248320" in message.replace(",", "")
    assert str(TINY_VOCAB) in message
    assert "resize" in message


def test_a_matching_vocabulary_passes(tokenizer_dir, corpus_file):
    from qwen_distill.training.tokenized_data import prepare_tokenized_corpus

    train, _, stats = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir,
        sequence_length=32, expected_vocab_size=TINY_VOCAB,
    )
    assert train
    assert stats.tokenizer["vocab_size"] == TINY_VOCAB


# --- empty and undersized corpora ----------------------------------------
def test_an_empty_corpus_fails_clearly(tokenizer_dir, tmp_path):
    from qwen_distill.training.tokenized_data import CorpusError, prepare_tokenized_corpus

    empty = tmp_path / "empty.txt"
    empty.write_text("   \n\n  \n", encoding="utf-8")
    with pytest.raises(CorpusError, match="no non-empty documents"):
        prepare_tokenized_corpus(
            text_path=empty, tokenizer_path=tokenizer_dir, sequence_length=32
        )


def test_a_missing_corpus_file_fails_clearly(tokenizer_dir, tmp_path):
    from qwen_distill.training.tokenized_data import CorpusError, prepare_tokenized_corpus

    with pytest.raises(CorpusError, match="not found"):
        prepare_tokenized_corpus(
            text_path=tmp_path / "absent.txt", tokenizer_path=tokenizer_dir, sequence_length=32
        )


def test_a_corpus_too_small_for_one_sequence_fails_clearly(tokenizer_dir, tmp_path):
    from qwen_distill.training.tokenized_data import CorpusError, prepare_tokenized_corpus

    tiny = tmp_path / "tiny.txt"
    tiny.write_text("short", encoding="utf-8")
    with pytest.raises(CorpusError, match="fewer than one sequence"):
        prepare_tokenized_corpus(
            text_path=tiny, tokenizer_path=tokenizer_dir, sequence_length=4096
        )


# --- determinism ----------------------------------------------------------
def test_the_packed_stream_is_deterministic(tokenizer_dir, corpus_file):
    """Same corpus, tokenizer and sequence length must give the same token stream, or a
    comparison between two runs measures the data as much as the change."""
    from qwen_distill.training.tokenized_data import prepare_tokenized_corpus

    first = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir, sequence_length=32
    )
    second = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir, sequence_length=32
    )
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[2].sha256 == second[2].sha256


def test_the_digest_changes_when_the_packing_changes(tokenizer_dir, corpus_file):
    """A digest that ignored sequence length would call two different corpora the same."""
    from qwen_distill.training.tokenized_data import prepare_tokenized_corpus

    a = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir, sequence_length=32
    )[2]
    b = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir, sequence_length=64
    )[2]
    assert a.sha256 != b.sha256


# --- documents ------------------------------------------------------------
def test_document_separators_split_as_documented(tmp_path):
    from qwen_distill.training.tokenized_data import read_documents

    path = tmp_path / "c.txt"
    path.write_text("alpha\nbeta\n\ngamma\n", encoding="utf-8")
    assert read_documents(path, separator="blank_line") == ["alpha\nbeta", "gamma"]
    assert read_documents(path, separator="line") == ["alpha", "beta", "gamma"]
    assert read_documents(path, separator="file") == ["alpha\nbeta\n\ngamma"]


def test_an_unknown_separator_is_refused(tmp_path):
    from qwen_distill.training.tokenized_data import CorpusError, read_documents

    path = tmp_path / "c.txt"
    path.write_text("alpha", encoding="utf-8")
    with pytest.raises(CorpusError, match="unknown document_separator"):
        read_documents(path, separator="paragraph")


def test_max_documents_bounds_a_smoke_test(tokenizer_dir, corpus_file):
    from qwen_distill.training.tokenized_data import prepare_tokenized_corpus

    _, _, stats = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir,
        sequence_length=16, max_documents=5,
    )
    assert stats.n_documents == 5


def test_max_tokens_bounds_a_smoke_test(tokenizer_dir, corpus_file):
    from qwen_distill.training.tokenized_data import prepare_tokenized_corpus

    _, _, stats = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir,
        sequence_length=16, max_tokens=200,
    )
    assert stats.n_tokens == 200


# --- provenance -----------------------------------------------------------
def test_stats_record_what_a_run_needs_to_be_reproducible(tokenizer_dir, corpus_file):
    from qwen_distill.training.tokenized_data import PACKING_VERSION, prepare_tokenized_corpus

    _, _, stats = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir, sequence_length=32,
        teacher_model="Qwen/Qwen3.8-27B", teacher_revision="0123abc",
    )
    recorded = stats.to_dict()
    assert recorded["packing_version"] == PACKING_VERSION
    assert recorded["document_separator"] == "blank_line"
    assert recorded["sequence_length"] == 32
    assert recorded["n_tokens"] > 0
    tokenizer = recorded["tokenizer"]
    assert tokenizer["tokenizer_class"]
    assert tokenizer["vocab_size"] == TINY_VOCAB
    assert tokenizer["teacher_model"] == "Qwen/Qwen3.8-27B"
    assert tokenizer["teacher_revision"] == "0123abc"
    assert tokenizer["is_pinned"] is True
    assert str(tokenizer_dir) in tokenizer["source"]
    # tokenizer files are hashed, so a substituted tokenizer is detectable
    assert tokenizer["file_sha256"]["tokenizer.json"]


def test_an_unpinned_teacher_revision_is_recorded_as_such(tokenizer_dir, corpus_file):
    from qwen_distill.training.tokenized_data import prepare_tokenized_corpus

    _, _, stats = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir, sequence_length=32,
        teacher_model="Qwen/Qwen3.8-27B",
    )
    tokenizer = stats.to_dict()["tokenizer"]
    assert tokenizer["is_pinned"] is False
    assert "pinning_note" in tokenizer


def test_stats_stay_compatible_with_the_byte_level_consumer(tokenizer_dir, corpus_file):
    """The trainer and the summary writer read `corpus_stats` structurally; the tokenised
    stats must not drop a field the byte-level path guaranteed."""
    from qwen_distill.training.text_data import CorpusStats
    from qwen_distill.training.tokenized_data import prepare_tokenized_corpus

    _, _, stats = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir, sequence_length=32
    )
    assert isinstance(stats, CorpusStats)
    for field in ("source", "n_bytes", "n_sequences", "sequence_length",
                  "sha256", "n_train", "n_validation"):
        assert field in stats.to_dict()


# --- configuration --------------------------------------------------------
def test_the_config_selects_the_tokenized_mode_explicitly():
    data = DataConfig(tokenized_text=True, text_path="c.txt", tokenizer_path="/models/teacher")
    assert data.mode == "tokenized"
    assert data.is_sequence_corpus


def test_legacy_modes_are_unchanged():
    assert DataConfig(text_corpus=True).mode == "text"
    assert DataConfig(synthetic=True).mode == "synthetic"
    assert DataConfig(train_path="d.jsonl").mode == "distillation"
    assert DataConfig(text_corpus=True).is_sequence_corpus
    assert not DataConfig(synthetic=True).is_sequence_corpus
    assert not DataConfig(train_path="d.jsonl").is_sequence_corpus


def _config(**data_kwargs) -> ExperimentConfig:
    return ExperimentConfig(
        name="t",
        model=ModelConfig(pretrained="checkpoint"),
        data=DataConfig(**data_kwargs),
    )


def test_tokenized_text_requires_both_paths():
    with pytest.raises(ValueError, match="needs data.tokenizer_path"):
        _config(tokenized_text=True, text_path="c.txt").validate()
    with pytest.raises(ValueError, match="needs data.text_path"):
        _config(tokenized_text=True, tokenizer_path="/models/teacher").validate()


def test_data_sources_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _config(
            tokenized_text=True, text_path="c.txt", tokenizer_path="/t", text_corpus=True
        ).validate()


def test_a_bad_separator_is_rejected_by_the_config():
    with pytest.raises(ValueError, match="document_separator"):
        _config(
            tokenized_text=True, text_path="c.txt", tokenizer_path="/t",
            document_separator="paragraph",
        ).validate()


def test_kd_over_a_tokenized_corpus_needs_an_online_teacher():
    """A corpus carries no stored teacher logits, byte-level or tokenised."""
    config = _config(tokenized_text=True, text_path="c.txt", tokenizer_path="/t")
    config.training.objective = "logit_kd"
    with pytest.raises(ValueError, match="no stored teacher logits"):
        config.validate()
    config.objective = {"signal_source": "online"}
    config.validate()


def test_a_valid_tokenized_config_passes():
    _config(
        tokenized_text=True, text_path="c.txt", tokenizer_path="/models/teacher",
        expected_vocab_size=248_320, max_sequence_length=4096,
    ).validate()


# --- trainer integration --------------------------------------------------
#
# The point of the whole path is that `train(...)` consumes it unchanged. These drive the
# real trainer on CPU with a tiny model, because a data layer that is correct in isolation
# and wrong at the trainer's boundary is exactly the failure this task exists to remove.

#: A real hybrid model small enough to train in seconds, with the *tokenizer's* vocabulary
#: rather than the byte vocabulary. That substitution is the blocker, in miniature.
TINY_MODEL = {
    "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
    "vocab_size": TINY_VOCAB, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 32, "linear_num_key_heads": 1, "linear_num_value_heads": 2,
    "linear_key_head_dim": 32, "linear_value_head_dim": 32,
    "full_attention_interval": 4, "tie_word_embeddings": True,
    "max_position_embeddings": 512,
}


def _training_config(output, tokenizer_dir, corpus_file, **overrides):
    config = ExperimentConfig(name="tokenized-smoke")
    config.model = ModelConfig(architecture=dict(TINY_MODEL))
    config.data = DataConfig(
        tokenized_text=True,
        text_path=str(corpus_file),
        tokenizer_path=str(tokenizer_dir),
        max_sequence_length=32,
        validation_fraction=0.2,
        expected_vocab_size=TINY_VOCAB,
        **overrides,
    )
    config.training.max_steps = 2
    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 1
    config.training.eval_every = 2
    config.training.save_every = 2
    config.training.log_every = 1
    config.training.precision = "fp32"
    config.training.strategy = "full"
    config.training.objective = "sft"
    config.training.gradient_checkpointing = False
    config.runtime.output_dir = str(output)
    config.runtime.device = "cpu"
    config.validate()
    return config


@pytest.mark.skipif(not HAS_STACK, reason="requires torch and transformers")
def test_the_trainer_consumes_the_tokenized_path(tmp_path, tokenizer_dir, corpus_file):
    """The integration claim: existing `train(...)`, tokenised data, no trainer redesign."""
    import json

    from qwen_distill.training.trainer import train

    output = tmp_path / "run"
    config = _training_config(output, tokenizer_dir, corpus_file)
    assert train(config, config.model.resolve_spec()) == 0

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["outcome"] == "completed"
    assert summary["steps"] == 2

    # The data is named honestly: a tokenised run is not "byte-level causal LM", and
    # bits-per-byte — which would be bits-per-token here — is absent.
    assert summary["objective"] == "tokenized causal LM"
    assert "best_validation_bits_per_byte" not in summary
    assert "uniform_baseline_bits_per_byte" not in summary
    records = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records
    assert all("bits_per_byte" not in record for record in records)


@pytest.mark.skipif(not HAS_STACK, reason="requires torch and transformers")
def test_the_run_summary_carries_the_tokenizer_provenance(tmp_path, tokenizer_dir, corpus_file):
    """Provenance rides the existing `corpus` block rather than a second record system."""
    import json

    from qwen_distill.training.trainer import train

    output = tmp_path / "run"
    config = _training_config(output, tokenizer_dir, corpus_file)
    config.teacher = {"model": "Qwen/Qwen3.8-27B", "revision": "0123abc"}
    assert train(config, config.model.resolve_spec()) == 0

    corpus = json.loads((output / "summary.json").read_text(encoding="utf-8"))["corpus"]
    assert corpus["sequence_length"] == 32
    assert corpus["packing_version"]
    assert corpus["n_tokens"] > 0
    assert corpus["tokenizer"]["vocab_size"] == TINY_VOCAB
    assert corpus["tokenizer"]["teacher_model"] == "Qwen/Qwen3.8-27B"
    assert corpus["tokenizer"]["teacher_revision"] == "0123abc"
    assert corpus["tokenizer"]["file_sha256"]["tokenizer.json"]


@pytest.mark.skipif(not HAS_STACK, reason="requires torch and transformers")
def test_a_vocabulary_mismatch_stops_the_run_before_training(tmp_path, tokenizer_dir, corpus_file):
    """The canonical failure: a 248,320-vocabulary student against another tokenizer."""
    from qwen_distill.training.tokenized_data import CorpusError
    from qwen_distill.training.trainer import train

    output = tmp_path / "run"
    config = _training_config(output, tokenizer_dir, corpus_file)
    config.data.expected_vocab_size = 248_320
    with pytest.raises(CorpusError, match="vocabulary"):
        train(config, config.model.resolve_spec())
    assert not (output / "summary.json").exists()


@pytest.mark.skipif(not HAS_STACK, reason="requires torch and transformers")
def test_the_byte_level_path_still_trains_unchanged(tmp_path):
    """Legacy regression: the Level-2 configuration must behave exactly as before."""
    import json

    from qwen_distill.training.trainer import train

    output = tmp_path / "legacy"
    config = ExperimentConfig(name="legacy-bytes")
    config.model = ModelConfig(architecture={**TINY_MODEL, "vocab_size": BYTE_VOCAB_SIZE})
    config.data = DataConfig(
        text_corpus=True, max_sequence_length=32, procedural_bytes=20_000,
        validation_fraction=0.2,
    )
    config.training.max_steps = 2
    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 1
    config.training.eval_every = 2
    config.training.save_every = 2
    config.training.log_every = 1
    config.training.precision = "fp32"
    config.training.strategy = "full"
    config.training.gradient_checkpointing = False
    config.runtime.output_dir = str(output)
    config.runtime.device = "cpu"
    config.validate()
    assert train(config, config.model.resolve_spec()) == 0

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    # Still byte-level, still reported in bits per byte against the 8.0 uniform baseline.
    assert summary["objective"] == "byte-level causal LM"
    assert summary["uniform_baseline_bits_per_byte"] == 8.0
    assert summary["corpus"]["n_bytes"] > 0


@pytest.mark.skipif(not HAS_STACK, reason="requires torch and transformers")
def test_the_batch_tensor_has_the_shape_and_dtype_the_trainer_builds(tokenizer_dir, corpus_file):
    """Integer ids cannot be NaN, so the finiteness that matters is downstream: the tensor
    converts cleanly at the dtype the embedding indexes with, and a forward pass over it
    produces a finite loss rather than a device-side assert."""
    import torch

    from qwen_distill.training.config import ExperimentConfig
    from qwen_distill.training.text_data import ResumableBatchSampler
    from qwen_distill.training.tokenized_data import prepare_tokenized_corpus
    from qwen_distill.training.trainer import build_model

    train, _, stats = prepare_tokenized_corpus(
        text_path=corpus_file, tokenizer_path=tokenizer_dir, sequence_length=32
    )
    sampler = ResumableBatchSampler(train, batch_size=2, seed=0)
    batch = torch.tensor(next(sampler), dtype=torch.long)
    assert batch.shape == (2, 32)
    assert batch.dtype == torch.long
    assert int(batch.min()) >= 0
    assert int(batch.max()) < stats.tokenizer["vocab_size"]

    config = ExperimentConfig(name="shapes")
    config.model = ModelConfig(architecture=dict(TINY_MODEL))
    model = build_model(config, config.model.resolve_spec())
    with torch.no_grad():
        loss = model(input_ids=batch, labels=batch).loss
    assert torch.isfinite(loss)
