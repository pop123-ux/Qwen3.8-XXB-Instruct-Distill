"""Tests for Level 2R corpus preparation and the generation sanity checks.

Level 2R changes exactly one thing about Level 2: the corpus. That makes two properties
load-bearing, and both are easy to get subtly wrong.

**The split must be at document level and fixed.** Level 2 held out a contiguous tail of
one text, which measures how well a model continues a passage it is already reading.
Holding out whole works measures generalisation to unseen prose. If a validation book
leaks into training, the number stops meaning that — and nothing else in the run would
say so.

**The corpus must be reproducible.** A result is tied to the exact bytes that produced
it, so two preparations of the same documents have to hash identically.

Everything here is CPU-only and offline: no download, no GPU. Gutenberg-shaped fixtures
stand in for real books.
"""

from __future__ import annotations

import json

import pytest

from qwen_distill.training.corpus import (
    DOCUMENT_SEPARATOR,
    PREPARATION_VERSION,
    Document,
    assign_splits,
    build_corpus,
    check_overlap,
    estimate_run,
    extract_metadata,
    load_documents_from_directory,
    normalise,
    sha256_text,
    strip_gutenberg_boilerplate,
)
from qwen_distill.training.sanity import (
    SANITY_PROMPTS,
    GenerationCheck,
    SanityReport,
    check_generation,
)

GUTENBERG_FIXTURE = (
    "The Project Gutenberg eBook of Some Book\r\n\r\n"
    "Title: Some Book\r\nAuthor: A Writer\r\nRelease Date: 1998\r\nLanguage: English\r\n\r\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK SOME BOOK ***\r\n\r\n"
    "Title: Some Book\r\nAuthor: A Writer\r\n\r\n\r\n\r\n\r\n"
    "It was the best of times, it was the worst of times.\r\n\r\n"
    "The river ran quietly past the old stone bridge.\r\n\r\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK SOME BOOK ***\r\n\r\n"
    "This eBook is for the use of anyone anywhere ... FULL LICENSE ...\r\n"
)


def make_document(identifier: str, body: str, split: str = "train") -> Document:
    return Document(identifier=identifier, text=normalise(body), split=split)


# --- normalisation --------------------------------------------------------
def test_gutenberg_boilerplate_is_removed():
    """The licence text is identical across every book — the most predictable content
    in the corpus, and the clearest contamination available."""
    cleaned = strip_gutenberg_boilerplate(GUTENBERG_FIXTURE)

    assert "PROJECT GUTENBERG" not in cleaned
    assert "FULL LICENSE" not in cleaned
    assert "Release Date" not in cleaned
    assert "It was the best of times" in cleaned


def test_residual_metadata_lines_are_dropped_but_only_at_the_top():
    """A 'Title:' inside the prose is part of the work and must survive."""
    text = strip_gutenberg_boilerplate(GUTENBERG_FIXTURE)
    assert not text.lstrip().startswith("Title:")

    inline = "Chapter One\n\nHe read the words Title: a curious thing, and paused.\n"
    assert "Title: a curious thing" in strip_gutenberg_boilerplate(inline)


def test_text_without_markers_is_returned_unchanged():
    """A non-Gutenberg source is legitimate; truncating it silently would be worse."""
    plain = "Just some prose with no markers at all.\n"
    assert strip_gutenberg_boilerplate(plain) == plain


def test_line_endings_are_normalised_so_the_hash_is_platform_independent():
    assert sha256_text(normalise("a\r\nb\r\n")) == sha256_text(normalise("a\nb\n"))


def test_unicode_is_nfc_normalised():
    """The same character as one code point or two is different bytes, and modelling
    both wastes capacity on an encoding artefact."""
    composed, decomposed = "café", "café"
    assert composed != decomposed
    assert normalise(composed) == normalise(decomposed)


def test_long_blank_runs_are_collapsed():
    assert "\n\n\n" not in normalise("a\n\n\n\n\n\nb")


def test_normalisation_does_not_lowercase_or_strip_punctuation():
    """The experiment is whether the model learns English as written."""
    text = normalise("The Cat, indeed! Was it?")
    assert "The Cat, indeed! Was it?" in text


def test_metadata_is_read_from_the_file_not_assumed():
    """So a wrong catalogue id shows up in the manifest rather than being papered over."""
    title, author = extract_metadata(GUTENBERG_FIXTURE)
    assert title == "Some Book"
    assert author == "A Writer"


def test_missing_metadata_is_none_rather_than_invented():
    title, author = extract_metadata("no headers here")
    assert title is None and author is None


# --- document-level split -------------------------------------------------
def test_whole_documents_go_to_one_side_only():
    """The property that makes validation BPB mean generalisation."""
    documents = [make_document(str(i), f"book {i} text " * 50) for i in range(5)]
    documents = assign_splits(documents, validation_ids=("1", "3"))

    train_text, validation_text, _ = build_corpus(
        documents, name="t", split_rule="test"
    )

    assert "book 1 text" in validation_text
    assert "book 1 text" not in train_text
    assert "book 0 text" in train_text
    assert "book 0 text" not in validation_text


def test_no_validation_passage_appears_in_the_training_text():
    documents = [make_document(str(i), f"unique body {i} " * 100) for i in range(6)]
    documents = assign_splits(documents, validation_ids=("2", "4"))
    train_text, validation_text, _ = build_corpus(documents, name="t", split_rule="r")

    assert check_overlap(train_text, validation_text)["ok"]


def test_a_duplicated_work_is_detected_by_the_overlap_check():
    """A document-level split should make this impossible, so a hit is a real problem —
    the same work under two ids, or an anthology reprinting a held-out text."""
    body = "the same passage repeated verbatim across two entries " * 40
    documents = [make_document("a", body), make_document("b", body)]
    documents = assign_splits(documents, validation_ids=("b",))
    train_text, validation_text, _ = build_corpus(documents, name="t", split_rule="r")

    result = check_overlap(train_text, validation_text)

    assert not result["ok"]
    assert result["overlaps"] > 0


def test_building_without_a_validation_document_is_refused():
    """Without a held-out set there is nothing measuring generalisation, which is the
    entire question Level 2R asks."""
    documents = [make_document("a", "text " * 50)]
    with pytest.raises(ValueError, match="no validation documents"):
        build_corpus(documents, name="t", split_rule="r")


def test_building_without_a_training_document_is_refused():
    documents = [make_document("a", "text " * 50, split="validation")]
    with pytest.raises(ValueError, match="no training documents"):
        build_corpus(documents, name="t", split_rule="r")


def test_the_split_is_by_explicit_id_not_by_fraction():
    """A fraction shifts when the document list is edited; a named set cannot."""
    documents = [make_document(str(i), f"body {i} " * 30) for i in range(4)]
    assign_splits(documents, validation_ids=("2",))
    assert [d.split for d in documents] == ["train", "train", "validation", "train"]

    # Adding a document must not move anything else.
    documents.append(make_document("9", "body 9 " * 30))
    assign_splits(documents, validation_ids=("2",))
    assert [d.identifier for d in documents if d.split == "validation"] == ["2"]


# --- determinism ----------------------------------------------------------
def test_two_preparations_of_the_same_documents_hash_identically():
    def prepare():
        docs = [make_document(str(i), f"body {i} " * 40) for i in range(5)]
        return build_corpus(
            assign_splits(docs, validation_ids=("3",)), name="t", split_rule="r"
        )[2]

    first, second = prepare(), prepare()
    assert first.train_sha256 == second.train_sha256
    assert first.validation_sha256 == second.validation_sha256


def test_ordering_is_by_identifier_not_by_input_order():
    """So two preparations from differently-ordered inputs still match."""
    bodies = {str(i): f"body {i} " * 30 for i in range(5)}
    forward = [make_document(i, bodies[i]) for i in sorted(bodies)]
    reverse = [make_document(i, bodies[i]) for i in sorted(bodies, reverse=True)]

    a = build_corpus(assign_splits(forward, validation_ids=("4",)), name="t", split_rule="r")
    b = build_corpus(assign_splits(reverse, validation_ids=("4",)), name="t", split_rule="r")

    assert a[2].train_sha256 == b[2].train_sha256


def test_changing_one_document_changes_the_hash():
    def prepare(marker: str):
        docs = [make_document(str(i), f"body {i} {marker} " * 30) for i in range(4)]
        return build_corpus(
            assign_splits(docs, validation_ids=("3",)), name="t", split_rule="r"
        )[2]

    assert prepare("alpha").train_sha256 != prepare("beta").train_sha256


# --- manifest -------------------------------------------------------------
def test_the_manifest_records_everything_needed_to_audit_the_corpus():
    documents = [
        Document(identifier="1342", text=normalise("prose " * 100), title="Pride",
                 author="Austen", source_url="https://example/1342",
                 public_domain_basis="pre-1929, copyright expired"),
        Document(identifier="55", text=normalise("other " * 100), title="Oz",
                 author="Baum", source_url="https://example/55",
                 public_domain_basis="pre-1929, copyright expired", split="validation"),
    ]
    _, _, manifest = build_corpus(
        documents, name="level2r", split_rule="document level",
        contamination_notes=("no benchmark data included",),
    )
    payload = manifest.to_dict()

    assert payload["preparation_version"] == PREPARATION_VERSION
    assert payload["n_documents"] == 2
    assert payload["n_train_documents"] == 1
    assert payload["n_validation_documents"] == 1
    assert payload["train_sha256"] and payload["validation_sha256"]
    assert payload["train_bytes"] > 0 and payload["validation_bytes"] > 0
    assert payload["split_rule"]
    assert payload["normalisation"], "the normalisation procedure must be recorded"
    assert payload["contamination_notes"]
    for record in payload["documents"]:
        for field in ("identifier", "title", "author", "source_url",
                      "public_domain_basis", "sha256", "n_bytes", "split"):
            assert field in record
    assert json.dumps(payload), "the manifest must be JSON-serialisable"


def test_the_manifest_reports_sizes_it_actually_measured():
    documents = [make_document(str(i), "x " * 200) for i in range(3)]
    train_text, validation_text, manifest = build_corpus(
        assign_splits(documents, validation_ids=("2",)), name="t", split_rule="r"
    )
    assert manifest.train_bytes == len(train_text.encode("utf-8"))
    assert manifest.validation_bytes == len(validation_text.encode("utf-8"))
    assert manifest.total_bytes == manifest.train_bytes + manifest.validation_bytes


def test_documents_are_joined_by_a_separator_that_looks_like_prose():
    """A distinctive delimiter would be a token the model could learn to exploit."""
    assert DOCUMENT_SEPARATOR.strip() == ""


# --- loading from disk ----------------------------------------------------
def test_documents_load_from_a_directory_of_text_files(tmp_path):
    (tmp_path / "1342.txt").write_text(GUTENBERG_FIXTURE, encoding="utf-8")
    (tmp_path / "55.txt").write_text(GUTENBERG_FIXTURE.replace("Some Book", "Other"),
                                     encoding="utf-8")

    documents = load_documents_from_directory(tmp_path, public_domain_basis="pd")

    assert [d.identifier for d in documents] == ["1342", "55"]
    assert documents[0].title == "Some Book"
    assert "PROJECT GUTENBERG" not in documents[0].text


def test_an_empty_directory_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match="no .txt files"):
        load_documents_from_directory(tmp_path)


def test_a_missing_directory_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="source directory not found"):
        load_documents_from_directory(tmp_path / "absent")


# --- run estimation -------------------------------------------------------
def test_the_run_estimate_uses_the_measured_level2_throughput():
    """Level 2 exhausted 8 MB in ~400 steps; knowing where that point falls is how
    Level 2R avoids repeating it."""
    estimate = estimate_run(50_000_000)

    assert estimate["measured_tokens_per_second"] == pytest.approx(2089.5)
    assert estimate["steps_per_epoch"] == pytest.approx(50_000_000 / 16_384, rel=1e-3)
    assert estimate["hours_per_1000_steps"] > 0


def test_a_level2_sized_corpus_is_shown_to_saturate_early():
    """8 MB is ~500 steps of data — which is what happened."""
    assert estimate_run(8_000_000)["steps_per_epoch"] < 600


# --- generation sanity ----------------------------------------------------
def test_the_level2_failure_mode_is_detected():
    """Validation BPB 1.270 coexisted with this. A loss curve will not warn you."""
    check = check_generation("The ", "and and and and and and and and and and")
    assert check.degenerate
    assert check.top_token == "and"
    assert any("Level-2 failure mode" in p for p in check.problems)


def test_a_short_repeating_cycle_is_detected():
    """No single token dominates 'the cat the cat the cat', so token counting misses it."""
    check = check_generation("The ", "the cat sat the cat sat the cat sat the cat sat "
                                     "the cat sat the cat sat the cat sat")
    assert check.degenerate
    assert check.cycle_share > 0.6


def test_empty_output_is_detected():
    assert check_generation("The ", "   \n  ").degenerate


def test_vocabulary_collapse_is_detected():
    check = check_generation("The ", "a b a b a b a b a b a b a b a b a b a b")
    assert check.degenerate


def test_ordinary_english_passes():
    check = check_generation(
        "The ", "morning was cold and the river ran quietly past the old stone bridge, "
                "where several children were playing near the water before breakfast."
    )
    assert not check.degenerate
    assert check.problems == []


def test_memorisation_is_detected_when_training_text_is_supplied():
    """Reproduction is not modelling, and BPB cannot tell the difference."""
    passage = ("It was the best of times, it was the worst of times, it was the age of "
               "wisdom, it was the age of foolishness, it was the epoch of belief.")
    check = check_generation("It was ", passage, training_text="..." + passage + "...")
    assert check.memorised
    assert check.degenerate


def test_memorisation_is_not_flagged_for_novel_text():
    check = check_generation(
        "The ", "quiet harbour town had changed a great deal since the previous winter, "
                "and the boats no longer came in before dawn as they once had done.",
        training_text="an entirely different body of training prose",
    )
    assert not check.memorised


def test_the_prompt_set_is_fixed_and_covers_the_requested_prompts():
    """Fixed across checkpoints, so generations are comparable over time."""
    for expected in ("The ", "In the beginning ", "It was ", "Once upon a time ",
                     "The most important ", "When the sun "):
        assert expected in SANITY_PROMPTS


def test_a_passing_report_does_not_claim_the_model_is_good():
    report = SanityReport(checkpoint="c", checks=[
        GenerationCheck(prompt="The ", completion="a perfectly ordinary sentence here")
    ])
    assert report.passed
    assert "does not say the" in report.render()
    assert "does NOT establish language capability" in report.to_dict()["interpretation"]


def test_a_report_with_no_checks_is_not_a_pass():
    """An empty report must never read as success."""
    assert not SanityReport(checkpoint="c").passed


def test_an_errored_report_is_not_a_pass():
    report = SanityReport(checkpoint="c", error="model would not load")
    assert not report.passed
    assert "ERROR" in report.render()
