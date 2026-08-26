"""Offline corpus verification.

Two things this must catch, because both have nearly happened here already: a wrong
Gutenberg id producing the wrong book, and the same work landing on both sides of the
split. Neither is visible from the manifest alone — the manifest agrees with itself.
"""

from __future__ import annotations

import json
import random
import unicodedata

import pytest

from qwen_distill.training.corpus import Document, assign_splits, build_corpus
from qwen_distill.training.corpus_verify import (
    BOILERPLATE_MARKERS,
    _titles_agree,
    verify_corpus,
)

WORDS = [
    "river", "morning", "harbour", "lantern", "quiet", "stone", "garden", "letter",
    "window", "winter", "shadow", "bridge", "orchard", "cabinet", "distant", "thunder",
    "meadow", "copper", "silence", "traveller", "candle", "ribbon", "summit", "hollow",
    "tavern", "parcel", "anchor", "gallery", "whisper", "lamplight",
]

BOOKS = [
    ("1342", "Pride and Prejudice"), ("84", "Frankenstein"), ("2701", "Moby Dick"),
    ("98", "A Tale of Two Cities"), ("1250", "Anthem"),
    ("55", "The Wonderful Wizard of Oz"),
]
VALIDATION = ("1250", "55")


def _prose(seed: int, sentences: int = 400) -> str:
    """Distinct prose per document — real text does not repeat across books, and a
    fixture that does would trip the contamination check for the wrong reason."""
    rng = random.Random(seed)
    return " ".join(
        " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 18))).capitalize() + "."
        for _ in range(sentences)
    )


def _documents(*, duplicate: tuple[str, str] | None = None) -> list[Document]:
    documents = []
    for n, (identifier, title) in enumerate(BOOKS):
        documents.append(Document(
            identifier=identifier,
            text=f"{title}\n\nChapter I\n\n{_prose(n + 1)}\n\nEnd of {title}.\n",
            title=title, author=f"Author {n}",
            source_url=f"https://www.gutenberg.org/ebooks/{identifier}",
            public_domain_basis="published before 1929",
        ))
    if duplicate:
        source_id, new_id = duplicate
        original = next(d for d in documents if d.identifier == source_id)
        documents.append(Document(
            identifier=new_id, text=original.text, title=original.title,
            author=original.author, public_domain_basis=original.public_domain_basis,
        ))
    return documents


def _write_corpus(root, *, documents=None, validation_ids=VALIDATION):
    root.mkdir(parents=True, exist_ok=True)
    documents = documents if documents is not None else _documents()
    assign_splits(documents, validation_ids=validation_ids)
    train, validation, manifest = build_corpus(
        documents, name="fixture", split_rule="whole works held out by explicit id"
    )
    (root / "train.txt").write_text(train, encoding="utf-8")
    (root / "validation.txt").write_text(validation, encoding="utf-8")
    (root / "corpus_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    return root


def _levels(report, check):
    return [f.level for f in report.findings if f.check == check]


# ----------------------------------------------------------------------------------
# the clean case
# ----------------------------------------------------------------------------------


def test_a_well_prepared_corpus_passes(tmp_path):
    report = verify_corpus(_write_corpus(tmp_path / "corpus"))
    assert report.passed, [f.message for f in report.errors]
    assert report.documents_checked == len(BOOKS)
    assert report.measured["overlap"]["ok"]


def test_nothing_touches_the_network(tmp_path):
    """The check must work in an environment where the catalogue is unreachable, which
    is the environment this tooling was built in."""
    report = verify_corpus(_write_corpus(tmp_path / "corpus"))
    assert report.to_dict()["network"] == "none — every check is local"
    assert "Every check is local" in report.render()


def test_run_estimate_is_reported(tmp_path):
    report = verify_corpus(_write_corpus(tmp_path / "corpus"), tokens_per_step=16_384)
    assert report.run_estimate["steps_per_epoch"] > 0
    assert "2,089" in report.run_estimate["basis"]


# ----------------------------------------------------------------------------------
# identity — is this the book you asked for?
# ----------------------------------------------------------------------------------


def test_a_wrong_gutenberg_id_is_caught_without_the_network(tmp_path):
    """The manifest agrees with itself. Comparing its titles against what was REQUESTED
    is what makes a wrong id visible offline."""
    root = _write_corpus(tmp_path / "corpus")
    report = verify_corpus(root, expected_titles={"55": "Alice in Wonderland"})
    assert not report.passed
    assert "identity" in [f.check for f in report.errors]
    assert "The Wonderful Wizard of Oz" in report.errors[0].message


def test_matching_titles_pass(tmp_path):
    root = _write_corpus(tmp_path / "corpus")
    report = verify_corpus(root, expected_titles={i: t for i, t in BOOKS})
    assert report.passed, [f.message for f in report.errors]


def test_without_expected_titles_the_check_is_reported_as_skipped(tmp_path):
    """'no titles checked' must not read as 'titles checked and fine'."""
    report = verify_corpus(_write_corpus(tmp_path / "corpus"))
    assert "NOTE" in _levels(report, "identity")
    assert "not checked" in report.render()


def test_titles_agree_tolerates_subtitles_and_editions():
    assert _titles_agree("Ulysses", "Ulysses — Joyce")
    assert _titles_agree("Alice's Adventures in Wonderland", "Alice's Adventures in Wonderland")
    assert _titles_agree("Grimms' Fairy Tales", "Grimms Fairy Tales")
    assert not _titles_agree("Moby Dick", "Anthem — Rand")
    assert not _titles_agree("", "Anthem")


# ----------------------------------------------------------------------------------
# contamination
# ----------------------------------------------------------------------------------


def test_the_same_work_under_two_ids_across_the_split_is_an_error(tmp_path):
    """The one way a document-level split still leaks. Validation BPB would then be
    measuring memorisation and would look like learning."""
    documents = _documents(duplicate=("1250", "9999"))
    root = _write_corpus(tmp_path / "corpus", documents=documents,
                         validation_ids=VALIDATION)
    report = verify_corpus(root)
    assert not report.passed
    messages = " ".join(f.message for f in report.errors)
    assert "byte-identical" in messages or "verbatim" in messages


def test_a_duplicate_within_one_split_is_only_a_warning(tmp_path):
    """Duplicated training data is waste, not contamination."""
    documents = _documents(duplicate=("1342", "8888"))
    root = _write_corpus(tmp_path / "corpus", documents=documents)
    report = verify_corpus(root)
    assert "WARNING" in _levels(report, "duplicates")
    assert "ERROR" not in _levels(report, "duplicates")


def test_verbatim_overlap_is_detected(tmp_path):
    root = _write_corpus(tmp_path / "corpus")
    validation = (root / "validation.txt").read_text(encoding="utf-8")
    train = (root / "train.txt").read_text(encoding="utf-8")
    (root / "train.txt").write_text(train + "\n\n" + validation, encoding="utf-8")
    report = verify_corpus(root)
    assert not report.passed
    assert any(f.check == "contamination" for f in report.errors)


# ----------------------------------------------------------------------------------
# the manifest is a claim; the bytes are the evidence
# ----------------------------------------------------------------------------------


def test_a_tampered_split_fails_its_digest(tmp_path):
    root = _write_corpus(tmp_path / "corpus")
    with open(root / "train.txt", "a", encoding="utf-8") as stream:
        stream.write("an extra sentence nobody recorded.\n")
    report = verify_corpus(root)
    assert not report.passed
    assert any(f.check == "digest" for f in report.errors)


def test_a_missing_manifest_is_an_error(tmp_path):
    root = _write_corpus(tmp_path / "corpus")
    (root / "corpus_manifest.json").unlink()
    report = verify_corpus(root)
    assert not report.manifest_present
    assert not report.passed
    assert any("cannot be reproduced" in f.message for f in report.errors)


def test_a_missing_split_file_is_an_error(tmp_path):
    root = _write_corpus(tmp_path / "corpus")
    (root / "validation.txt").unlink()
    report = verify_corpus(root)
    assert not report.passed
    assert any("validation.txt is missing" in f.message for f in report.errors)


def test_an_empty_split_is_an_error(tmp_path):
    root = _write_corpus(tmp_path / "corpus")
    (root / "validation.txt").write_text("   \n\n", encoding="utf-8")
    report = verify_corpus(root)
    assert not report.passed
    assert any(f.check in {"empty", "digest"} for f in report.errors)


def test_a_missing_directory_is_reported_not_raised(tmp_path):
    report = verify_corpus(tmp_path / "nope")
    assert not report.passed
    assert any(f.check == "directory" for f in report.errors)


# ----------------------------------------------------------------------------------
# preparation hygiene
# ----------------------------------------------------------------------------------


def test_unstripped_gutenberg_boilerplate_is_caught(tmp_path):
    root = _write_corpus(tmp_path / "corpus")
    text = (root / "validation.txt").read_text(encoding="utf-8")
    (root / "validation.txt").write_text(
        "*** START OF THE PROJECT GUTENBERG EBOOK ANTHEM ***\n" + text, encoding="utf-8"
    )
    report = verify_corpus(root)
    assert not report.passed
    assert any(f.check == "boilerplate" for f in report.errors)


def test_boilerplate_markers_cover_the_obvious_forms():
    assert any("GUTENBERG" in marker.upper() for marker in BOILERPLATE_MARKERS)
    assert any("gutenberg.org" in marker for marker in BOILERPLATE_MARKERS)


def test_carriage_returns_are_flagged(tmp_path):
    root = _write_corpus(tmp_path / "corpus")
    text = (root / "train.txt").read_text(encoding="utf-8")
    (root / "train.txt").write_text(text.replace("\n", "\r\n", 5), encoding="utf-8")
    report = verify_corpus(root)
    assert "WARNING" in _levels(report, "normalisation")


def test_non_nfc_text_is_flagged(tmp_path):
    """The model sees bytes, so visually identical text in two normal forms is two
    different inputs."""
    root = _write_corpus(tmp_path / "corpus")
    text = (root / "train.txt").read_text(encoding="utf-8")
    decomposed = unicodedata.normalize("NFD", "café résumé naïve ") * 4
    assert decomposed != unicodedata.normalize("NFC", decomposed)
    (root / "train.txt").write_text(decomposed + text, encoding="utf-8")
    report = verify_corpus(root)
    assert "WARNING" in _levels(report, "normalisation")


def test_a_lopsided_split_is_flagged(tmp_path):
    root = _write_corpus(tmp_path / "corpus", validation_ids=("1250",))
    report = verify_corpus(root)
    fraction = report.measured["validation_fraction"]
    if fraction < 0.02 or fraction > 0.20:
        assert "WARNING" in _levels(report, "split")


def test_a_tiny_corpus_warns_about_the_level2_shape(tmp_path):
    report = verify_corpus(_write_corpus(tmp_path / "corpus"))
    assert any("Level 2 exhausted 8 MB" in f.message for f in report.warnings)


# ----------------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------------


def test_warnings_do_not_block_a_run(tmp_path):
    report = verify_corpus(_write_corpus(tmp_path / "corpus"))
    assert report.warnings
    assert report.passed, "warnings must not be treated as errors"


def test_report_round_trips_to_json(tmp_path):
    report = verify_corpus(_write_corpus(tmp_path / "corpus"))
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["passed"] is True
    assert payload["n_errors"] == 0
    assert payload["documents_checked"] == len(BOOKS)


def test_render_never_claims_the_documents_were_authenticated(tmp_path):
    """The strongest claim available offline is 'the file agrees with what was asked
    for', and the report must not imply more."""
    rendered = verify_corpus(_write_corpus(tmp_path / "corpus")).render()
    assert "Nothing here confirms a document is the work its id" in rendered


@pytest.mark.parametrize("check", ["digest", "contamination", "identity", "boilerplate"])
def test_every_headline_check_is_reachable(tmp_path, check):
    """Guard against a check silently never firing."""
    root = _write_corpus(tmp_path / "corpus", documents=_documents(duplicate=("1250", "9999")))
    with open(root / "train.txt", "a", encoding="utf-8") as stream:
        stream.write("*** END OF THE PROJECT GUTENBERG EBOOK ***\n")
    report = verify_corpus(root, expected_titles={"55": "Alice in Wonderland"})
    assert check in {f.check for f in report.findings}
