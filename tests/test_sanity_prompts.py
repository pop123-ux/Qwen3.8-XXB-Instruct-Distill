"""The generation sanity check, and the record it leaves behind.

These tests never load a model or touch a GPU: ``run_sanity_checks`` imports its
generator lazily, so a stub can stand in. Level 2R is training on the only T4 available
and must not be disturbed.
"""

from __future__ import annotations

import json

import pytest

from qwen_distill.training import validate_checkpoint
from qwen_distill.training.sanity import (
    PROMPT_SET_VERSION,
    SANITY_PROMPTS,
    GenerationCheck,
    check_generation,
    run_sanity_checks,
)

#: The five prompts added in v2.0. Named here so a future edit that drops one fails.
V2_PROMPTS = (
    "The beginning of the story was",
    "It was a",
    "In the middle of the",
    "The most important thing",
    "Yesterday, I",
)

ENGLISH = (
    " a bright cold day in April, and the clocks were striking thirteen. Winston Smith, "
    "his chin nuzzled into his breast in an effort to escape the vile wind, slipped "
    "quickly through the glass doors."
)


class _StubModel:
    """Stands in for a language model. Returns fixed bytes per prompt."""

    def __init__(self, replies, *, stop_after=None):
        self.replies = replies
        self.stop_after = stop_after
        self.calls = []

    def reply(self, prompt, max_new_tokens):
        text = self.replies.get(prompt, self.replies.get("*", ENGLISH))
        ids = list(text.encode("utf-8"))[: self.stop_after or max_new_tokens]
        self.calls.append((prompt, max_new_tokens))
        return bytes(ids).decode("utf-8", errors="replace"), ids


@pytest.fixture
def stub(monkeypatch):
    def _install(replies, *, stop_after=None):
        model = _StubModel(replies, stop_after=stop_after)

        def _fake(_model, prompt, *, max_new_tokens=48, device="cpu"):
            return model.reply(prompt, max_new_tokens)

        monkeypatch.setattr(validate_checkpoint, "generate_bytes_detailed", _fake)
        return model

    return _install


# ----------------------------------------------------------------------------------
# the prompt set
# ----------------------------------------------------------------------------------


def test_the_five_new_prompts_are_present():
    for prompt in V2_PROMPTS:
        assert prompt in SANITY_PROMPTS, f"{prompt!r} is missing from the sanity set"


def test_the_original_prompts_survived():
    """The v1 prompts are how Level 2's failure was recorded. Dropping them would make
    the new reports incomparable with the published one."""
    for prompt in ("The ", "In the beginning ", "It was ", "Once upon a time "):
        assert prompt in SANITY_PROMPTS


def test_prompts_are_unique():
    assert len(set(SANITY_PROMPTS)) == len(SANITY_PROMPTS)


def test_prompt_set_is_versioned():
    assert PROMPT_SET_VERSION == "2.0"


# ----------------------------------------------------------------------------------
# what gets recorded
# ----------------------------------------------------------------------------------


def test_report_records_prompt_text_checkpoint_settings_tokens_and_time(stub):
    """The six things the record must carry to be reproducible."""
    stub({"*": ENGLISH})
    report = run_sanity_checks(
        object(), max_new_tokens=96, device="cpu",
        checkpoint="experiments/runs/x/checkpoints/step_000400", step=400,
    )
    payload = report.to_dict()

    assert payload["checkpoint"].endswith("step_000400")     # checkpoint
    assert payload["step"] == 400
    assert payload["settings"]["max_new_tokens"] == 96       # settings
    assert payload["settings"]["decoding"] == "greedy"
    assert payload["settings"]["device"] == "cpu"
    assert payload["generated_at"]                            # timestamp
    assert payload["prompt_set_version"] == PROMPT_SET_VERSION

    for check in payload["checks"]:
        assert check["prompt"] in SANITY_PROMPTS              # prompt
        assert isinstance(check["completion"], str)           # text
        assert check["n_generated_tokens"] > 0                # token count
        assert check["n_requested_tokens"] == 96
        assert check["generated_at"]
    assert payload["total_generated_tokens"] > 0


def test_token_count_comes_from_ids_not_from_characters(monkeypatch):
    """Bytes are not characters, so counting the decoded text would undercount tokens.

    Two ways they diverge, both routine for a byte-level model: a multi-byte UTF-8
    character is several tokens and one character, and a byte sequence that is not valid
    UTF-8 is replaced on decode. ``"héllo wörld"`` is 13 bytes and 11 characters.
    """
    raw = "héllo wörld".encode()
    assert len(raw) == 13 and len("héllo wörld") == 11

    def _fake(_model, prompt, *, max_new_tokens=48, device="cpu"):
        return raw.decode("utf-8", errors="replace"), list(raw)

    monkeypatch.setattr(validate_checkpoint, "generate_bytes_detailed", _fake)
    report = run_sanity_checks(object(), prompts=("The ",), max_new_tokens=16)

    check = report.checks[0]
    assert check.n_generated_tokens == 13, "token count must come from the ids"
    assert check.n_chars == 11, "the decoded text is shorter than the id sequence"


def test_stopping_early_is_recorded_not_treated_as_a_failure(stub):
    stub({"*": ENGLISH}, stop_after=12)
    report = run_sanity_checks(object(), prompts=("The ",), max_new_tokens=96)
    check = report.checks[0]
    assert check.n_generated_tokens == 12
    assert check.stopped_early
    assert "stopped early" in report.render()


def test_custom_prompts_are_not_labelled_v2(stub):
    """A pass rate over different prompts is a different measurement."""
    stub({"*": ENGLISH})
    report = run_sanity_checks(object(), prompts=("Hello ",), max_new_tokens=32)
    assert report.prompt_set_version == "custom"


def test_unchecked_memorisation_is_not_reported_as_clean(stub):
    """Without a training corpus nothing is compared, and 'memorised: no' would be a
    claim nobody made."""
    stub({"*": ENGLISH})
    report = run_sanity_checks(object(), prompts=("The ",), max_new_tokens=32)
    assert report.memorisation_checked is False
    assert "NOT CHECKED" in report.render()

    checked = run_sanity_checks(
        object(), prompts=("The ",), max_new_tokens=32, training_text="unrelated text"
    )
    assert checked.memorisation_checked is True
    assert "NOT CHECKED" not in checked.render()


def test_report_json_round_trips(stub):
    stub({"*": ENGLISH})
    report = run_sanity_checks(object(), max_new_tokens=64, checkpoint="ckpt", step=1)
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["n_prompts"] == len(SANITY_PROMPTS)


# ----------------------------------------------------------------------------------
# detection still works
# ----------------------------------------------------------------------------------


def test_level2_failure_is_caught_on_every_new_prompt(stub):
    """Level 2's exact output, against the v2 prompts."""
    stub({"*": "and and and and and and and and and and and and"})
    report = run_sanity_checks(object(), max_new_tokens=48)
    assert not report.passed
    assert report.n_degenerate == len(SANITY_PROMPTS)
    assert "DEGENERATE" in report.render()


def test_plausible_english_is_not_flagged(stub):
    stub({"*": ENGLISH})
    report = run_sanity_checks(object(), max_new_tokens=200)
    assert report.passed, [c.problems for c in report.checks if c.degenerate]
    assert "NOT OBVIOUSLY BROKEN" in report.render()
    assert "does not say the" in report.render()


def test_passing_never_claims_capability(stub):
    stub({"*": ENGLISH})
    payload = run_sanity_checks(object(), max_new_tokens=200).to_dict()
    assert "does NOT establish" in payload["interpretation"]


def test_provenance_is_recorded_but_never_scored():
    """A short generation is not thereby degenerate."""
    check = check_generation(
        "The ", ENGLISH, n_generated_tokens=4, n_requested_tokens=96,
    )
    assert check.stopped_early
    assert not check.degenerate


def test_generation_check_defaults_stamp_a_time():
    check = GenerationCheck(prompt="a", completion="b")
    assert check.generated_at is None  # only check_generation stamps it
    assert check_generation("a", ENGLISH).generated_at
