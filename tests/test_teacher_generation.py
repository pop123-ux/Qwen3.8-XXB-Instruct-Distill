"""Tests for teacher-data generation: sharding, manifests, resume, corruption.

Teacher generation is the most expensive step this project will run — a GPU big enough
for a 27B model, producing thousands of long generations. Losing it to a terminated
instance, or silently producing a dataset that is short, duplicated or synthetic, are
the failures worth engineering against.

Two properties get the most attention here:

* **A fake teacher must never stand in for a real one.** A synthetic dataset that looks
  real would train a student, produce numbers, and be worth nothing — and the failure
  would surface weeks later as "distillation doesn't work".
* **A partially written shard is never read as a complete one.** The difference between
  "1,000 examples" and "1,000 examples so far" is the difference between a usable
  dataset and a truncated one.

All CPU. No GPU, no teacher weights, no network.
"""

from __future__ import annotations

import json

import pytest

from qwen_distill.distillation.backends import (
    MOCK_BACKEND,
    MockTeacher,
    TransformersTeacher,
    make_backend,
)
from qwen_distill.distillation.generation import (
    Prompt,
    build_example,
    generate_dataset,
    iter_records,
    prompt_set_digest,
    read_prompts,
    scan_completed_ids,
    write_prompts,
)
from qwen_distill.distillation.manifest import DatasetManifest, ShardRecord, shard_name
from qwen_distill.distillation.reasoning_modes import resolve_mode


def make_prompts(n: int, prefix: str = "p") -> list[Prompt]:
    return [
        Prompt(id=f"{prefix}-{i:04d}", prompt=f"question {i}", category="math")
        for i in range(n)
    ]


def generate(prompts, directory, **kwargs):
    kwargs.setdefault("teacher_model", "Qwen/Qwen3.8-27B")
    return generate_dataset(
        prompts, make_backend(MOCK_BACKEND), resolve_mode("xhigh"), directory, **kwargs
    )


# --- the mock must be chosen, never fallen back to ------------------------
def test_the_real_backend_raises_rather_than_falling_back_to_synthetic():
    """A synthetic dataset that looks real is the most expensive failure here."""
    backend = make_backend("transformers", model="Qwen/Qwen3.8-27B")
    assert isinstance(backend, TransformersTeacher)
    with pytest.raises(NotImplementedError, match="not wired up yet"):
        backend.generate("hello", mode=resolve_mode("xhigh"))


def test_there_is_no_default_backend():
    with pytest.raises(ValueError, match="unknown teacher backend"):
        make_backend("")
    with pytest.raises(ValueError, match="never selected implicitly"):
        make_backend("something-else")


def test_the_mock_declares_itself_synthetic():
    described = MockTeacher().describe()
    assert described["is_synthetic"] is True
    assert "MOCK" in described["warning"]
    assert "never be used as real" in described["warning"]


def test_the_real_backend_does_not_declare_itself_synthetic():
    assert TransformersTeacher(model="x").describe()["is_synthetic"] is False


def test_a_synthetic_dataset_is_marked_in_its_manifest(tmp_path):
    """So a dataset cannot be mistaken for real teacher output later."""
    manifest, _ = generate(make_prompts(3), tmp_path)
    assert any("SYNTHETIC" in note for note in manifest.notes)
    assert manifest.run_manifest["teacher"]["is_synthetic"] is True


# --- the mock behaves usefully --------------------------------------------
def test_the_mock_is_deterministic():
    a, b = MockTeacher(seed=1), MockTeacher(seed=1)
    mode = resolve_mode("xhigh")
    first, second = a.generate("same prompt", mode=mode), b.generate("same prompt", mode=mode)
    assert first.answer == second.answer
    assert first.thinking_tokens == second.thinking_tokens


def test_higher_effort_modes_produce_more_reasoning():
    """The property a reasoning-cost sweep needs to have something to exercise."""
    mock = MockTeacher()
    counts = [
        mock.generate("q", mode=resolve_mode(m)).thinking_tokens
        for m in ("thinking_disabled", "low", "medium", "xhigh")
    ]
    assert counts[0] == 0, "thinking_disabled must produce no reasoning"
    assert counts == sorted(counts)
    assert counts[-1] > counts[1]


def test_mock_token_counts_are_labelled_as_not_tokenizer_counts():
    """Whitespace counts must never be compared against real tokenizer counts."""
    response = MockTeacher().generate("a b c", mode=resolve_mode("low"))
    assert "mock" in response.token_counting_method


def test_mock_token_parts_sum_to_the_total():
    response = MockTeacher().generate("q", mode=resolve_mode("xhigh"))
    assert response.thinking_tokens + response.answer_tokens == response.total_tokens


def test_the_mock_can_simulate_failures():
    mock = MockTeacher(failure_rate=1.0)
    response = mock.generate("q", mode=resolve_mode("low"))
    assert not response.ok
    assert response.error


# --- prompt sets ----------------------------------------------------------
def test_duplicate_prompt_ids_are_rejected(tmp_path):
    """Resume is by prompt id, so a duplicate makes "is this done?" ambiguous."""
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps({"id": "a", "prompt": "one"}) + "\n"
        + json.dumps({"id": "a", "prompt": "two"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate prompt id"):
        read_prompts(path)


def test_a_prompt_without_an_id_is_rejected(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"prompt": "no id"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no id"):
        read_prompts(path)


def test_an_empty_prompt_is_rejected(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"id": "a", "prompt": "   "}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty prompt"):
        read_prompts(path)


def test_corrupt_json_names_the_line(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"id": "a", "prompt": "ok"}\n{truncated\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        read_prompts(path)


def test_prompts_round_trip(tmp_path):
    prompts = make_prompts(5)
    path = tmp_path / "prompts.jsonl"
    assert write_prompts(prompts, path) == 5
    assert [p.id for p in read_prompts(path)] == [p.id for p in prompts]


def test_the_prompt_set_digest_changes_when_a_prompt_changes():
    """So a dataset generated over a different prompt set is detectable."""
    a = make_prompts(3)
    b = make_prompts(3)
    assert prompt_set_digest(a) == prompt_set_digest(b)
    b[1].prompt = "changed"
    assert prompt_set_digest(a) != prompt_set_digest(b)


# --- sharding and manifests -----------------------------------------------
def test_records_are_split_across_shards(tmp_path):
    manifest, stats = generate(make_prompts(25), tmp_path, shard_size=10)
    assert stats.generated == 25
    assert [s.name for s in manifest.shards] == [shard_name(i) for i in range(3)]
    assert manifest.n_records == 25


def test_every_complete_shard_is_checksummed(tmp_path):
    manifest, _ = generate(make_prompts(12), tmp_path, shard_size=5)
    complete = [s for s in manifest.shards if s.complete]
    assert complete
    assert all(s.sha256 and len(s.sha256) == 64 for s in complete)
    assert manifest.verify(tmp_path)["ok"]


def test_an_edited_shard_fails_verification(tmp_path):
    """A dataset that travelled over Drive can arrive changed; the checksum catches it."""
    manifest, _ = generate(make_prompts(6), tmp_path, shard_size=10)
    shard = tmp_path / shard_name(0)
    shard.write_text(shard.read_text(encoding="utf-8") + '{"example_id": "smuggled"}\n',
                     encoding="utf-8")

    verification = manifest.verify(tmp_path)

    assert not verification["ok"]
    assert any("checksum mismatch" in p for p in verification["problems"])


def test_a_shard_on_disk_that_the_manifest_does_not_know_about_is_flagged(tmp_path):
    """Usually an interrupted write; it must not be read as data."""
    manifest, _ = generate(make_prompts(4), tmp_path, shard_size=10)
    (tmp_path / shard_name(9)).write_text('{"example_id": "stray"}\n', encoding="utf-8")

    verification = manifest.verify(tmp_path)

    assert not verification["ok"]
    assert shard_name(9) in verification["unlisted_shards"]


def test_an_incomplete_shard_is_not_counted_in_the_record_total():
    manifest = DatasetManifest()
    manifest.add_shard(ShardRecord(name=shard_name(0), n_records=100, complete=True))
    manifest.add_shard(ShardRecord(name=shard_name(1), n_records=37, complete=False))
    assert manifest.n_records == 100, "an open shard is 'so far', not a total"
    assert manifest.n_incomplete_shards == 1


def test_the_manifest_records_failures_rather_than_hiding_them(tmp_path):
    manifest, stats = generate_dataset(
        make_prompts(20), make_backend(MOCK_BACKEND, failure_rate=1.0),
        resolve_mode("low"), tmp_path, teacher_model="t",
    )
    assert stats.generated == 0
    assert stats.failed == 20
    assert manifest.n_failed == 20
    assert len(manifest.failed_ids) == 20


def test_the_manifest_survives_a_round_trip(tmp_path):
    manifest, _ = generate(make_prompts(7), tmp_path, shard_size=3)
    manifest.write(tmp_path)
    reloaded = DatasetManifest.read(tmp_path)
    assert reloaded is not None
    assert reloaded.n_records == manifest.n_records
    assert [s.name for s in reloaded.shards] == [s.name for s in manifest.shards]


def test_a_corrupt_manifest_reads_as_absent_rather_than_raising(tmp_path):
    (tmp_path / "manifest.json").write_text("{truncated", encoding="utf-8")
    assert DatasetManifest.read(tmp_path) is None


# --- resume ---------------------------------------------------------------
def test_resume_skips_work_already_on_disk(tmp_path):
    """The rented-GPU case: an instance died, and the work must not be repeated."""
    generate(make_prompts(20), tmp_path, shard_size=8, limit=10)
    done, _ = scan_completed_ids(tmp_path)
    assert len(done) == 10

    _, stats = generate(make_prompts(20), tmp_path, shard_size=8)

    assert stats.skipped_existing == 10
    assert stats.generated == 10
    ids = [e.example_id for e in iter_records(tmp_path)]
    assert len(ids) == 20
    assert len(set(ids)) == 20, "resume must not duplicate records"


def test_resume_produces_the_same_records_as_an_uninterrupted_run(tmp_path):
    interrupted = tmp_path / "interrupted"
    straight = tmp_path / "straight"
    prompts = make_prompts(15)

    generate(prompts, interrupted, shard_size=6, limit=7)
    generate(prompts, interrupted, shard_size=6)
    generate(prompts, straight, shard_size=6)

    def payload(directory):
        return sorted(
            (e.example_id, e.teacher_answer, e.teacher_thinking_tokens)
            for e in iter_records(directory)
        )

    assert payload(interrupted) == payload(straight)


def test_no_resume_regenerates_everything(tmp_path):
    generate(make_prompts(5), tmp_path)
    _, stats = generate(make_prompts(5), tmp_path, resume=False)
    assert stats.skipped_existing == 0
    assert stats.generated == 5


def test_a_partial_final_line_costs_one_record_not_the_shard(tmp_path):
    """What a process killed mid-append leaves behind."""
    generate(make_prompts(6), tmp_path, shard_size=10)
    shard = tmp_path / shard_name(0)
    with shard.open("a", encoding="utf-8") as handle:
        handle.write('{"example_id": "p-0099", "prompt": "half')

    done, counts = scan_completed_ids(tmp_path)

    assert len(done) == 6, "the truncated record is skipped, the rest survive"
    assert counts[shard_name(0)] == 6


def test_scanning_an_empty_directory_is_not_an_error(tmp_path):
    done, counts = scan_completed_ids(tmp_path / "nothing-here")
    assert done == set()
    assert counts == {}


# --- provenance in every record -------------------------------------------
def test_every_record_carries_the_provenance_needed_to_trust_it(tmp_path):
    generate(make_prompts(3), tmp_path, teacher_model="Qwen/Qwen3.8-27B",
             teacher_revision="abc123", chat_template_sha256="deadbeef",
             generation_config={"temperature": 0.0})
    example = next(iter(iter_records(tmp_path)))

    assert example.teacher_model == "Qwen/Qwen3.8-27B"
    assert example.teacher_revision == "abc123"
    assert example.chat_template_sha256 == "deadbeef"
    assert example.generation_config_sha256
    assert example.teacher_reasoning_setting == "xhigh"
    assert example.reasoning_enabled is True
    assert example.created_at
    assert example.prompt_sha256 and example.response_sha256
    assert example.hashes_match()


def test_token_counts_are_split_and_consistent(tmp_path):
    generate(make_prompts(3), tmp_path)
    for example in iter_records(tmp_path):
        assert example.teacher_thinking_tokens is not None
        assert example.teacher_answer_tokens is not None
        assert (example.teacher_thinking_tokens + example.teacher_answer_tokens
                == example.teacher_total_tokens)
        assert example.validate() == []


def test_thinking_disabled_records_that_reasoning_was_off(tmp_path):
    generate_dataset(make_prompts(2), make_backend(MOCK_BACKEND),
                     resolve_mode("thinking_disabled"), tmp_path, teacher_model="t")
    for example in iter_records(tmp_path):
        assert example.reasoning_enabled is False
        assert example.teacher_thinking_tokens == 0


def test_an_edited_record_is_detectable(tmp_path):
    """The digests exist so a file changed after generation does not pass as teacher output."""
    generate(make_prompts(2), tmp_path)
    example = next(iter(iter_records(tmp_path)))
    assert example.hashes_match()
    example.teacher_answer = "tampered"
    assert not example.hashes_match()


def test_build_example_records_the_backend_that_produced_it():
    prompt = Prompt(id="x", prompt="q")
    response = MockTeacher().generate("q", mode=resolve_mode("low"))
    example = build_example(
        prompt, response, mode=resolve_mode("low"), teacher_model="t",
        teacher_revision=None, chat_template_sha256=None,
        generation_config_sha256=None, source="test", dataset_version="1.0",
    )
    assert example.teacher_metadata["backend"] == MOCK_BACKEND
    assert "mock" in example.teacher_metadata["token_counting_method"]


# --- reading back ---------------------------------------------------------
def test_iter_records_reads_only_complete_shards_by_default(tmp_path):
    manifest, _ = generate(make_prompts(10), tmp_path, shard_size=4)
    # A shard the manifest calls incomplete, as an interrupted run would leave.
    manifest.add_shard(ShardRecord(name=shard_name(0), n_records=4, complete=False))
    manifest.write(tmp_path)

    verified = list(iter_records(tmp_path, verified_only=True))
    everything = list(iter_records(tmp_path, verified_only=False))

    assert len(verified) < len(everything)
    assert len(everything) == 10
