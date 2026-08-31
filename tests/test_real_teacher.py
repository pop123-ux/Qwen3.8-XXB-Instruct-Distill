"""Tests for the real Qwen3.8-27B teacher backend, run against a tiny real model.

Nothing here downloads the 27B teacher. What it does instead is exercise the *actual code
paths* — `transformers` loading, the vendored chat template, real token ids, real logits —
against a small model of the same architecture family and a tokenizer carrying the
teacher's own template. That covers everything except scale.

The test this file exists for is
:func:`test_a_checkpoint_that_did_not_load_is_refused`. `transformers` does not raise when
a checkpoint's keys do not match the model class: it returns a freshly-initialised model
and prints a report. A 27B teacher "loaded" that way generates fluent nonsense and distils
a student that learns nothing, and no downstream artifact would show it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from qwen_distill.architecture.spec import HybridArchSpec  # noqa: E402
from qwen_distill.distillation.backends import (  # noqa: E402
    MOCK_BACKEND,
    TRANSFORMERS_BACKEND,
    MockTeacher,
    TransformersTeacher,
    make_backend,
)
from qwen_distill.distillation.real_teacher import (  # noqa: E402
    DEFAULT_TEACHER_MODEL,
    EXPECTED_ARCHITECTURE,
    TeacherLoadError,
    TeacherLoadPlan,
    TeacherNotLoaded,
    TemplateRejectedMode,
    describe_tokenizer,
    generate_once,
    load_verified_teacher,
    mode_changes_the_prompt,
    render_prompt,
    split_generated_tokens,
    teacher_logits,
    teacher_memory_estimate,
    verify_architecture,
)
from qwen_distill.distillation.reasoning_modes import resolve_mode, sweep_modes  # noqa: E402

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "qwen38-metadata"

#: The teacher's shape, scaled down. Same period-4 hybrid, same GQA and DeltaNet ratios.
TINY = dict(
    hidden_size=64, num_hidden_layers=4, intermediate_size=128,
    num_attention_heads=6, num_key_value_heads=1, head_dim=32,
    linear_num_key_heads=1, linear_num_value_heads=3,
    linear_key_head_dim=32, linear_value_head_dim=32,
    full_attention_interval=4, tie_word_embeddings=False,
    max_position_embeddings=512,
)


def build_tokenizer(directory: Path):
    """A real fast tokenizer carrying the teacher's own chat template."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    backing = Tokenizer(models.BPE(unk_token="<unk>"))
    backing.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backing.decoder = decoders.ByteLevel()
    backing.train_from_iterator(
        ["hello world", "the quick brown fox", "reasoning about things", "answer: 42"] * 40,
        trainers.BpeTrainer(vocab_size=180, special_tokens=["<unk>"], show_progress=False),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backing, unk_token="<unk>",
        eos_token="<|im_end|>", pad_token="<|endoftext|>",
    )
    tokenizer.add_special_tokens({"additional_special_tokens": [
        "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"]})
    tokenizer.chat_template = (VENDOR / "chat_template.jinja").read_text(encoding="utf-8")
    tokenizer.save_pretrained(directory)
    return tokenizer


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A small real qwen3_5 checkpoint plus a template-carrying tokenizer, on disk."""
    directory = tmp_path_factory.mktemp("teacher")
    tokenizer = build_tokenizer(directory)
    spec = HybridArchSpec(name="tiny-teacher", vocab_size=len(tokenizer), **TINY)
    fields = {k: v for k, v in spec.to_hf_text_config().items() if k != "model_type"}
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model("qwen3_5_text", **fields))
    model.save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def loaded(checkpoint):
    plan = TeacherLoadPlan(
        model="tiny/teacher", local_path=str(checkpoint), dtype="float32", device_map="cpu"
    )
    return load_verified_teacher(plan, strict_architecture=False)


# --- the gate this module exists for --------------------------------------
def test_a_checkpoint_that_did_not_load_is_refused(checkpoint, tmp_path):
    """transformers returns a random model instead of raising. This must not pass.

    Renaming every tensor is the extreme case; the realistic one is a partial download or
    a revision whose keys moved. Both look identical from the outside: a model that
    generates fluent text and teaches nothing.
    """
    from safetensors.torch import load_file, save_file

    broken = tmp_path / "broken"
    broken.mkdir()
    for item in checkpoint.iterdir():
        if item.is_file():
            (broken / item.name).write_bytes(item.read_bytes())
    weights = broken / "model.safetensors"
    save_file({f"garbage.{k}": v for k, v in load_file(weights).items()}, weights)

    plan = TeacherLoadPlan(
        model="tiny/teacher", local_path=str(broken), dtype="float32", device_map="cpu"
    )
    with pytest.raises(TeacherLoadError, match="which means those weights are RANDOM"):
        load_verified_teacher(plan, strict_architecture=False)


def test_a_partially_present_checkpoint_is_also_refused(checkpoint, tmp_path):
    """The realistic version: an interrupted download leaves most tensors in place."""
    from safetensors.torch import load_file, save_file

    partial = tmp_path / "partial"
    partial.mkdir()
    for item in checkpoint.iterdir():
        if item.is_file():
            (partial / item.name).write_bytes(item.read_bytes())
    weights = partial / "model.safetensors"
    kept = {k: v for k, v in load_file(weights).items() if ".layers.3." not in k}
    save_file(kept, weights)

    plan = TeacherLoadPlan(
        model="tiny/teacher", local_path=str(partial), dtype="float32", device_map="cpu"
    )
    with pytest.raises(TeacherLoadError, match="missing"):
        load_verified_teacher(plan, strict_architecture=False)


def test_a_good_checkpoint_reports_complete_weights(loaded):
    assert loaded.report.weights_complete
    assert loaded.report.missing_keys == []
    assert loaded.report.mismatched_keys == []
    assert loaded.report.model_class == "Qwen3_5ForCausalLM"


# --- architecture verification --------------------------------------------
def test_the_expected_architecture_matches_the_vendored_config():
    """EXPECTED_ARCHITECTURE is a hand-written constant; this proves it is not drifting
    from the config the project actually verified."""
    config = json.loads((VENDOR / "config.json").read_text(encoding="utf-8"))
    text = config["text_config"]
    for key, expected in EXPECTED_ARCHITECTURE.items():
        assert text[key] == expected, key


def test_a_different_architecture_is_refused_by_default(loaded):
    """Every transfer plan and memory estimate here was derived from one architecture.
    Loading a different one would leave all of them silently wrong."""
    problems = verify_architecture(loaded.model.config)
    assert problems, "the tiny fixture should not match the 27B architecture"
    assert any("hidden_size" in p for p in problems)


def test_the_real_teacher_defaults_to_the_project_teacher():
    assert DEFAULT_TEACHER_MODEL == "Qwen/Qwen3.8-27B"
    assert TransformersTeacher().model == DEFAULT_TEACHER_MODEL


# --- tokenizer -------------------------------------------------------------
def test_tokenizer_facts_are_measured_not_read_from_config(loaded):
    facts = loaded.tokenizer_facts
    assert facts.vocab_size == len(loaded.tokenizer)
    assert facts.think_close_id is not None
    assert facts.exact_reasoning_split is True
    assert isinstance(facts.adds_bos, bool)


def test_the_vendored_tokenizer_config_agrees_with_what_the_teacher_needs():
    """Pins the facts the alignment of every teacher signal depends on."""
    config = json.loads((VENDOR / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert config["add_bos_token"] is False
    assert config["bos_token"] is None
    assert config["eos_token"] == "<|im_end|>"
    assert config["model_max_length"] == 262144
    decoder = config["added_tokens_decoder"]
    assert decoder["248068"]["content"] == "<think>"
    assert decoder["248069"]["content"] == "</think>"


def test_a_tokenizer_larger_than_the_embedding_is_refused(checkpoint, tmp_path):
    """Token ids would index out of range, which surfaces as a CUDA assert much later."""
    oversized = tmp_path / "oversized"
    oversized.mkdir()
    for item in checkpoint.iterdir():
        if item.is_file():
            (oversized / item.name).write_bytes(item.read_bytes())
    config = json.loads((oversized / "config.json").read_text(encoding="utf-8"))
    config["vocab_size"] = 8
    (oversized / "config.json").write_text(json.dumps(config), encoding="utf-8")

    plan = TeacherLoadPlan(
        model="tiny/teacher", local_path=str(oversized), dtype="float32", device_map="cpu"
    )
    with pytest.raises(TeacherLoadError):
        load_verified_teacher(plan, strict_architecture=False)


# --- reasoning modes -------------------------------------------------------
def test_every_reasoning_mode_renders_a_distinct_prompt(loaded):
    """A control that leaves the prompt byte-identical is a control that does nothing."""
    result = mode_changes_the_prompt(loaded)
    assert result["all_distinct"], result["collisions"]
    assert set(result["rendered"]) == {m.name for m in sweep_modes()}


def test_medium_is_the_shortest_prompt_not_a_no_op(loaded):
    """A widely repeated secondary claim says medium is a no-op. The template refutes it:
    medium injects no reasoning instruction while the default injects the long xhigh one."""
    rendered = {m.name: render_prompt(loaded, "hi", mode=m) for m in sweep_modes()}
    assert len(rendered["medium"]) < len(rendered["xhigh"])
    assert len(rendered["medium"]) < len(rendered["low"])


def test_the_default_mode_renders_identically_to_xhigh(loaded):
    assert render_prompt(loaded, "hi", mode=resolve_mode(None)) == render_prompt(
        loaded, "hi", mode=resolve_mode("xhigh")
    )


def test_a_mode_the_template_rejects_raises_instead_of_falling_back(loaded):
    """The survey backend re-renders without the controls; the teacher must not. A record
    labelled with a mode the prompt never carried is worse than a failure."""
    from qwen_distill.distillation.reasoning_modes import ReasoningMode

    bogus = ReasoningMode(
        name="high", reasoning_effort="high", enable_thinking=None, description="rejected"
    )
    with pytest.raises(TemplateRejectedMode, match="high"):
        render_prompt(loaded, "hi", mode=bogus)


# --- generation ------------------------------------------------------------
def test_generation_returns_real_ids_and_exact_token_counts(loaded):
    result = generate_once(
        loaded, "hello", mode=resolve_mode("low"), max_new_tokens=8, temperature=0.0
    )
    assert result.prompt_tokens == len(result.prompt_token_ids)
    assert result.total_generated_tokens == len(result.generated_token_ids)
    assert result.thinking_tokens + result.answer_tokens <= result.total_generated_tokens
    assert "exact" in result.token_counting_method or "no </think>" in result.token_counting_method
    assert all(isinstance(i, int) for i in result.generated_token_ids)


def test_token_counts_never_come_from_whitespace(loaded):
    """The mock counts whitespace and says so. The real teacher must never do that."""
    result = generate_once(loaded, "hello", mode=resolve_mode("low"), max_new_tokens=4)
    assert "whitespace" not in result.token_counting_method


@pytest.mark.parametrize(
    ("ids", "close_id", "thinking", "answer"),
    [
        ([1, 2, 9, 3, 4], 9, 2, 2),      # split at the tag
        ([1, 2, 3], 9, 0, 3),            # tag never emitted: nothing was reasoning
        ([9, 1, 2], 9, 0, 2),            # tag first
        ([1, 2, 9], 9, 2, 0),            # tag last, no answer
    ],
)
def test_the_split_is_exact_on_token_ids(ids, close_id, thinking, answer):
    got_thinking, got_answer, method = split_generated_tokens(torch.tensor(ids), close_id)
    assert (got_thinking, got_answer) == (thinking, answer)
    assert "exact" in method


def test_without_a_think_token_the_split_is_declared_unavailable():
    _, answer, method = split_generated_tokens(torch.tensor([1, 2, 3]), None)
    assert answer == 3
    assert "no </think> token" in method


# --- logits and the KD signal ----------------------------------------------
def test_logits_have_the_shape_kd_needs(loaded):
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    logits = teacher_logits(loaded, ids)
    assert logits.shape[0] == 1
    assert logits.shape[1] == ids.shape[1]
    assert logits.shape[2] == loaded.model.config.vocab_size
    assert torch.isfinite(logits).all()


def test_logits_are_position_aligned_with_the_input(loaded):
    """Position t must be the teacher's prediction for the same token the student predicts
    at t. A prefix of the input must therefore give the same logits at shared positions."""
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    full = teacher_logits(loaded, ids)
    prefix = teacher_logits(loaded, ids[:, :3])
    assert torch.allclose(full[:, :3], prefix, atol=1e-4)


def test_logits_require_a_batch_dimension(loaded):
    with pytest.raises(ValueError, match="batch, positions"):
        teacher_logits(loaded, torch.tensor([1, 2, 3]))


def test_the_backend_produces_a_usable_teacher_signal(loaded):
    """The end of the chain this phase exists to reach: real weights -> TeacherSignal."""
    from qwen_distill.distillation.kd_loss import kd_divergence

    backend = TransformersTeacher(model="tiny/teacher", local_path=str(loaded.plan.local_path))
    backend.loaded = loaded
    provider = backend.signal_provider(top_k=8, temperature=1.0)

    ids = torch.tensor([[1, 2, 3, 4]])
    signal = provider.signal_for(ids)
    assert signal.k == 8
    assert signal.top_values.shape == (1, 4, 8)
    assert signal.logsumexp.shape == (1, 4)
    assert signal.metadata["teacher_model"] == "tiny/teacher"

    student_logits = torch.randn(1, 4, loaded.model.config.vocab_size)
    divergence, diagnostics = kd_divergence(student_logits, signal, tail="bucket")
    assert divergence.item() >= 0
    assert 0.0 <= diagnostics["tail_mass"] <= 1.0


def test_a_signal_from_the_teacher_matches_its_own_logits(loaded):
    """Self-distillation against the teacher's own distribution must be exactly zero."""
    from qwen_distill.distillation.kd_loss import kd_divergence

    ids = torch.tensor([[1, 2, 3, 4]])
    logits = teacher_logits(loaded, ids)
    backend = TransformersTeacher(model="tiny/teacher")
    backend.loaded = loaded
    signal = backend.signal_provider(top_k=None).signal_for(ids)
    divergence, _ = kd_divergence(logits, signal)
    assert divergence.item() == pytest.approx(0.0, abs=1e-5)


# --- provenance ------------------------------------------------------------
def test_every_operation_can_be_traced_to_a_pinned_identity(loaded):
    described = loaded.describe()
    assert described["is_synthetic"] is False
    identity = described["identity"]
    assert identity["model"] == "tiny/teacher"
    assert identity["config_sha256"]
    assert identity["chat_template_sha256"]
    assert identity["tokenizer_sha256"]


def test_an_unpinned_revision_is_flagged_rather_than_accepted(loaded):
    identity = loaded.describe()["identity"]
    assert identity["is_pinned"] is False
    assert "not reproducible" in identity["pinning_note"]


PINNED = "0f9e8d7c6b5a49382716051423f6e5d4c3b2a190"


def test_a_pinned_revision_reaches_the_plan():
    plan = TeacherLoadPlan(revision=PINNED)
    assert plan.to_dict()["revision"] == PINNED
    assert plan.validate() == []


# --- the revision gate ------------------------------------------------------
def test_a_hub_load_without_a_revision_is_refused_before_anything_downloads():
    """A repo id does not name weights: the same id serves different bytes over time. The
    check lives in validate() so every caller inherits it, and validate() runs before the
    first byte is fetched."""
    problems = TeacherLoadPlan(model=DEFAULT_TEACHER_MODEL).validate()
    assert any("--revision is required" in p for p in problems), problems


@pytest.mark.parametrize("revision", ["main", "master", "latest", "HEAD", "refs/heads/main"])
def test_a_moving_pointer_is_refused(revision):
    """These look like a pin and are not one: they resolve to whatever the repository holds
    at load time, which is the substitution the gate exists to prevent."""
    problems = TeacherLoadPlan(revision=revision).validate()
    assert any("moving pointer" in p for p in problems), problems


@pytest.mark.parametrize("revision", ["v1.0", "release-2", "main~1", "zzzzzzz", "abc12"])
def test_something_that_is_not_a_commit_id_is_refused(revision):
    """Tags and branches can be moved or deleted upstream; only a commit id cannot change
    what it names. 'abc12' is five characters, below the seven-character minimum."""
    problems = TeacherLoadPlan(revision=revision).validate()
    assert any("not a commit SHA" in p for p in problems), problems


@pytest.mark.parametrize("revision", ["a1b2c3d", PINNED, PINNED.upper()])
def test_a_commit_id_is_accepted(revision):
    assert TeacherLoadPlan(revision=revision).validate() == []


def test_a_local_load_may_omit_the_revision():
    """The bytes on disk are the pin. Requiring a SHA here would block every fixture."""
    plan = TeacherLoadPlan(local_path="/checkpoints/qwen", revision=None)
    assert plan.is_hub_load is False
    assert plan.validate() == []


def test_a_local_load_still_records_a_revision_when_one_is_given():
    """A directory does not say which upstream commit produced it, so the SHA is worth
    carrying even when it is not required."""
    plan = TeacherLoadPlan(local_path="/checkpoints/qwen", revision=PINNED)
    assert plan.validate() == []
    assert plan.to_dict()["revision"] == PINNED


def test_the_gate_does_not_weaken_the_missing_weights_gate():
    """Two independent safety properties; adding the revision check must not have replaced
    the one that rejects a checkpoint whose tensors did not load."""
    problems = TeacherLoadPlan(model="", revision=PINNED).validate()
    assert any("non-empty" in p for p in problems)


def test_the_load_report_records_what_was_discarded(loaded):
    """The vision tower and MTP head are discarded by design; that must be visible rather
    than silent, so a checkpoint that starts shipping something else is noticed."""
    described = loaded.describe()["load_report"]
    assert described["weights_complete"] is True
    assert "n_ignored_unexpected" in described


# --- the plan and its refusals ---------------------------------------------
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": ""}, "non-empty"),
        ({"quantization": "3bit"}, "unknown quantization"),
        ({"quantization": "4bit", "device_map": None}, "requires a device_map"),
        ({"offload_folder": "offload", "device_map": None}, "no effect without"),
    ],
)
def test_an_impossible_load_plan_is_rejected_before_anything_downloads(kwargs, message):
    problems = TeacherLoadPlan(**kwargs).validate()
    assert any(message in p for p in problems), problems


def test_memory_estimates_come_from_the_projects_own_model():
    """The teacher and the students must be sized by the same estimator."""
    bf16 = teacher_memory_estimate(4096, None)
    four_bit = teacher_memory_estimate(4096, "4bit")
    assert bf16["weights_gib"] > four_bit["weights_gib"] * 3
    assert four_bit["total_gib"] > 13.56, "4-bit teacher must not appear to fit a 16 GB card"


# --- the mock is still unreachable by accident -----------------------------
def test_the_mock_is_never_selected_implicitly():
    assert isinstance(make_backend(MOCK_BACKEND), MockTeacher)
    assert isinstance(make_backend(TRANSFORMERS_BACKEND), TransformersTeacher)
    with pytest.raises(ValueError, match="never selected implicitly"):
        make_backend("something-else")


def test_the_real_backend_refuses_to_work_unloaded():
    backend = make_backend(TRANSFORMERS_BACKEND)
    for call in (
        lambda: backend.generate("hi", mode=resolve_mode("low")),
        lambda: backend.logits(torch.tensor([[1, 2]])),
        lambda: backend.signal_provider(),
    ):
        with pytest.raises(TeacherNotLoaded):
            call()


def test_describe_before_loading_says_so_rather_than_inventing_hashes():
    described = make_backend(TRANSFORMERS_BACKEND).describe()
    assert described["loaded"] is False
    assert described["is_synthetic"] is False
    assert "not loaded yet" in described["note"]


def test_unloading_releases_the_model_and_blocks_further_use(checkpoint):
    plan = TeacherLoadPlan(
        model="tiny/teacher", local_path=str(checkpoint), dtype="float32", device_map="cpu"
    )
    backend = TransformersTeacher(model="tiny/teacher", local_path=str(checkpoint),
                                  dtype="float32", device="cpu", strict_architecture=False)
    backend.load()
    assert backend.loaded is not None
    backend.unload()
    assert backend.loaded is None
    with pytest.raises(TeacherNotLoaded):
        backend.logits(torch.tensor([[1, 2]]))
    assert plan.source == str(checkpoint)


def test_describe_after_loading_carries_the_real_identity(checkpoint):
    backend = TransformersTeacher(model="tiny/teacher", local_path=str(checkpoint),
                                  dtype="float32", device="cpu", strict_architecture=False)
    backend.load()
    try:
        described = backend.describe()
        assert described["backend"] == TRANSFORMERS_BACKEND
        assert described["is_synthetic"] is False
        assert described["load_report"]["weights_complete"] is True
        assert described["tokenizer"]["exact_reasoning_split"] is True
        assert described["generation"]["max_new_tokens"] == 2048
    finally:
        backend.unload()


def test_the_tokenizer_probe_reports_facts_for_any_tokenizer(loaded):
    facts = describe_tokenizer(loaded.tokenizer)
    assert facts.tokenizer_class
    assert facts.vocab_size > 0


def test_the_hybrid_period_is_derived_when_a_config_omits_it():
    """A config that stores layer_types explicitly need not keep the interval that
    generated it. Deriving it stops a correct checkpoint failing verification."""
    from qwen_distill.distillation.real_teacher import _interval_from_layer_types

    period4 = ["linear_attention"] * 3 + ["full_attention"]
    assert _interval_from_layer_types(period4 * 4) == 4
    assert _interval_from_layer_types(["full_attention"] * 4) == 1
    assert _interval_from_layer_types([]) is None
    assert _interval_from_layer_types(["linear_attention"] * 4) is None
    # Not periodic: refuses to report an interval rather than guessing one.
    assert _interval_from_layer_types(
        ["linear_attention", "full_attention", "linear_attention", "linear_attention"]
    ) is None


def test_the_real_vendored_config_passes_strict_verification():
    """The check that matters: loading the actual teacher must not fail spuriously."""
    from transformers import AutoConfig

    assert verify_architecture(AutoConfig.from_pretrained(VENDOR)) == []


def test_memory_figures_declare_themselves_unmeasured():
    """They are arithmetic from the analytical model, not observations of a real load.

    Presenting an estimate as a hardware requirement is how a project ends up sizing a
    student around a number nobody checked.
    """
    estimate = teacher_memory_estimate(4096, "4bit")
    assert estimate["measured"] is False
    assert "NOT measured" in estimate["basis"]
    for component in ("weights_gib", "kv_cache_gib", "recurrent_state_gib",
                      "activations_gib", "runtime_overhead_gib"):
        assert component in estimate, component
    total = sum(estimate[c] for c in ("weights_gib", "kv_cache_gib", "recurrent_state_gib",
                                      "activations_gib", "runtime_overhead_gib"))
    assert total == pytest.approx(estimate["total_gib"], abs=0.02)


def test_the_teachers_footprint_is_not_the_students_budget():
    """The two are separate budgets, and the estimate says so where it will be read."""
    assert "does not constrain the 16 GB student" in (
        teacher_memory_estimate(4096)["student_objective_note"]
    )


def test_the_kv_cache_estimate_grows_with_context():
    """The component that actually scales; the weights row does not."""
    short = teacher_memory_estimate(4096, "4bit")
    long = teacher_memory_estimate(65536, "4bit")
    assert long["kv_cache_gib"] > short["kv_cache_gib"] * 8
    assert long["weights_gib"] == short["weights_gib"]


def test_a_hub_load_without_a_revision_is_refused(capsys):
    """An unpinned hub load cannot be reproduced, and its tail-mass numbers could not be
    attributed to a specific checkpoint later. A local fixture needs no SHA."""
    from scripts_shim import load

    smoke = load("teacher_smoke_test")
    assert smoke.main(["--quantization", "4bit"]) == 2
    assert "--revision is required" in capsys.readouterr().err


def test_the_documented_smoke_test_flags_all_exist():
    """Guards the exact command in docs/REAL_TEACHER_RUN.md against CLI drift."""
    from scripts_shim import load

    parser_args = load("teacher_smoke_test").parse_args(
        ["--quantization", "4bit", "--revision", "abc123", "--json", "runs/teacher_smoke.json"]
    )
    assert parser_args.quantization == "4bit"
    assert parser_args.revision == "abc123"
    assert str(parser_args.json) == "runs/teacher_smoke.json"
    assert parser_args.model == DEFAULT_TEACHER_MODEL
