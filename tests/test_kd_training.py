"""End-to-end: does the trainer actually distil, and can it tell when it is not?

The KD path shares the loop, the checkpointing and the resume machinery with SFT, so what
these tests check is the part that is *not* shared: that a teacher signal reaches the
loss, that the loss reaching the optimizer is the KD one, and — the important one — that
a KD run with no teacher is refused rather than quietly becoming SFT.
"""

from __future__ import annotations

import json

import pytest
from conftest import HAS_STACK

pytestmark = pytest.mark.skipif(not HAS_STACK, reason="requires torch and transformers")

#: The same shape the lifecycle tests use: a real hybrid, small enough to train in seconds.
TINY = {
    "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
    "vocab_size": 256, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 32, "linear_num_key_heads": 1, "linear_num_value_heads": 2,
    "linear_key_head_dim": 32, "linear_value_head_dim": 32,
    "full_attention_interval": 4, "tie_word_embeddings": True,
    "max_position_embeddings": 512,
}


def make_config(output, *, objective="logit_kd", max_steps=4, **training):
    from qwen_distill.training.config import ExperimentConfig, ModelConfig

    config = ExperimentConfig(name="kd")
    config.model = ModelConfig(architecture=dict(TINY))
    config.data.text_corpus = True
    config.data.max_sequence_length = 64
    config.data.procedural_bytes = 20_000
    config.training.max_steps = max_steps
    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 1
    config.training.save_every = max_steps
    config.training.log_every = 1
    config.training.eval_every = max_steps
    config.training.precision = "fp32"
    config.training.strategy = "full"
    config.training.objective = objective
    config.training.kd_temperature = 1.0
    config.training.gradient_checkpointing = False
    config.objective = {"signal_source": "online"}
    config.runtime.output_dir = str(output)
    config.runtime.device = "cpu"
    for key, value in training.items():
        setattr(config.training, key, value)
    return config


def make_teacher(top_k=16, temperature=1.0, seed=1):
    """A small randomly-initialised stand-in. It teaches nothing; it *is* a distribution."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from qwen_distill.architecture.spec import HybridArchSpec
    from qwen_distill.distillation.teacher_signal import OnlineTeacher

    torch.manual_seed(seed)
    spec = HybridArchSpec(name="stand-in", **{**TINY, "hidden_size": 96, "num_hidden_layers": 8})
    fields = {k: v for k, v in spec.to_hf_text_config().items() if k != "model_type"}
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model("qwen3_5_text", **fields))
    return OnlineTeacher(model=model, top_k=top_k, temperature=temperature,
                         teacher_model="stand-in/tiny")


def run(config, teacher=None):
    from qwen_distill.training.trainer import train

    return train(config, config.model.resolve_spec(), teacher=teacher)


def summary(output):
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


def history(output):
    """Per-step records. The summary keeps endpoints; metrics.jsonl keeps every step."""
    lines = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# --- the refusal that matters ---------------------------------------------
def test_kd_without_a_teacher_is_refused_not_silently_downgraded(tmp_path):
    """The failure mode this whole module guards: a KD run that is really SFT.

    Nothing in the artifacts would show it — the loss falls, the checkpoints are valid,
    the summary says logit_kd — so it has to be impossible rather than detectable.
    """
    with pytest.raises(ValueError, match="needs a teacher signal provider"):
        run(make_config(tmp_path / "run"))


def test_sft_still_needs_no_teacher(tmp_path):
    assert run(make_config(tmp_path / "run", objective="sft")) == 0


def test_an_unknown_objective_is_still_refused(tmp_path):
    config = make_config(tmp_path / "run")
    config.training.objective = "self_play"
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        run(config)


def test_a_teacher_at_another_temperature_is_caught_before_training(tmp_path):
    """Caught at setup, not a thousand tokens in: the captured normaliser is not
    convertible to another temperature, so the two must agree from the start."""
    config = make_config(tmp_path / "run")
    config.training.kd_temperature = 2.0
    with pytest.raises(ValueError, match="same value"):
        run(config, teacher=make_teacher(temperature=1.0))


# --- it runs, and what it leaves behind ------------------------------------
def test_a_kd_run_completes_and_records_its_diagnostics(tmp_path):
    output = tmp_path / "run"
    assert run(make_config(output), teacher=make_teacher()) == 0

    steps = [record for record in history(output) if "kd_loss" in record]
    assert steps, "no step recorded KD diagnostics"
    for record in steps:
        assert record["kd_loss"] >= 0.0
        assert 0.0 <= record["top1_agreement"] <= 1.0
        assert 0.0 <= record["teacher_tail_mass"] <= 1.0
        assert record["ce_loss"] > 0.0


def test_the_summary_says_a_teacher_was_actually_reached(tmp_path):
    """A config echo is not evidence. The summary has to carry what the teacher gave.

    Without this block a KD run's artifact is indistinguishable from an SFT run's, which
    is the failure the objectives module exists to prevent, one layer further out.
    """
    output = tmp_path / "run"
    assert run(make_config(output), teacher=make_teacher(top_k=16)) == 0
    record = summary(output)

    assert record["objective"] == "logit_kd"
    block = record["distillation"]
    assert block is not None
    assert block["kd_alpha"] == 1.0
    assert block["kd_top_k"] == 64
    assert block["teacher"]["source"] == "online"
    assert block["teacher"]["teacher_model"] == "stand-in/tiny"
    assert block["n_logged_steps"] > 0
    for key in ("kd_loss", "ce_loss", "top1_agreement", "teacher_entropy", "teacher_tail_mass"):
        assert block[key]["first"] is not None, key
        assert block[key]["final"] is not None, key


def test_an_sft_run_carries_no_distillation_block(tmp_path):
    output = tmp_path / "run"
    assert run(make_config(output, objective="sft")) == 0
    assert summary(output)["distillation"] is None


def test_tail_mass_is_reported_so_top_k_can_be_chosen_from_data(tmp_path):
    """A larger k must leave less mass outside it. This is the measurement that decides
    whether an offline corpus at some k is good enough."""
    masses = {}
    for top_k in (2, 64):
        output = tmp_path / f"k{top_k}"
        assert run(make_config(output), teacher=make_teacher(top_k=top_k)) == 0
        masses[top_k] = summary(output)["distillation"]["teacher_tail_mass"]["mean"]
    assert masses[2] > masses[64]


def test_pure_kd_optimises_the_kd_term_and_mixed_kd_does_not(tmp_path):
    """`logit_kd` is alpha=1 by definition; `mixed_kd` reads kd_weight.

    Checked through the logged parts rather than by reading the config back, because the
    question is what the optimizer saw.
    """
    pure = tmp_path / "pure"
    assert run(make_config(pure, objective="logit_kd"), teacher=make_teacher()) == 0
    first = next(r for r in history(pure) if "kd_loss" in r)
    assert first["loss"] == pytest.approx(first["kd_loss"], abs=1e-3)

    mixed = tmp_path / "mixed"
    assert run(
        make_config(mixed, objective="mixed_kd", kd_weight=0.25), teacher=make_teacher()
    ) == 0
    record = next(r for r in history(mixed) if "kd_loss" in r)
    expected = 0.25 * record["kd_loss"] + 0.75 * record["ce_loss"]
    assert record["loss"] == pytest.approx(expected, abs=1e-3)


def test_mixed_kd_at_zero_weight_is_the_sft_control(tmp_path):
    """The control has to come out of the KD code path, or it is not a control."""
    output = tmp_path / "run"
    assert run(
        make_config(output, objective="mixed_kd", kd_weight=0.0), teacher=make_teacher()
    ) == 0
    record = next(r for r in history(output) if "kd_loss" in r)
    assert record["loss"] == pytest.approx(record["ce_loss"], abs=1e-3)


def test_a_kd_run_leaves_a_resumable_checkpoint(tmp_path):
    from qwen_distill.training.checkpoints import is_complete, resolve_checkpoint

    output = tmp_path / "run"
    assert run(make_config(output), teacher=make_teacher()) == 0
    resolved = resolve_checkpoint(output / "checkpoints", "latest")
    assert resolved is not None
    assert is_complete(resolved)


def test_the_teacher_is_left_in_the_mode_it_arrived_in(tmp_path):
    teacher = make_teacher()
    teacher.model.train()
    assert run(make_config(tmp_path / "run"), teacher=teacher) == 0
    assert teacher.model.training


def test_the_student_learns_to_agree_with_a_teacher_it_can_reach(tmp_path):
    """Distilling a student into *itself* must drive agreement up.

    A teacher the student can represent exactly is the only setting where "KD is working"
    has an unambiguous signature; against an unrelated random teacher, a flat agreement
    curve would prove nothing either way.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from qwen_distill.architecture.spec import HybridArchSpec
    from qwen_distill.distillation.teacher_signal import OnlineTeacher

    torch.manual_seed(11)
    spec = HybridArchSpec(name="twin", **TINY)
    fields = {k: v for k, v in spec.to_hf_text_config().items() if k != "model_type"}
    twin = AutoModelForCausalLM.from_config(AutoConfig.for_model("qwen3_5_text", **fields))

    output = tmp_path / "run"
    config = make_config(output, max_steps=30, log_every=1, learning_rate=3e-3)
    assert run(config, teacher=OnlineTeacher(model=twin, top_k=32, temperature=1.0)) == 0

    records = [r for r in history(output) if "kd_loss" in r]
    early = sum(r["kd_loss"] for r in records[:5]) / 5
    late = sum(r["kd_loss"] for r in records[-5:]) / 5
    assert late < early, f"KD loss did not fall: {early:.4f} -> {late:.4f}"

    # Agreement is asserted as a floor rather than a rise: against a small random teacher
    # it saturates inside the first logged step, so "it went up" is not a claim this setup
    # can support. The falling KD loss above is what carries the learning claim; this
    # guards the case where the loss falls while the student's argmax wanders off.
    agreement = [r["top1_agreement"] for r in records]
    assert agreement[-1] >= agreement[0]
    assert agreement[-1] > 0.9
