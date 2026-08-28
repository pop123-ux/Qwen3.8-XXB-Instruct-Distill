"""Tests for experiment configuration and the feasibility gate.

The gate is the point: `--dry-run` must catch a doomed configuration in seconds, on any
machine, before it consumes rented GPU time. So these tests run entirely on CPU with
synthetic VRAM figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_distill.architecture.spec import HybridArchSpec
from qwen_distill.training.config import ExperimentConfig, ModelConfig, TrainingConfig
from qwen_distill.training.data import (
    DistillationExample,
    read_jsonl,
    synthetic_corpus,
    write_jsonl,
)
from qwen_distill.training.feasibility import check_feasibility

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "experiments"


def toy_spec() -> HybridArchSpec:
    return HybridArchSpec(
        name="toy", hidden_size=256, num_hidden_layers=4, intermediate_size=704,
        vocab_size=4096, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=32,
        linear_value_head_dim=32, tie_word_embeddings=True,
    )


def base_config(**training) -> ExperimentConfig:
    config = ExperimentConfig(name="test")
    config.model = ModelConfig(architecture={"hidden_size": 256, "num_hidden_layers": 4,
                                             "intermediate_size": 704, "vocab_size": 4096,
                                             "num_attention_heads": 4, "num_key_value_heads": 2,
                                             "head_dim": 64, "linear_num_key_heads": 2,
                                             "linear_num_value_heads": 6})
    config.data.synthetic = True
    config.training = TrainingConfig(**training)
    return config


# --- shipped configs ------------------------------------------------------
def test_t4_prototype_config_is_valid():
    config = ExperimentConfig.load(CONFIG_DIR / "t4_prototype.yaml")
    assert config.training.strategy == "full"
    assert config.model.resolve_spec() is not None


def test_t4_config_uses_fp16_because_turing_has_no_bf16():
    """A bf16 config on a T4 fails at runtime; catching it here is the point."""
    config = ExperimentConfig.load(CONFIG_DIR / "t4_prototype.yaml")
    assert config.training.precision == "fp16"


def test_final_student_config_is_valid_and_marked_placeholder():
    config = ExperimentConfig.load(CONFIG_DIR / "final_student.yaml")
    assert config.training.strategy == "full", "architecture compression needs real parameters"
    assert "PLACEHOLDER" in (CONFIG_DIR / "final_student.yaml").read_text(encoding="utf-8")


def test_distillation_config_is_a_template_that_refuses_to_run_unfilled():
    """`pretrained: null` is deliberate; it must fail loudly, not silently train nothing."""
    with pytest.raises(ValueError, match="model is unset"):
        ExperimentConfig.load(CONFIG_DIR / "distillation_small.yaml")


# --- validation -----------------------------------------------------------
@pytest.mark.parametrize(
    "field,value,message",
    [("strategy", "magic", "strategy"), ("optimizer", "magic", "optimizer"),
     ("precision", "fp4", "precision"), ("objective", "telepathy", "objective"),
     ("batch_size", 0, "batch_size")],
)
def test_invalid_training_fields_are_rejected(field, value, message):
    config = base_config(**{field: value})
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_kd_on_the_synthetic_corpus_is_refused():
    """Synthetic data has no teacher distributions; a KD run on it would be a lie.

    Its tokens come from a deterministic induction rule, so "what the teacher believes
    the next token is" is not a question about anything the teacher models.
    """
    config = base_config(objective="mixed_kd")
    with pytest.raises(ValueError, match="synthetic induction corpus is meaningless"):
        config.validate()


def test_kd_over_a_text_corpus_is_allowed_with_a_resident_teacher():
    """The cheapest real distillation there is: plain text, teacher supplies the target.

    It needs no teacher-generated answers at all, so refusing it — as this validation
    once did — would have ruled out the first pilot the project can actually afford.
    """
    config = base_config(objective="logit_kd")
    config.data.synthetic = False
    config.data.text_corpus = True
    config.data.text_path = "corpus.txt"
    config.objective = {"signal_source": "online"}
    config.validate()

    config.objective = {"signal_source": "dataset"}
    with pytest.raises(ValueError, match="no stored teacher logits"):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kd_tail", "ignore", "kd_tail must be"),
        ("kd_top_k", 0, "kd_top_k must be"),
    ],
)
def test_kd_settings_are_validated(field, value, message):
    config = base_config(objective="logit_kd", **{field: value})
    config.data.synthetic = False
    config.data.text_corpus = True
    config.data.text_path = "corpus.txt"
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_spec_path_and_pretrained_are_mutually_exclusive():
    config = base_config()
    config.model = ModelConfig(spec_path="a.json", pretrained="b")
    with pytest.raises(ValueError, match="not both"):
        config.validate()


def test_effective_batch_size():
    assert TrainingConfig(batch_size=2, gradient_accumulation_steps=8).effective_batch_size == 16


def test_config_round_trips_through_dict():
    config = base_config()
    assert ExperimentConfig.from_dict(config.to_dict()).name == config.name


# --- feasibility ----------------------------------------------------------
def test_tiny_model_is_plausible_on_a_t4():
    report = check_feasibility(base_config(), toy_spec(), available_gib=15.0)
    assert report.status.startswith("PLAUSIBLE")
    assert report.fit.total_gib < 15.0


def test_teacher_full_training_is_not_feasible_on_a_t4():
    config = base_config(strategy="full", optimizer="adamw")
    config.data.max_sequence_length = 4096
    report = check_feasibility(config, HybridArchSpec(name="teacher"), available_gib=15.0)
    assert report.status == "NOT FEASIBLE"
    assert report.blockers
    assert report.fit.suggestions


def test_lora_carries_the_not_the_final_model_warning():
    """The distinction that stops LoRA being mistaken for architecture compression."""
    report = check_feasibility(
        base_config(strategy="qlora"), toy_spec(), available_gib=15.0
    )
    assert any("does NOT produce a smaller student" in w for w in report.warnings)


def test_no_spec_yields_unknown_not_a_false_pass():
    config = base_config()
    config.model = ModelConfig(pretrained="some/checkpoint")
    report = check_feasibility(config, None, available_gib=15.0)
    assert report.status == "UNKNOWN"
    assert report.blockers


def test_cpu_only_still_allows_a_toy_run():
    """Level 0 of the ladder is CPU work; a toy model must not be refused there."""
    report = check_feasibility(base_config(), toy_spec(), available_gib=None)
    assert report.status.startswith("PLAUSIBLE")
    assert any("CPU" in w for w in report.warnings)


def test_cpu_refuses_a_model_larger_than_ram():
    report = check_feasibility(
        base_config(strategy="full", optimizer="adamw"),
        HybridArchSpec(name="teacher"), available_gib=None,
    )
    assert report.status == "NOT FEASIBLE"


def test_report_renders_and_serialises():
    report = check_feasibility(base_config(), toy_spec(), available_gib=15.0)
    text = report.render()
    assert "TRAINING FEASIBILITY CHECK" in text
    assert "STATUS:" in text
    json.dumps(report.to_dict())


# --- distillation dataset --------------------------------------------------
def test_synthetic_corpus_is_deterministic():
    assert [e.to_dict() for e in synthetic_corpus(8, seed=1)] == [
        e.to_dict() for e in synthetic_corpus(8, seed=1)
    ]


def test_dataset_round_trip(tmp_path):
    path = tmp_path / "d.jsonl"
    write_jsonl(synthetic_corpus(5), path)
    assert len(list(read_jsonl(path))) == 5


def test_invalid_records_are_skipped_not_fatal(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps({"example_id": "a", "prompt": "p", "teacher_answer": "x"}) + "\n"
        "{not json\n"
        + json.dumps({"example_id": "b", "prompt": "", "teacher_answer": "x"}) + "\n",
        encoding="utf-8",
    )
    assert len(list(read_jsonl(path))) == 1


def test_invalid_records_can_be_made_fatal(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(read_jsonl(path, skip_invalid=False))


def test_unknown_fields_are_preserved_not_dropped(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps({"example_id": "a", "prompt": "p", "teacher_answer": "x",
                    "future_field": 42}) + "\n", encoding="utf-8",
    )
    example = next(read_jsonl(path))
    assert example.teacher_metadata["_unknown_fields"]["future_field"] == 42


def test_kd_targets_are_detected():
    plain = DistillationExample("a", "p", "x")
    with_logits = DistillationExample("b", "p", "x", teacher_top_logits=[[1.0, -0.5]])
    assert not plain.has_kd_targets
    assert with_logits.has_kd_targets


def test_reasoning_cost_fields_survive_a_round_trip(tmp_path):
    path = tmp_path / "d.jsonl"
    write_jsonl([DistillationExample(
        "a", "p", "x", teacher_reasoning_setting="xhigh",
        teacher_thinking_tokens=1234, teacher_total_tokens=1300, difficulty="hard",
    )], path)
    example = next(read_jsonl(path))
    assert example.teacher_thinking_tokens == 1234
    assert example.teacher_reasoning_setting == "xhigh"
    assert example.difficulty == "hard"
