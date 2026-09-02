"""The layer/intermediate KD objective — Run 003's control.

What these tests guard is the failure that would make the Run 002 / Run 003 comparison
meaningless: ``layer_kd`` quietly behaving like ``logit_kd``. So they check that the loss
reaching the optimizer is the *layer* term, that the objective refuses a teacher which
cannot supply hidden states rather than falling back, and that the run record carries the
exact definition — which representations, which mapping, which alignment, which reduction —
without which a layer-KD number cannot be interpreted or reproduced.
"""

from __future__ import annotations

import json

import pytest
from conftest import HAS_STACK

pytestmark = pytest.mark.skipif(not HAS_STACK, reason="requires torch and transformers")

#: Student and teacher share ``hidden_size``, as the real pair does (5120 both sides), so
#: the pointwise comparison needs no projection. The depths differ, which is the point.
STUDENT = {
    "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
    "vocab_size": 256, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 32, "linear_num_key_heads": 1, "linear_num_value_heads": 2,
    "linear_key_head_dim": 32, "linear_value_head_dim": 32,
    "full_attention_interval": 4, "tie_word_embeddings": True,
    "max_position_embeddings": 512,
}
TEACHER_LAYERS = 8


def make_config(output, *, objective="layer_kd", max_steps=3, **training):
    from qwen_distill.training.config import ExperimentConfig, ModelConfig

    config = ExperimentConfig(name="layer_kd")
    config.model = ModelConfig(architecture=dict(STUDENT))
    config.data.text_corpus = True
    config.data.max_sequence_length = 32
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


def make_teacher(*, hidden_size=64, capture_hidden_states=True, seed=1):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from qwen_distill.architecture.spec import HybridArchSpec
    from qwen_distill.distillation.teacher_signal import OnlineTeacher

    torch.manual_seed(seed)
    spec = HybridArchSpec(name="stand-in", **{**STUDENT, "hidden_size": hidden_size,
                                              "num_hidden_layers": TEACHER_LAYERS})
    fields = {k: v for k, v in spec.to_hf_text_config().items() if k != "model_type"}
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model("qwen3_5_text", **fields))
    return OnlineTeacher(model=model, top_k=16, temperature=1.0,
                         capture_hidden_states=capture_hidden_states,
                         teacher_model="stand-in/tiny")


def run(config, teacher):
    from qwen_distill.training.trainer import train

    return train(config, config.model.resolve_spec(), teacher=teacher)


def history(output):
    lines = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def summary(output):
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# it is a real objective, and it is not logit KD
# ---------------------------------------------------------------------------
def test_layer_kd_trains_and_records_the_layer_term(tmp_path):
    assert run(make_config(tmp_path), make_teacher()) == 0
    steps = [r for r in history(tmp_path) if r.get("status") == "completed_step"]
    assert steps
    for record in steps:
        for field in ("layer_kd_loss", "layer_magnitude", "layer_direction",
                      "layer_norm_ratio", "layer_pairs"):
            assert record.get(field) is not None, f"{field} was not recorded"
        assert record["layer_pairs"] == STUDENT["num_hidden_layers"]


def test_the_optimised_loss_is_the_layer_term_not_the_kd_divergence(tmp_path):
    """The failure this exists for: ``layer_kd`` running as logit KD under a new name.

    The recorded ``loss`` is what was handed to ``backward()``. It must equal the layer
    term and — because the two are computed from different quantities — differ from the
    KD divergence that is reported alongside it as a diagnostic.
    """
    assert run(make_config(tmp_path), make_teacher()) == 0
    steps = [r for r in history(tmp_path) if r.get("status") == "completed_step"]
    for record in steps:
        assert record["loss"] == pytest.approx(record["layer_kd_loss"], abs=1e-5)
        assert abs(record["loss"] - record["kd_loss"]) > 1e-3


def test_the_logit_diagnostics_are_still_reported_so_the_arms_are_comparable(tmp_path):
    """Run 002 is read on kd_loss, ce_loss, agreement, entropy and tail mass. Run 003 must
    report the same five or the two arms cannot be put on one axis."""
    assert run(make_config(tmp_path), make_teacher()) == 0
    steps = [r for r in history(tmp_path) if r.get("status") == "completed_step"]
    for field in ("kd_loss", "ce_loss", "top1_agreement", "teacher_entropy",
                  "teacher_tail_mass"):
        assert all(r.get(field) is not None for r in steps), f"{field} missing"


# ---------------------------------------------------------------------------
# it refuses rather than degrading
# ---------------------------------------------------------------------------
def test_a_teacher_that_returns_no_hidden_states_is_refused(tmp_path):
    with pytest.raises(ValueError, match="capture_hidden_states"):
        run(make_config(tmp_path), make_teacher(capture_hidden_states=False))


def test_a_width_mismatch_raises_rather_than_projecting_by_guesswork(tmp_path):
    """No learned projection exists, and inventing one mid-run would be an untested
    modelling choice presented as a control."""
    with pytest.raises(ValueError, match="widths must match"):
        run(make_config(tmp_path), make_teacher(hidden_size=96))


def test_layer_kd_cannot_run_off_a_stored_logit_corpus():
    from qwen_distill.distillation.objectives import LAYER_KD, ObjectiveConfig

    problems = ObjectiveConfig(type=LAYER_KD, signal_source="dataset").validate()
    assert any("hidden states" in problem for problem in problems)
    assert ObjectiveConfig(type=LAYER_KD, signal_source="online").validate() == []


# ---------------------------------------------------------------------------
# the definition is written down
# ---------------------------------------------------------------------------
def test_the_run_record_carries_the_exact_layer_kd_definition(tmp_path):
    assert run(make_config(tmp_path), make_teacher()) == 0
    definition = summary(tmp_path)["distillation"]["layer_kd_definition"]
    for field in ("teacher_representation", "student_representation", "mapping_strategy",
                  "mapping", "removed_teacher_layers", "projection", "normalisation",
                  "loss", "direction_weight", "loss_weight", "topology_mismatch"):
        assert definition.get(field) not in (None, ""), f"{field} is not documented"
    assert definition["mode"] == "pointwise"
    assert definition["n_supervised_pairs"] == STUDENT["num_hidden_layers"]
    assert len(definition["mapping"]) == STUDENT["num_hidden_layers"]


def test_the_mapping_is_derived_from_the_depths_actually_seen():
    """A mapping that assumed 64 teacher layers and got 40 would supervise the wrong pairs
    while every downstream number still looked plausible."""
    from qwen_distill.training.trainer import _layer_mapping

    mapping = _layer_mapping(48, 64, "group")
    assert len(mapping.mapping) == 48
    assert len(mapping.removed_teacher_layers) == 16
    assert mapping.problems == []
    # Never the identity: teacher layer i is not student layer i beyond the first group.
    assert mapping.mapping[47] == 63
    assert any(student != teacher for student, teacher in mapping.mapping.items())


def test_a_depth_that_is_not_whole_hybrid_groups_is_refused():
    """Six student layers cannot tile [DeltaNet x3, attention] groups, so there is no
    type-preserving correspondence. Refusing beats inventing one."""
    from qwen_distill.training.trainer import _layer_mapping

    with pytest.raises(ValueError, match="whole number of 4-layer hybrid groups"):
        _layer_mapping(6, 64, "group")


def test_the_summary_endpoints_cover_the_layer_term(tmp_path):
    assert run(make_config(tmp_path), make_teacher()) == 0
    distillation = summary(tmp_path)["distillation"]
    assert distillation["objective"] == "layer_kd"
    for key in ("layer_kd_loss", "layer_magnitude", "layer_direction", "layer_norm_ratio"):
        assert distillation[key]["first"] is not None
        assert distillation[key]["final"] is not None
