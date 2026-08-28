"""Tests for turning a transfer plan into real student weights.

The thing being guarded against is not a crash. A wrong reduction produces a student
that loads cleanly, trains without error, and is quietly worse than random
initialisation — so these tests check *values*, not just shapes, and the strongest of
them (``test_identity_transfer_reproduces_the_teacher_exactly``) pins the whole pipeline
against a teacher forward pass.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from qwen_distill.architecture.materialize import (  # noqa: E402
    SafetensorsSource,
    StateDictSource,
    UnsupportedReduction,
    apply_transfer_plan,
    initialise_student,
    strip_layer_prefix,
    tensor_axes,
)
from qwen_distill.architecture.spec import HybridArchSpec  # noqa: E402
from qwen_distill.architecture.transfer import build_transfer_plan  # noqa: E402

# Small enough to build repeatedly on CPU, but the teacher's *structure*: a period-4
# hybrid, GQA at 4 query heads per KV head, 3 DeltaNet value heads per key head.
SHARED = dict(
    vocab_size=256,
    head_dim=16,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    max_position_embeddings=512,
    tie_word_embeddings=True,
    num_attention_heads=8,
    num_key_value_heads=2,
    linear_num_key_heads=4,
    linear_num_value_heads=12,
)


def spec(name: str, **overrides) -> HybridArchSpec:
    fields = dict(name=name, hidden_size=96, num_hidden_layers=16, intermediate_size=256, **SHARED)
    fields.update(overrides)
    return HybridArchSpec(**fields)


def build(architecture: HybridArchSpec):
    fields = {k: v for k, v in architecture.to_hf_text_config().items() if k != "model_type"}
    return AutoModelForCausalLM.from_config(AutoConfig.for_model("qwen3_5_text", **fields))


@pytest.fixture(scope="module")
def teacher():
    torch.manual_seed(0)
    return build(spec("teacher"))


@pytest.fixture(scope="module")
def source(teacher):
    return StateDictSource(teacher.state_dict())


TEACHER_SPEC = spec("teacher")


# --- the axis table -------------------------------------------------------
def test_layer_prefix_round_trip():
    assert strip_layer_prefix("model.layers.7.mlp.up_proj.weight") == ("mlp.up_proj.weight", 7)
    assert strip_layer_prefix("model.norm.weight") == ("model.norm.weight", None)


def test_q_proj_rows_are_blocked_by_two_head_dims_not_one():
    """The layout fact that makes naive head slicing wrong.

    ``q_proj`` is viewed as ``(..., num_heads, head_dim * 2)`` and chunked, so each head
    owns ``2 * head_dim`` consecutive rows: ``[query | gate]``. A ``head_dim``-blocked
    selection would keep half the heads and pair each with its own gate.
    """
    student = spec("s", num_attention_heads=4, num_key_value_heads=1)
    rows = tensor_axes("self_attn.q_proj.weight", TEACHER_SPEC, student)[0]
    assert rows.segments[0].block == TEACHER_SPEC.head_dim * 2
    assert rows.teacher_size == 8 * 16 * 2
    assert rows.student_size == 4 * 16 * 2


def test_in_proj_qkv_is_three_segments_not_one_axis():
    """Row-slicing this matrix would drop the value segment entirely."""
    student = spec("s", linear_num_key_heads=2, linear_num_value_heads=6)
    rows = tensor_axes("linear_attn.in_proj_qkv.weight", TEACHER_SPEC, student)[0]
    assert len(rows.segments) == 3
    assert [s.role for s in rows.segments] == ["dn_key_head", "dn_key_head", "dn_value_head"]
    assert rows.teacher_size == TEACHER_SPEC.linear_conv_dim
    assert rows.student_size == student.linear_conv_dim


def test_unknown_tensor_names_are_not_guessed():
    assert tensor_axes("model.some_future_adapter.weight", TEACHER_SPEC, TEACHER_SPEC) is None


# --- correctness of the values written ------------------------------------
def test_identity_transfer_reproduces_the_teacher_exactly(teacher, source):
    """The strongest available check: same architecture in, same logits out.

    Every reduction path is a no-op here, so any error in naming, ordering, layer mapping
    or tying shows up as a non-zero logit delta.
    """
    student_spec = spec("identical")
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    student = build(student_spec)
    report = initialise_student(student, plan, TEACHER_SPEC, student_spec, source)

    assert report.parameter_coverage == 1.0
    assert not [w for w in report.warnings if "matched nothing" in w]

    ids = torch.randint(0, SHARED["vocab_size"], (2, 48))
    teacher.eval()
    student.eval()
    with torch.no_grad():
        delta = (teacher(input_ids=ids).logits - student(input_ids=ids).logits).abs().max()
    assert delta.item() == 0.0


def test_depth_only_transfer_is_a_bit_exact_copy_of_the_selected_layers(teacher, source):
    student_spec = spec("shallow", num_hidden_layers=8)
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    student = build(student_spec)
    initialise_student(student, plan, TEACHER_SPEC, student_spec, source)

    teacher_state, student_state = teacher.state_dict(), student.state_dict()
    for student_layer, teacher_layer in plan.layer_map.items():
        prefix = f"model.layers.{student_layer}."
        for name in (n for n in student_state if n.startswith(prefix)):
            origin = name.replace(prefix, f"model.layers.{teacher_layer}.", 1)
            assert torch.equal(student_state[name], teacher_state[origin]), name


def test_head_reduction_keeps_whole_gqa_groups_and_query_gate_pairs(teacher, source):
    student_spec = spec(
        "narrow_heads",
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
    )
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    student = build(student_spec)
    initialise_student(student, plan, TEACHER_SPEC, student_spec, source)
    teacher_state, student_state = teacher.state_dict(), student.state_dict()

    attention_layer = 3  # period 4, so index 3 is the full-attention slot
    origin = plan.layer_map[attention_layer]
    head_dim = SHARED["head_dim"]

    q_teacher = teacher_state[f"model.layers.{origin}.self_attn.q_proj.weight"]
    q_student = student_state[f"model.layers.{attention_layer}.self_attn.q_proj.weight"]
    # KV head 0 survives, so its four query heads do: rows 0 .. 4 * (2 * head_dim).
    assert torch.equal(q_student, q_teacher[: 4 * 2 * head_dim])
    # And the two blockings really do disagree, or the assertion above proves nothing.
    assert not torch.equal(q_teacher[head_dim : 2 * head_dim], q_teacher[2 * head_dim : 3 * head_dim])

    o_teacher = teacher_state[f"model.layers.{origin}.self_attn.o_proj.weight"]
    o_student = student_state[f"model.layers.{attention_layer}.self_attn.o_proj.weight"]
    assert torch.equal(o_student, o_teacher[:, : 4 * head_dim])

    k_teacher = teacher_state[f"model.layers.{origin}.self_attn.k_proj.weight"]
    k_student = student_state[f"model.layers.{attention_layer}.self_attn.k_proj.weight"]
    assert torch.equal(k_student, k_teacher[:head_dim])


def test_deltanet_segments_are_reduced_independently(teacher, source):
    student_spec = spec(
        "narrow_dn",
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
    )
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    student = build(student_spec)
    initialise_student(student, plan, TEACHER_SPEC, student_spec, source)
    teacher_state, student_state = teacher.state_dict(), student.state_dict()

    origin = plan.layer_map[0]  # index 0 is a DeltaNet slot
    key_dim = TEACHER_SPEC.linear_key_dim
    value_dim = TEACHER_SPEC.linear_value_dim
    kept_key = 2 * TEACHER_SPEC.linear_key_head_dim
    kept_value = 6 * TEACHER_SPEC.linear_value_head_dim

    for name in ("in_proj_qkv.weight", "conv1d.weight"):
        full = teacher_state[f"model.layers.{origin}.linear_attn.{name}"]
        expected = torch.cat(
            [
                full[:kept_key],
                full[key_dim : key_dim + kept_key],
                full[2 * key_dim : 2 * key_dim + kept_value],
            ],
            dim=0,
        )
        got = student_state[f"model.layers.0.linear_attn.{name}"]
        assert torch.equal(got, expected), name
    assert value_dim == 12 * TEACHER_SPEC.linear_value_head_dim


def test_the_transferred_student_runs_a_forward_pass(source):
    student_spec = spec("runnable", num_hidden_layers=8, hidden_size=48, intermediate_size=128)
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    student = build(student_spec)
    initialise_student(student, plan, TEACHER_SPEC, student_spec, source)
    ids = torch.randint(0, SHARED["vocab_size"], (2, 32))
    with torch.no_grad():
        output = student(input_ids=ids, labels=ids)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.loss)


# --- the reduction methods ------------------------------------------------
def test_mean_pool_handles_a_ratio_that_does_not_divide(source):
    """96 -> 60 is not an integer ratio; a reshape-and-mean could not do it at all."""
    student_spec = spec("pooled", num_hidden_layers=8, hidden_size=60, intermediate_size=160)
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    state, report = apply_transfer_plan(
        plan, TEACHER_SPEC, student_spec, source, width_reduction="mean_pool"
    )
    assert report.parameter_coverage == 1.0
    assert state["model.norm.weight"].shape == (60,)


def test_importance_differs_from_slicing_and_is_still_complete(source):
    student_spec = spec("important", num_hidden_layers=8, hidden_size=48, intermediate_size=128)
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    by_norm, norm_report = apply_transfer_plan(
        plan, TEACHER_SPEC, student_spec, source, width_reduction="importance"
    )
    by_slice, _ = apply_transfer_plan(
        plan, TEACHER_SPEC, student_spec, source, width_reduction="slice"
    )
    assert norm_report.parameter_coverage == 1.0
    assert set(by_norm) == set(by_slice)
    differing = [k for k in by_norm if not torch.equal(by_norm[k], by_slice[k])]
    assert differing, "importance selection produced exactly the slice, so it selected nothing"


def test_a_role_keeps_one_index_set_across_every_tensor_that_uses_it(source):
    """The failure that would leave shapes right and the model scrambled.

    ``down_proj`` reads the intermediate dimension that ``gate_proj``/``up_proj`` write.
    If the three chose channels independently the student would route each FFN's output
    through the wrong neurons.
    """
    student_spec = spec("consistent", num_hidden_layers=8, intermediate_size=128)
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    state, _ = apply_transfer_plan(
        plan, TEACHER_SPEC, student_spec, source, width_reduction="importance"
    )
    teacher_state = source._state_dict
    origin = plan.layer_map[0]
    up = state["model.layers.0.mlp.up_proj.weight"]
    down = state["model.layers.0.mlp.down_proj.weight"]

    teacher_up = teacher_state[f"model.layers.{origin}.mlp.up_proj.weight"]
    teacher_down = teacher_state[f"model.layers.{origin}.mlp.down_proj.weight"]
    kept = [
        i for i in range(TEACHER_SPEC.intermediate_size)
        if any(torch.equal(teacher_up[i], row) for row in up)
    ]
    assert len(kept) == student_spec.intermediate_size
    assert torch.equal(down, teacher_down[:, kept])


# --- what it refuses to do ------------------------------------------------
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"head_dim": 8}, "head_dim differs"),
        ({"vocab_size": 128}, "vocabulary differs"),
        ({"num_key_value_heads": 4}, "GQA group size differs"),
        ({"linear_num_key_heads": 6}, "value-per-key-head ratio differs"),
        ({"linear_conv_kernel_dim": 8}, "linear_conv_kernel_dim differs"),
    ],
)
def test_structural_incompatibilities_raise_rather_than_half_apply(source, overrides, message):
    student_spec = spec("incompatible", num_hidden_layers=8, **overrides)
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    with pytest.raises(UnsupportedReduction, match=message):
        apply_transfer_plan(plan, TEACHER_SPEC, student_spec, source)


@pytest.mark.parametrize(
    "overrides",
    [{"hidden_size": 192}, {"intermediate_size": 512}, {"num_attention_heads": 16, "num_key_value_heads": 4}],
)
def test_growth_is_skipped_and_counted_rather_than_invented(source, overrides):
    student_spec = spec("wider", num_hidden_layers=8, **overrides)
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    _, report = apply_transfer_plan(plan, TEACHER_SPEC, student_spec, source)
    assert report.parameter_coverage < 1.0
    assert any("cannot invent" in reason for _, reason in report.skipped)


def test_a_tied_head_is_not_written_twice(source):
    student_spec = spec("tied", num_hidden_layers=8)
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    state, report = apply_transfer_plan(plan, TEACHER_SPEC, student_spec, source)
    assert "lm_head.weight" not in state
    assert any("tied" in reason for name, reason in report.skipped if name == "lm_head.weight")


def test_missing_teacher_tensors_are_reported_not_silently_dropped():
    student_spec = spec("partial", num_hidden_layers=8)
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    torch.manual_seed(1)
    full = build(TEACHER_SPEC).state_dict()
    holes = {k: v for k, v in full.items() if "mlp.up_proj" not in k}
    _, report = apply_transfer_plan(plan, TEACHER_SPEC, student_spec, StateDictSource(holes))
    assert any("has no tensor" in reason for _, reason in report.skipped)
    assert report.parameter_coverage < 1.0


# --- reporting ------------------------------------------------------------
def test_parameter_coverage_is_not_tensor_coverage(source):
    """A transfer that misses the embedding must not report itself as nearly complete."""
    student_spec = spec("no_embed", num_hidden_layers=8)
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    torch.manual_seed(2)
    full = build(TEACHER_SPEC).state_dict()
    without_embedding = {k: v for k, v in full.items() if k != "model.embed_tokens.weight"}
    _, report = apply_transfer_plan(
        plan, TEACHER_SPEC, student_spec, StateDictSource(without_embedding)
    )
    assert report.tensor_coverage > 0.98
    assert report.parameter_coverage < report.tensor_coverage
    assert "TRANSFER REPORT" in report.render()


def test_report_is_json_serialisable(source):
    import json

    student_spec = spec("json", num_hidden_layers=8)
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    _, report = apply_transfer_plan(plan, TEACHER_SPEC, student_spec, source)
    json.dumps(report.to_dict())


# --- the on-disk source ---------------------------------------------------
def test_safetensors_source_reads_a_sharded_checkpoint(tmp_path, source):
    safetensors = pytest.importorskip("safetensors.torch")
    import json

    state = dict(source._state_dict)
    names = sorted(state)
    half = len(names) // 2
    shards = {"a.safetensors": names[:half], "b.safetensors": names[half:]}
    weight_map = {}
    for shard, keys in shards.items():
        safetensors.save_file({k: state[k].contiguous().clone() for k in keys}, tmp_path / shard)
        weight_map.update(dict.fromkeys(keys, shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )

    student_spec = spec("from_disk", num_hidden_layers=8)
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    with SafetensorsSource(tmp_path) as on_disk:
        assert on_disk.names() == set(state)
        from_disk, report = apply_transfer_plan(plan, TEACHER_SPEC, student_spec, on_disk)
    in_memory, _ = apply_transfer_plan(plan, TEACHER_SPEC, student_spec, source)

    assert report.parameter_coverage == 1.0
    assert set(from_disk) == set(in_memory)
    assert all(torch.equal(from_disk[k], in_memory[k]) for k in from_disk)


# --- multimodal checkpoints -----------------------------------------------
def test_a_text_tower_stored_under_language_model_is_still_readable(tmp_path, source):
    """The real Qwen3.8-27B layout, which would otherwise report 0% coverage.

    The checkpoint declares ``Qwen3_5ForConditionalGeneration`` and stores its text weights
    as ``model.language_model.*``; every transfer plan here is written against
    ``model.layers.*``. ``from_pretrained`` remaps between them, but reading the shards
    directly does not — so without the alias a transfer against the real teacher would find
    every tensor missing and say so honestly while producing nothing.
    """
    safetensors = pytest.importorskip("safetensors.torch")

    state = dict(source._state_dict)
    relocated = {
        ("model.language_model." + k[len("model."):] if k.startswith("model.") else k): v.contiguous().clone()
        for k, v in state.items()
    }
    # A vision tower alongside it, as the real checkpoint has.
    relocated["model.visual.blocks.0.norm1.weight"] = torch.ones(8)
    safetensors.save_file(relocated, tmp_path / "model.safetensors")

    student_spec = spec("multimodal", num_hidden_layers=8)
    plan = build_transfer_plan(TEACHER_SPEC, student_spec, layer_selection="group")
    with SafetensorsSource(tmp_path) as on_disk:
        assert on_disk.prefix_in_use() == "model.language_model."
        assert "model.layers.0.mlp.up_proj.weight" in on_disk.names()
        from_disk, report = apply_transfer_plan(plan, TEACHER_SPEC, student_spec, on_disk)

    in_memory, _ = apply_transfer_plan(plan, TEACHER_SPEC, student_spec, source)
    assert report.parameter_coverage == 1.0
    assert all(torch.equal(from_disk[k], in_memory[k]) for k in from_disk)


def test_a_plain_checkpoint_reports_no_prefix(tmp_path, source):
    safetensors = pytest.importorskip("safetensors.torch")

    safetensors.save_file(
        {k: v.contiguous().clone() for k, v in source._state_dict.items()},
        tmp_path / "model.safetensors",
    )
    with SafetensorsSource(tmp_path) as on_disk:
        assert on_disk.prefix_in_use() is None
