"""Regression tests for the developer chain self-test — ``scripts/chain_selftest.py``.

These drive a small *dense* student on purpose: the point is that the transfer, KD and
checkpoint machinery survives a geometry change, which is why that script keeps geometry
flags and the research pilot does not. The research pilot — real teacher to the canonical
frozen student — is covered in ``tests/test_distill_pilot.py``.

The Stage-0 pilot, end to end.

One property here is worth the runtime on its own: **the transferred weights must reach
training.** The trainer builds its student from the config, so a pilot that hands it an
architecture spec rebuilds a random model and silently discards the transfer — while
printing a transfer report, a cold evaluation and a falling KD loss the whole way. Every
number would look right and the experiment would be meaningless.
"""
from __future__ import annotations

import json

import pytest
from conftest import HAS_STACK
from scripts_shim import load

pytestmark = pytest.mark.skipif(not HAS_STACK, reason="requires torch and transformers")


@pytest.fixture(scope="module")
def pilot():
    return load("chain_selftest")


def arguments(output, **overrides) -> list[str]:
    args = {
        "--stand-in": None, "--layers": "8", "--hidden": "64", "--ffn": "192",
        "--kv-heads": "1", "--dn-key-heads": "2", "--steps": "2", "--batch-size": "2",
        "--seq-len": "32", "--top-k": "8", "--device": "cpu", "--output": str(output),
    }
    args.update(overrides)
    flat: list[str] = []
    for key, value in args.items():
        flat.append(key)
        if value is not None:
            flat.append(value)
    return flat


def test_a_dry_run_plans_without_loading_anything(pilot, tmp_path):
    assert pilot.main(arguments(tmp_path / "run") + ["--dry-run"]) == 0
    record = json.loads((tmp_path / "run" / "pilot_record.json").read_text(encoding="utf-8"))
    assert record["plan"]["coverage"] == 1.0
    assert "transfer" not in record          # nothing was materialised
    assert not (tmp_path / "run" / "checkpoints").exists()


def test_transfer_only_writes_a_student_carrying_teacher_weights(pilot, tmp_path):
    output = tmp_path / "run"
    assert pilot.main(arguments(output) + ["--transfer-only"]) == 0

    record = json.loads((output / "pilot_record.json").read_text(encoding="utf-8"))
    assert record["transfer"]["parameter_coverage"] == 1.0
    assert (output / "transferred" / "config.json").exists()
    assert not (output / "checkpoints").exists()


@pytest.fixture(scope="module")
def completed(pilot, tmp_path_factory):
    """One full pilot run, shared by every assertion that only reads its artifacts.

    Running it per test meant five CPU trainings for one run's worth of evidence, which is
    how a test file ends up too slow to be run.
    """
    output = tmp_path_factory.mktemp("pilot") / "run"
    assert pilot.main(arguments(output)) == 0
    return output


def test_the_transferred_weights_are_what_gets_trained(completed):
    """The regression this file exists for.

    Two independent checks, because either alone could pass while the bug is present: the
    resolved config must name the transferred checkpoint, and two steps at the default
    learning rate must leave the trained weights recognisably close to it — far closer
    than a fresh random model of the same shape would be.
    """
    import torch
    from transformers import AutoModelForCausalLM

    output = completed
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["model"]["pretrained"] == str(output / "transferred")
    assert summary["config"]["model"]["architecture"] in (None, {}, )

    from qwen_distill.training.checkpoints import load_checkpoint, resolve_checkpoint

    # `from_pretrained` works on the transferred student (it was written with
    # `save_pretrained`) but NOT on a training checkpoint, whose config.json is the
    # experiment config. Loading each the way it was written is the point.
    transferred = AutoModelForCausalLM.from_pretrained(output / "transferred")
    trained = AutoModelForCausalLM.from_config(transferred.config)
    load_checkpoint(resolve_checkpoint(output / "checkpoints", "latest"), model=trained)
    fresh = AutoModelForCausalLM.from_config(transferred.config)

    start = transferred.state_dict()
    after = trained.state_dict()
    unrelated = fresh.state_dict()
    name = "model.layers.0.mlp.up_proj.weight"
    moved = (after[name] - start[name]).abs().mean()
    apart = (unrelated[name] - start[name]).abs().mean()
    assert moved < apart / 10, f"trained weights are {moved:.3e} from the transfer, random is {apart:.3e}"
    assert torch.isfinite(after[name]).all()


def test_the_record_carries_the_cold_measurement_and_its_null_result(completed):
    """Against a random teacher the transfer must be worth ~nothing. A gain would mean
    the measurement is wrong, not that the transfer is good."""
    record = json.loads((completed / "pilot_record.json").read_text(encoding="utf-8"))

    cold = record["cold_evaluation"]
    assert set(cold) == {"transferred_nats", "random_init_nats", "delta"}
    assert abs(cold["delta"]) < 0.5, "a random teacher should not produce a real transfer gain"
    assert "warm_evaluation" in record
    assert record["training_exit_code"] == 0


def test_the_run_is_recorded_as_distillation_not_as_the_corpus_type(completed):
    summary = json.loads((completed / "summary.json").read_text(encoding="utf-8"))
    assert summary["objective"] == "logit_kd"
    assert summary["distillation"]["teacher"]["source"] == "online"


def test_a_vocabulary_the_corpus_cannot_produce_is_refused(pilot, tmp_path, capsys):
    """The real teacher's 248,320-entry vocabulary against a byte-level corpus. Stopping
    beats training on mis-encoded text."""
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_text", "hidden_size": 128, "num_hidden_layers": 8,
        "intermediate_size": 256, "vocab_size": 4096, "num_attention_heads": 4,
        "num_key_value_heads": 1, "head_dim": 32, "linear_num_key_heads": 2,
        "linear_num_value_heads": 6, "linear_key_head_dim": 16,
        "linear_value_head_dim": 16, "full_attention_interval": 4,
    }), encoding="utf-8")

    code = pilot.main([
        "--teacher", str(teacher), "--layers", "8", "--device", "cpu",
        "--output", str(tmp_path / "run"),
    ])
    assert code == 2
    assert "tokenizer" in capsys.readouterr().err


def test_a_missing_teacher_directory_is_reported(pilot, tmp_path, capsys):
    code = pilot.main(["--teacher", str(tmp_path / "nowhere"), "--output", str(tmp_path / "run")])
    assert code == 2
    assert "config.json" in capsys.readouterr().err


def test_a_training_checkpoint_is_not_a_from_pretrained_directory(completed):
    """A training checkpoint's config.json is the *experiment* config.

    It looks like a model directory — it has a config.json and a model.safetensors — and
    `from_pretrained` fails on it with a message about `model_type` that says nothing about
    why. Pinned so nothing reaches for the auto-loader here again.
    """
    from transformers import AutoModelForCausalLM

    from qwen_distill.training.checkpoints import resolve_checkpoint

    checkpoint = resolve_checkpoint(completed / "checkpoints", "latest")
    assert (checkpoint / "config.json").exists()
    assert (checkpoint / "model.safetensors").exists()
    with pytest.raises(ValueError, match="model_type"):
        AutoModelForCausalLM.from_pretrained(checkpoint)


# --- the real-teacher path -------------------------------------------------
@pytest.fixture(scope="module")
def tiny_teacher(tmp_path_factory):
    """A small real qwen3_5 checkpoint at the byte vocabulary the corpus emits.

    Vocab 256 so the pilot's tokenizer check passes and the whole chain runs; the
    tokenizer itself is smaller, which is the normal padded-embedding case.
    """
    import torch
    from test_real_teacher import TINY, build_tokenizer
    from transformers import AutoConfig, AutoModelForCausalLM

    from qwen_distill.architecture.spec import HybridArchSpec

    directory = tmp_path_factory.mktemp("real_teacher")
    build_tokenizer(directory)
    spec = HybridArchSpec(name="tiny-teacher", vocab_size=256, **TINY)
    fields = {k: v for k, v in spec.to_hf_text_config().items() if k != "model_type"}
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(
        AutoConfig.for_model("qwen3_5_text", **fields)
    ).save_pretrained(directory)
    return directory


def real_teacher_args(teacher, output, **overrides) -> list[str]:
    args = {
        "--teacher": str(teacher), "--layers": "4", "--kv-heads": "1",
        "--dn-key-heads": "1", "--steps": "1", "--batch-size": "2", "--seq-len": "32",
        "--top-k": "8", "--device": "cpu", "--teacher-dtype": "float32",
        "--precision": "fp32", "--output": str(output), "--revision": "abc123",
    }
    args.update(overrides)
    flat: list[str] = []
    for key, value in args.items():
        if value is None:      # an override of None drops the flag entirely
            continue
        flat.extend([key, value])
    return flat


def test_a_teacher_that_did_not_load_never_reaches_the_kd_loss(pilot, tiny_teacher, tmp_path):
    """The gate must hold on the pilot path, not only in the smoke test.

    This is the path that would otherwise distil against random weights: transformers
    returns a freshly-initialised model rather than raising, and the resulting KD loss is
    finite, falls, and means nothing.
    """
    from safetensors.torch import load_file, save_file

    broken = tmp_path / "broken"
    broken.mkdir()
    for item in tiny_teacher.iterdir():
        if item.is_file():
            (broken / item.name).write_bytes(item.read_bytes())
    weights = broken / "model.safetensors"
    save_file({f"garbage.{k}": v for k, v in load_file(weights).items()}, weights)

    assert pilot.main(real_teacher_args(broken, tmp_path / "run")) == 2
    assert not (tmp_path / "run" / "checkpoints").exists()


def test_the_real_teacher_chain_runs_and_records_provenance(pilot, tiny_teacher, tmp_path):
    """teacher -> transfer -> TeacherSignal -> KD loss -> one optimizer step -> checkpoint."""
    output = tmp_path / "run"
    assert pilot.main(real_teacher_args(tiny_teacher, output)) == 0

    record = json.loads((output / "pilot_record.json").read_text(encoding="utf-8"))
    provenance = record["teacher_provenance"]
    assert provenance["is_synthetic"] is False
    assert provenance["load_report"]["weights_complete"] is True
    assert provenance["identity"]["revision"] == "abc123"
    assert provenance["identity"]["is_pinned"] is True
    assert provenance["identity"]["config_sha256"]

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["objective"] == "logit_kd"
    block = summary["distillation"]
    assert block["teacher"]["source"] == "online"
    assert block["teacher"]["teacher_revision"] == "abc123"
    assert block["kd_loss"]["final"] is not None
    assert block["n_logged_steps"] >= 1

    from qwen_distill.training.checkpoints import is_complete, resolve_checkpoint

    checkpoint = resolve_checkpoint(output / "checkpoints", "latest")
    assert checkpoint is not None and is_complete(checkpoint)


def test_an_unpinned_real_teacher_run_says_so(pilot, tiny_teacher, tmp_path, capsys):
    output = tmp_path / "run"
    assert pilot.main(real_teacher_args(tiny_teacher, output, **{"--revision": None})) == 0
    assert "revision unpinned" in capsys.readouterr().err
    record = json.loads((output / "pilot_record.json").read_text(encoding="utf-8"))
    assert record["teacher_provenance"]["identity"]["is_pinned"] is False
