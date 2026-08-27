"""The Stage-0 pilot, end to end.

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
    return load("distill_pilot")


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

    from qwen_distill.training.checkpoints import resolve_checkpoint

    transferred = AutoModelForCausalLM.from_pretrained(output / "transferred")
    trained = AutoModelForCausalLM.from_pretrained(
        resolve_checkpoint(output / "checkpoints", "latest")
    )
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
