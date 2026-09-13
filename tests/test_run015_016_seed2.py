"""Run015 / Run016: the seed-2 A3 pair may differ from seed 1 only in seed, name and output path."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
PERMITTED = {"training.seed", "name", "runtime.output_dir"}


@pytest.fixture(scope="module")
def launcher():
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import importlib

    return importlib.import_module("run015_016_A3_seed2")


@pytest.fixture(scope="module")
def built(launcher):
    matrix = launcher.load_matrix()
    return {row[2]: launcher.build(matrix, row[0], row[1], row[2], row[3]) for row in launcher.PLAN}


def test_preregistered_order_and_references(launcher):
    assert [row[2] for row in launcher.PLAN] == [
        "run015_A3_behavioural_delta_raw_seed2",
        "run016_A3_behavioural_delta_normalised_seed2",
    ]
    assert [row[3] for row in launcher.PLAN] == [
        "run013_A3_behavioural_delta_raw_seed1",
        "run013_A3_behavioural_delta_normalised_seed1",
    ]
    assert [row[1] for row in launcher.PLAN] == ["raw", "normalised"]
    assert launcher.SEED == 2


@pytest.mark.parametrize("which", ["diff_vs_built_seed1", "diff_vs_archived_seed1"])
def test_only_permitted_fields_change_vs_seed1(built, which):
    for run_id, b in built.items():
        assert set(b[which]) == PERMITTED, (run_id, b[which])
        assert b[which]["training.seed"] == [1, 2]
        assert b["clean"] is True


def test_comparison_is_substantive_not_vacuous(launcher, built):
    for row in launcher.PLAN:
        archived = json.loads((REPO / "experiments" / row[3] / "summary.json").read_text())["config"]
        live = launcher.flatten(built[row[2]]["cfg"])
        shared = set(live) & set(launcher.flatten(archived))
        assert len(shared) >= 60, len(shared)
        for key in ("training.composite_weights.hidden_delta", "training.kd_top_k",
                    "training.layer_kd_chunk_pairs", "training.learning_rate", "data.max_tokens"):
            assert key in shared, key


def test_seed2_raw_and_normalised_differ_only_in_normalisation(launcher, built):
    raw, norm = (built[row[2]]["cfg"] for row in launcher.PLAN)
    pair = launcher.diff(launcher.flatten(raw), launcher.flatten(norm))
    assert set(pair) == {"training.layer_kd_normalise", "name", "runtime.output_dir"}
    assert pair["training.layer_kd_normalise"] == [False, True]


def test_frozen_coefficients_and_envelope(built):
    for b in built.values():
        t = b["cfg"].training
        assert t.objective == "composite"
        assert dict(t.composite_weights) == {"ce": 0.1, "logit_kd": 1.0,
                                             "hidden_delta": 0.5, "router_balance": 1.0}
        assert (t.max_steps, t.batch_size, t.gradient_accumulation_steps) == (128, 1, 1)
        assert (t.learning_rate, t.optimizer, t.scheduler, t.warmup_steps) == (2e-4, "adamw", "cosine", 10)
        assert (t.lora_rank, t.lora_alpha, t.lora_dropout, t.precision) == (16, 32, 0.05, "bf16")
        assert (t.kd_temperature, t.kd_top_k, t.layer_kd_chunk_pairs) == (2.0, 64, 4)
        assert (t.eval_every, t.save_every, t.log_every) == (32, 64, 1)
        assert b["cfg"].data.max_tokens == 700000 and b["cfg"].data.max_sequence_length == 1536


def test_diff_detects_an_unpermitted_change(launcher, built):
    import copy

    b = built["run015_A3_behavioural_delta_raw_seed2"]
    tampered = copy.deepcopy(b["cfg"])
    tampered.training.learning_rate = 1e-4
    d = launcher.diff(launcher.flatten(b["cfg"]), launcher.flatten(tampered))
    assert set(d) == {"training.learning_rate"}


def test_implementation_unchanged_since_seed1():
    files = ["src/qwen_distill/training/trainer.py", "src/qwen_distill/distillation/behavioral.py",
             "scripts/kd_run.py", "src/qwen_distill/research/ablations.py",
             "src/qwen_distill/training/config.py"]
    def blob(rev, f):
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", f"{rev}:{f}"],
                              capture_output=True, text=True, check=True).stdout.strip()
    for f in files:
        assert blob("e3c525c", f) == blob("0f3ec06", f) == blob("HEAD", f), f


def test_historical_seed1_evidence_untouched():
    protected = ["experiments/run013_A3_behavioural_delta_raw_seed1",
                 "experiments/run013_A3_behavioural_delta_normalised_seed1",
                 "experiments/run014_normalized_delta_gradmatched_seed1",
                 "experiments/_session_2026-09-11/run_matrix_composite.py"]
    out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *protected],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "", out


def test_only_selects_a_single_arm_and_still_verifies_both(launcher):
    import inspect

    src = inspect.getsource(launcher.main)
    assert '"--only"' in src and "selected" in src
    # both arms are still built and verified before any selected arm trains
    assert src.index("for run_id, b in built.items()") < src.index("for position, (arm, tag, run_id, ref_id) in enumerate(selected)")
    assert "gc.collect()" in src and "torch.cuda.empty_cache()" in src and "import torch" in src
    assert "run016_A3_behavioural_delta_normalised_seed2" in launcher.EXECUTION_NOTES
