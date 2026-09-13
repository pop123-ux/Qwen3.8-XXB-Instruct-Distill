"""Focused tests for Run014, the gradient-scale-matched normalised residual-delta control.

Every property Run014's causal reading depends on is checked here without loading the 13B student
or the 27B teacher: which sequences are calibrated, how they are hashed, which parameters a
gradient norm counts, that calibration never updates a parameter, that raw and normalised start
from identical state, that the multiplier is exactly the median, that it scales the hidden
gradient exactly once, that mismatched provenance is refused, that Run014 is otherwise Run011, and
that the historical controls are untouched.
"""

from __future__ import annotations

import copy
import hashlib
import json
import statistics
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from qwen_distill.distillation.behavioral import behavioral_loss_chunked  # noqa: E402
from qwen_distill.research import gradient_calibration as gc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FREEZE = REPO / "experiments" / "run014_normalized_delta_gradmatched_seed1" / "sample_freeze.json"


# ---------------------------------------------------------------------------------------------
# tiny network mirroring the residual stream the delta objective reads
# ---------------------------------------------------------------------------------------------
class _Tiny(torch.nn.Module):
    def __init__(self, width=8, layers=4, frozen=True):
        super().__init__()
        self.blocks = torch.nn.ModuleList(torch.nn.Linear(width, width) for _ in range(layers))
        self.dropout = torch.nn.Dropout(0.5)
        self.frozen = torch.nn.Linear(width, width)
        if frozen:
            self.frozen.weight.requires_grad_(False)
            self.frozen.bias.requires_grad_(False)

    def forward(self, x):
        hidden, h = [x], x
        for block in self.blocks:
            h = h + torch.tanh(block(self.dropout(h)))
            hidden.append(h)
        return tuple(hidden)


def _fixture(seed=0):
    torch.manual_seed(seed)
    model = _Tiny()
    x = torch.randn(2, 5, 8)
    teacher = tuple(torch.randn(2, 5, 8) * (1 + i) for i in range(len(model.blocks) + 1))
    return model, x, teacher


def _measure(model, x, teacher, *, normalise, rng_seed, loss_scale=1.0, chunked=None):
    """One calibration measurement: forward, chunked delta loss, propagate, read the norm."""
    chunked = chunked or behavioral_loss_chunked
    model.zero_grad(set_to_none=True)
    torch.manual_seed(rng_seed)
    hidden = model(x)
    mapping = {i: i for i in range(len(model.blocks))}
    held = chunked(hidden, teacher, mapping, mode="delta", normalise=normalise,
                   teacher_layers=len(teacher) - 1, chunk_pairs=2, loss_scale=loss_scale)
    held.backward()
    return gc.trainable_grad_norm(model)


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


# 1 ---------------------------------------------------------------------------------------------
def test_calibration_constructs_no_optimizer_and_takes_no_step(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("calibration must not construct an optimiser or scheduler")

    monkeypatch.setattr(torch.optim, "AdamW", refuse)
    monkeypatch.setattr(torch.optim, "SGD", refuse)
    monkeypatch.setattr(torch.optim.lr_scheduler, "LambdaLR", refuse)
    model, x, teacher = _fixture()
    for normalise in (False, True):
        assert _measure(model, x, teacher, normalise=normalise, rng_seed=7) > 0
    source = (REPO / "scripts" / "run014_calibrate.py").read_text()
    for forbidden in ("build_optimizer", "optimizer.step", "make_schedule", "scheduler.step",
                      "torch.optim."):
        assert forbidden not in source, f"calibration script references {forbidden}"


# 2 ---------------------------------------------------------------------------------------------
def test_calibration_does_not_change_trainable_parameters():
    model, x, teacher = _fixture()
    before = gc.trainable_state_digest(model)
    snapshot = copy.deepcopy(model.state_dict())
    for normalise in (False, True, False, True):
        _measure(model, x, teacher, normalise=normalise, rng_seed=3)
    assert gc.trainable_state_digest(model) == before
    for name, value in model.state_dict().items():
        assert torch.equal(value, snapshot[name]), name


# 3 ---------------------------------------------------------------------------------------------
def test_gradient_norm_counts_only_trainable_parameters():
    model, x, teacher = _fixture()
    _measure(model, x, teacher, normalise=False, rng_seed=1)
    manual = sum(float(p.grad.pow(2).sum()) for p in model.parameters()
                 if p.requires_grad and p.grad is not None) ** 0.5
    # A stray gradient on a frozen parameter must not be counted.
    model.frozen.weight.grad = torch.full_like(model.frozen.weight, 1e6)
    assert gc.trainable_grad_norm(model) == pytest.approx(manual)
    assert all(not p.requires_grad for p in model.frozen.parameters())


# 4 ---------------------------------------------------------------------------------------------
def test_calibration_indices_are_frozen_and_deterministic():
    assert gc.CALIBRATION_INDICES == (0, 144, 288, 432)
    train = [[i] * 1536 for i in range(433)]
    a = gc.select_calibration_samples(train)
    b = gc.select_calibration_samples(train)
    assert [s["index"] for s in a] == [0, 144, 288, 432] == [s["index"] for s in b]
    freeze = json.loads(FREEZE.read_text())
    assert tuple(freeze["calibration_indices"]) == gc.CALIBRATION_INDICES


# 5 ---------------------------------------------------------------------------------------------
def test_token_hashes_are_deterministic_and_little_endian_uint32():
    tokens = [0, 1, 248045, 2**32 - 1]
    expected = hashlib.sha256(b"".join(t.to_bytes(4, "little") for t in tokens)).hexdigest()
    assert gc.sequence_token_sha256(tokens) == expected == gc.sequence_token_sha256(list(tokens))
    assert gc.sequence_token_sha256([1, 0]) != gc.sequence_token_sha256([0, 1])
    with pytest.raises(gc.CalibrationError):
        gc.sequence_token_sha256([2**32])
    with pytest.raises(gc.CalibrationError):
        gc.sequence_token_sha256([-1])


# 6 ---------------------------------------------------------------------------------------------
def test_calibration_samples_all_come_from_training():
    assert max(gc.CALIBRATION_INDICES) < gc.EXPECTED_N_TRAIN
    train = [[i] * 1536 for i in range(433)]
    with pytest.raises(gc.CalibrationError, match="not a training index"):
        gc.select_calibration_samples(train, indices=(0, 433))
    with pytest.raises(gc.CalibrationError, match="training split has"):
        gc.select_calibration_samples(train + [[0] * 1536])
    with pytest.raises(gc.CalibrationError, match="tokens, expected"):
        gc.select_calibration_samples([[0] * 1535] + train[1:])
    freeze = json.loads(FREEZE.read_text())
    assert freeze["n_train"] == 433 and freeze["n_validation"] == 22
    assert "TRAINING" in freeze["split"]


# 7 ---------------------------------------------------------------------------------------------
def test_packed_corpus_sha_is_validated():
    launcher = _launcher()
    ok = _preflight_inputs(launcher)
    assert launcher.preflight(**ok) > 0
    bad = dict(ok, packed_sha256="0" * 64)
    with pytest.raises(gc.CalibrationError, match="packed corpus sha"):
        launcher.preflight(**bad)
    freeze = json.loads(FREEZE.read_text())
    assert freeze["packed_corpus_sha256"] == gc.PACKED_CORPUS_SHA256


# 8 ---------------------------------------------------------------------------------------------
def test_raw_and_normalised_start_from_identical_initialisation_and_dropout():
    model, x, teacher = _fixture()
    digest = gc.trainable_state_digest(model)
    _measure(model, x, teacher, normalise=False, rng_seed=11)
    assert gc.trainable_state_digest(model) == digest
    _measure(model, x, teacher, normalise=True, rng_seed=11)
    assert gc.trainable_state_digest(model) == digest
    # Same RNG seed -> identical dropout masks -> identical forward hidden states.
    torch.manual_seed(11); h1 = model(x)
    torch.manual_seed(11); h2 = model(x)
    assert all(torch.equal(a, b) for a, b in zip(h1, h2, strict=True))
    assert gc.calibration_seed(1, 144) == gc.calibration_seed(1, 144) == 1 * 1_000_003 + 144


# 9 ---------------------------------------------------------------------------------------------
def test_multiplier_is_exactly_the_median_of_raw_over_normalised():
    raw, norm = [8.0, 3.0, 10.0, 1.0], [2.0, 1.0, 2.0, 0.5]
    ratios, m = gc.median_multiplier(raw, norm)
    assert ratios == [4.0, 3.0, 5.0, 2.0]
    assert m == statistics.median([4.0, 3.0, 5.0, 2.0]) == 3.5  # mean of 2nd and 3rd sorted
    for bad_raw, bad_norm in ([[0.0], [1.0]], [[1.0], [0.0]], [[float("nan")], [1.0]],
                              [[1.0], [float("inf")]], [[-1.0], [1.0]]):
        with pytest.raises(gc.CalibrationError):
            gc.median_multiplier(bad_raw, bad_norm)
    with pytest.raises(gc.CalibrationError):
        gc.median_multiplier([1.0, 2.0], [1.0])


# 10 --------------------------------------------------------------------------------------------
def test_multiplier_scales_the_hidden_gradient_exactly_once():
    m = 3.7
    model, x, teacher = _fixture()

    trainer_like = types.SimpleNamespace(behavioral_loss_chunked=behavioral_loss_chunked)

    # Emulate run004's delta patch, which Run014 applies first, then the multiplier patch.
    original = trainer_like.behavioral_loss_chunked

    def delta_chunked(*args, **kwargs):
        kwargs["mode"] = "delta"
        return original(*args, **kwargs)

    trainer_like.behavioral_loss_chunked = delta_chunked

    def call(scale_ctx):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(5)
        hidden = model(x)
        mapping = {i: i for i in range(len(model.blocks))}
        with scale_ctx as calls:
            held = trainer_like.behavioral_loss_chunked(
                hidden, teacher, mapping, mode="pointwise", normalise=True,
                teacher_layers=len(teacher) - 1, chunk_pairs=2, loss_scale=1.0)
        held.backward()
        return _grads(model), held.output.total, calls

    from contextlib import nullcontext

    base, base_total, _ = call(nullcontext({"n": 0}))
    scaled, scaled_total, calls = call(gc.scaled_hidden_loss(trainer_like, m))

    assert calls["n"] == 1
    assert set(base) == set(scaled)
    for name in base:
        # Relative error of the whole tensor, at float32 precision. Element-wise atol is the
        # wrong tool: near-zero elements carry large relative float32 error.
        rel = float((scaled[name] - m * base[name]).norm() / (m * base[name]).norm())
        assert rel < 1e-5, (name, rel)
        # Discriminate "exactly once" from "not at all" (1x) and "twice" (m^2), by a wide margin.
        ratio = float(scaled[name].norm() / base[name].norm())
        assert abs(ratio - m) < 1e-4, (name, ratio)
        assert abs(ratio - 1.0) > 1.0 and abs(ratio - m * m) > 1.0, (name, ratio)
    # The reported total is detached and therefore NOT scaled: the log stays comparable to Run011.
    assert scaled_total == pytest.approx(base_total)
    # The patch is removed afterwards, and a bad multiplier is refused.
    assert trainer_like.behavioral_loss_chunked is delta_chunked
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(gc.CalibrationError):
            with gc.scaled_hidden_loss(trainer_like, bad):
                pass


# 11 --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("field,value", [
    ("git_commit", "f" * 40),
    ("teacher_revision", "0" * 40),
    ("seed", 0),
    ("method_version", "run014-gradcal-0.9"),
    ("preregistration_sha256", "e" * 64),
    ("trainable_parameters", 1),
    ("calibration_indices", [0, 144, 288, 431]),
    ("packed_corpus_sha256", "0" * 64),
    ("lora_rank", 8),
])
def test_manifest_provenance_mismatch_fails_closed(field, value):
    launcher = _launcher()
    ok = _preflight_inputs(launcher)
    tampered = copy.deepcopy(ok["manifest"])
    tampered[field] = value
    with pytest.raises(gc.CalibrationError):
        launcher.preflight(**dict(ok, manifest=tampered))


def test_manifest_hash_multiplier_and_state_tampering_fail_closed():
    launcher = _launcher()
    ok = _preflight_inputs(launcher)

    wrong_hash = copy.deepcopy(ok["manifest"])
    wrong_hash["token_hashes"]["144"] = "a" * 64
    with pytest.raises(gc.CalibrationError, match="token_hashes"):
        launcher.preflight(**dict(ok, manifest=wrong_hash))

    edited_multiplier = copy.deepcopy(ok["manifest"])
    edited_multiplier["multiplier"] *= 1.5
    with pytest.raises(gc.CalibrationError, match="recomputed"):
        launcher.preflight(**dict(ok, manifest=edited_multiplier))

    mutated = copy.deepcopy(ok["manifest"])
    mutated["parameters_mutated"] = True
    with pytest.raises(gc.CalibrationError, match="parameters_mutated"):
        launcher.preflight(**dict(ok, manifest=mutated))

    incompatible = {k: v for k, v in ok["manifest"].items() if k != "method_version"}
    with pytest.raises(gc.CalibrationError, match="missing fields"):
        launcher.preflight(**dict(ok, manifest=incompatible))

    for kwargs, pattern in (
        (dict(worktree_status=" M src/x.py"), "dirty"),
        (dict(manifest=None), "missing"),
        (dict(head="1" * 40), "HEAD"),
    ):
        with pytest.raises(gc.CalibrationError, match=pattern):
            launcher.preflight(**dict(ok, **kwargs))


# 12 --------------------------------------------------------------------------------------------
def test_run014_is_otherwise_matched_to_run011():
    launcher = _launcher()
    run011 = json.loads((REPO / "experiments" / "run011_normalized_delta_seed1"
                         / "arm_manifest.json").read_text())["command"]
    argv = launcher.command()
    assert launcher._strip_artifacts(argv) == launcher._strip_artifacts(run011)
    assert "--layer-kd-no-normalise" not in argv
    diff = [(a, b) for a, b in zip(argv, run011, strict=True) if a != b]
    assert diff == [("/workspace/runs/run014_normalized_delta_gradmatched_seed1",
                     "/workspace/runs/run011_normalized_delta_seed1"),
                    ("run014_normalized_delta_gradmatched_seed1",
                     "run011_normalized_delta_seed1")]

    ok = _preflight_inputs(launcher)
    for flag, value in (("--seed", "0"), ("--revision", "0" * 40), ("--steps", "64"),
                        ("--layer-kd-chunk-pairs", "2"), ("--learning-rate", "0.0001")):
        bad = list(argv)
        bad[bad.index(flag) + 1] = value
        with pytest.raises(gc.CalibrationError):
            launcher.preflight(**dict(ok, argv=bad))
    with pytest.raises(gc.CalibrationError, match="NORMALISED"):
        launcher.preflight(**dict(ok, argv=argv + ["--layer-kd-no-normalise"]))


# 13 --------------------------------------------------------------------------------------------
def test_historical_experiment_artifacts_are_untouched():
    protected = [
        "experiments/run008_raw_delta_seed1",
        "experiments/run009_pointwise_seed1",
        "experiments/run011_normalized_delta_seed1",
        "scripts/run008_raw_delta_seed1.py",
        "experiments/_session_2026-09-11/run011_normalized_delta_seed1.py",
    ]
    for path in protected:
        assert (REPO / path).exists(), path
    changed = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--", *protected],
        capture_output=True, text=True, check=True).stdout.strip()
    assert changed == "", f"historical artifacts modified:\n{changed}"
    committed = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", *protected],
        capture_output=True)
    assert committed.returncode == 0


# ---------------------------------------------------------------------------------------------
def _launcher():
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import importlib

    return importlib.import_module("run014_normalized_delta_gradmatched_seed1")


def _preflight_inputs(launcher):
    """A fully consistent set of preflight inputs, from which each test perturbs one thing."""
    freeze = json.loads(FREEZE.read_text())
    head = "a" * 40
    prereg = "b" * 64
    raw, norm = [2.0, 4.0, 6.0, 8.0], [1.0, 1.0, 2.0, 2.0]
    ratios, m = gc.median_multiplier(raw, norm)
    manifest = launcher.expected_provenance(git_commit=head, prereg_sha256=prereg, freeze=freeze)
    manifest.update({
        "samples": [{"index": i, "raw_grad_norm": r, "normalised_grad_norm": n}
                    for i, r, n in zip(gc.CALIBRATION_INDICES, raw, norm, strict=True)],
        "ratios": ratios, "multiplier": m, "parameters_mutated": False,
    })
    run011 = json.loads((REPO / "experiments" / "run011_normalized_delta_seed1"
                         / "arm_manifest.json").read_text())["command"]
    return dict(worktree_status="", head=head, manifest=manifest, prereg_sha256=prereg,
                freeze=freeze, packed_sha256=gc.PACKED_CORPUS_SHA256,
                argv=launcher.command(), run011_argv=run011)
