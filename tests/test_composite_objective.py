"""Gradient-level tests for the preregistered composite objective.

These exist because the composite path assembles gradients from two sources -- the
chunked hidden term, which backwards internally onto detached stand-ins, and a graph
tensor carrying CE / logit KD / router balance. The risks are double-backward, a weight
applied twice or not at all, and a term that silently contributes nothing. Each is
checked directly on parameter gradients rather than on reported scalars.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from qwen_distill.distillation.behavioral import (  # noqa: E402
    CE, HIDDEN_DELTA, HIDDEN_POINTWISE, LOGIT_KD, ROUTER_BALANCE,
)
from qwen_distill.training.config import OBJECTIVES  # noqa: E402


def test_composite_is_a_registered_objective():
    assert "composite" in OBJECTIVES


# --------------------------------------------------------------------------------------
# A miniature stand-in for the trainer's composite step. It exercises the SAME assembly
# rules the trainer uses: the hidden weight rides in loss_scale, CE/KD/router ride on a
# graph tensor, and one torch.autograd.backward seeds both.
# --------------------------------------------------------------------------------------
from qwen_distill.distillation.behavioral import behavioral_loss_chunked  # noqa: E402


class _Tiny(torch.nn.Module):
    """Two 'layers' over a hidden stream, plus a head. Small enough to be exact."""

    def __init__(self, width=8, layers=4):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            torch.nn.Linear(width, width) for _ in range(layers)
        )
        self.head = torch.nn.Linear(width, width)
        self.router = torch.nn.Linear(width, 4)

    def forward(self, x):
        hidden = [x]
        h = x
        for block in self.blocks:
            h = h + torch.tanh(block(h))
            hidden.append(h)
        return self.head(h), tuple(hidden), self.router(h).float().pow(2).mean()


def _assemble(model, x, teacher_hidden, weights, *, accum=1, chunk_pairs=2):
    """Mirror of trainer._composite_step's gradient wiring."""
    logits, hidden, aux = model(x)
    mapping = {i: i for i in range(len(model.blocks))}

    layer_backward = None
    if HIDDEN_DELTA in weights or HIDDEN_POINTWISE in weights:
        term = HIDDEN_DELTA if HIDDEN_DELTA in weights else HIDDEN_POINTWISE
        mode = "delta" if term == HIDDEN_DELTA else "pointwise"
        layer_backward = behavioral_loss_chunked(
            hidden, teacher_hidden, mapping, mode=mode,
            teacher_layers=len(teacher_hidden) - 1,
            chunk_pairs=chunk_pairs,
            loss_scale=weights[term] / accum,
            backward=lambda t: t.backward(),
        )

    terms = []
    if weights.get(LOGIT_KD):
        terms.append(weights[LOGIT_KD] * logits.float().pow(2).mean())
    if weights.get(CE):
        terms.append(weights[CE] * logits.float().abs().mean())
    if weights.get(ROUTER_BALANCE):
        terms.append(weights[ROUTER_BALANCE] * 0.001 * aux)

    total = None
    for t in terms:
        total = t if total is None else total + t

    if layer_backward is not None and total is not None:
        loss = total / accum
        live = [(t, g) for t, g in zip(layer_backward.sources, layer_backward.grads)
                if t.requires_grad]
        torch.autograd.backward(
            [t for t, _ in live] + [loss],
            [g for _, g in live] + [torch.ones_like(loss)],
        )
    elif layer_backward is not None:
        layer_backward.backward()
    else:
        (total / accum).backward()


def _fixture(seed=0):
    torch.manual_seed(seed)
    model = _Tiny()
    x = torch.randn(2, 5, 8)
    teacher_hidden = tuple(torch.randn(2, 5, 8) for _ in range(len(model.blocks) + 1))
    return model, x, teacher_hidden


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.grad is not None}


def _run(weights, *, accum=1, seed=0, chunk_pairs=2):
    model, x, th = _fixture(seed)
    model.zero_grad(set_to_none=True)
    _assemble(model, x, th, weights, accum=accum, chunk_pairs=chunk_pairs)
    return model, _grads(model)


def test_every_enabled_term_contributes_nonzero_gradient():
    """Each term alone must move parameters it is responsible for."""
    _, g_hidden = _run({HIDDEN_DELTA: 0.5})
    assert any(v.abs().sum() > 0 for k, v in g_hidden.items() if "blocks" in k)

    _, g_logit = _run({LOGIT_KD: 1.0})
    assert g_logit["head.weight"].abs().sum() > 0

    _, g_ce = _run({CE: 0.1})
    assert g_ce["head.weight"].abs().sum() > 0

    _, g_router = _run({ROUTER_BALANCE: 1.0})
    assert g_router["router.weight"].abs().sum() > 0


def test_disabling_a_term_removes_its_gradient_contribution():
    """Dropping router_balance must leave the router untouched, and change nothing else."""
    _, with_router = _run({CE: 0.1, LOGIT_KD: 1.0, ROUTER_BALANCE: 1.0})
    _, without = _run({CE: 0.1, LOGIT_KD: 1.0})

    assert with_router["router.weight"].abs().sum() > 0
    assert "router.weight" not in without or without["router.weight"].abs().sum() == 0
    # the shared head is reached only by ce/logit here, so it must be identical
    torch.testing.assert_close(with_router["head.weight"], without["head.weight"])


def test_weights_are_applied_exactly_once():
    """Doubling a weight doubles that term's gradient -- not 1x and not 4x."""
    _, single = _run({LOGIT_KD: 1.0})
    _, double = _run({LOGIT_KD: 2.0})
    torch.testing.assert_close(double["head.weight"], single["head.weight"] * 2.0)

    _, h1 = _run({HIDDEN_DELTA: 0.5})
    _, h2 = _run({HIDDEN_DELTA: 1.0})
    for k in h1:
        if "blocks" in k:
            torch.testing.assert_close(h2[k], h1[k] * 2.0, rtol=1e-4, atol=1e-6)


def test_composite_equals_sum_of_its_parts():
    """The combined backward must equal summing the terms' individual gradients.

    This is the property that would break under a double backward, a dropped term, or a
    weight applied to an already-detached scalar.
    """
    weights = {CE: 0.1, LOGIT_KD: 1.0, HIDDEN_DELTA: 0.5, ROUTER_BALANCE: 1.0}
    _, both = _run(weights)

    accum_grads: dict[str, torch.Tensor] = {}
    for single in ({CE: 0.1}, {LOGIT_KD: 1.0}, {HIDDEN_DELTA: 0.5}, {ROUTER_BALANCE: 1.0}):
        _, g = _run(single)
        for k, v in g.items():
            accum_grads[k] = accum_grads.get(k, torch.zeros_like(v)) + v

    for k, v in both.items():
        torch.testing.assert_close(v, accum_grads[k], rtol=1e-4, atol=1e-6,
                                   msg=f"composite != sum of parts for {k}")


def test_gradient_accumulation_scales_correctly():
    """accum=2 must halve each micro-batch's contribution, so two of them equal one accum=1."""
    weights = {CE: 0.1, LOGIT_KD: 1.0, HIDDEN_DELTA: 0.5, ROUTER_BALANCE: 1.0}
    _, once = _run(weights, accum=1)

    model, x, th = _fixture(0)
    model.zero_grad(set_to_none=True)
    for _ in range(2):
        _assemble(model, x, th, weights, accum=2)
    twice = _grads(model)

    for k, v in once.items():
        torch.testing.assert_close(twice[k], v, rtol=1e-4, atol=1e-6,
                                   msg=f"accumulation mis-scaled for {k}")


def test_chunking_does_not_change_the_hidden_gradient():
    """chunk_pairs selects how the hidden term is evaluated, never what it is."""
    _, a = _run({HIDDEN_DELTA: 0.5}, chunk_pairs=1)
    _, b = _run({HIDDEN_DELTA: 0.5}, chunk_pairs=4)
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-4, atol=1e-6)


def test_config_rejects_composite_without_weights():
    from qwen_distill.training.config import (
        DataConfig, ExperimentConfig, ModelConfig, RuntimeConfig, TrainingConfig,
    )
    cfg = ExperimentConfig(
        name="x", objective={"signal_source": "online"},
        teacher={"model": "m", "revision": "r"},
        model=ModelConfig(pretrained="/tmp/x"),
        data=DataConfig(tokenized_text=True, text_path="/tmp/t", tokenizer_path="/tmp/k"),
        training=TrainingConfig(objective="composite"),
        runtime=RuntimeConfig(output_dir="/tmp/o"),
    )
    with pytest.raises(Exception) as excinfo:
        cfg.validate()
    assert "composite_weights" in str(excinfo.value)


def test_config_accepts_the_preregistered_arms():
    from qwen_distill.research.ablations import ARMS, DEFAULT_LOSS_WEIGHTS
    from qwen_distill.training.config import (
        DataConfig, ExperimentConfig, ModelConfig, RuntimeConfig, TrainingConfig,
    )
    for name in ("A0", "A2", "A1", "A3"):
        arm = ARMS[name]
        weights = dict(arm.loss_weights or DEFAULT_LOSS_WEIGHTS)
        cfg = ExperimentConfig(
            name=name, objective={"signal_source": "online"},
            teacher={"model": "m", "revision": "r"},
            model=ModelConfig(pretrained="/tmp/x"),
            data=DataConfig(tokenized_text=True, text_path="/tmp/t", tokenizer_path="/tmp/k"),
            training=TrainingConfig(objective="composite", composite_weights=weights),
            runtime=RuntimeConfig(output_dir="/tmp/o"),
        )
        cfg.validate()
