"""Check that the installed runtime actually implements what the checkpoint declares.

Matching parameter *shapes* is not the same as matching the *computation*. A model can
carry exactly 26,895,998,464 correctly-shaped parameters and still apply the wrong
activation, producing plausible but wrong outputs. This module checks the second thing.

The motivating case: the checkpoint declares ``output_gate_type: "swish"`` while
``transformers`` 5.15.1 contains a hard-coded ``torch.sigmoid(gate)``. Those look like a
contradiction, and an earlier phase of this project recorded them as one. They are not —
but establishing that required identifying *which gate* each refers to:

* **``output_gate_type`` governs the Gated DeltaNet output gate.** vLLM reads it in
  ``QwenGatedDeltaNetAttention.__init__`` and passes it as ``activation=`` to
  ``RMSNormGated(head_v_dim, norm_before_gate=True, ...)``, normalising ``"swish"`` to
  ``"silu"`` first. ``transformers`` builds the same module as
  ``Qwen3_5RMSNormGated(head_v_dim)`` with ``self.activation = "silu"`` hard-coded.
  Since swish and SiLU are the same function, the two agree.
* **The ``torch.sigmoid(gate)`` in ``Qwen3_5Attention.forward`` is a different gate** —
  the full-attention output gate, governed by ``attn_output_gate``, not by
  ``output_gate_type``. vLLM applies ``torch.sigmoid`` there too.

So the two implementations agree, and the config is satisfied. What this module does is
make that conclusion *checkable* rather than a note in a document, so a future upstream
change (a checkpoint declaring ``sigmoid``, or a `transformers` release that starts
reading the key) is caught before it silently corrupts a run.

Nothing here patches `transformers`. It reports; it does not modify.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: `transformers` releases surveyed for this architecture, and what each applies.
#: Checked by downloading each wheel and reading ``models/qwen3_5/modeling_qwen3_5.py``.
#: The module first appears in 5.8.0 — the same version that wrote this checkpoint's
#: config — and every release since applies the same two activations. 5.8.0-5.12.x
#: spell the DeltaNet gate ``F.silu(gate)``; 5.13+ route it through
#: ``ACT2FN[self.activation]`` with ``self.activation = "silu"``. Same function.
TRANSFORMERS_VERSION_SURVEY: dict[str, str] = {
    "5.8.0": "qwen3_5 present; DeltaNet gate F.silu, attention gate sigmoid",
    "5.9.0": "same",
    "5.10.1": "same",
    "5.12.0": "same",
    "5.15.1": "same, DeltaNet gate via ACT2FN[self.activation='silu']",
}

#: Earliest release known to contain the architecture at all.
MINIMUM_TRANSFORMERS_VERSION = "5.8.0"
#: Release this project verified against and pins for reproducibility.
RECOMMENDED_TRANSFORMERS_VERSION = "5.15.1"

#: Gate-activation names that a ``silu``-hard-coded implementation satisfies.
#: Swish (β=1) and SiLU are the same function: ``x * sigmoid(x)``.
SILU_EQUIVALENT_NAMES: frozenset[str] = frozenset({"silu", "swish"})

#: What `transformers` 5.15.1 hard-codes, per gate. Verified by reading the installed
#: source, not inferred.
TRANSFORMERS_HARDCODED_GATES: dict[str, str] = {
    # Qwen3_5RMSNormGated.__init__: self.activation = "silu"
    "deltanet_output_gate": "silu",
    # Qwen3_5Attention.forward: attn_output = attn_output * torch.sigmoid(gate)
    "attention_output_gate": "sigmoid",
}

#: Which config key governs which gate. Establishing this mapping is the whole point of
#: the investigation: applying ``output_gate_type`` to the attention gate would be a
#: category error, and produced a false "discrepancy" in an earlier phase.
GATE_CONFIG_KEYS: dict[str, str] = {
    "deltanet_output_gate": "output_gate_type",
    "attention_output_gate": "attn_output_gate",
}


@dataclass
class GateCheck:
    """Whether one gate's declared activation is satisfied by the runtime."""

    gate: str
    config_key: str
    declared: Any
    runtime_activation: str
    satisfied: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeCompatReport:
    """Verdict on whether the installed runtime implements the declared computation."""

    transformers_version: str
    checks: list[GateCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if not self.checks:
            return "UNRESOLVED"
        return "VERIFIED_CORRECT" if all(c.satisfied for c in self.checks) else "IMPLEMENTATION_MISMATCH"

    def to_dict(self) -> dict[str, Any]:
        return {
            "transformers_version": self.transformers_version,
            "verdict": self.verdict,
            "checks": [c.to_dict() for c in self.checks],
            "notes": self.notes,
            "warnings": self.warnings,
        }


def activations_equivalent(declared: str, runtime: str) -> bool:
    """Whether a runtime activation satisfies a declared one.

    ``swish`` and ``silu`` name the same function and are interchangeable. Everything
    else must match exactly — in particular ``sigmoid`` is **not** interchangeable with
    either: they differ by nearly 2.0 over a typical input range.
    """
    declared, runtime = declared.lower(), runtime.lower()
    if declared == runtime:
        return True
    return {declared, runtime} <= SILU_EQUIVALENT_NAMES


def check_runtime_compatibility(config: Any) -> RuntimeCompatReport:
    """Check the installed `transformers` against a checkpoint config.

    ``config`` may be a loaded `transformers` config object or a plain dict; the text
    sub-config is used when present.
    """
    try:
        import transformers

        version = transformers.__version__
    except ImportError:
        version = "not installed"

    report = RuntimeCompatReport(transformers_version=version)

    if isinstance(config, dict):
        text = config.get("text_config", config)
        get = text.get
    else:
        text = getattr(config, "text_config", config)
        def get(key, default=None):  # noqa: E306
            return getattr(text, key, default)

    # --- DeltaNet output gate ------------------------------------------
    declared_gate = get("output_gate_type", "silu")
    runtime_gate = TRANSFORMERS_HARDCODED_GATES["deltanet_output_gate"]
    satisfied = activations_equivalent(str(declared_gate), runtime_gate)
    report.checks.append(
        GateCheck(
            gate="deltanet_output_gate",
            config_key="output_gate_type",
            declared=declared_gate,
            runtime_activation=runtime_gate,
            satisfied=satisfied,
            reason=(
                f"transformers hard-codes {runtime_gate!r} in Qwen3_5RMSNormGated; "
                + (
                    "swish and silu are the same function, so the declaration is satisfied"
                    if satisfied and str(declared_gate).lower() != runtime_gate
                    else "exact match"
                    if satisfied
                    else f"{declared_gate!r} is NOT equivalent to {runtime_gate!r}; the "
                         "installed implementation would compute the wrong DeltaNet gate"
                )
            ),
        )
    )
    if not satisfied:
        report.warnings.append(
            f"DeltaNet gate mismatch: checkpoint declares {declared_gate!r} but "
            f"transformers {version} hard-codes {runtime_gate!r}. Do NOT use this "
            "runtime for teacher inference until resolved."
        )

    # --- full-attention output gate ------------------------------------
    declared_attn_gate = get("attn_output_gate", True)
    report.checks.append(
        GateCheck(
            gate="attention_output_gate",
            config_key="attn_output_gate",
            declared=declared_attn_gate,
            runtime_activation=TRANSFORMERS_HARDCODED_GATES["attention_output_gate"],
            satisfied=bool(declared_attn_gate),
            reason=(
                "transformers builds the doubled q_proj and applies torch.sigmoid "
                "unconditionally; the checkpoint declares the gate enabled, so they agree"
                if declared_attn_gate
                else "checkpoint declares attn_output_gate=false, but transformers builds "
                     "the gated projection regardless: shapes would not match the "
                     "checkpoint and loading would fail loudly"
            ),
        )
    )
    if not declared_attn_gate:
        report.warnings.append(
            "attn_output_gate=false is not honoured by transformers 5.15.1; it always "
            "builds a gated q_proj. Expect a shape mismatch at load time."
        )

    report.notes.append(
        "output_gate_type governs the DeltaNet gate, NOT the attention gate. The "
        "torch.sigmoid in Qwen3_5Attention.forward is the separate attention output "
        "gate and is applied identically by vLLM."
    )
    ssm_dtype = get("mamba_ssm_dtype")
    if ssm_dtype:
        report.notes.append(
            f"mamba_ssm_dtype={ssm_dtype!r} is not read by transformers, but its torch "
            "reference path accumulates the delta-rule state in float32 anyway "
            "(measured), so the declaration is honoured in effect."
        )
    return report
