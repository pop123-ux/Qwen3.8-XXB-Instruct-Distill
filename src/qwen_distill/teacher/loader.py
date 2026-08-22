"""Verify what a checkpoint *actually* loads as, under the intended software stack.

Phase 0 established the architecture by reading the reference implementation. That
answers "what does the `qwen3_5` family look like"; it does **not** answer "what does
the `Qwen3.8-27B` checkpoint resolve to". Those are different questions, and the
second one can only be answered by pointing `transformers` at the real checkpoint.

This module does that resolution and records the evidence:

* the ``model_type`` the config declares,
* the concrete class ``AutoModelForCausalLM`` would build, and the module it lives in,
* whether ``trust_remote_code`` is required (i.e. the checkpoint ships its own
  modeling code via ``auto_map`` or bundled ``.py`` files),
* the exact package versions involved.

Class resolution is done **without instantiating weights**, so it works on a laptop
and on a config-only download.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoaderReport:
    """What actually happens when the intended stack loads this checkpoint."""

    source: str
    #: Present only when the check ran against a real config.
    model_type: str | None = None
    config_class: str | None = None
    config_module: str | None = None
    declared_architectures: list[str] = field(default_factory=list)
    resolved_model_class: str | None = None
    resolved_model_module: str | None = None
    #: True when the config declares ``auto_map`` or the directory bundles .py modules.
    requires_trust_remote_code: bool = False
    remote_code_evidence: list[str] = field(default_factory=list)
    #: True when the resolving module is inside the installed `transformers` package.
    uses_native_transformers: bool = False
    tokenizer_class: str | None = None
    versions: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.errors:
            return "FAILED"
        if self.resolved_model_class is None:
            return "UNRESOLVED"
        if self.requires_trust_remote_code:
            return "LOADS_WITH_REMOTE_CODE"
        if self.uses_native_transformers:
            return "LOADS_NATIVELY"
        return "LOADS_UNKNOWN_SOURCE"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict
        return data


def collect_versions() -> dict[str, str]:
    """Record the versions that determine whether a checkpoint loads."""
    versions = {"python": sys.version.split()[0]}
    for name in ("transformers", "torch", "tokenizers", "huggingface_hub", "accelerate", "vllm"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def detect_remote_code(source: str, config_dict: dict[str, Any]) -> tuple[bool, list[str]]:
    """Decide whether loading this checkpoint needs ``trust_remote_code=True``.

    Two independent signals: an ``auto_map`` entry in the config (the mechanism
    `transformers` uses to point at checkpoint-provided classes), and ``.py`` files
    shipped alongside the weights.
    """
    evidence: list[str] = []
    auto_map = config_dict.get("auto_map")
    if auto_map:
        evidence.append(f"config.auto_map present: {json.dumps(auto_map)}")
    for key in ("text_config", "vision_config"):
        sub = config_dict.get(key)
        if isinstance(sub, dict) and sub.get("auto_map"):
            evidence.append(f"config.{key}.auto_map present: {json.dumps(sub['auto_map'])}")

    path = Path(source)
    if path.is_dir():
        modules = sorted(p.name for p in path.glob("*.py"))
        if modules:
            evidence.append(f"checkpoint bundles python modules: {', '.join(modules)}")
    return bool(evidence), evidence


def verify_loader(source: str, *, trust_remote_code: bool = False) -> LoaderReport:
    """Resolve the config and model class for ``source`` without building weights."""
    report = LoaderReport(source=source, versions=collect_versions())

    try:
        from transformers import AutoConfig
    except ImportError as exc:
        report.errors.append(f"transformers is not installed: {exc}")
        return report

    # --- config ------------------------------------------------------------
    try:
        config = AutoConfig.from_pretrained(source, trust_remote_code=trust_remote_code)
    except Exception as exc:  # noqa: BLE001 - report any failure verbatim
        report.errors.append(f"AutoConfig.from_pretrained failed: {type(exc).__name__}: {exc}")
        return report

    report.model_type = getattr(config, "model_type", None)
    report.config_class = type(config).__name__
    report.config_module = type(config).__module__
    report.declared_architectures = list(getattr(config, "architectures", None) or [])

    config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
    report.requires_trust_remote_code, report.remote_code_evidence = detect_remote_code(
        source, config_dict
    )

    # --- model class, resolved from the config without materialising weights ---
    try:
        from transformers import AutoModelForCausalLM

        mapping = AutoModelForCausalLM._model_mapping

        def lookup(cfg):
            # _LazyAutoMapping raises KeyError rather than exposing a stable .get().
            try:
                return mapping[type(cfg)]
            except KeyError:
                return None

        model_class = lookup(config)
        if model_class is None:
            # Multimodal checkpoints often map their causal-LM head off the text config.
            text_config = getattr(config, "text_config", None)
            if text_config is not None and lookup(text_config) is not None:
                model_class = lookup(text_config)
                report.warnings.append(
                    "causal-LM class resolved from text_config, not the top-level config; "
                    "the checkpoint is multimodal and the text tower loads separately"
                )
        if model_class is None:
            report.warnings.append(
                f"no AutoModelForCausalLM mapping for {type(config).__name__}; "
                "the checkpoint may need a newer transformers or trust_remote_code"
            )
        else:
            report.resolved_model_class = model_class.__name__
            report.resolved_model_module = model_class.__module__
            report.uses_native_transformers = model_class.__module__.startswith("transformers.")
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"model class resolution failed: {type(exc).__name__}: {exc}")

    # --- tokenizer (class only; no vocabulary download beyond what is cached) ---
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)
        report.tokenizer_class = type(tokenizer).__name__
    except Exception as exc:  # noqa: BLE001
        report.warnings.append(f"tokenizer could not be loaded: {type(exc).__name__}: {exc}")

    return report


def instantiate_on_meta(config_source: str | dict[str, Any], *, trust_remote_code: bool = False):
    """Build the model structure on the ``meta`` device and return (model, config).

    ``meta`` tensors carry shape and dtype but allocate **no storage**, so the full
    27B structure can be materialised on a laptop. This is how we check our analytical
    parameter formulas against what `transformers` would actually construct, without
    a GPU and without downloading weights.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if isinstance(config_source, dict):
        config = AutoConfig.for_model(**config_source)
    else:
        config = AutoConfig.from_pretrained(config_source, trust_remote_code=trust_remote_code)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    return model, config


def summarise_meta_parameters(model) -> dict[str, Any]:
    """Count parameters of a meta-device model and group them by component.

    Grouping mirrors :class:`qwen_distill.architecture.params.ParamBreakdown` so the
    two can be compared term by term rather than only in total.
    """
    groups = {
        "embedding": 0, "lm_head": 0, "final_norm": 0, "layer_norms": 0,
        "mlp": 0, "full_attention": 0, "linear_attention": 0, "other": 0,
    }
    per_tensor: dict[str, list[int]] = {}
    total = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        per_tensor[name] = list(param.shape)
        if "embed_tokens" in name:
            groups["embedding"] += n
        elif "lm_head" in name:
            groups["lm_head"] += n
        elif ".mlp." in name:
            groups["mlp"] += n
        elif ".self_attn." in name:
            groups["full_attention"] += n
        elif ".linear_attn." in name:
            groups["linear_attention"] += n
        elif "input_layernorm" in name or "post_attention_layernorm" in name:
            groups["layer_norms"] += n
        elif name.endswith("model.norm.weight") or name.endswith("norm.weight"):
            groups["final_norm"] += n
        else:
            groups["other"] += n
    return {"total": total, "groups": groups, "n_tensors": len(per_tensor), "tensors": per_tensor}
