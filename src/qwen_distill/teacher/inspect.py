"""Teacher inspection: derive a verified architecture report from a real checkpoint.

The project rule is that no architectural claim enters the repository unless it came
from the checkpoint or the reference implementation. This module is how that rule is
enforced mechanically: point it at a local checkpoint directory or a Hugging Face
repo id, and it reports what is actually there.

It deliberately does **not** require ``torch``: reading ``config.json`` and the
``safetensors`` header is enough to recover every shape, and that works on a laptop
with no GPU and no 50 GB download when ``--config-only`` is used.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..architecture.params import count_parameters, format_params
from ..architecture.spec import HybridArchSpec


@dataclass
class TeacherReport:
    """Everything we could establish about a teacher checkpoint."""

    source: str
    model_type: str | None = None
    architectures: list[str] = field(default_factory=list)
    torch_dtype: str | None = None
    is_multimodal: bool = False
    vision_config: dict[str, Any] | None = None
    spec: HybridArchSpec | None = None
    #: Tensor name -> (dtype, shape), read from safetensors headers when available.
    tensors: dict[str, tuple[str, list[int]]] = field(default_factory=dict)
    checkpoint_param_count: int | None = None
    mtp_tensors: list[str] = field(default_factory=list)
    vision_tensors: list[str] = field(default_factory=list)
    generation_config: dict[str, Any] | None = None
    chat_template_excerpt: str | None = None
    reasoning_controls: list[str] = field(default_factory=list)
    tokenizer_vocab_size: int | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model_type": self.model_type,
            "architectures": self.architectures,
            "torch_dtype": self.torch_dtype,
            "is_multimodal": self.is_multimodal,
            "vision_config": self.vision_config,
            "spec": self.spec.to_dict() if self.spec else None,
            "analytical_param_count": count_parameters(self.spec).as_dict() if self.spec else None,
            "checkpoint_param_count": self.checkpoint_param_count,
            "n_tensors": len(self.tensors),
            "n_mtp_tensors": len(self.mtp_tensors),
            "mtp_tensors_sample": self.mtp_tensors[:10],
            "n_vision_tensors": len(self.vision_tensors),
            "generation_config": self.generation_config,
            "reasoning_controls": self.reasoning_controls,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "warnings": self.warnings,
        }


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read a ``.safetensors`` header without loading any tensor data.

    Layout: 8-byte little-endian header length, then that many bytes of JSON.
    """
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        (header_len,) = struct.unpack("<Q", raw_len)
        header = fh.read(header_len)
    return json.loads(header)


_DTYPE_BYTES = {
    "F64": 8, "I64": 8, "F32": 4, "I32": 4, "F16": 2, "BF16": 2,
    "I16": 2, "U16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}


def _numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


REASONING_CONTROL_MARKERS = (
    "reasoning_effort",
    "enable_thinking",
    "preserve_thinking",
    "xhigh",
    "<think>",
    "</think>",
    "thinking",
)


def inspect_local(path: str | Path, *, config_only: bool = False) -> TeacherReport:
    """Inspect a checkpoint directory on disk."""
    root = Path(path)
    report = TeacherReport(source=str(root))
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    config_path = root / "config.json"
    if not config_path.is_file():
        report.warnings.append("config.json not found; nothing structural could be verified")
        return report

    config = json.loads(config_path.read_text(encoding="utf-8"))
    report.model_type = config.get("model_type")
    report.architectures = config.get("architectures", []) or []
    text_cfg = config.get("text_config", config)
    report.torch_dtype = config.get("torch_dtype") or text_cfg.get("torch_dtype")
    report.vision_config = config.get("vision_config")
    report.is_multimodal = report.vision_config is not None or any(
        "ConditionalGeneration" in a or "VL" in a for a in report.architectures
    )

    try:
        report.spec = HybridArchSpec.from_hf_config(config, name=root.name)
    except (KeyError, ValueError) as exc:
        report.warnings.append(f"could not build HybridArchSpec from config.json: {exc}")

    gen_path = root / "generation_config.json"
    if gen_path.is_file():
        report.generation_config = json.loads(gen_path.read_text(encoding="utf-8"))

    # Reasoning controls live in the chat template, which may sit in its own file
    # or inside tokenizer_config.json depending on the release.
    template_text = ""
    for candidate in ("chat_template.jinja", "chat_template.json"):
        p = root / candidate
        if p.is_file():
            template_text = p.read_text(encoding="utf-8")
            break
    if not template_text:
        tok_cfg_path = root / "tokenizer_config.json"
        if tok_cfg_path.is_file():
            tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
            template = tok_cfg.get("chat_template")
            if isinstance(template, list):  # multiple named templates
                template_text = json.dumps(template)
            elif isinstance(template, str):
                template_text = template
    if template_text:
        report.chat_template_excerpt = template_text[:4000]
        report.reasoning_controls = sorted(
            {m for m in REASONING_CONTROL_MARKERS if m in template_text}
        )
    else:
        report.warnings.append("no chat template found; reasoning controls unverified")

    tok_path = root / "tokenizer.json"
    if tok_path.is_file():
        try:
            tok = json.loads(tok_path.read_text(encoding="utf-8"))
            vocab = tok.get("model", {}).get("vocab")
            added = tok.get("added_tokens", [])
            if vocab is not None:
                report.tokenizer_vocab_size = len(vocab) + len(added)
        except (json.JSONDecodeError, AttributeError) as exc:
            report.warnings.append(f"tokenizer.json present but unreadable: {exc}")

    if config_only:
        return report

    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        report.warnings.append("no .safetensors shards found; tensor shapes unverified")
        return report

    total_params = 0
    for shard in shards:
        try:
            header = read_safetensors_header(shard)
        except (ValueError, json.JSONDecodeError) as exc:
            report.warnings.append(f"unreadable shard {shard.name}: {exc}")
            continue
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = meta.get("dtype", "?")
            shape = meta.get("shape", [])
            report.tensors[name] = (dtype, shape)
            total_params += _numel(shape)
    report.checkpoint_param_count = total_params
    report.mtp_tensors = sorted(n for n in report.tensors if n.startswith("mtp"))
    report.vision_tensors = sorted(
        n for n in report.tensors if ".visual." in n or n.startswith("visual.")
    )
    return report


def inspect_hub(repo_id: str, *, config_only: bool = True, revision: str | None = None) -> TeacherReport:
    """Inspect a checkpoint on the Hugging Face Hub.

    Downloads only the small metadata files by default. Requires network access to
    ``huggingface.co``; if that host is blocked, download the checkpoint elsewhere
    and use :func:`inspect_local`.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "huggingface_hub is required for --repo-id; install requirements/base.txt "
            "or pass a local --path instead"
        ) from exc

    from ..utils.hub import HubAccessError, diagnose_hub_error

    patterns = ["*.json", "*.jinja", "*.txt"]
    if not config_only:
        patterns.append("*.safetensors")
    try:
        local = snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=patterns)
    except Exception as exc:  # noqa: BLE001 - classify rather than leak a traceback
        raise HubAccessError(diagnose_hub_error(exc, repo_id)) from exc
    report = inspect_local(local, config_only=config_only)
    report.source = f"hf://{repo_id}" + (f"@{revision}" if revision else "")
    return report


def cross_check(report: TeacherReport) -> list[str]:
    """Compare the checkpoint against our analytical model and flag discrepancies.

    This is the guard that keeps ``qwen_distill.architecture`` honest: if upstream
    changes a projection shape, the analytical parameter count stops matching the
    checkpoint and this reports it.
    """
    findings: list[str] = []
    if report.spec is None:
        return ["no spec could be derived; cross-check skipped"]

    analytical = count_parameters(report.spec).total
    findings.append(f"analytical text-tower parameters: {format_params(analytical)} ({analytical:,})")

    if report.checkpoint_param_count is not None:
        ckpt = report.checkpoint_param_count
        vision = sum(_numel(s) for n, (_, s) in report.tensors.items() if n in set(report.vision_tensors))
        mtp = sum(_numel(s) for n, (_, s) in report.tensors.items() if n in set(report.mtp_tensors))
        text_only = ckpt - vision - mtp
        findings.append(f"checkpoint total parameters:      {format_params(ckpt)} ({ckpt:,})")
        findings.append(f"  of which vision tower:          {format_params(vision)}")
        findings.append(f"  of which MTP head:              {format_params(mtp)}")
        findings.append(f"  text tower (derived):           {format_params(text_only)}")
        if text_only:
            drift = abs(text_only - analytical) / text_only
            verdict = "MATCH" if drift < 0.005 else "MISMATCH"
            findings.append(f"  analytical vs checkpoint drift: {drift:.3%} -> {verdict}")
            if drift >= 0.005:
                findings.append(
                    "  ACTION: the analytical model in qwen_distill.architecture.params "
                    "no longer matches upstream; reconcile before trusting any estimate"
                )
    else:
        findings.append("checkpoint tensor data not inspected; run without --config-only to cross-check")

    if (
        report.tokenizer_vocab_size
        and report.spec
        and report.tokenizer_vocab_size > report.spec.vocab_size
    ):
        findings.append(
                f"WARNING: tokenizer vocab {report.tokenizer_vocab_size} exceeds "
                f"config vocab_size {report.spec.vocab_size}"
            )
    return findings
