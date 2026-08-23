"""Offline ingestion and validation of locally supplied checkpoint metadata.

The teacher's small metadata files — config, tokenizer, generation config, chat
template — are a few megabytes and settle most of Phase 1's open questions. The weights
are ~50 GB and settle almost none of them. So the two are separated here: this module
consumes a **local directory** of metadata files and never touches the network.

Design rules:

* **Never assume a file exists.** Real checkpoints vary; the chat template may live in
  ``chat_template.jinja``, in ``tokenizer_config.json``, or in neither.
* **Presence is not verification.** A field is FOUND only when it was parsed and its
  value read. ``validate_metadata`` reports the value, not just the filename.
* **Distinguish "absent" from "unknowable".** A field the metadata simply does not
  carry (MISSING) is different from one that cannot be settled without weights or a
  runtime experiment (UNKNOWN). Collapsing them would overstate what we know.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Status = Literal["FOUND", "MISSING", "OPTIONAL", "UNKNOWN"]

#: Files we look for, and whether their absence blocks Phase 1 verification.
#: ``blocks`` names what cannot be answered without the file.
EXPECTED_FILES: tuple[tuple[str, bool, str], ...] = (
    ("config.json", True, "architecture, model_type, layer layout, parameter count"),
    ("tokenizer_config.json", True, "tokenizer class, special tokens, chat template"),
    ("tokenizer.json", False, "exact vocabulary size and merges"),
    ("generation_config.json", False, "default sampling and stop tokens"),
    ("special_tokens_map.json", False, "BOS/EOS/PAD identity"),
    ("chat_template.jinja", False, "chat template, if not inside tokenizer_config.json"),
    ("chat_template.json", False, "chat template, alternative location"),
    ("preprocessor_config.json", False, "vision preprocessing, if multimodal"),
    ("LICENSE", False, "upstream licence terms"),
)

#: Config fields that materially affect parameter count, memory, or loading.
#: (key, required, note)
ARCHITECTURE_FIELDS: tuple[tuple[str, bool, str], ...] = (
    ("model_type", True, "which implementation loads this checkpoint"),
    ("architectures", True, "declared model classes"),
    ("hidden_size", True, "dominant term in every parameter count"),
    ("num_hidden_layers", True, "depth"),
    ("intermediate_size", True, "FFN width; ~64% of teacher parameters"),
    ("vocab_size", True, "embedding and LM head size"),
    ("num_attention_heads", True, "gated attention query heads"),
    ("num_key_value_heads", True, "GQA groups; sets KV cache size"),
    ("head_dim", True, "attention head dimension"),
    ("linear_num_key_heads", True, "DeltaNet key heads"),
    ("linear_num_value_heads", True, "DeltaNet value heads"),
    ("linear_key_head_dim", True, "DeltaNet key head dimension"),
    ("linear_value_head_dim", True, "DeltaNet value head dimension"),
    ("linear_conv_kernel_dim", False, "DeltaNet depthwise conv kernel"),
    ("layer_types", False, "explicit hybrid layout; derivable from full_attention_interval"),
    ("full_attention_interval", False, "hybrid layout period; superseded by layer_types"),
    ("tie_word_embeddings", True, "whether the LM head duplicates the embedding"),
    ("max_position_embeddings", True, "native context length"),
    ("rope_parameters", False, "RoPE configuration"),
    ("rope_theta", False, "RoPE base frequency"),
    ("rope_scaling", False, "context extension (e.g. YaRN)"),
    ("partial_rotary_factor", False, "fraction of head_dim that is rotated"),
    ("use_cache", False, "default caching behaviour"),
    ("rms_norm_eps", False, "normalisation epsilon"),
    ("hidden_act", False, "activation function"),
    ("attention_bias", False, "whether attention projections carry bias"),
    ("dtype", False, "checkpoint storage dtype"),
    ("torch_dtype", False, "checkpoint storage dtype (legacy key)"),
    ("auto_map", False, "presence means trust_remote_code is required"),
    ("quantization_config", False, "whether the checkpoint is pre-quantised"),
)

#: MTP-related keys. Absent here does not mean absent from the checkpoint: the tensors
#: can ship without a config entry, which only the weights would reveal.
MTP_FIELDS: tuple[str, ...] = (
    "mtp_num_hidden_layers", "num_nextn_predict_layers", "mtp_config", "num_mtp_modules",
)

#: Markers searched for in the chat template to identify reasoning controls.
REASONING_MARKERS: tuple[str, ...] = (
    "reasoning_effort", "enable_thinking", "preserve_thinking", "thinking",
    "xhigh", "high", "medium", "low", "<think>", "</think>", "tools", "tool_call",
)


@dataclass
class MetadataFileReport:
    """One expected file: present or not, parsed or not."""

    name: str
    required: bool
    present: bool
    path: str | None = None
    size_bytes: int | None = None
    parsed: bool = False
    parse_error: str | None = None
    blocks: str = ""

    @property
    def status(self) -> Status:
        if not self.present:
            return "MISSING" if self.required else "OPTIONAL"
        if self.parse_error:
            return "UNKNOWN"
        return "FOUND"


@dataclass
class FieldReport:
    """One field we tried to read, and what we found."""

    name: str
    status: Status
    value: Any = None
    source: str | None = None
    note: str = ""

    def display_value(self, limit: int = 60) -> str:
        if self.value is None:
            return "-"
        text = json.dumps(self.value) if not isinstance(self.value, str) else self.value
        return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class TeacherMetadata:
    """Everything readable from a local metadata directory."""

    root: Path
    files: list[MetadataFileReport] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    text_config: dict[str, Any] = field(default_factory=dict)
    vision_config: dict[str, Any] | None = None
    tokenizer_config: dict[str, Any] = field(default_factory=dict)
    generation_config: dict[str, Any] = field(default_factory=dict)
    special_tokens_map: dict[str, Any] = field(default_factory=dict)
    chat_template: str | None = None
    chat_template_source: str | None = None
    license_text: str | None = None
    tokenizer_vocab_size: int | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def has_config(self) -> bool:
        return bool(self.config)

    def file_report(self, name: str) -> MetadataFileReport | None:
        return next((f for f in self.files if f.name == name), None)


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    except OSError as exc:
        return None, f"unreadable: {exc}"
    if not isinstance(data, dict):
        return None, f"expected a JSON object, got {type(data).__name__}"
    return data, None


def load_metadata(path: str | Path) -> TeacherMetadata:
    """Read every metadata file present in ``path``. Never touches the network."""
    root = Path(path)
    metadata = TeacherMetadata(root=root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    parsed: dict[str, dict[str, Any]] = {}
    for name, required, blocks in EXPECTED_FILES:
        file_path = root / name
        report = MetadataFileReport(
            name=name, required=required, present=file_path.is_file(), blocks=blocks
        )
        if report.present:
            report.path = str(file_path)
            report.size_bytes = file_path.stat().st_size
            if name == "LICENSE":
                try:
                    metadata.license_text = file_path.read_text(encoding="utf-8")
                    report.parsed = True
                except OSError as exc:
                    report.parse_error = f"unreadable: {exc}"
            elif name == "chat_template.jinja":
                try:
                    metadata.chat_template = file_path.read_text(encoding="utf-8")
                    metadata.chat_template_source = name
                    report.parsed = True
                except OSError as exc:
                    report.parse_error = f"unreadable: {exc}"
            else:
                data, error = _load_json(file_path)
                if error:
                    report.parse_error = error
                    metadata.errors.append(f"{name}: {error}")
                else:
                    parsed[name] = data
                    report.parsed = True
        metadata.files.append(report)

    metadata.config = parsed.get("config.json", {})
    metadata.tokenizer_config = parsed.get("tokenizer_config.json", {})
    metadata.generation_config = parsed.get("generation_config.json", {})
    metadata.special_tokens_map = parsed.get("special_tokens_map.json", {})

    # Multimodal checkpoints nest the language model under text_config.
    if isinstance(metadata.config.get("text_config"), dict):
        metadata.text_config = metadata.config["text_config"]
    else:
        metadata.text_config = metadata.config
    if isinstance(metadata.config.get("vision_config"), dict):
        metadata.vision_config = metadata.config["vision_config"]

    # Chat template: three possible homes, in precedence order.
    if metadata.chat_template is None:
        template_json = parsed.get("chat_template.json")
        if template_json is not None:
            candidate = template_json.get("chat_template")
            if isinstance(candidate, str):
                metadata.chat_template = candidate
                metadata.chat_template_source = "chat_template.json"
    if metadata.chat_template is None:
        candidate = metadata.tokenizer_config.get("chat_template")
        if isinstance(candidate, str):
            metadata.chat_template = candidate
            metadata.chat_template_source = "tokenizer_config.json"
        elif isinstance(candidate, list):
            # Some checkpoints ship several named templates.
            metadata.chat_template = json.dumps(candidate, ensure_ascii=False)
            metadata.chat_template_source = "tokenizer_config.json (named templates)"

    tokenizer_json = parsed.get("tokenizer.json")
    if tokenizer_json is not None:
        vocab = tokenizer_json.get("model", {}).get("vocab")
        added = tokenizer_json.get("added_tokens", [])
        if isinstance(vocab, (dict, list)):
            metadata.tokenizer_vocab_size = len(vocab) + len(added)

    return metadata


def _lookup(metadata: TeacherMetadata, key: str) -> tuple[Any, str | None]:
    """Find ``key`` in the text config, then the top-level config."""
    if key in metadata.text_config:
        source = "config.json:text_config" if metadata.text_config is not metadata.config else "config.json"
        return metadata.text_config[key], source
    if key in metadata.config:
        return metadata.config[key], "config.json"
    return None, None


def validate_metadata(metadata: TeacherMetadata) -> tuple[list[MetadataFileReport], list[FieldReport]]:
    """Classify every expected file and field.

    A field is FOUND only when it was actually parsed and its value read — never
    because the file containing it exists.
    """
    fields: list[FieldReport] = []

    if not metadata.has_config:
        fields.append(
            FieldReport(
                "*", "MISSING", note="config.json absent or unparseable; no field can be read"
            )
        )
        return metadata.files, fields

    for key, required, note in ARCHITECTURE_FIELDS:
        value, source = _lookup(metadata, key)
        if value is not None:
            fields.append(FieldReport(key, "FOUND", value, source, note))
        elif required:
            fields.append(FieldReport(key, "MISSING", None, None, note))
        else:
            fields.append(FieldReport(key, "OPTIONAL", None, None, note))

    # layer_types: explicit, derivable, or neither. Report which.
    explicit, _ = _lookup(metadata, "layer_types")
    interval, _ = _lookup(metadata, "full_attention_interval")
    if explicit is not None:
        fields.append(
            FieldReport("layer_layout", "FOUND", f"explicit ({len(explicit)} layers)",
                        "config.json", "layer_types given directly")
        )
    elif interval is not None:
        fields.append(
            FieldReport("layer_layout", "FOUND", f"derived from interval {interval}",
                        "config.json", "expanded by the reference implementation's rule")
        )
    else:
        fields.append(
            FieldReport("layer_layout", "UNKNOWN", None, None,
                        "neither layer_types nor full_attention_interval present; "
                        "layout would come from the implementation default")
        )

    # MTP: config keys only. Tensors can ship without a config entry.
    mtp_found = {k: _lookup(metadata, k)[0] for k in MTP_FIELDS}
    mtp_present = {k: v for k, v in mtp_found.items() if v is not None}
    if mtp_present:
        fields.append(FieldReport("mtp_config", "FOUND", mtp_present, "config.json",
                                  "MTP declared in config"))
    else:
        fields.append(
            FieldReport("mtp_config", "UNKNOWN", None, None,
                        "no MTP key in config; the checkpoint may still ship mtp.* "
                        "tensors, which only the weights would reveal")
        )

    # Vision tower.
    if metadata.vision_config is not None:
        fields.append(FieldReport("vision_config", "FOUND", sorted(metadata.vision_config)[:6],
                                  "config.json", "checkpoint declares a vision tower"))
    else:
        fields.append(FieldReport("vision_config", "OPTIONAL", None, None,
                                  "no vision_config; checkpoint appears text-only"))

    # Tokenizer and template.
    if metadata.tokenizer_config:
        fields.append(FieldReport("tokenizer_class", _status(metadata.tokenizer_config.get("tokenizer_class")),
                                  metadata.tokenizer_config.get("tokenizer_class"),
                                  "tokenizer_config.json", "declared tokenizer class"))
        for token in ("bos_token", "eos_token", "pad_token", "unk_token"):
            value = metadata.tokenizer_config.get(token) or metadata.special_tokens_map.get(token)
            fields.append(FieldReport(token, _status(value), _token_text(value),
                                      "tokenizer_config.json/special_tokens_map.json", ""))
    else:
        fields.append(FieldReport("tokenizer_class", "MISSING", None, None,
                                  "tokenizer_config.json absent"))

    if metadata.tokenizer_vocab_size is not None:
        fields.append(FieldReport("tokenizer_vocab_size", "FOUND", metadata.tokenizer_vocab_size,
                                  "tokenizer.json", "counted from vocab + added tokens"))
    else:
        fields.append(FieldReport("tokenizer_vocab_size", "OPTIONAL", None, None,
                                  "tokenizer.json absent; config vocab_size stands alone"))

    if metadata.chat_template:
        fields.append(FieldReport("chat_template", "FOUND",
                                  f"{len(metadata.chat_template)} chars",
                                  metadata.chat_template_source, "template text available"))
        found_markers = [m for m in REASONING_MARKERS if m in metadata.chat_template]
        fields.append(
            FieldReport("reasoning_controls",
                        "FOUND" if found_markers else "MISSING",
                        found_markers or None, metadata.chat_template_source,
                        "markers present in the template; whether a control has an "
                        "*effect* needs the template diff, not just its presence")
        )
    else:
        fields.append(FieldReport("chat_template", "MISSING", None, None,
                                  "no template in chat_template.jinja/.json or tokenizer_config.json"))
        fields.append(FieldReport("reasoning_controls", "UNKNOWN", None, None,
                                  "cannot be read without the chat template"))

    # Licence.
    if metadata.license_text:
        fields.append(FieldReport("license", "FOUND", _license_hint(metadata.license_text),
                                  "LICENSE", "text supplied locally"))
    else:
        fields.append(FieldReport("license", "MISSING", None, None,
                                  "upstream LICENSE file not supplied; licence stays UNKNOWN"))

    # Things metadata fundamentally cannot settle.
    fields.append(FieldReport("state_dict_parameter_count", "UNKNOWN", None, None,
                              "requires the weights"))
    fields.append(FieldReport("runtime_generation", "UNKNOWN", None, None,
                              "requires the weights; config parsing is not proof the model runs"))
    fields.append(FieldReport("medium_reasoning_is_noop", "UNKNOWN", None, None,
                              "decidable from the template diff (scripts/inspect_chat_template.py); "
                              "confirming behaviour needs a runtime experiment"))

    return metadata.files, fields


def _status(value: Any) -> Status:
    return "FOUND" if value not in (None, "", [], {}) else "MISSING"


def _token_text(value: Any) -> Any:
    """Special tokens are sometimes plain strings, sometimes AddedToken dicts."""
    if isinstance(value, dict):
        return value.get("content", value)
    return value


def _license_hint(text: str) -> str:
    """Name the licence if a well-known one is recognisable; never guess silently."""
    head = text[:4000].lower()
    for needle, name in (
        ("apache license", "Apache-2.0 (by header)"),
        ("mit license", "MIT (by header)"),
        ("tongyi qianwen", "Tongyi Qianwen licence (by header)"),
        ("gnu general public", "GPL (by header)"),
        ("bsd ", "BSD (by header)"),
    ):
        if needle in head:
            return name
    return "unrecognised licence text; read it manually"


def summarise_counts(files: list[MetadataFileReport], fields: list[FieldReport]) -> dict[str, int]:
    """Count statuses across files and fields."""
    counts = {"FOUND": 0, "MISSING": 0, "OPTIONAL": 0, "UNKNOWN": 0}
    for item in (*files, *fields):
        counts[item.status] += 1
    return counts


def blocking_gaps(files: list[MetadataFileReport], fields: list[FieldReport]) -> list[str]:
    """What is missing that actually blocks Phase 1 verification."""
    gaps = [f"{f.name} is required ({f.blocks})" for f in files if f.status == "MISSING"]
    gaps += [f"{f.name} could not be read: {f.note}" for f in fields if f.status == "MISSING"]
    return gaps
