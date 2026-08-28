"""Loading the actual Qwen3.8-27B, and refusing to pretend when it did not load.

This module exists because of one measured fact about `transformers`: **a checkpoint whose
keys do not match the model class loads without raising.** It prints a report and returns
a model whose weights are freshly initialised. Verified here against 5.15.1 by renaming
every tensor in a small checkpoint:

===========================  =======  ==========  ================
checkpoint keys              missing  unexpected  weights restored
===========================  =======  ==========  ================
untouched                          0           0  yes
``model.language_model.*``         0           0  yes  (remapped)
``garbage.*``                     56          55  **no**
half the layers removed           25           0  partially
===========================  =======  ==========  ================

Only the third and fourth rows are failures, and neither raised. A 27B model that loads
"successfully" with random weights will generate fluent text, produce a plausible loss
curve, and distil a student that learns nothing — and nothing downstream would reveal it.
So :func:`load_verified_teacher` treats a non-empty ``missing_keys`` as fatal.

The second row is worth stating separately because it looked like the dangerous one. The
Qwen3.8-27B checkpoint declares ``Qwen3_5ForConditionalGeneration`` and stores its text
tower under ``model.language_model.*``, while ``Qwen3_5ForCausalLM`` — the class we want,
since the project distils the text model — expects ``model.layers.*``. `transformers`
remaps between them, and ``Qwen3_5ForCausalLM._keys_to_ignore_on_load_unexpected`` already
discards ``^model.visual.*`` and ``^mtp.*``. Loading the text tower out of the multimodal
checkpoint is therefore correct and lossless; this module asserts it rather than assuming
it.

Everything real about the teacher lives here. :mod:`.backends` keeps the interface, the
mock and the dispatch, so the rule that a fake teacher is never reachable by accident stays
where it can be read in one screen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .provenance import TeacherIdentity, package_versions, sha256_text
from .reasoning_modes import ReasoningMode

#: The teacher this project distils. Recorded as a constant so a substitution is a diff.
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3.8-27B"

#: Metadata vendored into the repository, used to check that whatever gets loaded is the
#: architecture this project verified against.
VENDORED_METADATA = Path(__file__).resolve().parents[3] / "vendor" / "qwen38-metadata"

#: Architecture facts the loaded model must match, read from the vendored config.json.
#: A mismatch means a different checkpoint is being loaded under the expected name.
EXPECTED_ARCHITECTURE: dict[str, int] = {
    "vocab_size": 248320,
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "intermediate_size": 17408,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "full_attention_interval": 4,
}

#: Token ids the reasoning split depends on, from the vendored tokenizer_config.json.
#: Both are single tokens in this tokenizer, which is what makes an exact split possible.
THINK_OPEN_TOKEN = "<think>"
THINK_CLOSE_TOKEN = "</think>"

#: Quantisation schemes this loader knows how to configure, and what each needs.
QUANTIZATION_SCHEMES: dict[str, str] = {
    "4bit": "bitsandbytes NF4; ~15 GiB of weights for this teacher",
    "8bit": "bitsandbytes int8; ~27 GiB of weights for this teacher",
}


class TeacherLoadError(RuntimeError):
    """The teacher could not be loaded in a state fit to produce real outputs."""


class TeacherNotLoaded(RuntimeError):
    """An operation needing weights was called before :meth:`load`."""


@dataclass(frozen=True)
class TeacherLoadPlan:
    """Exactly how the teacher is to be loaded. Every field is explicit on purpose.

    ``dtype="auto"`` is passed to `transformers` **as the string**. Passing ``None``
    instead makes it load in float32 regardless of what the checkpoint declares, which for
    this bf16 27B checkpoint is ~108 GB rather than ~54 GB and fits nowhere.
    """

    model: str = DEFAULT_TEACHER_MODEL
    #: Commit SHA. ``None`` means unpinned, which is recorded as a reproducibility gap
    #: rather than treated as acceptable — the same repo id serves different weights over
    #: time.
    revision: str | None = None
    dtype: str = "auto"
    #: ``accelerate`` device map, or ``"cpu"`` to load without it. ``None`` leaves the
    #: model on whatever device the caller places it on.
    device_map: str | None = "auto"
    quantization: str | None = None
    max_memory: dict[str, str] | None = None
    offload_folder: str | None = None
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    #: Where to look for weights. Defaults to ``model``; set to a local directory to load
    #: an already-downloaded checkpoint without touching the network.
    local_path: str | None = None

    @property
    def source(self) -> str:
        return self.local_path or self.model

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.model:
            problems.append("model must be a non-empty repo id or path")
        if self.quantization is not None and self.quantization not in QUANTIZATION_SCHEMES:
            problems.append(
                f"unknown quantization {self.quantization!r}; known: "
                f"{', '.join(sorted(QUANTIZATION_SCHEMES))}"
            )
        if self.quantization is not None and self.device_map is None:
            problems.append("quantization requires a device_map (bitsandbytes places weights)")
        if self.offload_folder and self.device_map is None:
            problems.append("offload_folder has no effect without a device_map")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model, "revision": self.revision, "dtype": self.dtype,
            "device_map": self.device_map, "quantization": self.quantization,
            "max_memory": self.max_memory, "offload_folder": self.offload_folder,
            "trust_remote_code": self.trust_remote_code,
            "attn_implementation": self.attn_implementation,
            "local_path": self.local_path, "source": self.source,
        }


@dataclass
class LoadReport:
    """What actually happened during loading, including what did not load."""

    model_class: str = ""
    tokenizer_class: str = ""
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    mismatched_keys: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    load_seconds: float = 0.0
    #: Unexpected keys that are *expected* to be unexpected: the vision tower and the MTP
    #: head, which the causal-LM class discards by design.
    ignored_unexpected: list[str] = field(default_factory=list)

    @property
    def weights_complete(self) -> bool:
        return not self.missing_keys and not self.mismatched_keys

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_class": self.model_class, "tokenizer_class": self.tokenizer_class,
            "n_missing": len(self.missing_keys), "n_unexpected": len(self.unexpected_keys),
            "n_mismatched": len(self.mismatched_keys),
            "n_ignored_unexpected": len(self.ignored_unexpected),
            "missing_keys": self.missing_keys[:20],
            "unexpected_keys": self.unexpected_keys[:20],
            "mismatched_keys": self.mismatched_keys[:20],
            "error_messages": self.error_messages,
            "weights_complete": self.weights_complete,
            "load_seconds": round(self.load_seconds, 2),
        }


@dataclass
class TokenizerFacts:
    """What the tokenizer actually does, read off the loaded object rather than assumed."""

    tokenizer_class: str
    vocab_size: int
    model_max_length: int
    bos_token: str | None
    bos_token_id: int | None
    eos_token: str | None
    eos_token_id: int | None
    pad_token: str | None
    pad_token_id: int | None
    adds_bos: bool
    think_open_id: int | None
    think_close_id: int | None
    #: Whether ``</think>`` is a single token. When it is, thinking/answer token counts
    #: are exact; when it is not, they can only be approximated and that is recorded.
    exact_reasoning_split: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_class": self.tokenizer_class, "vocab_size": self.vocab_size,
            "model_max_length": self.model_max_length,
            "bos_token": self.bos_token, "bos_token_id": self.bos_token_id,
            "eos_token": self.eos_token, "eos_token_id": self.eos_token_id,
            "pad_token": self.pad_token, "pad_token_id": self.pad_token_id,
            "adds_bos": self.adds_bos,
            "think_open_id": self.think_open_id, "think_close_id": self.think_close_id,
            "exact_reasoning_split": self.exact_reasoning_split,
        }


def describe_tokenizer(tokenizer: Any) -> TokenizerFacts:
    """Read the tokenizer's real behaviour, including whether it prepends a BOS.

    ``adds_bos`` is *measured* by encoding a probe string rather than read from a config
    flag, because the flag and the behaviour can disagree and the alignment of every
    teacher signal depends on the behaviour.
    """
    probe = tokenizer("a", add_special_tokens=True)["input_ids"]
    bare = tokenizer("a", add_special_tokens=False)["input_ids"]
    adds_bos = len(probe) > len(bare) and probe[0] != bare[0]

    def token_id(token: str) -> int | None:
        try:
            resolved = tokenizer.convert_tokens_to_ids(token)
        except Exception:  # noqa: BLE001 - tokenizer-specific; absence is not an error
            return None
        if not isinstance(resolved, int) or resolved < 0:
            return None
        if resolved == getattr(tokenizer, "unk_token_id", None):
            return None
        return resolved

    close_id = token_id(THINK_CLOSE_TOKEN)
    return TokenizerFacts(
        tokenizer_class=type(tokenizer).__name__,
        vocab_size=len(tokenizer),
        model_max_length=int(getattr(tokenizer, "model_max_length", 0)),
        bos_token=getattr(tokenizer, "bos_token", None),
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token=getattr(tokenizer, "eos_token", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token=getattr(tokenizer, "pad_token", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        adds_bos=adds_bos,
        think_open_id=token_id(THINK_OPEN_TOKEN),
        think_close_id=close_id,
        exact_reasoning_split=close_id is not None,
    )


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
#: Unexpected-key patterns the causal-LM class discards by design. Matching these is not
#: a problem; anything else unexpected is reported so a checkpoint change is visible.
BENIGN_UNEXPECTED_PREFIXES = ("model.visual.", "visual.", "mtp.", "model.mtp.")


def _is_benign_unexpected(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in BENIGN_UNEXPECTED_PREFIXES)


def _tied_keys(model: Any) -> set[str]:
    """Keys the model itself declares tied, which may legitimately be absent."""
    declared = getattr(type(model), "_tied_weights_keys", None) or {}
    if isinstance(declared, dict):
        return set(declared)
    return set(declared)


@dataclass
class LoadedTeacher:
    """A real teacher, its tokenizer, and the evidence that it loaded correctly."""

    model: Any
    tokenizer: Any
    plan: TeacherLoadPlan
    report: LoadReport
    tokenizer_facts: TokenizerFacts
    identity: TeacherIdentity
    architecture: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, str | None] = field(default_factory=dict)

    @property
    def device(self) -> Any:
        return self.model.device

    def describe(self) -> dict[str, Any]:
        """Everything needed to reproduce or audit an operation by this teacher."""
        return {
            "backend": "transformers",
            "is_synthetic": False,
            "plan": self.plan.to_dict(),
            "identity": self.identity.to_dict(),
            "load_report": self.report.to_dict(),
            "tokenizer": self.tokenizer_facts.to_dict(),
            "architecture": self.architecture,
            "versions": self.versions,
            "device": str(self.device),
            "dtype": str(getattr(self.model, "dtype", "unknown")),
        }

    def unload(self) -> None:
        """Release the weights. Needed before a student is loaded in the same process."""
        self.model = None
        self.tokenizer = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _quantization_config(scheme: str | None) -> Any:
    if scheme is None:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise TeacherLoadError(
            f"quantization={scheme!r} needs a transformers build exposing "
            "BitsAndBytesConfig, and bitsandbytes itself (pip install bitsandbytes)."
        ) from exc
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise TeacherLoadError(
            f"quantization={scheme!r} requires bitsandbytes (pip install bitsandbytes). "
            "Load without quantization, or on hardware with enough memory for bf16."
        ) from exc
    import torch

    if scheme == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def teacher_identity(plan: TeacherLoadPlan) -> TeacherIdentity:
    """Hash the files that decide how a prompt is rendered and tokenised.

    Prefers the checkpoint being loaded; falls back to the vendored metadata so an
    identity is always produced, and records which source it came from.
    """
    source = Path(plan.local_path) if plan.local_path else VENDORED_METADATA
    if not (source / "config.json").exists() and VENDORED_METADATA.exists():
        source = VENDORED_METADATA
    return TeacherIdentity.from_metadata_dir(plan.model, source, revision=plan.revision)


def verify_architecture(config: Any) -> list[str]:
    """Compare the loaded config against the architecture this project verified.

    A mismatch does not mean the model is broken; it means it is **not the model this
    project's analysis, transfer plans and memory estimates were built against**, which is
    worse than broken because everything downstream would still run.
    """
    text = getattr(config, "text_config", config)
    problems = []
    for key, expected in EXPECTED_ARCHITECTURE.items():
        actual = getattr(text, key, None)
        if actual is None:
            actual = getattr(config, key, None)
        if actual is None and key == "full_attention_interval":
            # Derivable from the expanded layout, and worth deriving: a config that stores
            # `layer_types` explicitly need not keep the interval that generated it, and a
            # future normalisation upstream would otherwise fail this check spuriously on
            # a checkpoint that is in fact correct.
            actual = _interval_from_layer_types(getattr(text, "layer_types", None))
        if actual is None:
            problems.append(f"{key}: absent from the loaded config (expected {expected})")
        elif int(actual) != expected:
            problems.append(f"{key}: loaded {actual}, expected {expected}")
    return problems


def _interval_from_layer_types(layer_types: Any) -> int | None:
    """The hybrid period implied by an expanded ``layer_types`` list, if it is periodic."""
    if not layer_types:
        return None
    positions = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
    if not positions:
        return None
    interval = positions[0] + 1
    expected = list(range(interval - 1, len(layer_types), interval))
    return interval if positions == expected else None


def load_verified_teacher(
    plan: TeacherLoadPlan, *, strict_architecture: bool = True
) -> LoadedTeacher:
    """Load the teacher, or raise saying precisely what was wrong.

    The gate that matters: a non-empty ``missing_keys`` is fatal. `transformers` returns a
    model with freshly-initialised weights in that case and only prints a report, so
    without this check a 27B teacher can be "loaded" and generate fluent nonsense.
    """
    problems = plan.validate()
    if problems:
        raise TeacherLoadError("invalid teacher load plan:\n  - " + "\n  - ".join(problems))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    report = LoadReport()
    started = time.perf_counter()

    common: dict[str, Any] = {"trust_remote_code": plan.trust_remote_code}
    if plan.revision is not None:
        common["revision"] = plan.revision

    try:
        tokenizer = AutoTokenizer.from_pretrained(plan.source, **common)
    except Exception as exc:  # noqa: BLE001 - the failure is the result
        raise TeacherLoadError(
            f"could not load the tokenizer for {plan.source!r}: {type(exc).__name__}: {exc}\n"
            "The distillation path needs the teacher's own tokenizer — the vendored "
            "metadata carries tokenizer_config.json but no tokenizer.json, so it cannot "
            "substitute."
        ) from exc

    # "auto" must reach transformers verbatim; anything else names a torch dtype.
    dtype = plan.dtype if plan.dtype == "auto" else getattr(torch, plan.dtype)
    kwargs: dict[str, Any] = {**common, "dtype": dtype, "output_loading_info": True}
    if plan.device_map is not None and plan.device_map != "cpu":
        try:
            import accelerate  # noqa: F401
        except ImportError as exc:
            raise TeacherLoadError(
                f"device_map={plan.device_map!r} requires accelerate "
                "(pip install accelerate). Use device_map='cpu' to load without it."
            ) from exc
        kwargs["device_map"] = plan.device_map
    if plan.max_memory:
        kwargs["max_memory"] = plan.max_memory
    if plan.offload_folder:
        kwargs["offload_folder"] = plan.offload_folder
    if plan.attn_implementation:
        kwargs["attn_implementation"] = plan.attn_implementation
    quantization = _quantization_config(plan.quantization)
    if quantization is not None:
        kwargs["quantization_config"] = quantization

    try:
        loaded = AutoModelForCausalLM.from_pretrained(plan.source, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the failure is the result
        raise TeacherLoadError(
            f"could not load {plan.source!r}: {type(exc).__name__}: {exc}"
        ) from exc

    model, info = loaded if isinstance(loaded, tuple) else (loaded, {})
    report.load_seconds = time.perf_counter() - started
    report.model_class = type(model).__name__
    report.tokenizer_class = type(tokenizer).__name__

    tied = _tied_keys(model)
    unexpected = sorted(info.get("unexpected_keys", []) or [])
    report.missing_keys = sorted(k for k in (info.get("missing_keys", []) or []) if k not in tied)
    report.unexpected_keys = [k for k in unexpected if not _is_benign_unexpected(k)]
    report.ignored_unexpected = [k for k in unexpected if _is_benign_unexpected(k)]
    report.mismatched_keys = sorted(str(k) for k in (info.get("mismatched_keys", []) or []))
    report.error_messages = [str(m) for m in (info.get("error_msgs", []) or [])]

    if not report.weights_complete:
        raise TeacherLoadError(
            f"{plan.source!r} loaded with {len(report.missing_keys)} missing and "
            f"{len(report.mismatched_keys)} mismatched tensor(s), which means those "
            "weights are RANDOM.\n"
            f"  first missing: {report.missing_keys[:5]}\n"
            f"  first mismatched: {report.mismatched_keys[:5]}\n"
            "transformers does not raise on this — it returns a freshly-initialised model "
            "and prints a report — so a teacher loaded this way would generate fluent "
            "nonsense and distil a student that learns nothing. Refusing.\n"
            "Check that the revision matches the config, that the download completed, and "
            "that the checkpoint is the text-tower-compatible one."
        )

    model.eval()
    architecture_problems = verify_architecture(model.config)
    if architecture_problems and strict_architecture:
        raise TeacherLoadError(
            f"{plan.source!r} is not the architecture this project verified against:\n  - "
            + "\n  - ".join(architecture_problems)
            + "\nEvery transfer plan, memory estimate and parameter count in this "
            "repository was derived from the expected architecture. Pass "
            "strict_architecture=False to load a variant deliberately."
        )

    facts = describe_tokenizer(tokenizer)
    text_config = getattr(model.config, "text_config", model.config)
    architecture = {
        "verified_against_expected": not architecture_problems,
        "differences": architecture_problems,
        "vocab_size": getattr(text_config, "vocab_size", None),
        "hidden_size": getattr(text_config, "hidden_size", None),
        "num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
        "config_class": type(model.config).__name__,
        "declared_architectures": getattr(model.config, "architectures", None),
    }
    if facts.vocab_size != architecture["vocab_size"]:
        # Not fatal — tokenizers commonly carry fewer entries than the padded embedding —
        # but a *larger* tokenizer would index past the embedding, so it is checked.
        if architecture["vocab_size"] is not None and facts.vocab_size > architecture["vocab_size"]:
            raise TeacherLoadError(
                f"the tokenizer has {facts.vocab_size} entries but the model embedding has "
                f"{architecture['vocab_size']}: token ids would index out of range."
            )
        architecture["tokenizer_vocab_note"] = (
            f"tokenizer has {facts.vocab_size} entries against an embedding of "
            f"{architecture['vocab_size']}; the surplus rows are padding, which is normal"
        )

    return LoadedTeacher(
        model=model, tokenizer=tokenizer, plan=plan, report=report,
        tokenizer_facts=facts, identity=teacher_identity(plan),
        architecture=architecture, versions=package_versions(),
    )


# ---------------------------------------------------------------------------
# prompt rendering
# ---------------------------------------------------------------------------
class TemplateRejectedMode(TeacherLoadError):
    """The chat template would not accept a reasoning mode's kwargs."""


def render_prompt(
    loaded: LoadedTeacher,
    prompt: str,
    *,
    mode: ReasoningMode,
    system_prompt: str | None = None,
) -> str:
    """Render the chat template with the mode's controls, or raise.

    Deliberately **not** tolerant. ``evaluation.runner`` catches ``TypeError`` here and
    re-renders without the controls, which is right for a survey backend that wants to
    measure whether a control does anything. It is wrong for the teacher: silently
    dropping ``reasoning_effort`` would produce records labelled with a mode the prompt
    never carried, and the reasoning-cost comparison this project exists to make would be
    measuring nothing.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = mode.template_kwargs()
    try:
        return loaded.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    except TypeError as exc:
        raise TemplateRejectedMode(
            f"the chat template rejected the kwargs for reasoning mode {mode.name!r} "
            f"({kwargs}): {exc}. The teacher will not fall back to an uncontrolled prompt, "
            "because the record would then claim a mode the prompt never carried."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - jinja raises its own types
        raise TemplateRejectedMode(
            f"the chat template raised while rendering reasoning mode {mode.name!r} "
            f"({kwargs}): {type(exc).__name__}: {exc}"
        ) from exc


def mode_changes_the_prompt(loaded: LoadedTeacher, probe: str = "Hello.") -> dict[str, Any]:
    """Whether each reasoning mode actually renders a distinct prompt.

    A control that leaves the prompt byte-identical is a control that does nothing, and
    that is far better discovered here than inferred from flat token counts after a
    generation run. Returns the rendered hash per mode plus which modes collide.
    """
    from .reasoning_modes import sweep_modes

    rendered = {}
    for mode in sweep_modes():
        text = render_prompt(loaded, probe, mode=mode)
        rendered[mode.name] = {"sha256": sha256_text(text), "length": len(text)}
    by_hash: dict[str, list[str]] = {}
    for name, info in rendered.items():
        by_hash.setdefault(info["sha256"], []).append(name)
    collisions = [names for names in by_hash.values() if len(names) > 1]
    return {"rendered": rendered, "collisions": collisions, "all_distinct": not collisions}


# ---------------------------------------------------------------------------
# generation and logits
# ---------------------------------------------------------------------------
def split_generated_tokens(
    token_ids: Any, close_id: int | None
) -> tuple[int, int, str]:
    """Split generated token ids at ``</think>``: (thinking, answer, method).

    Splitting on the ids is exact. Re-tokenising decoded strings is not — whitespace and
    special-token handling mean the parts need not sum to the whole — so when the closing
    tag is not a single token the method is recorded as approximate rather than the counts
    being presented as if they were exact. Nothing here infers a split from whitespace.
    """
    total = int(token_ids.shape[-1])
    if close_id is None:
        return 0, total, "no </think> token in this tokenizer; all tokens counted as answer"
    positions = (token_ids == close_id).nonzero()
    if positions.numel() == 0:
        return 0, total, "exact (token ids); no </think> emitted, so nothing was reasoning"
    first = positions[0]
    cut = int(first.item()) if first.dim() == 0 else int(first[0].item())
    return cut, max(0, total - cut - 1), "exact (token ids split at </think>)"


@dataclass
class TeacherGeneration:
    """One real generation, with the ids kept so nothing downstream has to re-tokenise."""

    prompt_text: str
    rendered_prompt: str
    thinking: str
    answer: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    thinking_tokens: int
    answer_tokens: int
    finish_reason: str
    latency_s: float
    token_counting_method: str

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def total_generated_tokens(self) -> int:
        return len(self.generated_token_ids)


def generate_once(
    loaded: LoadedTeacher,
    prompt: str,
    *,
    mode: ReasoningMode,
    system_prompt: str | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    seed: int = 0,
) -> TeacherGeneration:
    """Generate one real response, returning ids as well as text."""
    import torch

    if loaded.model is None:
        raise TeacherNotLoaded("the teacher has been unloaded")

    rendered = render_prompt(loaded, prompt, mode=mode, system_prompt=system_prompt)
    encoded = loaded.tokenizer(rendered, return_tensors="pt")
    device = loaded.model.device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_ids = encoded["input_ids"][0]

    do_sample = temperature > 0
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": loaded.tokenizer.pad_token_id or loaded.tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

    torch.manual_seed(seed)
    started = time.perf_counter()
    with torch.no_grad():
        output = loaded.model.generate(**encoded, **gen_kwargs)
    latency = time.perf_counter() - started

    new_tokens = output[0][prompt_ids.shape[-1]:]
    close_id = loaded.tokenizer_facts.think_close_id
    thinking_tokens, answer_tokens, method = split_generated_tokens(new_tokens, close_id)

    decoded = loaded.tokenizer.decode(new_tokens, skip_special_tokens=True)
    if close_id is not None and thinking_tokens:
        thinking = loaded.tokenizer.decode(new_tokens[:thinking_tokens], skip_special_tokens=True)
        answer = loaded.tokenizer.decode(new_tokens[thinking_tokens + 1:], skip_special_tokens=True)
    else:
        thinking, answer = "", decoded

    return TeacherGeneration(
        prompt_text=prompt, rendered_prompt=rendered,
        thinking=thinking, answer=answer,
        prompt_token_ids=prompt_ids.tolist(),
        generated_token_ids=new_tokens.tolist(),
        thinking_tokens=thinking_tokens, answer_tokens=answer_tokens,
        finish_reason="length" if int(new_tokens.shape[-1]) >= max_new_tokens else "stop",
        latency_s=latency, token_counting_method=method,
    )


def teacher_logits(loaded: LoadedTeacher, input_ids: Any) -> Any:
    """Full logits for a batch of token ids, under ``no_grad``.

    ``input_ids`` must already be the ids the *student* will see, so that position ``t``
    of the returned logits is the teacher's prediction for the same token the student
    predicts at ``t``. Nothing is re-tokenised here; alignment is the caller's to preserve
    and this signature is what makes that possible.
    """
    import torch

    if loaded.model is None:
        raise TeacherNotLoaded("the teacher has been unloaded")
    if input_ids.dim() != 2:
        raise ValueError(
            f"expected input_ids of shape (batch, positions), got {tuple(input_ids.shape)}"
        )
    with torch.no_grad():
        return loaded.model(input_ids=input_ids.to(loaded.model.device)).logits


def teacher_memory_estimate(context: int = 4096, quantization: str | None = None) -> dict[str, Any]:
    """What loading this teacher costs, from the project's own memory model.

    Reuses :mod:`qwen_distill.architecture.memory` rather than restating the arithmetic, so
    the teacher and the students are sized by the same estimator.
    """
    from ..architecture.memory import GIB, DeploymentConfig, estimate_memory
    from ..architecture.spec import HybridArchSpec

    quant = {None: "bf16", "8bit": "int8", "4bit": "int4"}[quantization]
    spec = HybridArchSpec(name=DEFAULT_TEACHER_MODEL)
    estimate = estimate_memory(
        spec, DeploymentConfig(context_length=context, weight_quant=quant)
    )
    return {
        "measured": False,
        "basis": (
            "analytical estimate from qwen_distill.architecture.memory; NOT measured on "
            "hardware. Treat as sizing guidance, not a hardware requirement."
        ),
        "quantization": quantization or "bf16",
        "context": context,
        # Reported separately because they scale differently and are confirmed
        # differently: weights are arithmetic, the cache grows with context, and the
        # runtime overhead is the term an analytical model is least able to predict.
        "weights_gib": round(estimate.weights / GIB, 2),
        "kv_cache_gib": round(estimate.kv_cache / GIB, 3),
        "recurrent_state_gib": round(estimate.recurrent_state / GIB, 3),
        "activations_gib": round(estimate.activations / GIB, 3),
        "runtime_overhead_gib": round(estimate.runtime_overhead / GIB, 3),
        "total_gib": round(estimate.total_gib, 2),
        "note": QUANTIZATION_SCHEMES.get(quantization or "", "unquantised bf16 weights"),
        "student_objective_note": (
            "The teacher's footprint does not constrain the 16 GB student target. The "
            "teacher runs once, on larger rented hardware, to produce the distillation "
            "signal; only the student has to fit 16 GB."
        ),
    }
