"""Build a tiny but complete Qwen3.5-family checkpoint on disk, offline.

Every piece of infrastructure in this repository is meant to run against a real
teacher checkpoint we do not currently have. That makes it easy to ship code that
has never executed. This module removes that excuse: it writes a miniature but
*structurally faithful* checkpoint — hybrid DeltaNet/attention layout, GQA, a real
tokenizer, and a chat template with Qwen-style reasoning controls — so the loader,
the evaluator, and the reasoning benchmark can all be exercised end to end with no
network access and no GPU.

The weights are random, so generations are gibberish. That is fine: these fixtures
validate *plumbing*, not capability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .architecture.spec import HybridArchSpec

#: A small spec that keeps every structural feature that matters: a 3:1 hybrid
#: layout, grouped-query attention, and DeltaNet value heads outnumbering key heads.
TINY_SPEC = HybridArchSpec(
    name="tiny-hybrid",
    hidden_size=128,
    num_hidden_layers=4,
    intermediate_size=256,
    vocab_size=512,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    full_attention_interval=4,
    max_position_embeddings=4096,
    notes="Synthetic fixture for offline testing. Random weights.",
)

#: A chat template modelled on the Qwen family's reasoning controls, including a
#: `medium` branch that deliberately injects nothing — mirroring the behaviour
#: reported upstream so `benchmark_reasoning.py` can be shown to detect a no-op.
CHAT_TEMPLATE = """{%- for message in messages %}
{%- if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% endif %}
{%- if message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant
{%- if enable_thinking is defined and not enable_thinking %}
{%- else %}
{%- if reasoning_effort is defined and reasoning_effort == 'xhigh' %}
Check your assumptions and consider alternatives carefully.
{%- elif reasoning_effort is defined and reasoning_effort == 'low' %}
Think briefly.
{%- endif %}
<think>
{%- endif %}
{%- endif %}"""


def build_tokenizer(vocab_size: int = 512):
    """Create a minimal byte-level BPE tokenizer with Qwen-style special tokens.

    Built programmatically rather than downloaded, so it works with no network.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    vocab: dict[str, int] = {tok: i for i, tok in enumerate(specials)}
    # Single characters cover any input; unknown bytes fall back to <|endoftext|>.
    for code in range(32, 127):
        char = chr(code)
        if char not in vocab:
            vocab[char] = len(vocab)
    while len(vocab) < vocab_size:
        vocab[f"<unused{len(vocab)}>"] = len(vocab)

    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="<|endoftext|>", max_input_chars_per_word=1))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    return tokenizer


def write_tiny_checkpoint(
    path: str | Path,
    spec: HybridArchSpec = TINY_SPEC,
    *,
    with_weights: bool = True,
    with_mtp: bool = False,
    seed: int = 0,
) -> Path:
    """Write a loadable miniature checkpoint to ``path`` and return the directory.

    With ``with_weights=False`` only config/tokenizer files are written, which is
    enough to exercise config-only inspection paths.
    """
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)

    text_config = {k: v for k, v in spec.to_hf_text_config().items() if k != "model_type"}
    config: dict[str, Any] = {
        "model_type": "qwen3_5_text",
        "architectures": ["Qwen3_5ForCausalLM"],
        "dtype": "float32",
        **text_config,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    (root / "config.json").write_text(json.dumps(config, indent=2))
    (root / "generation_config.json").write_text(
        json.dumps({"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}, indent=2)
    )
    (root / "chat_template.jinja").write_text(CHAT_TEMPLATE)

    tokenizer = build_tokenizer(spec.vocab_size)
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<|im_start|>",
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
                "model_max_length": spec.max_position_embeddings,
                "chat_template": CHAT_TEMPLATE,
            },
            indent=2,
        )
    )
    (root / "special_tokens_map.json").write_text(
        json.dumps(
            {"bos_token": "<|im_start|>", "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"},
            indent=2,
        )
    )

    if with_weights:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoConfig, AutoModelForCausalLM

        torch.manual_seed(seed)
        hf_config = AutoConfig.for_model("qwen3_5_text", **text_config)
        model = AutoModelForCausalLM.from_config(hf_config)
        state = {k: v.contiguous() for k, v in model.state_dict().items()}
        if with_mtp:
            # Mirror the real checkpoint layout: MTP tensors sit under an `mtp.` prefix
            # and stock transformers discards them on load.
            h = spec.hidden_size
            state["mtp.fc.weight"] = torch.randn(h, 2 * h)
            state["mtp.norm.weight"] = torch.randn(h)
            state["mtp.pre_fc_norm_hidden.weight"] = torch.randn(h)
            state["mtp.pre_fc_norm_embedding.weight"] = torch.randn(h)
        save_file(state, str(root / "model.safetensors"), metadata={"format": "pt"})

    return root
