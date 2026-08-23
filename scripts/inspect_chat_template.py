#!/usr/bin/env python3
"""Render the real chat template for representative conversations.

The template is never hand-written or reconstructed — it is read from the supplied
metadata and rendered by `transformers`, so what you see is what the model would
receive. Runs entirely offline against a local directory.

Renders: plain user, system + user, multi-turn, thinking disabled, thinking enabled,
each reasoning-effort level, and a tool-call conversation if the template supports one.

The reasoning-effort renderings are hashed and grouped. Two settings that produce
**byte-identical prompts** are indistinguishable at the template level — one of them
cannot change behaviour via the prompt, which is the reported `medium` behaviour. That
is a template-level proof; whether a control affects a *trained* model's behaviour by
another route still needs a runtime experiment.

Example::

    python scripts/inspect_chat_template.py --path vendor/qwen38-metadata
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.teacher.metadata import load_metadata
from qwen_distill.utils.hub import diagnose_hub_error
from qwen_distill.utils.offline import offline_mode

CONVERSATIONS: tuple[tuple[str, list[dict[str, str]]], ...] = (
    ("plain_user", [{"role": "user", "content": "What is 15 * 7?"}]),
    (
        "system_and_user",
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is 15 * 7?"},
        ],
    ),
    (
        "multi_turn",
        [
            {"role": "user", "content": "What is 15 * 7?"},
            {"role": "assistant", "content": "105."},
            {"role": "user", "content": "And times 2?"},
        ],
    ),
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--path", type=Path, default=Path("vendor/qwen38-metadata"))
    parser.add_argument(
        "--efforts", nargs="+", default=["low", "medium", "high", "xhigh"],
        help="reasoning_effort values to try (all are hypotheses until rendered)",
    )
    parser.add_argument("--show-full", action="store_true", help="print whole prompts")
    parser.add_argument("--json", type=Path)
    return parser


def _render(tokenizer, messages, **kwargs) -> tuple[str | None, str | None]:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        ), None
    except Exception as exc:  # noqa: BLE001 - a rejected kwarg is itself a finding
        return None, f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.path.is_dir():
        print(f"Metadata directory not found: {args.path}", file=sys.stderr)
        print("See vendor/README.md for how to supply it.", file=sys.stderr)
        return 2

    with offline_mode():
        metadata = load_metadata(args.path)
        if not metadata.chat_template:
            print(
                "No chat template found in the supplied metadata "
                "(looked in chat_template.jinja, chat_template.json and "
                "tokenizer_config.json).\nChat formatting stays UNKNOWN.",
                file=sys.stderr,
            )
            return 1

        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(args.path), local_files_only=True)
        except Exception as exc:  # noqa: BLE001
            print(diagnose_hub_error(exc, str(args.path)).render(), file=sys.stderr)
            print(
                "\nThe template text was found but no tokenizer could be built from the "
                "supplied files; tokenizer.json and tokenizer_config.json are both needed.",
                file=sys.stderr,
            )
            return 1

        results: dict[str, object] = {
            "path": str(args.path),
            "chat_template_source": metadata.chat_template_source,
            "chat_template_chars": len(metadata.chat_template),
            "conversations": {},
            "reasoning": {},
        }

        print(f"chat template source : {metadata.chat_template_source}")
        print(f"template length      : {len(metadata.chat_template):,} chars")
        print(f"tokenizer class      : {type(tokenizer).__name__}\n")

        print("=== conversation shapes ===")
        for name, messages in CONVERSATIONS:
            rendered, error = _render(tokenizer, messages)
            entry = {"error": error} if error else {
                "chars": len(rendered),
                "sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "rendered": rendered if args.show_full else rendered[:400],
            }
            results["conversations"][name] = entry
            print(f"\n--- {name} ---")
            print(error if error else (rendered if args.show_full else rendered[:400]))

        print("\n\n=== tool-call formatting ===")
        rendered, error = _render(tokenizer, CONVERSATIONS[0][1], tools=TOOLS)
        if error:
            print(f"  template rejected `tools`: {error}")
            results["conversations"]["tools"] = {"error": error}
        else:
            plain = _render(tokenizer, CONVERSATIONS[0][1])[0]
            supported = rendered != plain
            print(f"  tools change the prompt: {supported}")
            if not supported:
                print("  => the template accepts `tools` but ignores it; no tool formatting.")
            print(rendered if args.show_full else rendered[:400])
            results["conversations"]["tools"] = {
                "changes_prompt": supported,
                "chars": len(rendered),
                "rendered": rendered if args.show_full else rendered[:400],
            }

        print("\n\n=== reasoning controls ===")
        renderings: dict[str, str] = {}
        for label, kwargs in (
            ("(no control)", {}),
            ("thinking_disabled", {"enable_thinking": False}),
            ("thinking_enabled", {"enable_thinking": True}),
            *[(f"effort={e}", {"reasoning_effort": e}) for e in args.efforts],
        ):
            rendered, error = _render(tokenizer, CONVERSATIONS[0][1], **kwargs)
            if error:
                print(f"  {label:<22} REJECTED  {error[:80]}")
                results["reasoning"][label] = {"error": error}
                continue
            digest = hashlib.sha256(rendered.encode()).hexdigest()
            renderings[label] = digest
            print(f"  {label:<22} {digest[:12]}  {len(rendered):>6} chars")
            results["reasoning"][label] = {
                "sha256": digest, "chars": len(rendered),
                "rendered": rendered if args.show_full else rendered[:300],
            }

        groups: dict[str, list[str]] = {}
        for label, digest in renderings.items():
            groups.setdefault(digest, []).append(label)
        identical = [g for g in groups.values() if len(g) > 1]
        results["identical_groups"] = identical
        results["n_distinct_prompts"] = len(groups)

        print(f"\n  distinct prompts: {len(groups)} of {len(renderings)}")
        if identical:
            for group in identical:
                print(f"  IDENTICAL: {' == '.join(group)}")
            print("\n  These settings are indistinguishable at the template level: the model")
            print("  receives the same input, so one cannot change behaviour via the prompt.")
            print("  Whether it affects a trained model by another route needs a runtime test.")
        else:
            print("  every reasoning setting renders a distinct prompt")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
