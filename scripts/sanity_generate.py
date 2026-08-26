#!/usr/bin/env python3
"""Generate from a checkpoint and check for obvious degeneracy.

Level 2 finished with validation BPB 1.270 — a healthy-looking number — and generated
`"and and and and"`. The loss curve never warned. This looks at the output instead.

Not a benchmark. It answers one question: *is the model obviously broken?* Passing does
not establish language capability; failing establishes that something is wrong, cheaply
and early.

**Prompt set v2.0** — eleven prompts at two lengths. The six short ones leave the model
nearly unconstrained, which is where Level 2's collapse showed up fastest. Five longer
ones supply real syntactic context, so a model that learned local structure has something
to continue and one that learned only unigram frequencies has nowhere to hide::

    "The beginning of the story was"     "It was a"
    "In the middle of the"               "The most important thing"
    "Yesterday, I"

Every generation is recorded with what produced it — prompt, text, checkpoint, step,
decoding settings, token count (from the generated ids, not the decoded characters) and
timestamp — so a report can be reproduced or contradicted later. Passing ``--prompts``
marks the report ``custom``, because a pass rate over different prompts is a different
measurement.

Deterministic: greedy decoding, so two runs at one checkpoint produce identical text and
any change between checkpoints is a change in the model.

Examples::

    python scripts/sanity_generate.py experiments/runs/t4_level2r_100m_real_english
    python scripts/sanity_generate.py <checkpoint> --training-text data/level2r/train.txt
    python scripts/sanity_generate.py <checkpoint> --json sanity.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.training.sanity import SANITY_PROMPTS, run_sanity_checks

RULE = "=" * 72


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=Path,
                        help="checkpoint directory, or a run directory (resolves to latest)")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompts", nargs="+", default=list(SANITY_PROMPTS))
    parser.add_argument("--training-text", type=Path,
                        help="training corpus, to detect verbatim memorisation. Without "
                             "it the memorisation check does not run, and the report "
                             "says so rather than reporting 'not memorised'")
    parser.add_argument("--json", type=Path, help="write the report here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from qwen_distill.training.validate_checkpoint import (
        _build_from_config,
        resolve_checkpoint_argument,
    )

    checkpoint = resolve_checkpoint_argument(args.checkpoint)
    config_file = checkpoint / "config.json"
    if not config_file.is_file():
        print(f"no config.json in {checkpoint} — is that a checkpoint directory?",
              file=sys.stderr)
        return 2

    print(f"{RULE}\nGENERATION SANITY CHECK\n{RULE}\n")

    try:
        import torch
        from safetensors.torch import load_model

        model, _spec = _build_from_config(config_file, args.device)
        weights = checkpoint / "model.safetensors"
        if weights.is_file():
            load_model(model, str(weights), strict=False, device=args.device)
        else:
            legacy = checkpoint / "training_state.pt"
            if not legacy.is_file():
                print(f"no weights in {checkpoint}", file=sys.stderr)
                return 2
            payload = torch.load(legacy, map_location=args.device, weights_only=False)
            model.load_state_dict(payload["model"], strict=False)
    except Exception as exc:  # noqa: BLE001 - report rather than traceback
        print(f"could not load the checkpoint: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    training_text = None
    if args.training_text and args.training_text.is_file():
        # Only a prefix is needed: memorisation shows up in the opening characters, and
        # holding a 50 MB corpus in memory to check eleven generations is wasteful.
        training_text = args.training_text.read_text(
            encoding="utf-8", errors="replace"
        )[:20_000_000]

    step = None
    metadata = checkpoint / "metadata.json"
    if metadata.is_file():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            step = json.loads(metadata.read_text(encoding="utf-8")).get("step")

    report = run_sanity_checks(
        model, prompts=tuple(args.prompts), max_new_tokens=args.max_new_tokens,
        device=args.device, training_text=training_text,
        checkpoint=str(checkpoint), step=step,
    )
    print(report.render())

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
