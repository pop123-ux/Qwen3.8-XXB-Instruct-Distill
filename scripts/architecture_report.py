#!/usr/bin/env python3
"""Can this architecture be served on 16 GB — and is it worth what it costs?

The project's destination is a Qwen3.8-27B alternative that runs locally on **16 GB of
VRAM**, with a credible path to **12 GB**. This is the command that answers, before any
GPU hours are spent, whether a candidate architecture belongs on that path.

Three modes:

``--presets`` / ``--spec``
    assess named or derived architectures: parameters, weights, KV cache, DeltaNet state,
    activations and overhead, at every precision, across the 4K-256K context ladder, for
    both targets.

``--sweep``
    several architectures side by side, with training memory too, so a candidate that
    cannot be trained on the primary target is rejected in milliseconds rather than
    discovered eight hours in.

``--summary``
    the ARCHITECTURE RESEARCH SUMMARY: what each completed experiment measured, what the
    step between two of them bought, and what it cost to serve. Reads committed
    ``RESULT.json`` records — it measures nothing itself and invents nothing.

Every memory figure is an **estimate**, from the same analytical model the rest of the
project uses. It is not a benchmark. Confirm a candidate with
``scripts/benchmark_memory.py`` on the real card before committing hours to it.

Examples::

    python scripts/architecture_report.py --presets level2r level3
    python scripts/architecture_report.py --sweep level2r level3 teacher
    python scripts/architecture_report.py --spec level3:hidden_size=1280,num_hidden_layers=20
    python scripts/architecture_report.py --summary \\
        experiments/runs/t4_level2r_100m_real_english \\
        experiments/runs/t4_level3_236m_real_english
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.analysis.compare import load_run_facts
from qwen_distill.analysis.deployment import (
    PRIMARY_TARGET,
    SECONDARY_TARGET,
    assess,
    sweep,
)
from qwen_distill.analysis.returns import build_step, research_summary
from qwen_distill.architecture.presets import (
    derive,
    get_preset,
    get_spec,
    preset_names,
)

RULE = "=" * 78


def _parse_spec(token: str):
    """``base:field=value,...`` — a derived architecture in one argument.

    Deliberately terse because the point is a one-line diff against a known baseline:
    ``level3:hidden_size=1280`` says exactly what differs and from what.
    """
    base, _, changes = token.partition(":")
    if base not in preset_names():
        raise SystemExit(f"unknown preset {base!r}; known: {', '.join(preset_names())}")
    if not changes:
        return base, get_spec(base)
    edits: dict[str, object] = {}
    for pair in changes.split(","):
        key, _, value = pair.partition("=")
        key, value = key.strip(), value.strip()
        if not key or not value:
            raise SystemExit(f"could not parse {pair!r}; expected field=value")
        if value.lower() in ("true", "false"):
            edits[key] = value.lower() == "true"
        else:
            try:
                edits[key] = int(value)
            except ValueError:
                edits[key] = value
    name = f"{base}+" + ",".join(f"{k}={v}" for k, v in edits.items())
    return name, derive(base, name=name, **edits)


def _collect(tokens: list[str]) -> dict[str, object]:
    return dict(_parse_spec(token) for token in tokens)


def _render_assessment(name: str, spec) -> None:
    result = assess(spec, name=name)
    print(f"\n{RULE}\n{name}\n{RULE}")
    print(f"  parameters        : {result.parameters:,}")
    print(f"  non-embedding     : {result.non_embedding_parameters:,}")
    print(f"  embedding         : {result.embedding_parameters:,}")
    arch = result.architecture
    print(f"  hidden / layers   : {arch['hidden_size']} / {arch['num_layers']}")
    print(f"  layout            : {arch['deltanet_layers']} DeltaNet + "
          f"{arch['attention_layers']} attention  ({arch['deltanet_to_attention']}:1)")
    print(f"  FFN               : {arch['intermediate_size']} "
          f"({arch['ffn_expansion']}x expansion)")
    print(f"  attention heads   : {arch['attention_heads']} q / {arch['kv_heads']} kv, "
          f"head_dim {arch['head_dim']}")
    print(f"  DeltaNet heads    : {arch['deltanet_key_heads']} key / "
          f"{arch['deltanet_value_heads']} value")
    print(f"  vocab / context   : {arch['vocab_size']} / {arch['context_length']:,}")

    for target in (PRIMARY_TARGET, SECONDARY_TARGET):
        print(f"\n  {target.name} ({target.priority}) — {target.usable_gib} GiB usable")
        print(f"    {'precision':<10}{'ctx':>10}{'total':>9}{'weights':>9}{'KV':>8}"
              f"{'state':>8}{'act':>8}   verdict")
        for entry in result.feasibility:
            if entry.target.name != target.name:
                continue
            for cell in entry.contexts:
                print(f"    {entry.precision:<10}{cell.context_length:>10,}"
                      f"{cell.total_gib:>9.2f}{cell.weights_gib:>9.2f}"
                      f"{cell.kv_cache_gib:>8.2f}{cell.state_gib:>8.2f}"
                      f"{cell.activations_gib:>8.2f}   {cell.verdict}")
        best = result.best_precision_for(target.name)
        if best and best.max_fitting_context:
            print(f"    -> best: {best.precision} up to {best.max_fitting_context:,} context")
        else:
            print("    -> DOES NOT FIT at any precision or context")
    for note in result.notes:
        print(f"\n  ! {note}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--presets", nargs="+", metavar="SPEC",
                        help="architectures to assess in detail (preset, or base:field=value,...)")
    parser.add_argument("--spec", nargs="+", metavar="SPEC", dest="presets",
                        help=argparse.SUPPRESS)
    parser.add_argument("--sweep", nargs="+", metavar="SPEC",
                        help="architectures to compare side by side")
    parser.add_argument("--summary", nargs="+", type=Path, metavar="RUN",
                        help="run directories with RESULT.json, oldest first")
    parser.add_argument("--list", action="store_true", help="list the known presets")
    parser.add_argument("--json", type=Path, help="write the full report here")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload: dict[str, object] = {}

    if args.list:
        print(f"{RULE}\nARCHITECTURE PRESETS\n{RULE}\n")
        for name in preset_names():
            preset = get_preset(name)
            print(f"  {name:<12} {preset.parameters:>15,}  [{preset.kind}]")
            print(f"  {'':<12} {preset.summary}")
            if preset.config:
                print(f"  {'':<12} {preset.config}")
            print()
        print("  Derive a variant with base:field=value, e.g. level3:hidden_size=1280")
        print("  Presets are experiments that RAN, plus the teacher. Future architectures")
        print("  are derived, not added here until they have a result.")
        return 0

    if not (args.presets or args.sweep or args.summary):
        parser.error("one of --presets, --sweep, --summary or --list is required")

    if args.presets:
        specs = _collect(args.presets)
        for name, spec in specs.items():
            _render_assessment(name, spec)
        payload["assessments"] = {
            name: assess(spec, name=name).to_dict() for name, spec in specs.items()
        }

    if args.sweep:
        result = sweep(_collect(args.sweep))
        print()
        print(result.render())
        payload["sweep"] = result.to_dict()

    if args.summary:
        status = _render_summary(args.summary, payload)
        if status:
            return status

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                             encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


def _render_summary(runs: list[Path], payload: dict[str, object]) -> int:
    """The research summary, built only from what the run records actually contain."""
    facts, rungs = [], []
    for run in runs:
        if not run.exists():
            print(f"no such run directory: {run}", file=sys.stderr)
            return 2
        try:
            record = load_run_facts(run)
        except (OSError, ValueError, KeyError) as exc:
            print(f"could not read {run}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        facts.append(record)

        rung: dict[str, object] = {
            "name": record.name,
            "parameters": record.get("parameters"),
            "validation_bits_per_byte": record.get("validation_bits_per_byte"),
            "mean_repeated_3gram": record.get("mean_repeated_3gram"),
            "status": record.status,
        }
        # Deployment cost is estimated from the architecture the record names, when that
        # architecture is one this repository knows. It is never guessed from parameters.
        matched = next(
            (n for n in preset_names()
             if get_preset(n).experiment and Path(get_preset(n).experiment).name == run.name),
            None,
        )
        if matched:
            estimate = assess(get_spec(matched), name=matched)
            best16 = estimate.best_precision_for(PRIMARY_TARGET.name)
            best12 = estimate.best_precision_for(SECONDARY_TARGET.name)
            rung["inference_gib"] = (
                best16.contexts[0].total_gib if best16 and best16.contexts else None
            )
            rung["status_16gb"] = best16.verdict if best16 else "DOES NOT FIT"
            rung["max_context_16gb"] = best16.max_fitting_context if best16 else None
            rung["status_12gb"] = best12.verdict if best12 else "DOES NOT FIT"
            rung["max_context_12gb"] = best12.max_fitting_context if best12 else None
        rungs.append(rung)

    # Consecutive pairs. Indexed rather than zipped: the offset lists are deliberately
    # different lengths, and a single run must produce zero steps rather than raise.
    steps = [
        build_step(
            facts[i], facts[i + 1],
            baseline_inference_gib=rungs[i].get("inference_gib"),
            candidate_inference_gib=rungs[i + 1].get("inference_gib"),
        )
        for i in range(len(facts) - 1)
    ]

    open_questions = []
    if any(r.get("validation_bits_per_byte") is None for r in rungs):
        open_questions.append(
            "at least one run has no validation bits/byte in its record — the primary "
            "capability comparison cannot be made until it does"
        )
    if not steps:
        open_questions.append("only one run supplied; a step needs two")

    summary = research_summary(rungs, steps, open_questions)
    print()
    print(summary.render())
    payload["research_summary"] = summary.to_dict()
    return 0


if __name__ == "__main__":
    sys.exit(main())
