#!/usr/bin/env python3
"""Build the paper figure set.

    python plots/make_figures.py                     # every figure, both profiles
    python plots/make_figures.py --status real       # only the data-backed ones
    python plots/make_figures.py F03 F04 --profile paper
    python plots/make_figures.py --write-registry    # regenerate plots/REGISTRY.md
    python plots/make_figures.py --list              # the register, as a table

A figure with no artifact behind it exits 2 from its builder, is reported as
``missing-data`` and does not stop the run. The exit code is non-zero only when a figure
the registry calls ``real`` fails to render, or a figure it calls ``unavailable``
unexpectedly succeeds — either of those means the registry has drifted from the truth.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

PLOTS = Path(__file__).resolve().parent
if str(PLOTS) not in sys.path:
    sys.path.insert(0, str(PLOTS))
if str(PLOTS.parent / "src") not in sys.path:
    sys.path.insert(0, str(PLOTS.parent / "src"))

from common import (  # noqa: E402
    OUTPUTS,
    PROFILES,
    MissingData,
    Profile,
    display_path,
    repo_commit,
)
from registry import (  # noqa: E402
    FIGURES,
    REAL,
    STATUSES,
    UNAVAILABLE,
    FigureSpec,
    check_integrity,
    get,
    render_markdown,
)

RENDERED = "rendered"
MISSING = "missing-data"
FAILED = "failed"


def build_one(spec: FigureSpec, profile: Profile) -> dict:
    """Run one builder for one profile and classify the outcome."""
    module_name, function_name = spec.builder.split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return {"outcome": FAILED, "detail": f"cannot import {module_name}: {exc}"}
    builder = getattr(module, function_name, None)
    if builder is None:
        return {"outcome": FAILED,
                "detail": f"{module_name} has no {function_name}()"}
    try:
        written = builder(profile)
    except MissingData as exc:
        return {"outcome": MISSING, "detail": f"{exc.what}"}
    except SystemExit as exc:  # a builder that exited 2 through another path
        if exc.code == 2:
            return {"outcome": MISSING, "detail": "builder exited 2"}
        raise
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return {"outcome": FAILED, "detail": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}
    return {"outcome": RENDERED, "files": [display_path(p) for p in written]}


def run(specs: list[FigureSpec], profiles: list[Profile], *, quiet: bool = False) -> dict:
    results: dict[str, dict] = {}
    for spec in specs:
        results[spec.id] = {"title": spec.title, "declared_status": spec.status,
                            "builder": spec.builder, "profiles": {}}
        if not quiet:
            print(f"{spec.id}  {spec.title}")
        for profile in profiles:
            if profile.name not in spec.profiles:
                continue
            outcome = build_one(spec, profile)
            results[spec.id]["profiles"][profile.name] = outcome
            if not quiet and outcome["outcome"] != RENDERED:
                print(f"  [{profile.name}] {outcome['outcome']}: {outcome['detail']}")
                if outcome.get("traceback"):
                    print(outcome["traceback"], file=sys.stderr)
    return results


def reconcile(specs: list[FigureSpec], results: dict) -> list[str]:
    """Where the registry's claimed status disagrees with what happened."""
    problems = []
    for spec in specs:
        outcomes = {p: r["outcome"] for p, r in results[spec.id]["profiles"].items()}
        if not outcomes:
            continue
        if spec.status == REAL and any(o != RENDERED for o in outcomes.values()):
            problems.append(f"{spec.id} is registered 'real' but did not render: {outcomes}")
        if spec.status == UNAVAILABLE and any(o == RENDERED for o in outcomes.values()):
            problems.append(
                f"{spec.id} is registered 'unavailable' but rendered: {outcomes}. "
                f"Its data now exists — update plots/registry.py."
            )
        if any(o == FAILED for o in outcomes.values()):
            problems.append(f"{spec.id} failed to build: {outcomes}")
    return problems


def write_manifest(results: dict, profiles: list[Profile]) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / "manifest.json"
    path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plotting_commit": repo_commit(),
        "profiles": [p.name for p in profiles],
        "figures": results,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def list_table() -> str:
    width = max(len(f.title) for f in FIGURES)
    lines = [f"  {'id':<5}{'status':<14}{'values':<20}{'figure':<{width}}  question"]
    for spec in FIGURES:
        lines.append(f"  {spec.id:<5}{spec.status:<14}{spec.value_kind:<20}"
                     f"{spec.title:<{width}}  {spec.question}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figures", nargs="*", help="figure ids (default: all)")
    parser.add_argument("--profile", action="append", choices=sorted(PROFILES),
                        help="output profile; repeatable (default: all)")
    parser.add_argument("--status", action="append", choices=list(STATUSES),
                        help="only figures with this registered status; repeatable")
    parser.add_argument("--list", action="store_true", help="print the register and exit")
    parser.add_argument("--write-registry", action="store_true",
                        help="regenerate plots/REGISTRY.md and exit")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    integrity = check_integrity()
    if integrity:
        for problem in integrity:
            print(f"  registry: {problem}", file=sys.stderr)
        return 1

    if args.list:
        print(list_table())
        return 0
    if args.write_registry:
        path = PLOTS / "REGISTRY.md"
        path.write_text(render_markdown(), encoding="utf-8")
        print(f"  wrote {display_path(path)}")
        return 0

    specs = [get(f) for f in args.figures] if args.figures else list(FIGURES)
    if args.status:
        specs = [s for s in specs if s.status in args.status]
    profiles = [PROFILES[p] for p in (args.profile or sorted(PROFILES))]

    results = run(specs, profiles, quiet=args.quiet)
    problems = reconcile(specs, results)
    manifest = write_manifest(results, profiles)

    counts = {RENDERED: 0, MISSING: 0, FAILED: 0}
    for entry in results.values():
        for outcome in entry["profiles"].values():
            counts[outcome["outcome"]] += 1
    print(f"\n  {counts[RENDERED]} rendered, {counts[MISSING]} awaiting data, "
          f"{counts[FAILED]} failed  ->  {display_path(manifest)}")
    for problem in problems:
        print(f"  MISMATCH: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
