#!/usr/bin/env python3
"""What GPU do I have, and what can I actually do with it?

Answers both questions this project needs answered on a new machine:

* for someone who wants to *run* a model — what fits, at what quantisation and context;
* for someone who wants to *experiment* — which training configurations are plausible.

Works on NVIDIA, on AMD/ROCm where PyTorch exposes the information, and on CPU-only
machines. It never fails merely because there is no GPU: that is a normal answer.

Every memory figure is an **analytical estimate** from the project's memory model until
``--calibrate`` measures the model against this specific machine.

Examples::

    python scripts/hardware_info.py
    python scripts/hardware_info.py --model Qwen3.8-27B --matrix
    python scripts/hardware_info.py --spec configs/student/candidate_ctx32k.json
    python scripts/hardware_info.py --recommend --json hardware_report.json
    python scripts/hardware_info.py --calibrate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.architecture.params import count_parameters, format_params
from qwen_distill.architecture.spec import HybridArchSpec
from qwen_distill.diagnostics import (
    analyse_inference_fit,
    calibrate,
    classify,
    collect_devices,
    collect_system,
    cpu_device,
    estimate_training_memory,
    fit_matrix,
    recommend,
    tier_for_devices,
)

RULE = "=" * 64
BUILTIN_MODELS = {"Qwen3.8-27B": HybridArchSpec(name="Qwen3.8-27B")}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _fmt(value: float | None, unit: str = "GiB", places: int = 2) -> str:
    return "unknown" if value is None else f"{value:.{places}f} {unit}"


def _yn(value: bool | None) -> str:
    return {True: "YES", False: "NO", None: "unknown"}[value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=sorted(BUILTIN_MODELS), help="built-in model to analyse")
    parser.add_argument("--spec", type=Path, help="path to a saved HybridArchSpec JSON")
    parser.add_argument("--matrix", action="store_true", help="print a quantisation x context fit grid")
    parser.add_argument("--recommend", action="store_true", help="print experiment recommendations")
    parser.add_argument(
        "--calibrate", action="store_true",
        help="measure the analytical memory model against this machine (needs CUDA)",
    )
    parser.add_argument(
        "--reserved", type=float, default=1.0,
        help="GiB withheld for driver/desktop when judging fit (default 1.0)",
    )
    parser.add_argument("--device-index", type=int, default=0, help="which accelerator to analyse")
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    parser.add_argument(
        "--simulate-vram", type=float, metavar="GIB",
        help="analyse a hypothetical GPU of this size instead of the detected one "
             "(e.g. 16 to preview a T4 from a machine without one). Detection output "
             "is still shown and clearly separated from the simulation.",
    )
    parser.add_argument(
        "--simulate-name", default="simulated GPU", help="name to display when simulating"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    system = collect_system()
    devices = collect_devices(system)
    if not devices:
        devices = [cpu_device(system)]
    tier = tier_for_devices(devices)

    payload: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "system": system.to_dict(),
        "devices": [d.to_dict() for d in devices],
        "tier": {"level": tier.level, "name": tier.name, "summary": tier.summary},
        "note": "memory figures are ANALYTICAL ESTIMATES unless --calibrate was run",
    }

    print(RULE)
    print("QWEN DISTILL - HARDWARE DIAGNOSTICS")
    print(RULE)
    print("\nSystem")
    print(f"  {'OS':<20}{system.os} {system.os_release} ({system.machine})")
    print(f"  {'Python':<20}{system.python_version}")
    print(f"  {'PyTorch':<20}{system.torch_version or 'not installed'}")
    print(f"  {'Transformers':<20}{system.transformers_version or 'not installed'}")
    print(f"  {'CPU cores':<20}{system.cpu_count or 'unknown'}")
    print(f"  {'System RAM':<20}{_fmt(system.total_ram_gib)}")
    print(f"  {'CUDA available':<20}{_yn(system.cuda_available)}"
          f"{'  (runtime ' + system.cuda_runtime_version + ')' if system.cuda_runtime_version else ''}")
    print(f"  {'ROCm available':<20}{_yn(system.rocm_available)}")
    print(f"  {'MPS available':<20}{_yn(system.mps_available)}")
    for note in system.notes:
        print(f"  ! {note}")

    for device in devices:
        print(f"\n{device.backend.upper()} device {device.index}")
        print(f"  {'Name':<20}{device.name}")
        print(f"  {'Vendor':<20}{device.vendor}")
        if device.total_memory_gib is not None:
            used = (device.total_memory_gib - device.free_memory_gib) if device.free_memory_gib else None
            label = "VRAM" if device.vendor != "cpu" else "RAM"
            print(f"  {label:<20}{_fmt(used)} used / {_fmt(device.total_memory_gib)} total")
            print(f"  {'Free':<20}{_fmt(device.free_memory_gib)}")
        print(f"  {'Allocated':<20}{_fmt(device.allocated_memory_gib)}")
        print(f"  {'Reserved':<20}{_fmt(device.reserved_memory_gib)}")
        print(f"  {'Peak allocated':<20}{_fmt(device.peak_allocated_gib)}")
        if device.compute_capability:
            print(f"  {'Compute capability':<20}{device.compute_capability}")
        if device.multi_processor_count:
            print(f"  {'SM count':<20}{device.multi_processor_count}")
        if device.driver_version:
            print(f"  {'Driver':<20}{device.driver_version}")
        print(f"  {'BF16':<20}{_yn(device.supports_bf16)}")
        print(f"  {'FP16':<20}{_yn(device.supports_fp16)}")
        print(f"  {'FP8':<20}{_yn(device.supports_fp8)}")
        print(f"  {'INT8':<20}{_yn(device.supports_int8)}")
        print(f"  {'Tensor cores':<20}{_yn(device.has_tensor_cores)}")
        for note in device.notes:
            print(f"  ! {note}")

    print(f"\n  {'Capability tier':<20}Tier {tier.level} - {tier.name}")
    print(f"  {'':<20}{tier.summary}")
    print(f"  {'':<20}examples: {tier.examples}")

    primary = devices[min(args.device_index, len(devices) - 1)]
    # On a CPU-only machine total_memory_gib holds system RAM. It is shown above for
    # context, but it is not VRAM, and treating it as a memory budget would tell the
    # user a 27B model "fits" on a machine with no GPU at all.
    is_accelerator = primary.vendor != "cpu"
    total = (primary.total_memory_gib or 0.0) if is_accelerator else 0.0
    device_label = primary.name

    if args.simulate_vram is not None:
        total = args.simulate_vram
        is_accelerator = True
        device_label = args.simulate_name
        tier = classify(total)
        payload["simulated"] = {"vram_gib": total, "name": device_label}
        payload["tier"] = {"level": tier.level, "name": tier.name, "summary": tier.summary}
        print(f"\n{RULE}")
        print(f"SIMULATING A {total:.0f} GiB DEVICE: {device_label}")
        print("Detected hardware above is unchanged; everything below is hypothetical.")
        print(RULE)
        print(f"\n  Capability tier     Tier {tier.level} - {tier.name}")
        print(f"  {'':<20}{tier.summary}")

    usable = max(0.0, total - args.reserved)

    # --- model fit ------------------------------------------------------
    specs: list[HybridArchSpec] = []
    if args.spec:
        specs.append(HybridArchSpec.load(args.spec))
    if args.model:
        specs.append(BUILTIN_MODELS[args.model])
    if not specs and not args.recommend and not args.calibrate:
        specs.append(BUILTIN_MODELS["Qwen3.8-27B"])

    if specs:
        print(f"\n{RULE}")
        print("MODEL DEPLOYMENT ANALYSIS  (ANALYTICAL ESTIMATE)")
        print(RULE)
        if is_accelerator:
            print(f"\n  budget: {total:.2f} GiB total - {args.reserved:.2f} reserved "
                  f"= {usable:.2f} GiB usable\n")
        else:
            print("\n  NO ACCELERATOR DETECTED. Every model below is reported as not\n"
                  "  fitting because there is no VRAM budget at all. The system RAM\n"
                  "  shown above is not a substitute; run this on the target GPU.\n")
        fits = []
        for spec in specs:
            params = count_parameters(spec)
            print(f"{spec.name}  ({format_params(params.total)} parameters)")
            for quant in ("bf16", "int8", "q6_k", "q5_k_m", "q4_k_m"):
                fit = analyse_inference_fit(
                    spec, usable, quantization=quant, context_length=8192
                )
                fits.append(fit.to_dict())
                print(f"  {quant:<8} weights {fit.weights_gib:6.2f}  total @8k "
                      f"{fit.total_gib:6.2f} GiB   {fit.verdict}")
            print()
        payload["inference_fits"] = fits

        if args.matrix:
            for spec in specs:
                print(f"{RULE}\nCONTEXT x QUANTISATION - {spec.name}  (ANALYTICAL ESTIMATE)\n{RULE}")
                grid = fit_matrix(spec, usable)
                contexts = sorted(next(iter(grid.values())))
                header = f"  {'quant':<10}" + "".join(f"{c // 1024:>8}k" for c in contexts)
                print(header)
                for quant, row in grid.items():
                    cells = []
                    for ctx in contexts:
                        v = row[ctx].verdict
                        cells.append({"FITS": "  OK", "TIGHT": " TIGHT", "DOES NOT FIT": "   NO"}[v].rjust(9))
                    print(f"  {quant:<10}" + "".join(cells))
                print()
                payload.setdefault("fit_matrix", {})[spec.name] = {  # type: ignore[index]
                    q: {str(c): row[c].verdict for c in contexts} for q, row in grid.items()
                }

            for spec in specs:
                print(f"{RULE}\nTRAINING FEASIBILITY - {spec.name}  (ANALYTICAL ESTIMATE)\n{RULE}")
                print(f"  {'strategy':<10}{'optimizer':<14}{'seq':>6}{'est GiB':>10}   verdict")
                rows = []
                for strategy, optimizer in (
                    ("full", "adamw"), ("full", "adamw_8bit"),
                    ("lora", "adamw_8bit"), ("qlora", "adamw_8bit"),
                ):
                    fit = estimate_training_memory(
                        spec, usable, strategy=strategy, optimizer=optimizer,
                        sequence_length=2048, gradient_checkpointing=True,
                    )
                    rows.append(fit.to_dict())
                    print(f"  {strategy:<10}{optimizer:<14}{2048:>6}{fit.total_gib:>10.2f}   {fit.verdict}")
                print()
                payload.setdefault("training_fits", {})[spec.name] = rows  # type: ignore[index]

    # --- recommendations -------------------------------------------------
    if args.recommend:
        result = recommend(total or None, device_label, reserved_gib=args.reserved)
        payload["recommendations"] = result.to_dict()
        print(f"{RULE}\nRECOMMENDATION\n{RULE}\n")
        print(f"  Tier {result.tier.level} - {result.tier.name}: {result.tier.summary}\n")
        for title, items in (
            ("GOOD", result.good), ("POSSIBLE WITH CARE", result.with_care),
            ("NOT REALISTIC", result.not_realistic),
        ):
            print(f"  {title}")
            for item in items or ["(nothing)"]:
                print(f"    - {item}")
            print()
        for title, items in (
            ("Inference", result.inference), ("Training", result.training),
            ("Evaluation", result.evaluation), ("Architecture experiments", result.architecture),
        ):
            print(f"  {title}")
            for item in items or ["(nothing)"]:
                print(f"    - {item}")
            print()

    # --- calibration ------------------------------------------------------
    if args.calibrate:
        print(f"{RULE}\nMEMORY MODEL CALIBRATION\n{RULE}\n")
        report = calibrate()
        payload["calibration"] = report.to_dict()
        print(f"  device : {report.device}")
        print(f"  verdict: {report.verdict}")
        if report.points:
            print(f"\n  {'phase':<44}{'measured':>10}{'estimated':>11}{'ratio':>8}")
            for point in report.points:
                ratio = f"{point.ratio:.3f}" if point.ratio else "-"
                print(f"  {point.phase[:43]:<44}{point.measured_peak_gib:>10.3f}"
                      f"{point.estimated_gib:>11.3f}{ratio:>8}")
        if report.mean_ratio:
            print(f"\n  mean measured/estimated: {report.mean_ratio:.3f}")
        for note in report.notes:
            print(f"  ! {note}")
        if report.error:
            print(f"  ERROR: {report.error}")
        print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
