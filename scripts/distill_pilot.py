#!/usr/bin/env python3
"""The research pilot: real Qwen3.8-27B -> the canonical frozen student.

One path, one student. The architecture is :data:`FROZEN_STUDENT`
(``qwen38_19b_h5120_l48_moe``) and this script exposes **no way to change it** — no hidden
size, no layer count, no expert count. A pilot that could quietly run a different
architecture would produce a result nobody could attribute, and the mechanism-level
knob-turning lives in ``scripts/chain_selftest.py`` where it belongs.

What it does, in order::

    teacher checkpoint  ->  verified load (missing weights are fatal)
                        ->  64 -> 48 group-aligned layer mapping
                        ->  materialise: copy, KV-merge 4 -> 2, decompose dense FFN -> MoE
                        ->  report coverage, audit, 16 GB verdict
                        ->  checkpoint

Every step is a routine that is separately tested and separately measured; this script only
sequences them and reports what happened.

Reproducibility: a Hub load requires ``--revision <commit SHA>`` and is refused without one,
before anything downloads. A ``--teacher DIR`` load reads bytes already on disk, so the
revision is optional there — pass it anyway and it is recorded, because the directory alone
does not say which upstream commit produced it.

Exit codes: ``0`` the transfer ran, ``1`` it did not, ``2`` the request could not be set up.

Examples::

    # what would happen, loading and writing nothing
    python scripts/distill_pilot.py --teacher ./qwen3.8-27b --dry-run

    # materialise the canonical student from a downloaded checkpoint
    python scripts/distill_pilot.py --teacher ./qwen3.8-27b \\
        --revision <EXACT_QWEN_COMMIT_SHA> --output runs/pilot1

    # from the Hub, which requires the pin
    python scripts/distill_pilot.py --revision <EXACT_QWEN_COMMIT_SHA> --output runs/pilot1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.architecture.materialize import SafetensorsSource
from qwen_distill.architecture.moe_init import map_layers, materialise_student
from qwen_distill.architecture.moe_student import (
    FROZEN_STUDENT,
    MTP_STATUS,
    STUDENT_ID,
    TEACHER_ID,
    TEACHER_KV_HEADS,
    TEACHER_LAYERS,
    audit,
    build_model,
)
from qwen_distill.distillation.real_teacher import TeacherLoadPlan
from qwen_distill.research.memory import RuntimeConfig, account

RULE = "=" * 78


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    teacher = parser.add_argument_group("teacher")
    teacher.add_argument("--teacher", type=Path, default=None,
                         help="directory holding the teacher's config.json and safetensors. "
                              "Omit to load from the Hub, which requires --revision.")
    teacher.add_argument("--revision", default=None,
                         help="exact commit SHA. Required for a Hub load; recorded for a "
                              "local one.")
    teacher.add_argument("--teacher-model", default=TEACHER_ID,
                         help=argparse.SUPPRESS)

    # There is deliberately no student group. The student is FROZEN_STUDENT.
    build = parser.add_argument_group("transfer")
    build.add_argument("--layer-strategy", default="group", choices=("group", "importance"),
                       help="'group' keeps whole hybrid groups; 'importance' needs a "
                            "measured per-group score and is not wired to a scorer yet")
    build.add_argument("--kv-merge", default="mean", choices=("mean", "weighted", "first"),
                       help="how the teacher's 4 KV heads become the student's 2")
    build.add_argument("--ffn-method", default="importance_partition",
                       choices=("importance_partition", "contiguous_partition"))
    build.add_argument("--no-gate-compensation", action="store_true",
                       help="skip the routing-weight compensation; the block then starts at "
                            "half the teacher's FFN output scale")

    run = parser.add_argument_group("run")
    run.add_argument("--output", type=Path, default=Path("runs/pilot"))
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--dry-run", action="store_true",
                     help="report the plan and the budget; load and write nothing")
    run.add_argument("--json", type=Path, help="also write the record here")
    return parser.parse_args(argv)


def _budget_line() -> tuple[str, dict]:
    report = audit(FROZEN_STUDENT)
    acc = account(FROZEN_STUDENT,
                  RuntimeConfig(context_length=32_768, expert_quant="q4_k_m",
                                dense_quant="q4_k_m", embedding_quant="q4_k_m"),
                  report["components"])
    text = (f"    Q4 @ 32,768 tokens : {acc.total_gib:.2f} GiB "
            f"({acc.headroom_gib():+.2f} GiB headroom)  {acc.verdict()}")
    return text, {"audit": report, "q4_32k": acc.to_dict()}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record: dict[str, object] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "student_id": STUDENT_ID,
    }

    # --- the teacher plan, gated before anything downloads ---------------
    plan = TeacherLoadPlan(
        model=args.teacher_model,
        revision=args.revision,
        local_path=str(args.teacher) if args.teacher else None,
    )
    problems = plan.validate()
    if problems:
        print("  invalid teacher load plan:\n    - " + "\n    - ".join(problems),
              file=sys.stderr)
        return 2
    record["teacher_plan"] = plan.to_dict()

    print(f"\n{RULE}\nDISTILLATION PILOT\n{RULE}")
    print(f"  teacher : {plan.source}")
    print(f"  revision: {args.revision or 'local checkpoint (bytes on disk are the pin)'}")
    print(f"  student : {STUDENT_ID}")

    budget_text, budget = _budget_line()
    report = budget["audit"]
    print("\n  STUDENT")
    print(f"    parameters      : {report['exact_parameter_count']:,}")
    print(f"    active / token  : {report['active_parameters_per_token']:,}")
    print(f"    layers          : {report['num_layers']} "
          f"({report['deltanet_layers']} DeltaNet + {report['attention_layers']} attention)")
    print(f"    experts         : {FROZEN_STUDENT.num_experts} routed "
          f"(top-{FROZEN_STUDENT.num_experts_per_tok}) + 1 shared, width "
          f"{FROZEN_STUDENT.moe_intermediate_size}")
    print("\n  16 GB BUDGET")
    print(budget_text)
    record["student"] = budget

    mapping = map_layers(FROZEN_STUDENT, teacher_layers=TEACHER_LAYERS,
                         strategy=args.layer_strategy) if args.layer_strategy == "group" else None
    if mapping is None:
        print("  --layer-strategy=importance needs a measured per-group score, and no "
              "scorer is wired\n  to this script yet. Use 'group'.", file=sys.stderr)
        return 2
    print("\n  LAYER MAPPING")
    print(f"    {TEACHER_LAYERS} -> {FROZEN_STUDENT.num_hidden_layers} layers, "
          f"{len(mapping.removed_teacher_layers)} teacher layers absorbed")
    print(f"    block types preserved: {mapping.block_types_preserved}")
    record["layer_mapping"] = mapping.to_dict()

    if args.dry_run:
        _write(args, record)
        print("\n  dry run: nothing was loaded, transferred or written.")
        return 0

    if args.teacher is None:
        print("\n  a Hub load is not wired into this script yet: materialisation streams "
              "shards\n  from disk. Download the pinned revision first, then pass "
              "--teacher DIR.", file=sys.stderr)
        return 2
    if not (args.teacher / "config.json").exists():
        print(f"\n  no config.json in {args.teacher}", file=sys.stderr)
        return 2

    # --- materialise -----------------------------------------------------
    print("\n  MATERIALISING (streaming one teacher tensor at a time) ...")
    source = SafetensorsSource(args.teacher)
    # Build the student in bf16, not the from_config fp32 default. The teacher is
    # bf16 on disk, so an fp32 student is a pointless upcast that (a) doubles the
    # 24 GiB materialisation to 48 GiB and OOM-kills on a 51 GiB-capped container,
    # and (b) makes the KV-merge and FFN-decomposition arithmetic fp32, so the
    # saved checkpoint would not be bit-identical to one built in bf16. The
    # canonical pilot001 student is bf16; this keeps a re-materialisation matching it.
    import torch
    torch.set_default_dtype(torch.bfloat16)
    model = build_model(FROZEN_STUDENT, meta=False)
    torch.set_default_dtype(torch.float32)
    try:
        result = materialise_student(
            model, source, FROZEN_STUDENT,
            teacher_layers=TEACHER_LAYERS, teacher_kv_heads=TEACHER_KV_HEADS,
            mapping=mapping, kv_method=args.kv_merge, ffn_method=args.ffn_method,
            compensate=not args.no_gate_compensation, seed=args.seed,
        )
    except (KeyError, ValueError) as exc:
        print(f"\n  materialisation refused: {exc}", file=sys.stderr)
        return 2
    finally:
        if hasattr(source, "close"):
            source.close()

    print(f"    copied     : {len(result.copied)}")
    print(f"    KV-merged  : {len(result.merged)}")
    print(f"    decomposed : {len(result.decomposed)}")
    print(f"    initialised: {len(result.initialised)}  (router and shared-expert gate — "
          "no teacher counterpart)")
    print(f"    coverage   : {result.coverage:.4%} of student parameters came from the teacher")
    print(f"    FFN channels transferred: {result.measurements.get('ffn_coverage', 0):.1%} "
          f"of the teacher's, active width {result.measurements.get('ffn_active_width')}")
    record["materialisation"] = result.to_dict()

    if not result.complete:
        print(f"\n  INCOMPLETE: {len(result.missing)} student tensors were never written. "
              "The first few:", file=sys.stderr)
        for name in result.missing[:10]:
            print(f"    {name}", file=sys.stderr)
        _write(args, record)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "transferred"
    model.save_pretrained(destination)
    print(f"\n  written to {destination}")
    record["transferred_path"] = str(destination)
    record["mtp_status"] = MTP_STATUS
    _write(args, record)
    print("\n  transfer complete. Distillation is the next step and is not run here.")
    return 0


def _write(args: argparse.Namespace, record: dict) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    for path in filter(None, (args.output / "pilot_record.json", args.json)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(record, indent=2, default=str) + "\n",
                              encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
