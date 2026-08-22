#!/usr/bin/env python3
"""Compare a student against the teacher on identical prompts.

Aggregate scores hide what this project cares about. Running both models over the same
items lets us say *"the student matched the teacher on 94% of items using 38% of the
reasoning tokens, and the 6% it lost was concentrated in the hardest quartile"* — which
is a far more useful statement than two accuracy numbers.

The comparison enforces the rule from ``docs/reasoning-efficiency.md``: a token saving
only counts as a win if hard-stratum accuracy held up. ``efficiency_win`` in the output
encodes exactly that, so a capability regression cannot be reported as an efficiency
gain.

The student does not exist yet; this interface is built now so the teacher baseline is
recorded in a directly comparable form.

Example::

    python scripts/paired_eval.py --teacher /models/Qwen3.8-27B --student /models/student \\
        --suite reasoning --output evaluations/paired/run1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.evaluation.metrics import compare, format_summary, summarise
from qwen_distill.evaluation.runner import TransformersBackend, run_tasks
from qwen_distill.evaluation.tasks import long_context_suite, reasoning_dev_set
from qwen_distill.utils.hardware import collect_hardware

SUITES = {
    "reasoning": reasoning_dev_set,
    "long_context": lambda: long_context_suite(),
    "tier1": lambda: reasoning_dev_set() + long_context_suite((1024, 4096), (0.1, 0.5, 0.9)),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--teacher", required=True, help="reference model path or repo id")
    parser.add_argument("--student", required=True, help="candidate model path or repo id")
    parser.add_argument("--suite", default="reasoning", choices=sorted(SUITES))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--teacher-reasoning-effort")
    parser.add_argument("--student-reasoning-effort")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = SUITES[args.suite]()
    if args.limit is not None:
        tasks = tasks[: args.limit]

    def make(path: str, effort: str | None) -> TransformersBackend:
        return TransformersBackend(
            path, device=args.device, dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            seed=args.seed, reasoning_effort=effort,
        )

    output_dir = args.output
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== teacher: {args.teacher} ===")
    teacher_results = run_tasks(
        make(args.teacher, args.teacher_reasoning_effort), tasks,
        output_path=(output_dir / "teacher_generations.jsonl") if output_dir else None,
    )
    print(f"\n=== student: {args.student} ===")
    student_results = run_tasks(
        make(args.student, args.student_reasoning_effort), tasks,
        output_path=(output_dir / "student_generations.jsonl") if output_dir else None,
    )

    teacher_summary = summarise(teacher_results)
    student_summary = summarise(student_results)

    print("\n\n=== TEACHER ===")
    print(format_summary(teacher_summary))
    print("\n=== STUDENT ===")
    print(format_summary(student_summary))

    comparison = compare(teacher_summary, student_summary)
    print("\n=== COMPARISON ===")
    for key, value in comparison.items():
        if isinstance(value, float):
            print(f"  {key:<32}{value:.4f}")
        else:
            print(f"  {key:<32}{value}")

    ratio = comparison["thinking_token_ratio"]
    if comparison["efficiency_win"] is False and ratio is not None and ratio < 1.0:
        print("\n  WARNING: the student used fewer reasoning tokens but hard-task accuracy")
        print("           dropped. This is a capability regression, not an efficiency win.")

    print("\n=== PER-ITEM ===")
    print(f"  {'task_id':<20}{'diff':<11}{'T ok':>5}{'S ok':>5}{'T think':>9}{'S think':>9}")
    student_by_id = {r.task_id: r for r in student_results}
    for t in teacher_results:
        s = student_by_id.get(t.task_id)
        if s is None:
            continue
        fmt = {True: "yes", False: "NO", None: "-"}
        print(
            f"  {t.task_id:<20}{t.difficulty:<11}{fmt[t.correct]:>5}{fmt[s.correct]:>5}"
            f"{t.thinking_tokens:>9}{s.thinking_tokens:>9}"
        )

    if output_dir is not None:
        (output_dir / "paired_results.json").write_text(
            json.dumps(
                {
                    "teacher": args.teacher,
                    "student": args.student,
                    "suite": args.suite,
                    "task_ids": [t.task_id for t in tasks],
                    "teacher_summary": teacher_summary.to_dict(),
                    "student_summary": student_summary.to_dict(),
                    "comparison": comparison,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "hardware": collect_hardware().to_dict(),
                },
                indent=2,
            ) + "\n"
        )
        print(f"\nwrote {output_dir}/paired_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
