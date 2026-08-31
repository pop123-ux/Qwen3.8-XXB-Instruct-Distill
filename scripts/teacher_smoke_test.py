#!/usr/bin/env python3
"""Can we really access Qwen3.8-27B correctly? Ten checks, one command, no dataset.

This is the integration test the unit suite cannot be: it needs the actual teacher, real
weights on real hardware. It generates nothing reusable and trains nothing. It answers one
question — **is the teacher operational** — and answers it in a way that cannot be faked,
because every check reads something the model itself produced.

    1  load the teacher, with the missing-weight gate armed
    2  load the tokenizer and read its real behaviour
    3  render one prompt through the real chat template, per reasoning mode
    4  tokenize it and report exact ids
    5  generate one answer
    6  obtain logits for a short sequence
    7  construct a TeacherSignal
    8  verify signal dimensions
    9  verify teacher/student token alignment
    10 verify provenance, then unload

The teacher does not fit a 16 GB card. At bf16 it needs ~48 GiB and at 4-bit ~16.3 GiB, so
this runs on rented or borrowed hardware — see docs/TEACHER_INTERFACE.md. Against a small
local checkpoint (``--local-path``) it runs anywhere, which is how the mechanism is checked
without spending anything.

Exit codes: ``0`` every check passed, ``1`` a check failed, ``2`` could not start.

Examples::

    # the real teacher, on a 24 GB card
    python scripts/teacher_smoke_test.py --quantization 4bit

    # the real teacher, unquantised, across two GPUs
    python scripts/teacher_smoke_test.py --device auto --dtype bfloat16

    # the mechanism only, on any machine, against a small local checkpoint
    python scripts/teacher_smoke_test.py --local-path ./tiny-qwen3_5 --lenient-architecture
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.distillation.kd_loss import kd_divergence
from qwen_distill.distillation.real_teacher import (
    DEFAULT_TEACHER_MODEL,
    TeacherLoadError,
    TeacherLoadPlan,
    generate_once,
    load_verified_teacher,
    mode_changes_the_prompt,
    render_prompt,
    teacher_logits,
    teacher_memory_estimate,
)
from qwen_distill.distillation.reasoning_modes import resolve_mode

RULE = "=" * 72


class CheckFailed(RuntimeError):
    """One of the ten checks did not hold."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--revision", default=None,
                        help="exact commit SHA. Required for a hub load; a repo id alone "
                             "does not pin weights. Omit only with --local-path.")
    parser.add_argument("--local-path", default=None,
                        help="load from a directory instead of the hub")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto", help="device_map, or 'cpu'")
    parser.add_argument("--quantization", choices=("4bit", "8bit"), default=None)
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--max-memory", default=None,
                        help='JSON, e.g. \'{"0":"22GiB","cpu":"64GiB"}\'')
    parser.add_argument("--prompt", default="What is 17 * 23? Answer with the number only.")
    parser.add_argument("--mode", default="low", help="reasoning mode for the generation")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=64, help="0 for the dense signal")
    parser.add_argument("--lenient-architecture", action="store_true",
                        help="allow an architecture other than the verified 27B one")
    parser.add_argument("--json", type=Path, help="write the full report here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report: dict = {"checks": {}}

    try:
        max_memory = json.loads(args.max_memory) if args.max_memory else None
    except json.JSONDecodeError as exc:
        print(f"  --max-memory is not valid JSON: {exc}", file=sys.stderr)
        return 2

    plan = TeacherLoadPlan(
        model=args.model, revision=args.revision, local_path=args.local_path,
        dtype=args.dtype, device_map=None if args.device == "cpu" else args.device,
        quantization=args.quantization, max_memory=max_memory,
        offload_folder=args.offload_folder,
    )
    # The revision gate lives in TeacherLoadPlan.validate() so every caller gets it, not
    # only this script. It runs before anything downloads.
    problems = plan.validate()
    if problems:
        print("  invalid load plan:\n    - " + "\n    - ".join(problems), file=sys.stderr)
        return 2

    # Only meaningful for the real teacher: the estimate is derived from the 27B
    # architecture, and printing it beside a small local checkpoint would misdescribe what
    # is about to be loaded.
    is_project_teacher = args.local_path is None and args.model == DEFAULT_TEACHER_MODEL
    estimate = teacher_memory_estimate(4096, args.quantization)
    print(f"\n{RULE}\nTEACHER SMOKE TEST\n{RULE}")
    print(f"  model          : {plan.source}")
    print(f"  revision       : {args.revision or 'local checkpoint — the bytes on disk are the pin'}")
    print(f"  dtype / quant  : {args.dtype} / {args.quantization or 'none'}")
    print(f"  device map     : {plan.device_map or 'cpu'}")
    if is_project_teacher:
        print(f"  expected size  : ~{estimate['weights_gib']:.1f} GiB of weights "
              f"({estimate['total_gib']:.1f} GiB total at 4k context)")
    else:
        print(f"  expected size  : unknown for {plan.source!r} — the {estimate['weights_gib']:.0f} "
              f"GiB figure applies to {DEFAULT_TEACHER_MODEL}, not this checkpoint")
    report["plan"] = plan.to_dict()
    report["memory_estimate"] = estimate

    from qwen_distill.utils.hardware import collect_hardware, measure_memory, reset_peak_memory

    hardware = collect_hardware()
    print(f"  runtime        : {hardware.gpu_name or 'no CUDA device'}"
          + (f" x{hardware.gpu_count}, {hardware.total_vram_gib:.1f} GiB total"
             if hardware.total_vram_gib else "")
          + f" | torch {hardware.versions.get('torch', '?')}")
    report["hardware"] = hardware.to_dict()

    loaded = None
    try:
        # 1-2 -------------------------------------------------------------
        print("\n  [1/10] loading the teacher (the missing-weight gate is armed) ...")
        reset_peak_memory()
        loaded = load_verified_teacher(plan, strict_architecture=not args.lenient_architecture)
        print(f"         {loaded.report.model_class} in {loaded.report.load_seconds:.1f}s, "
              f"{len(loaded.report.ignored_unexpected)} non-text tensor(s) discarded")
        report["checks"]["load"] = loaded.report.to_dict()

        print("  [2/10] tokenizer")
        facts = loaded.tokenizer_facts
        # An unset model_max_length is a sentinel near 2**63, not a real limit.
        max_length = (f"{facts.model_max_length:,}" if facts.model_max_length < 10**9
                      else "unset")
        print(f"         {facts.tokenizer_class}, vocab {facts.vocab_size:,}, "
              f"max length {max_length}")
        print(f"         bos={facts.bos_token!r} (adds_bos={facts.adds_bos}) "
              f"eos={facts.eos_token!r} pad={facts.pad_token!r}")
        print(f"         </think> id {facts.think_close_id} -> "
              f"{'exact' if facts.exact_reasoning_split else 'APPROXIMATE'} reasoning split")
        report["checks"]["tokenizer"] = facts.to_dict()
        if not facts.exact_reasoning_split:
            print("         ! thinking/answer counts will be approximate and recorded as such")

        # 3 ---------------------------------------------------------------
        print("  [3/10] chat template, one render per reasoning mode")
        distinct = mode_changes_the_prompt(loaded, args.prompt)
        for name, info in distinct["rendered"].items():
            print(f"         {name:<18}{info['length']:>6} chars  {info['sha256'][:16]}")
        if not distinct["all_distinct"]:
            raise CheckFailed(
                f"reasoning modes render identical prompts: {distinct['collisions']}. "
                "The controls are doing nothing, so any reasoning-cost comparison built on "
                "them would be measuring noise."
            )
        report["checks"]["template"] = distinct

        # 4 ---------------------------------------------------------------
        mode = resolve_mode(args.mode)
        rendered = render_prompt(loaded, args.prompt, mode=mode)
        prompt_ids = loaded.tokenizer(rendered, return_tensors="pt")["input_ids"]
        print(f"  [4/10] tokenized: {prompt_ids.shape[-1]} ids, first 8 {prompt_ids[0][:8].tolist()}")
        report["checks"]["tokenize"] = {
            "mode": mode.name, "n_prompt_tokens": int(prompt_ids.shape[-1]),
            "first_ids": prompt_ids[0][:8].tolist(),
        }

        # 5 ---------------------------------------------------------------
        print(f"  [5/10] generating (mode={mode.name}, max_new_tokens={args.max_new_tokens}) ...")
        generation = generate_once(
            loaded, args.prompt, mode=mode, max_new_tokens=args.max_new_tokens
        )
        print(f"         {generation.total_generated_tokens} tokens in "
              f"{generation.latency_s:.1f}s ({generation.finish_reason})")
        print(f"         thinking {generation.thinking_tokens} / answer "
              f"{generation.answer_tokens}  [{generation.token_counting_method}]")
        print(f"         answer: {generation.answer.strip()[:200]!r}")
        if generation.total_generated_tokens == 0:
            raise CheckFailed("the teacher generated zero tokens")
        report["checks"]["generate"] = {
            "generated_tokens": generation.total_generated_tokens,
            "thinking_tokens": generation.thinking_tokens,
            "answer_tokens": generation.answer_tokens,
            "token_counting_method": generation.token_counting_method,
            "finish_reason": generation.finish_reason,
            "latency_s": round(generation.latency_s, 2),
            "answer_preview": generation.answer.strip()[:500],
        }

        # 6-8 -------------------------------------------------------------
        import torch

        from qwen_distill.distillation.teacher_signal import OnlineTeacher

        short = prompt_ids[:, : min(16, prompt_ids.shape[-1])]
        print(f"  [6/10] logits for a {short.shape[-1]}-token sequence ...")
        logits = teacher_logits(loaded, short)
        vocab = int(loaded.model.config.vocab_size)
        print(f"         {tuple(logits.shape)}  finite={bool(torch.isfinite(logits).all())}")
        if tuple(logits.shape) != (1, short.shape[-1], vocab):
            raise CheckFailed(
                f"logits are {tuple(logits.shape)}, expected {(1, short.shape[-1], vocab)}"
            )

        top_k = args.top_k or None
        print(f"  [7/10] building a TeacherSignal (top_k={top_k or 'dense'}) ...")
        provider = OnlineTeacher(
            model=loaded.model, top_k=top_k, temperature=1.0,
            teacher_model=args.model, teacher_revision=args.revision,
        )
        signal = provider.signal_for(short)

        print("  [8/10] verifying signal dimensions")
        if signal.is_dense:
            expected = (1, short.shape[-1], vocab)
            if tuple(signal.logits.shape) != expected:
                raise CheckFailed(f"dense signal is {tuple(signal.logits.shape)}, expected {expected}")
        else:
            expected = (1, short.shape[-1], top_k)
            if tuple(signal.top_values.shape) != expected:
                raise CheckFailed(f"top_values is {tuple(signal.top_values.shape)}, expected {expected}")
            if tuple(signal.logsumexp.shape) != (1, short.shape[-1]):
                raise CheckFailed("logsumexp does not cover every position")
            if int(signal.top_indices.max()) >= vocab:
                raise CheckFailed("a top-k index points past the vocabulary")
        print(f"         ok — {'dense' if signal.is_dense else f'top-{top_k}'} over {vocab:,} tokens")

        # 9 ---------------------------------------------------------------
        print("  [9/10] verifying teacher/student alignment")
        divergence, diagnostics = kd_divergence(logits, signal, tail="bucket")
        if abs(divergence.item()) > 1e-4:
            raise CheckFailed(
                f"the teacher's own logits diverge from its own signal by "
                f"{divergence.item():.3e}; positions are misaligned"
            )
        prefix = teacher_logits(loaded, short[:, : short.shape[-1] // 2])
        if not torch.allclose(logits[:, : prefix.shape[1]], prefix, atol=1e-2):
            raise CheckFailed(
                "logits for a prefix differ from the same positions of the full sequence; "
                "position t is not the prediction for the token the student predicts at t"
            )
        print(f"         self-divergence {divergence.item():.2e}, prefix-consistent")
        print(f"         self-consistent at top-{top_k or 'dense'}")

        # The number the offline-vs-online decision turns on, swept rather than taken at
        # one k: how much teacher probability mass falls outside the stored shortlist.
        # Small at k=64 means an offline corpus loses almost nothing; large means either
        # a bigger k or staying online.
        from qwen_distill.distillation.kd_loss import signal_bytes_per_token

        print(f"\n         {'k':>6}{'tail mass':>12}{'entropy':>10}{'top1 agree':>12}"
              f"{'bytes/token':>13}{'GiB / 10M':>11}")
        sweep = {}
        for k in (8, 16, 32, 64, 128, 256):
            if k > vocab:
                continue
            probe = OnlineTeacher(model=loaded.model, top_k=k, temperature=1.0).signal_for(short)
            _, stats = kd_divergence(logits, probe, tail="bucket")
            per_token = signal_bytes_per_token(k)
            sweep[k] = {
                "tail_mass": stats["tail_mass"], "entropy": stats["teacher_entropy"],
                "top1_agreement": stats["top1_agreement"], "bytes_per_token": per_token,
                "gib_per_10m_tokens": per_token * 10_000_000 / 1024**3,
            }
            print(f"         {k:>6}{stats['tail_mass']:>12.4f}{stats['teacher_entropy']:>10.3f}"
                  f"{stats['top1_agreement']:>12.3f}{per_token:>13}"
                  f"{sweep[k]['gib_per_10m_tokens']:>11.2f}")
        print("\n         (measured on ONE short sequence — indicative, not a corpus statistic)")
        report["checks"]["alignment"] = {
            "self_divergence": divergence.item(),
            "tail_mass": diagnostics["tail_mass"],
            "teacher_entropy": diagnostics["teacher_entropy"],
            "top_k_sweep": sweep,
            "sweep_caveat": "one short sequence; run over a real prompt set before choosing k",
        }

        # 10 --------------------------------------------------------------
        print("  [10/10] provenance")
        identity = loaded.identity.to_dict()
        for key in ("model", "revision", "config_sha256", "chat_template_sha256",
                    "tokenizer_sha256"):
            value = identity.get(key)
            print(f"         {key:<22}{(str(value)[:32] if value else 'MISSING')}")
        missing = [k for k in ("config_sha256", "chat_template_sha256") if not identity.get(k)]
        if missing:
            raise CheckFailed(f"provenance is incomplete: {missing} could not be hashed")
        if not identity["is_pinned"]:
            print("         ! revision unpinned: reproducible only while the repo id serves "
                  "these weights")
        report["checks"]["provenance"] = identity
        report["describe"] = loaded.describe()

    except (TeacherLoadError, CheckFailed) as exc:
        print(f"\n  FAILED\n  {exc}", file=sys.stderr)
        report["error"] = str(exc)
        _write(args, report)
        return 1
    except Exception as exc:  # noqa: BLE001 - the failure is the result
        print(f"\n  FAILED\n  {type(exc).__name__}: {exc}", file=sys.stderr)
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write(args, report)
        return 1
    finally:
        if loaded is not None:
            loaded.unload()
            print("\n  teacher unloaded")

    # The only real memory number this run produces. Reported beside the estimate so the
    # analytical model can be checked rather than trusted.
    if hardware.cuda_available:
        peak = measure_memory("smoke_test").torch_peak_reserved_gib
        print(f"\n  MEASURED peak GPU memory: {peak:.2f} GiB"
              + (f"  (estimate for this quantisation was {estimate['total_gib']:.2f} GiB "
                 f"at 4k context)" if is_project_teacher else ""))
        report["measured_peak_gpu_gib"] = peak
    else:
        print("\n  no CUDA device: peak GPU memory is unavailable, not zero. Every memory "
              "figure above\n  is an analytical estimate.")
        report["measured_peak_gpu_gib"] = None

    print(f"\n{RULE}\n  ALL TEN CHECKS PASSED\n{RULE}")
    print("  The teacher is operational: real weights, real tokenizer, real template,")
    print("  real generation, real logits, and a TeacherSignal the KD loss accepts.")
    print("  Next: teacher -> chosen student -> one-step real KD pilot.")
    _write(args, report)
    return 0


def _write(args: argparse.Namespace, report: dict) -> None:
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"  report: {args.json}")


if __name__ == "__main__":
    raise SystemExit(main())
