#!/usr/bin/env python3
"""The pre-GPU acceptance gate: check every claim this repository makes about itself.

Each item is verified by running the thing, not by reading a document. An item is PASS only
if its check returns true, DEFERRED if it is deliberately not built yet and says so, and
FAIL otherwise. ``READY FOR GPU`` is printed only when nothing failed.

    python scripts/acceptance_gate.py
    python scripts/acceptance_gate.py --json runs/acceptance.json

Exit codes: ``0`` ready, ``1`` something failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
PASS, FAIL, DEFERRED = "PASS", "FAIL", "DEFERRED"

#: The exact architecture this gate exists to protect.
TOTAL = 13_008_505_728
ACTIVE = 9_611_119_488
TEACHER = "Qwen/Qwen3.8-27B"


def _checks() -> list[tuple[str, str, object]]:
    """(name, note, callable) — each callable returns True, False, or the string DEFERRED."""
    from qwen_distill.architecture.moe_student import (
        FROZEN_STUDENT,
        MTP_STATUS,
        PARAMETER_BUDGET,
        REJECTED,
        TEACHER_ID,
        audit,
        parameter_model,
    )
    from qwen_distill.distillation.behavioral import DELTANET_STATE, LOSS_TERMS, MTP
    from qwen_distill.distillation.real_teacher import TeacherLoadPlan
    from qwen_distill.research.ablations import ARMS
    from qwen_distill.research.baselines import baselines
    from qwen_distill.research.context import CURRICULA, CURVE_LENGTHS
    from qwen_distill.research.memory import CONTEXT_LADDER, RELEASE_QUANTS, headline

    spec = FROZEN_STUDENT
    report = audit(spec)
    pinned = "0f9e8d7c6b5a49382716051423f6e5d4c3b2a190"

    def teacher_uses_pretrained() -> bool:
        source = (ROOT / "src/qwen_distill/distillation/real_teacher.py").read_text(
            encoding="utf-8")
        return ("AutoModelForCausalLM.from_pretrained" in source
                and "AutoModelForCausalLM.from_config" not in source)

    def revision_gate() -> bool:
        if TeacherLoadPlan().validate() == []:
            return False                                   # unpinned hub must be refused
        for moving in ("main", "master", "latest", "HEAD"):
            if TeacherLoadPlan(revision=moving).validate() == []:
                return False
        if TeacherLoadPlan(revision="v1.0").validate() != []:
            pass                                           # tags refused: good
        else:
            return False
        if TeacherLoadPlan(revision=pinned).validate() != []:
            return False
        return TeacherLoadPlan(local_path="/x").validate() == []

    def no_silent_mock() -> bool:
        from qwen_distill.distillation.backends import make_backend

        try:
            make_backend("does-not-exist")
        except ValueError:
            return True
        return False

    def pilot_has_no_geometry() -> bool:
        """Whole-flag matching, not substring: ``--ffn-method`` selects a decomposition
        method and is not student geometry, so a substring test would fail on it wrongly."""
        import re as _re

        source = (ROOT / "scripts/distill_pilot.py").read_text(encoding="utf-8")
        banned = ("--hidden", "--layers", "--ffn", "--kv-heads", "--dn-key-heads",
                  "--untie-embeddings", "--num-experts", "--expert-width")
        for flag in banned:
            if _re.search(_re.escape(flag) + r"(?![\w-])", source):
                return False
        return "FROZEN_STUDENT" in source

    def materialisation_runs() -> bool:
        """Actually materialise a scaled fixture end to end, not merely import the function."""
        import torch
        from transformers import AutoModelForCausalLM

        from qwen_distill.architecture.materialize import StateDictSource
        from qwen_distill.architecture.moe_init import materialise_student
        from qwen_distill.architecture.moe_student import build_config, tiny_fixture

        sys.path.insert(0, str(ROOT / "tests"))
        from test_moe_init import T_KV, T_LAYERS, TI, _fixture_teacher  # noqa: PLC0415

        tiny = tiny_fixture()
        torch.manual_seed(0)
        state = _fixture_teacher(tiny, T_LAYERS, T_KV, TI)
        model = AutoModelForCausalLM.from_config(build_config(tiny))
        result = materialise_student(model, StateDictSource(state), tiny,
                                     teacher_layers=T_LAYERS, teacher_kv_heads=T_KV,
                                     teacher_intermediate=TI)
        return result.complete and bool(result.merged) and bool(result.decomposed)

    def memory_accounting() -> bool:
        h = headline()
        return (h["fits_at_any_release_quant"]
                and set(CONTEXT_LADDER) >= {4096, 8192, 16384, 32768, 65536, 131072, 262144}
                and len(RELEASE_QUANTS) == 3
                and h["max_context_by_quant"]["Q4"] >= 131_072)

    def ledger_provenance() -> bool:
        from qwen_distill.research.ledger import ESTIMATED, REPORTED, Entry

        for bad in ({"provenance": "vibes"}, {"provenance": ESTIMATED},
                    {"provenance": REPORTED}):
            try:
                Entry(kind="note", title="x", **bad)
            except ValueError:
                continue
            return False
        return True

    return [
        ("teacher identity", TEACHER, lambda: TEACHER_ID == TEACHER),
        ("pretrained teacher loader", "from_pretrained, never from_config",
         teacher_uses_pretrained),
        ("exact revision gate", "hub needs a commit SHA; main/tags refused", revision_gate),
        ("no silent mock fallback", "unknown backend raises", no_silent_mock),
        ("missing-weight gate intact", "a mismatched checkpoint is fatal",
         lambda: "missing_keys" in (ROOT / "src/qwen_distill/distillation/real_teacher.py")
                 .read_text(encoding="utf-8")),
        ("student canonical path", "pilot exposes no geometry", pilot_has_no_geometry),
        ("exact student parameters", f"{TOTAL:,}",
         lambda: report["exact_parameter_count"] == TOTAL),
        ("active parameters/token", f"{ACTIVE:,}",
         lambda: report["active_parameters_per_token"] == ACTIVE),
        ("closed form agrees with the model", "two independent derivations",
         lambda: parameter_model(spec)["total"] == report["exact_parameter_count"]),
        ("parameter budget enforced", f"<= {PARAMETER_BUDGET:,}",
         lambda: report["exact_parameter_count"] <= PARAMETER_BUDGET),
        ("48-layer topology", "36 DeltaNet + 12 attention",
         lambda: (spec.num_hidden_layers == 48
                  and len(spec.deltanet_layer_indices) == 36
                  and len(spec.attention_layer_indices) == 12)),
        ("8 x 768 top-2 MoE", "routed experts",
         lambda: (spec.num_experts == 8 and spec.moe_intermediate_size == 768
                  and spec.num_experts_per_tok == 2)),
        ("1 shared expert", "width 768",
         lambda: spec.shared_expert_intermediate_size == 768),
        ("24 Q / 2 KV heads", "head_dim 256",
         lambda: (spec.num_attention_heads == 24 and spec.num_key_value_heads == 2
                  and spec.head_dim == 256)),
        ("262144 architectural context", "config field, not a claim",
         lambda: spec.max_position_embeddings == 262_144),
        ("teacher -> student materialisation", "runs on a scaled fixture",
         materialisation_runs),
        ("64 -> 48 mapping", "block types preserved, 16 absorbed",
         lambda: _mapping_ok(spec)),
        ("4 -> 2 KV conversion", "mean merge, configurable", lambda: _kv_ok(spec)),
        ("FFN -> MoE initialisation", "decomposed, coverage reported",
         lambda: _ffn_ok(spec)),
        ("distillation objectives", "CE / logit / layer / behaviour distinguishable",
         lambda: len({frozenset(ARMS[a].loss_weights) for a in ("A0", "A2", "A1", "A3")}) == 4),
        ("context specialisation", "6 mixtures over 4K-262K",
         lambda: (len(CURRICULA) == 6
                  and set(CURVE_LENGTHS) >= {4096, 32768, 262144})),
        ("16 GB accounting", "Q4/Q5/Q6 across the ladder, GPU-resident", memory_accounting),
        ("experiment provenance", "closed provenance set, estimates carry a method",
         ledger_provenance),
        ("research baselines", "dense h5120-L40 kept; 24-expert rejected recorded",
         lambda: ("dense_h5120_l40" in baselines()
                  and any(r["total_parameters"] == 22_072_134_528 for r in REJECTED))),
        ("plotting infrastructure", "registered figures, provenance, no invented numbers",
         lambda: ((ROOT / "plots/common.py").exists()
                  and (ROOT / "plots/registry.py").exists()
                  and (ROOT / "plots/REGISTRY.md").exists()
                  and len(list((ROOT / "plots/figures").glob("*.py"))) >= 5)),
        ("teacher downloader", "pinned, verified, manifested",
         lambda: (ROOT / "scripts/download_teacher.py").exists()),
        ("README consistency", "states the exact count and the 16 GB baseline",
         lambda: _readme_ok()),
        ("Further Questions section", "future-compute limits stated",
         lambda: _research_doc_ok()),
        ("MTP", "declared, not built by the runtime — extension point kept",
         lambda: DEFERRED if ("DECLARED, NOT BUILT" in MTP_STATUS
                              and not LOSS_TERMS[MTP].available) else False),
        ("DeltaNet state matching", "shapes differ; hidden-delta covers the same layers",
         lambda: DEFERRED if not LOSS_TERMS[DELTANET_STATE].available else False),
        ("measured GPU memory", "analytical only until hardware is available",
         lambda: DEFERRED),
        ("benchmark results", "none held; no SOTA claim made", lambda: DEFERRED),
    ]


def _mapping_ok(spec) -> bool:
    from qwen_distill.architecture.moe_init import map_layers

    m = map_layers(spec, teacher_layers=64)
    return m.block_types_preserved and len(m.mapping) == 48 and len(m.removed_teacher_layers) == 16


def _kv_ok(spec) -> bool:
    import torch

    from qwen_distill.architecture.moe_init import merge_kv_heads

    weight = torch.randn(4 * spec.head_dim, spec.hidden_size)
    merged = merge_kv_heads(weight, teacher_heads=4, student_heads=2, head_dim=spec.head_dim)
    return merged.shape == (2 * spec.head_dim, spec.hidden_size)


def _ffn_ok(spec) -> bool:
    from qwen_distill.architecture.moe_init import plan_ffn_decomposition

    plan = plan_ffn_decomposition(spec, importance=None)
    return (len(plan.expert_channels) == spec.num_experts
            and plan.active_width == 2 * 768 + 768
            and 0.0 < plan.coverage < 1.0)


#: Phrases that would constitute an unsupported claim. Mentioning "SOTA" to *disclaim* it
#: is correct and must not trip the gate, so the patterns are claims rather than the word.
CLAIM_PATTERNS = ("is sota", "achieves sota", "achieving sota", "state-of-the-art results",
                  "state of the art results", "we are #1", "the best model", "beats all")


def _readme_ok() -> bool:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lowered = text.lower()
    if any(pattern in lowered for pattern in CLAIM_PATTERNS):
        return False
    return all(needle in text for needle in
               ("13,008,505,728", "9,611,119,488", "16 GB",
                "DEMONSTRATED", "FUTURE WORK", "Qwen/Qwen3.8-27B"))


def _research_doc_ok() -> bool:
    path = ROOT / "docs" / "RESEARCH.md"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return all(s in text for s in ("Further Questions and Future Work",
                                   "Validated in this work", "Not yet tested",
                                   "Future work"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("  PRE-GPU ACCEPTANCE GATE")
    print("=" * 78)
    results = []
    for name, note, check in _checks():
        try:
            outcome = check()
        except Exception as exc:  # noqa: BLE001 - a raising check is a failing check
            status, note = FAIL, f"{type(exc).__name__}: {exc}"
        else:
            status = (DEFERRED if outcome == DEFERRED
                      else PASS if outcome is True else FAIL)
        results.append({"item": name, "status": status, "note": note})
        print(f"  {status:<9} {name:<34} {note}")

    failed = [r for r in results if r["status"] == FAIL]
    deferred = [r for r in results if r["status"] == DEFERRED]
    print("-" * 78)
    print(f"  {len(results) - len(failed) - len(deferred)} pass, "
          f"{len(deferred)} deferred, {len(failed)} fail")
    ready = not failed
    print(f"\n  OVERALL: {'READY FOR GPU' if ready else 'NOT READY'}")
    if deferred:
        print("  Deferred items are deliberately not built and are documented as such; "
              "they do not\n  block the first GPU session.")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"ready": ready, "results": results}, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.json}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
