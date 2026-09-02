#!/usr/bin/env python3
"""Write this session's findings into the experiment ledger.

One entry per finding, each with its own provenance, so a later reader can tell what was
measured on hardware from what was computed from a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.research.ledger import Ledger

REVISION = "dbdc473dea0d6a9763042881cc33d6058d1742d2"
SUPERSEDED = "72a217afab8029b39e4af1c7273a829995a3dbaf"


def main() -> int:
    ledger = Ledger(Path("experiments/ledger.jsonl"))
    summary = json.loads(Path("runs/kd_preflight/summary.json").read_text())
    dist = summary["distillation"]

    ledger.measured(
        "teacher_pin",
        "Qwen3.8-27B: the weights-upload commit is not the right research pin",
        {
            "resolved_full_sha_of_72a217a": SUPERSEDED,
            "adopted_revision": REVISION,
            "weights_identical": True,
            "weights_evidence": (
                "all 18 safetensors shards carry identical LFS oids at both revisions; no "
                "weight commit exists between them"
            ),
            "why_the_upload_commit_is_wrong": (
                "72a217a is the folder upload (2026-08-13T08:23:30Z). Upstream then "
                "replaced tokenizer_config.json, chat_template.jinja and "
                "generation_config.json about two hours later. Pinning the upload commit "
                "therefore pins superseded metadata alongside correct weights."
            ),
            "observed_consequence": (
                "at 72a217a the smoke test fails check 3: the chat template accepts "
                "('low','high','xhigh') and raises TemplateError on 'medium'. The "
                "project's reasoning_modes.py documents the corrected contract "
                "('low','medium','xhigh', with 'high' raising), so every reasoning-mode "
                "experiment would have been run against a template that contradicts it."
            ),
            "metadata_sha256_at_adopted_revision_matches_vendor": True,
            "also_differs": "tokenizer model_max_length 131,072 at 72a217a, 262,144 at "
                            "the adopted revision",
            "closes": "B2",
        },
        tags=["teacher", "provenance", "B2"],
    )

    ledger.measured(
        "teacher_verification",
        "Qwen3.8-27B loads as the intended teacher on an A40 at 4-bit",
        {
            "revision": REVISION,
            "model_class": "Qwen3_5ForCausalLM",
            "declared_architecture": "Qwen3_5ForConditionalGeneration",
            "n_missing": 0, "n_unexpected": 0, "n_mismatched": 0,
            "non_text_tensors_discarded": 0,
            "weights_complete": True,
            "load_seconds_cold": 611.9,
            "load_seconds_warm": 95.7,
            "tokenizer_class": "Qwen2Tokenizer",
            "tokenizer_vocab_size": 248077,
            "config_vocab_size": 248320,
            "generation_check": "answered '391' to 17*23 in mode 'low' — trained weights, "
                                "not a freshly-initialised model",
            "reasoning_modes_render_distinctly": {
                "thinking_disabled": 114, "low": 269, "medium": 103, "xhigh": 340,
            },
            "measured_teacher_resident_gib": 16.456,
            "analytical_estimate_gib": 16.31,
            "gpu": "NVIDIA A40, 44.43 GiB",
            "smoke_test_checks_passed": "1-8 of 10; check 9's prefix assertion is "
                                        "discussed in its own entry",
        },
        tags=["teacher", "16gb", "smoke_test"],
    )

    ledger.measured(
        "kd_mechanism",
        "First real teacher-in-the-loop KD steps against Qwen3.8-27B",
        {
            "status": "MECHANISM CHECK, NOT RUN 001",
            "why_not_run_001": (
                "the canonical 13.01B student does not fit this GPU for training; see the "
                "run001_feasibility entry. This proves the chain, and its loss says "
                "nothing about capability."
            ),
            "student": "moe_student.tiny_fixture at the teacher's vocabulary",
            "student_class": "Qwen3_5MoeForCausalLM",
            "student_parameters": 32_310_312,
            "student_vocab_size": 248320,
            "teacher": {"model": "Qwen/Qwen3.8-27B", "revision": REVISION,
                        "quantization": "4bit", "signal": "online", "top_k": 64},
            "chain": "teacher forward -> student forward -> KD loss -> backward -> "
                     "optimizer step",
            "objective": "logit_kd", "kd_alpha": 1.0, "kd_temperature": 2.0,
            "kd_tail": "bucket",
            "steps": summary["steps"], "exit_code": 0,
            "runtime_s": summary["runtime_s"],
            "tokens_per_second": summary["tokens_per_second"],
            "sequence_length": 512, "batch_size": 1,
            "corpus": "Project Gutenberg public-domain English via "
                      "scripts/prepare_level2r_dataset.py, tokenised by the teacher's own "
                      "tokenizer",
            "corpus_sha256": summary["corpus"]["sha256"],
            "ce_loss_first": dist["ce_loss"]["first"],
            "ce_loss_final": dist["ce_loss"]["final"],
            "validation_first": summary["first_validation_loss"],
            "validation_final": summary["final_validation_loss"],
            "top1_agreement_first": dist["top1_agreement"]["first"],
            "top1_agreement_final": dist["top1_agreement"]["final"],
            "measured_peak_allocated_gib": 19.019,
            "measured_peak_reserved_gib": 20.271,
        },
        tags=["kd", "mechanism", "B4"],
    )

    ledger.measured(
        "teacher_signal",
        "Teacher tail mass at k=64 is large at T=2.0",
        {
            "revision": REVISION,
            "top_k": 64,
            "temperature": 2.0,
            "tail_mass_mean": dist["teacher_tail_mass"]["mean"],
            "tail_mass_first": dist["teacher_tail_mass"]["first"],
            "tail_mass_final": dist["teacher_tail_mass"]["final"],
            "entropy_mean": dist["teacher_entropy"]["mean"],
            "measured_over": "100 KD steps, 512-token sequences of public-domain English",
            "reading": (
                "about 67% of the teacher's probability mass at T=2.0 falls outside the "
                "top-64 shortlist. The bucket tail treatment keeps the objective exact "
                "regardless, because the full-vocabulary logsumexp is captured before "
                "truncation — but an OFFLINE corpus storing only k=64 would discard that "
                "mass, so this argues against a small-k offline format."
            ),
            "IMPORTANT_CAVEAT": (
                "this is measured at the KD temperature T=2.0, which flattens the "
                "distribution and inflates tail mass. The offline-vs-online decision (B6) "
                "needs the T=1.0 sweep from smoke-test check 9, which did not complete. "
                "Do not treat this number as the B6 input."
            ),
            "does_not_close": "B6",
        },
        tags=["kd", "teacher_signal", "B6"],
    )

    ledger.estimated(
        "run001_feasibility",
        "Run 001 at canonical scale does not fit a single A40",
        {
            "verdict": "INFEASIBLE on this hardware; Run 001 was NOT executed",
            "student_parameters": 13_008_505_728,
            "gpu": "NVIDIA A40",
            "measured_total_vram_gib": 44.431640625,
            "usable_after_reserved_vram_gib_1_0": 43.43,
            "resident_teacher_gib_measured": 16.456,
            "budget_left_for_student_gib": 26.98,
            "required_gib_full_adamw": 193.8,
            "required_gib_full_adamw_with_teacher": 210.2,
            "over_by_factor": 4.8,
            "breakdown_gib": {"weights": 48.5, "gradients": 48.5, "adamw_moments": 96.9},
            "why_no_cheaper_path_exists_in_this_code": [
                "trainer.py raises NotImplementedError for strategy 'lora'/'qlora'; only "
                "'full' is implemented, so there is no adapter path",
                "trainer.py hardcodes torch.optim.AdamW; training.optimizer 'adamw_8bit' "
                "changes the memory estimate and the record but not the optimizer",
                "KD over a text corpus requires objective.signal_source 'online', so the "
                "teacher must stay resident; build_provider('offline') raises",
                "nothing sets requires_grad=False, so a pretrained student trains fully",
            ],
            "largest_full_parameter_student_that_would_fit": (
                "~1.7B parameters at 16 bytes/param with the teacher resident, before "
                "activations — an order of magnitude below the frozen target"
            ),
            "what_would_unblock_it": [
                "a GPU (or multi-GPU/offload setup) with roughly 5x this capacity, or",
                "implementing the declared-but-absent LoRA/QLoRA strategy, or",
                "an offline logit corpus so the teacher need not be resident — itself "
                "gated on the B6 tail-mass measurement",
            ],
            "note": "Run 002 was not launched, as instructed.",
        },
        method=(
            "the project's own memory model: TRAINABLE_FRACTION, OPTIMIZER_MOMENT_BYTES "
            "and PRECISION_SCHEMES from qwen_distill.diagnostics.fit, applied to the "
            "audited parameter count, against the measured card capacity and the measured "
            "resident-teacher footprint"
        ),
        tags=["run001", "16gb", "blocker"],
    )

    print(f"ledger entries: {len(ledger.entries())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
