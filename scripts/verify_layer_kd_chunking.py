#!/usr/bin/env python3
"""Prove the chunked layer objective is the unchunked one, on Run 003's own batch.

Run 003's 1536-token calibration cleared its step and failed its memory gate at 42.5354
GiB allocated against 42.0. The excess is entirely in the loss: ``mse_loss`` saves both of
its normalised fp32 inputs, so holding all 48 mapped pairs at 1536 positions costs roughly
4 GiB that cannot be released until the gradient exists.
:func:`~qwen_distill.distillation.behavioral.behavioral_loss_chunked` takes that gradient a
few pairs at a time instead.

That is a change to *when the gradient is taken*, not to what is optimised — and this
script is the evidence, measured rather than argued, on the real teacher, the real student
and the exact batch the calibration saw. Not a synthetic tensor and not a smaller sequence:
the unit tests already cover the algebra, and what they cannot cover is whether it still
holds at 1536 positions in bf16 through a 4-bit teacher.

Two stages, run separately so neither has to share the card with the other.

``--stage hidden``
    Both models forward under ``no_grad``; the mapped hidden states become leaves; both
    forms of the objective are evaluated on the *same* tensors. Compares the scalar terms,
    the per-layer diagnostics, and ``d(objective)/d(h_student)`` at all 48 supervised
    layers. This is the objective's own equivalence, isolated from the student's backward.

``--stage parameters``
    The end-to-end claim: two complete forward/backward passes over the same batch, one
    per form, comparing the gradient that actually reaches the optimizer on all LoRA
    parameters. Slower and heavier — the unchunked pass is the 42.5 GiB one — but it is
    the gradient the run is trained by.

Exit codes: ``0`` equivalent within tolerance, ``1`` a difference exceeded it, ``2``
refused before loading anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import kd_run

from qwen_distill.distillation.backends import TransformersTeacher
from qwen_distill.distillation.behavioral import behavioral_loss, behavioral_loss_chunked
from qwen_distill.training.text_data import ResumableBatchSampler
from qwen_distill.training.tokenized_data import prepare_tokenized_corpus
from qwen_distill.training.trainer import _layer_mapping, build_model

#: Both forms reduce a mean over 48 pairs; they differ only in summation order, so the
#: scalar is expected to agree to float32's own resolution and no better.
VALUE_TOLERANCE = 1e-6
#: Every pair's term reaches ``backward`` with the identical coefficient 1/48 through the
#: identical kernels, so gradients are expected to agree far more closely than the scalar.
#: Reported as a relative difference against the reference gradient's own scale, because
#: an absolute threshold on a bf16 gradient says nothing without knowing its magnitude.
GRADIENT_TOLERANCE = 1e-5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("hidden", "parameters"), default="hidden")
    parser.add_argument("--repeats", type=int, default=3,
                        help="passes per form in the parameters stage. The student's "
                             "backward is not bit-reproducible, so a single pass of each "
                             "form cannot separate the objective from the hardware")
    parser.add_argument("--chunk-pairs", type=int, default=4,
                        help="the chunk width to validate; must be the width the run uses")
    parser.add_argument("--record", type=Path, default=None,
                        help="write the measured differences here as JSON")
    # Everything below is passed through to kd_run's own argument parser so the
    # configuration under test is built by the same code that builds the run's.
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--text-path", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=1536)
    parser.add_argument("--max-tokens", type=int, default=700_000)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--kd-top-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def run_configuration(args):
    """Run 003's configuration, built by ``kd_run.build_config`` rather than restated."""
    run_args = kd_run.parse_args([
        "--teacher", str(args.teacher), "--revision", args.revision,
        "--student", "canonical", "--pretrained", str(args.pretrained),
        "--text-path", str(args.text_path),
        "--sequence-length", str(args.sequence_length),
        "--max-tokens", str(args.max_tokens),
        "--steps", "1", "--batch-size", "1", "--gradient-accumulation-steps", "1",
        "--objective", "layer_kd", "--kd-weight", "1.0",
        "--kd-temperature", str(args.kd_temperature), "--kd-top-k", str(args.kd_top_k),
        "--strategy", "qlora", "--optimizer", "adamw",
        "--lora-rank", "16", "--lora-alpha", "32", "--precision", "bf16",
        "--seed", str(args.seed), "--log-every", "1",
        "--name", "run003_chunking_equivalence",
        "--output", "/tmp/run003_chunking_equivalence",
    ])
    return kd_run.build_config(run_args, args.pretrained)


def calibration_batch(config, device):
    """The exact first batch Run 003's calibration trained on.

    Same corpus, same tokenizer, same sequence length, same sampler, same seed — read
    through the trainer's own corpus and sampler code, so a batch that differed from the
    run's would have to come from a change the run would see too.
    """
    import torch

    train_sequences, _, stats = prepare_tokenized_corpus(
        text_path=config.data.text_path,
        tokenizer_path=config.data.tokenizer_path,
        sequence_length=config.data.max_sequence_length,
        validation_fraction=config.data.validation_fraction,
        document_separator=config.data.document_separator,
        max_documents=config.data.max_documents,
        max_tokens=config.data.max_tokens,
        max_bytes=config.data.max_corpus_bytes,
        expected_vocab_size=config.data.expected_vocab_size,
        teacher_model=config.teacher.get("model"),
        teacher_revision=config.teacher.get("revision"),
        trust_remote_code=config.model.trust_remote_code,
    )
    batches = ResumableBatchSampler(
        train_sequences, config.training.batch_size, seed=config.training.seed)
    batch = torch.tensor(next(batches), dtype=torch.long, device=device)
    print(f"  corpus  : {stats.n_bytes:,} bytes, sha256 {stats.sha256[:16]}, "
          f"{stats.n_sequences} sequences of {stats.sequence_length}")
    print(f"  batch   : {tuple(batch.shape)}, first ids {batch[0, :8].tolist()}")
    return batch, stats


def load_teacher(args, config):
    backend = TransformersTeacher(
        model=config.teacher["model"], revision=config.teacher["revision"],
        local_path=str(args.teacher), quantization="4bit", strict_architecture=True,
    )
    backend.load()
    return backend.signal_provider(
        top_k=args.kd_top_k or None, temperature=args.kd_temperature,
        capture_hidden_states=True,
    )


def build_student(config, device):
    """The student exactly as the trainer builds it: 4-bit base, LoRA, checkpointing on."""
    model = build_model(config, None)
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    trainable = [p for _, p in sorted(
        (n, p) for n, p in model.named_parameters() if p.requires_grad)]
    print(f"  student : {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{sum(p.numel() for p in trainable):,} trainable")
    return model, trainable


def difference(a, b):
    """``(max absolute difference, relative to the reference's own largest magnitude)``."""
    d = float((a.float() - b.float()).abs().max())
    scale = float(a.float().abs().max())
    return d, (d / scale if scale else 0.0)


# ---------------------------------------------------------------------------
# stage: the objective and its gradient with respect to the hidden states
# ---------------------------------------------------------------------------
def stage_hidden(args, config, device):
    import torch

    teacher = load_teacher(args, config)
    batch, stats = calibration_batch(config, device)
    model, _ = build_student(config, device)

    print("\n  forwards (both under no_grad: only the hidden states are wanted)")
    with torch.no_grad():
        signal = teacher.signal_for(batch)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            student_out = model(input_ids=batch, output_hidden_states=True)
    student_hidden = [h.detach() for h in student_out.hidden_states]
    teacher_hidden = [h.detach() for h in signal.hidden_states]
    del student_out, signal
    # The student is no longer needed: this stage compares the loss function, and holding
    # 13B of weights through it only narrows the headroom the comparison runs in.
    del model
    torch.cuda.empty_cache()

    mapping = _layer_mapping(len(student_hidden) - 1, len(teacher_hidden) - 1,
                             config.training.layer_kd_map_strategy)
    print(f"  mapping : {len(mapping.mapping)} pairs, "
          f"{len(mapping.removed_teacher_layers)} teacher layers unsupervised")
    print(f"  states  : student {tuple(student_hidden[1].shape)} "
          f"{student_hidden[1].dtype}, teacher {tuple(teacher_hidden[1].shape)} "
          f"{teacher_hidden[1].dtype}")

    kwargs = dict(mode="pointwise",
                  direction_weight=config.training.layer_kd_direction_weight,
                  normalise=config.training.layer_kd_normalise)

    print("\n  reference: behavioral_loss, all pairs held to one backward")
    leaves = [h.clone().requires_grad_(True) for h in student_hidden]
    torch.cuda.reset_peak_memory_stats()
    reference = behavioral_loss(leaves, teacher_hidden, mapping.mapping, **kwargs)
    reference.total.backward()
    reference_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    reference_values = {k: float(v) for k, v in
                        (("total", reference.total.detach()),
                         ("magnitude", reference.magnitude.detach()),
                         ("direction", reference.direction.detach()))}
    reference_per_layer = dict(reference.per_layer)
    reference_norms = (reference.student_norm, reference.teacher_norm)
    reference_grads = [None if leaf.grad is None else leaf.grad.clone() for leaf in leaves]
    del reference, leaves
    torch.cuda.empty_cache()
    print(f"    total {reference_values['total']:.6f}  "
          f"magnitude {reference_values['magnitude']:.6f}  "
          f"direction {reference_values['direction']:.6f}   "
          f"peak {reference_peak:.4f} GiB")

    print(f"\n  chunked: behavioral_loss_chunked, {args.chunk_pairs} pair(s) per gradient")
    leaves = [h.clone().requires_grad_(True) for h in student_hidden]
    torch.cuda.reset_peak_memory_stats()
    chunked = behavioral_loss_chunked(leaves, teacher_hidden, mapping.mapping,
                                      chunk_pairs=args.chunk_pairs, **kwargs)
    chunked.backward()
    chunked_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    chunked_values = {k: float(v) for k, v in
                      (("total", chunked.output.total),
                       ("magnitude", chunked.output.magnitude),
                       ("direction", chunked.output.direction))}
    print(f"    total {chunked_values['total']:.6f}  "
          f"magnitude {chunked_values['magnitude']:.6f}  "
          f"direction {chunked_values['direction']:.6f}   "
          f"peak {chunked_peak:.4f} GiB  ({chunked.n_chunks} chunks)")
    # Both peaks are absolute allocations with the teacher's weights and both sides'
    # hidden states already resident, and the chunked one additionally carries the
    # reference's 48 saved gradients. They are indicative only. The memory evidence is
    # the calibration run's own step profile, not this stage.

    # --- compare -------------------------------------------------------------------
    value_diffs = {k: abs(chunked_values[k] - reference_values[k])
                   for k in reference_values}
    value_rel = {k: (value_diffs[k] / abs(reference_values[k]) if reference_values[k]
                     else value_diffs[k]) for k in reference_values}
    per_layer_diff = max(
        (abs(chunked.output.per_layer[s] - reference_per_layer[s])
         for s in reference_per_layer), default=0.0)

    worst_abs = worst_rel = 0.0
    worst_layer = None
    compared = 0
    for i, (want, got) in enumerate(zip(reference_grads, leaves, strict=True)):
        if want is None:
            assert got.grad is None or float(got.grad.abs().max()) == 0.0, \
                f"the chunked form put a gradient on hidden state {i} and the reference did not"
            continue
        assert got.grad is not None, f"the chunked form left hidden state {i} ungraded"
        compared += 1
        d_abs, d_rel = difference(want, got.grad)
        if d_rel > worst_rel:
            worst_abs, worst_rel, worst_layer = d_abs, d_rel, i
    assert compared == len(mapping.mapping), \
        f"compared {compared} gradients, expected {len(mapping.mapping)}"

    print(f"\n  gradient: {compared} supervised hidden states compared")
    print(f"    worst absolute difference  {worst_abs:.6e}  (hidden state {worst_layer})")
    print(f"    worst relative difference  {worst_rel:.6e}")
    print(f"  value   : total {value_rel['total']:.3e} relative, "
          f"per-layer worst {per_layer_diff:.3e} absolute")

    ok = (max(value_rel.values()) <= VALUE_TOLERANCE
          and worst_rel <= GRADIENT_TOLERANCE)
    return ok, {
        "stage": "hidden",
        "chunk_pairs": args.chunk_pairs,
        "n_chunks": chunked.n_chunks,
        "n_pairs": len(mapping.mapping),
        "n_gradients_compared": compared,
        "sequence_length": config.data.max_sequence_length,
        "corpus_sha256": stats.sha256,
        "hidden_dtype": str(student_hidden[1].dtype),
        "reference": reference_values | {
            "student_norm": reference_norms[0], "teacher_norm": reference_norms[1]},
        "chunked": chunked_values | {
            "student_norm": chunked.output.student_norm,
            "teacher_norm": chunked.output.teacher_norm},
        "value_absolute_difference": value_diffs,
        "value_relative_difference": value_rel,
        "per_layer_worst_absolute_difference": per_layer_diff,
        "gradient_worst_absolute_difference": worst_abs,
        "gradient_worst_relative_difference": worst_rel,
        "gradient_worst_at_hidden_state": worst_layer,
        "loss_peak_allocated_gib": {"reference": reference_peak, "chunked": chunked_peak},
        "tolerance": {"value_relative": VALUE_TOLERANCE,
                      "gradient_relative": GRADIENT_TOLERANCE},
    }


# ---------------------------------------------------------------------------
# stage: the gradient the optimizer actually receives
# ---------------------------------------------------------------------------
def stage_parameters(args, config, device):
    """Three complete forward/backward passes, comparing every LoRA parameter's gradient.

    The hidden stage proves the loss function agrees. This proves the *run* agrees: the
    same batch, the same student, the same teacher, and the gradient that reaches AdamW.

    Two things have to be controlled for, and a two-pass version of this got both wrong.

    **Dropout.** ``lora_dropout`` is 0.05 and the model is in ``train()`` mode. LoRA's B is
    zero-initialised, so the dropout mask cannot affect the forward — the loss is identical
    to the last decimal either way — but ``grad_B = grad_out @ dropout(x) A^T`` depends on
    it entirely. Two passes that resample the mask disagree on every ``lora_B`` gradient by
    tens of percent, and none of it has anything to do with chunking. Every pass therefore
    restores the same RNG state before it starts.

    **The student's own reproducibility.** Even with the mask fixed, a 13B MoE with DeltaNet
    kernels and gradient checkpointing need not produce a bit-identical backward twice:
    atomic accumulation and routing scatters are order-dependent on CUDA. So the second
    pass repeats the *unchunked* form, establishing the floor that the chunked form's
    difference has to be read against. A chunked-vs-unchunked difference is only evidence
    of a real disagreement if it exceeds what the unchunked form already shows against
    itself.
    """
    import torch

    teacher = load_teacher(args, config)
    batch, stats = calibration_batch(config, device)
    model, trainable = build_student(config, device)
    names = [n for n, p in sorted(model.named_parameters()) if p.requires_grad]
    rng = {"cpu": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state() if device == "cuda" else None}

    def one_pass(chunk_pairs):
        model.zero_grad(set_to_none=True)
        # Same dropout masks in every pass. Without this the comparison measures the RNG.
        torch.set_rng_state(rng["cpu"])
        if rng["cuda"] is not None:
            torch.cuda.set_rng_state(rng["cuda"])
        torch.cuda.reset_peak_memory_stats()
        signal = teacher.signal_for(batch)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            out = model(input_ids=batch, output_hidden_states=True)
            mapping = _layer_mapping(len(out.hidden_states) - 1,
                                     len(signal.hidden_states) - 1,
                                     config.training.layer_kd_map_strategy)
            kwargs = dict(mode="pointwise",
                          direction_weight=config.training.layer_kd_direction_weight,
                          normalise=config.training.layer_kd_normalise)
            if chunk_pairs is None:
                output = behavioral_loss(out.hidden_states, signal.hidden_states,
                                         mapping.mapping, **kwargs)
                signal.hidden_states = None
                total = float(output.total.detach())
                output.total.backward()
            else:
                held = behavioral_loss_chunked(
                    out.hidden_states, signal.hidden_states, mapping.mapping,
                    chunk_pairs=chunk_pairs, **kwargs)
                signal.hidden_states = None
                total = float(held.output.total)
                held.backward()
                del held
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        grads = [None if p.grad is None else p.grad.detach().to("cpu", torch.float32)
                 for p in trainable]
        del out, signal
        torch.cuda.empty_cache()
        return total, grads, peak

    def per_parameter(a, b):
        """Absolute gradient difference for every LoRA parameter, by name."""
        out = {}
        for name, want, got in zip(names, a, b, strict=True):
            if want is None and got is None:
                continue
            assert want is not None and got is not None, \
                f"{name}: one pass produced no gradient"
            out[name] = difference(want, got)
        return out

    passes = {"unchunked": [], "chunked": []}
    for i in range(args.repeats):
        print(f"\n  unchunked pass {i + 1} of {args.repeats}")
        passes["unchunked"].append(one_pass(None))
        print(f"    layer term {passes['unchunked'][-1][0]:.6f}   step peak "
              f"{passes['unchunked'][-1][2]:.4f} GiB allocated")
    for i in range(args.repeats):
        print(f"\n  chunked pass {i + 1} of {args.repeats}  "
              f"({args.chunk_pairs} pair(s) per gradient)")
        passes["chunked"].append(one_pass(args.chunk_pairs))
        print(f"    layer term {passes['chunked'][-1][0]:.6f}   step peak "
              f"{passes['chunked'][-1][2]:.4f} GiB allocated")

    grads = [g for _, g, _ in passes["unchunked"] + passes["chunked"]]
    labels = ["unchunked"] * args.repeats + ["chunked"] * args.repeats
    totals = {k: [t for t, _, _ in v] for k, v in passes.items()}
    peaks = {k: [p for _, _, p in v] for k, v in passes.items()}
    n = len(grads)

    # Every pairwise difference, computed once. Everything below is a re-reading of this
    # table under different groupings of the same six passes.
    table = {}
    for i in range(n):
        for j in range(i + 1, n):
            table[(i, j)] = per_parameter(grads[i], grads[j])
    names_seen = sorted({k for d in table.values() for k in d})
    varying = [k for k in names_seen
               if any(d.get(k, (0.0, 0.0))[0] > 0 for d in table.values())]

    def pair(i, j):
        return table[(min(i, j), max(i, j))]

    def cross(group_a, group_b):
        """Per-parameter worst absolute difference across two groups of passes."""
        out = {}
        for i in group_a:
            for j in group_b:
                for name, (d_abs, _) in pair(i, j).items():
                    out[name] = max(out.get(name, 0.0), d_abs)
        return out

    def statistics(group_a, group_b):
        c = cross(group_a, group_b)
        interesting = [c[k] for k in varying] or [0.0]
        return {"max": max(interesting), "mean": sum(interesting) / len(interesting)}

    # Exchangeability test. If the chunked form and the unchunked form differ only by the
    # hardware's own non-reproducibility, the labels carry no information: splitting these
    # six passes into two groups of three any other way should separate them just as much.
    # So the same statistic is computed for every 3/3 split, and the true split's rank
    # among them is the result. Comparing "between forms" against "within a form" directly
    # would not do: the between set has nine pairs and each within set has three, and the
    # maximum of more draws is larger for no reason but the counting.
    from itertools import combinations

    everything = list(range(n))
    true_group = tuple(i for i in everything if labels[i] == "unchunked")
    splits, seen = [], set()
    for group in combinations(everything, args.repeats):
        other = tuple(i for i in everything if i not in group)
        key = frozenset((group, other))
        if key in seen:
            continue
        seen.add(key)
        splits.append((group, other))
    scored = [(statistics(a, b), a, b) for a, b in splits]
    true_stats = statistics(true_group, tuple(i for i in everything if i not in true_group))
    rank_max = 1 + sum(1 for st, _, _ in scored if st["max"] > true_stats["max"])
    rank_mean = 1 + sum(1 for st, _, _ in scored if st["mean"] > true_stats["mean"])
    p_max = sum(1 for st, _, _ in scored if st["max"] >= true_stats["max"]) / len(scored)
    p_mean = sum(1 for st, _, _ in scored if st["mean"] >= true_stats["mean"]) / len(scored)

    def within(group):
        """The same statistic over a single group's own pairs."""
        out = {}
        for i, j in combinations(group, 2):
            for name, (d_abs, _) in pair(i, j).items():
                out[name] = max(out.get(name, 0.0), d_abs)
        interesting = [out[k] for k in varying if k in out] or [0.0]
        return {"max": max(interesting), "mean": sum(interesting) / len(interesting)}

    other_group = tuple(i for i in everything if i not in true_group)
    within_unchunked = within(true_group)
    within_chunked = within(other_group)

    value_spread = max(abs(t - totals["unchunked"][0])
                       for t in totals["unchunked"] + totals["chunked"])
    value_rel = (value_spread / abs(totals["unchunked"][0])
                 if totals["unchunked"][0] else 0.0)

    print(f"\n  gradient: {len(names_seen)} LoRA parameters, "
          f"{len(names_seen) - len(varying)} identical in all {n} passes")
    print(f"    the student's backward is not bit-reproducible: two identical unchunked "
          f"passes differ by up to {within_unchunked['max']:.3e}")
    print(f"    true split (unchunked | chunked)   max {true_stats['max']:.6e}   "
          f"mean {true_stats['mean']:.6e}")
    print(f"    over all {len(scored)} {args.repeats}/{args.repeats} splits of the same "
          f"{n} passes:")
    print(f"      rank of the true split by max   {rank_max} of {len(scored)}  "
          f"(p = {p_max:.2f})")
    print(f"      rank of the true split by mean  {rank_mean} of {len(scored)}  "
          f"(p = {p_mean:.2f})")
    print(f"  value   : layer term spread {value_rel:.3e} relative over all {n} passes")
    print(f"  memory  : {peaks['unchunked'][0]:.4f} -> {peaks['chunked'][0]:.4f} GiB "
          f"allocated ({peaks['chunked'][0] - peaks['unchunked'][0]:+.4f})")

    # Not extreme means the label explains nothing the hardware does not already explain.
    indistinguishable = p_max > 0.1 and p_mean > 0.1
    ok = value_rel <= VALUE_TOLERANCE and indistinguishable
    print("  verdict : value "
          + ("ok" if value_rel <= VALUE_TOLERANCE else "FAILED")
          + ", gradients "
          + ("indistinguishable from the hardware's own non-reproducibility"
             if indistinguishable else
             "SEPARATE by form — the chunked path changes the gradient"))

    return ok, {
        "stage": "parameters",
        "chunk_pairs": args.chunk_pairs,
        "repeats_per_form": args.repeats,
        "sequence_length": config.data.max_sequence_length,
        "corpus_sha256": stats.sha256,
        "lora_dropout": config.training.lora_dropout,
        "rng_state_restored_before_each_pass": True,
        "method": (
            "the student's backward is not bit-reproducible on this hardware — a 13B MoE "
            "with DeltaNet kernels and gradient checkpointing accumulates through "
            "order-dependent atomics and routing scatters — so a single unchunked pass is "
            "not a fixed reference and 'between forms' cannot be compared against 'within "
            "a form' directly, the between set having three times the pairs. Instead all "
            "passes are split evenly every possible way and the true split's separation "
            "is ranked among them. A typical rank means the form label explains nothing "
            "the hardware does not already explain."
        ),
        "layer_term": totals,
        "value_relative_spread": value_rel,
        "n_parameters": len(names_seen),
        "n_parameters_identical_in_all_passes": len(names_seen) - len(varying),
        "within_unchunked": within_unchunked,
        "within_chunked": within_chunked,
        "true_split": true_stats,
        "n_splits": len(scored),
        "rank_of_true_split_by_max": rank_max,
        "rank_of_true_split_by_mean": rank_mean,
        "p_max": p_max,
        "p_mean": p_mean,
        "indistinguishable": indistinguishable,
        "all_split_statistics": sorted((st["max"] for st, _, _ in scored), reverse=True),
        "step_peak_allocated_gib": peaks,
        "tolerance": {"value_relative": VALUE_TOLERANCE},
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if not (args.teacher / "config.json").is_file():
        print(f"  REFUSED: no teacher checkpoint at {args.teacher}", file=sys.stderr)
        return 2
    if not args.text_path.is_file():
        print(f"  REFUSED: no corpus at {args.text_path}", file=sys.stderr)
        return 2
    if args.chunk_pairs < 1:
        print("  REFUSED: --chunk-pairs must be at least 1", file=sys.stderr)
        return 2

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = run_configuration(args)
    started = time.time()
    print(f"\n  LAYER-KD CHUNKING EQUIVALENCE  stage '{args.stage}'  device {device}")
    print(f"  objective : {config.training.objective}, pointwise, direction weight "
          f"{config.training.layer_kd_direction_weight}, normalise "
          f"{config.training.layer_kd_normalise}")

    stage = stage_hidden if args.stage == "hidden" else stage_parameters
    ok, record = stage(args, config, device)

    record["elapsed_s"] = round(time.time() - started, 1)
    record["equivalent"] = ok
    record["teacher"] = {"model": config.teacher["model"],
                         "revision": config.teacher["revision"], "quantization": "4bit"}
    record["student_checkpoint"] = str(args.pretrained)

    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\n  wrote {args.record}")

    if ok:
        print("\n  EQUIVALENT within the declared tolerance.")
        return 0
    print("\n  NOT EQUIVALENT: a difference exceeded the declared tolerance. The chunked "
          "form must not be used for Run 003.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
