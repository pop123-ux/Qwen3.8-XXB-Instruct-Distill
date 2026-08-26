"""Candidate architectures for the next scale step, and where each of them fits.

Level 2 established one point: a 94.48M hybrid trains on a T4 at ~2,090 tok/s. The next
question is what a 250M–500M version costs, and on which cards.

**No new estimator is defined here.** Every memory number comes from
:func:`qwen_distill.diagnostics.fit.estimate_training_memory`, which already models the
term that mattered — the Gated DeltaNet activation explosion whose absence caused the
Level-2 OOM — and is calibrated against six measured configurations. Inventing a second
estimator would produce two numbers that disagree and no way to tell which is wrong. This
module supplies architectures and configurations; ``fit.py`` supplies the arithmetic.

**Parameter counts are measured, not chosen.** Each candidate is built from a shape rule
and its parameter count is read out of :func:`qwen_distill.architecture.params.count_parameters`.
The 250M/350M/500M figures are *scale classes*, not targets to be hit — the candidates
land where the shape rule puts them, and the measured count is reported. A rounder number
would mean the shape had been distorted to produce it.

**The candidates bracket each class rather than centring on it.** Two per class, one wider
and shallower, one narrower and deeper. Two architectures of the same size that differ in
aspect ratio have visibly different activation costs, and picking one per class would hide
that.

What this module cannot tell you: whether any of these sizes is *worth* training. That is
what the scaling study in ``docs/experiments/SCALING_STUDY.md`` is for, and it needs
measurements this analysis does not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..architecture.params import count_parameters
from ..architecture.spec import HybridArchSpec
from ..diagnostics.fit import TrainingFit, estimate_training_memory
from ..diagnostics.tiers import classify

# ----------------------------------------------------------------------------------
# the shape rule
# ----------------------------------------------------------------------------------

#: Ratios read off the Level-2 94.48M architecture and held fixed as it scales. Holding
#: them fixed is what makes the ladder a scaling study rather than six unrelated models:
#: if the aspect ratios drift, a difference between two rungs could be the size or could
#: be the shape, and nothing in the result says which.
#:
#: Level 2: hidden 640, 16 layers, ff 2176, 10 heads / 2 kv, head_dim 64,
#: DeltaNet 4 key heads / 12 value heads at 64 dim, full-attention every 4th layer.
FF_RATIO = 3.4                      # 2176 / 640
HEAD_DIM = 64                       # 640 / 10
DELTANET_KEY_HEAD_DIVISOR = 160     # 640 / 4
DELTANET_VALUE_PER_KEY_HEAD = 3     # 12 / 4
FULL_ATTENTION_INTERVAL = 4
GQA_TARGET_RATIO = 5                # 10 heads / 2 kv

#: Byte-level throughout: a token is a byte, so the vocabulary contributes almost nothing
#: to the parameter count and scaling is dominated by the blocks. Keeping it at 256 also
#: keeps every rung's bits-per-byte on the same scale.
BYTE_VOCAB_SIZE = 256


def _key_value_heads(num_attention_heads: int) -> int:
    """Largest divisor of ``num_attention_heads`` at or below the GQA target ratio.

    GQA requires divisibility. A head count that is prime would otherwise collapse to
    multi-query attention (1 kv head), which is a different architecture, not a rounding
    difference — so candidates whose head count forces that are excluded rather than
    silently accepted.
    """
    target = max(2, num_attention_heads // GQA_TARGET_RATIO)
    for candidate in range(target, 1, -1):
        if num_attention_heads % candidate == 0:
            return candidate
    return 1


def scaled_spec(hidden_size: int, num_hidden_layers: int, *, name: str) -> HybridArchSpec:
    """Build a candidate by applying the Level-2 shape rule at a new size.

    Raises rather than adjusting anything if the result would not be a valid GQA
    configuration: a candidate that has to be fudged into existence is not a member of
    this family.
    """
    num_attention_heads = hidden_size // HEAD_DIM
    if num_attention_heads * HEAD_DIM != hidden_size:
        raise ValueError(f"hidden_size {hidden_size} is not a multiple of head_dim {HEAD_DIM}")
    key_value_heads = _key_value_heads(num_attention_heads)
    if key_value_heads < 2:
        raise ValueError(
            f"hidden_size {hidden_size} gives {num_attention_heads} attention heads, which "
            f"has no divisor below the GQA ratio — that would be multi-query attention, a "
            f"different architecture from Level 2's"
        )
    if num_hidden_layers % FULL_ATTENTION_INTERVAL:
        raise ValueError(
            f"num_hidden_layers {num_hidden_layers} is not a multiple of "
            f"{FULL_ATTENTION_INTERVAL}: the last layer would be linear attention, which "
            f"changes the layout rather than scaling it"
        )
    linear_key_heads = max(2, hidden_size // DELTANET_KEY_HEAD_DIVISOR)
    return HybridArchSpec(
        name=name,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=round(hidden_size * FF_RATIO / 128) * 128,
        vocab_size=BYTE_VOCAB_SIZE,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=key_value_heads,
        head_dim=HEAD_DIM,
        linear_num_key_heads=linear_key_heads,
        linear_num_value_heads=DELTANET_VALUE_PER_KEY_HEAD * linear_key_heads,
        linear_key_head_dim=HEAD_DIM,
        linear_value_head_dim=HEAD_DIM,
        linear_conv_kernel_dim=4,
        full_attention_interval=FULL_ATTENTION_INTERVAL,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        provenance="shape rule derived from the Level-2 94.48M architecture",
    )


#: The Level-2 architecture, rebuilt through the same rule so the ladder starts from a
#: measured rung rather than an assumed one.
LEVEL2_SPEC = scaled_spec(640, 16, name="level2_94m")


@dataclass(frozen=True)
class ScaleClass:
    """A size band, and the candidates that bracket it."""

    label: str
    target_parameters: int
    shapes: tuple[tuple[int, int], ...]   # (hidden_size, num_hidden_layers)


#: Two candidates per class, bracketing the nominal size from below and above where the
#: shape rule allows. Named for the class, not for their parameter count, because the
#: count is an output.
SCALE_CLASSES: tuple[ScaleClass, ...] = (
    ScaleClass("250M", 250_000_000, ((1024, 16), (960, 20))),
    ScaleClass("350M", 350_000_000, ((1024, 24), (1152, 20))),
    ScaleClass("500M", 500_000_000, ((1280, 20), (1152, 28))),
)


def _candidate_name(scale: ScaleClass, hidden: int, layers: int) -> str:
    return f"{scale.label}_h{hidden}_L{layers}"


def build_candidates() -> list[HybridArchSpec]:
    """Every candidate, in class order."""
    specs = []
    for scale in SCALE_CLASSES:
        for hidden, layers in scale.shapes:
            specs.append(scaled_spec(hidden, layers, name=_candidate_name(scale, hidden, layers)))
    return specs


# ----------------------------------------------------------------------------------
# devices
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceBudget:
    """One VRAM budget, with the gap between the marketing number and the real one.

    A "16 GB" T4 reports **14.56 GiB**. That is not a rounding difference — it is 1.44 GiB
    of the exact resource being budgeted, and planning against 16 is how a configuration
    that "obviously fits" OOMs. ``total_gib`` is the number the driver reports;
    ``reserved_gib`` is what is left for the driver, the display and any compositor.
    """

    label: str
    nominal_gb: int
    total_gib: float
    reserved_gib: float
    examples: str
    #: ``measured`` means this project read it off the device. Anything else is a
    #: vendor figure converted to GiB and should be treated as approximate.
    source: str

    @property
    def usable_gib(self) -> float:
        return round(self.total_gib - self.reserved_gib, 2)

    @property
    def tier(self) -> str:
        return classify(self.total_gib).name

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "nominal_gb": self.nominal_gb,
            "total_gib": self.total_gib, "reserved_gib": self.reserved_gib,
            "usable_gib": self.usable_gib, "examples": self.examples,
            "source": self.source, "tier": self.tier,
        }


DEVICE_BUDGETS: tuple[DeviceBudget, ...] = (
    DeviceBudget(
        "12 GB", 12, 11.76, 1.0, "RTX 3060 12GB, RTX 2060 12GB",
        "vendor spec converted to GiB — not measured by this project",
    ),
    DeviceBudget(
        # The one figure here this project actually measured: Level 2's T4 reported
        # 14.56 GiB total, recorded in its RESULT.json.
        "16 GB", 16, 14.56, 1.0, "Tesla T4 (Colab free tier) — the deployment target",
        "measured on the Level-2 T4",
    ),
    DeviceBudget(
        "24 GB", 24, 23.69, 1.0, "RTX 3090, RTX 4090, A5000, L4",
        "vendor spec converted to GiB — not measured by this project",
    ),
    DeviceBudget(
        "48 GB", 48, 47.5, 1.5, "RTX A6000, L40S, A40",
        "vendor spec converted to GiB — not measured by this project",
    ),
)


# ----------------------------------------------------------------------------------
# training configurations
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """One set of the knobs that decide whether a size fits."""

    sequence_length: int
    batch_size: int
    gradient_checkpointing: bool = True
    optimizer: str = "adamw"
    precision: str = "fp16"
    strategy: str = "full"

    @property
    def tokens_per_micro_step(self) -> int:
        return self.sequence_length * self.batch_size

    @property
    def label(self) -> str:
        checkpointing = "ckpt" if self.gradient_checkpointing else "NO-ckpt"
        return (
            f"seq{self.sequence_length}xb{self.batch_size} {checkpointing} "
            f"{self.optimizer} {self.precision}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length, "batch_size": self.batch_size,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optimizer": self.optimizer, "precision": self.precision,
            "strategy": self.strategy,
            "tokens_per_micro_step": self.tokens_per_micro_step,
        }


#: Level 2's configuration exactly. Every sweep includes it, so each candidate has one
#: rung whose cost is anchored to something that actually ran.
LEVEL2_CONFIG = TrainingConfig(sequence_length=1024, batch_size=4)


def default_sweep(*, precision: str = "fp16", include_no_checkpointing: bool = True) -> list[TrainingConfig]:
    """The configurations worth trying, largest first.

    Ordered by tokens per micro-step descending so the first feasible entry is the
    largest that fits. Gradient checkpointing stays ON except for a single control at
    Level 2's shape: turning it off multiplies retained activations by roughly 67x for
    this architecture, and the point of including it is to show that, not to suggest it.
    """
    configs = [
        TrainingConfig(sequence_length=seq, batch_size=batch,
                       optimizer=optimizer, precision=precision)
        for seq in (2048, 1024, 512)
        for batch in (8, 4, 2, 1)
        for optimizer in ("adamw", "adamw_8bit")
    ]
    if include_no_checkpointing:
        configs.append(
            TrainingConfig(sequence_length=1024, batch_size=4,
                           gradient_checkpointing=False, precision=precision)
        )
    configs.sort(key=lambda c: (-c.tokens_per_micro_step, c.optimizer != "adamw"))
    return configs


# ----------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------

#: Level 2's measured run-wide throughput, on a T4 at seq 1024 / batch 4 / checkpointing
#: on. VERIFIED — 32,768,000 tokens in 15,684.6 s. It is the *only* throughput
#: measurement this project has, which is exactly why extrapolating from it is arithmetic
#: rather than prediction.
LEVEL2_TOKENS_PER_SECOND = 2089.2

#: The verdict ``estimate_training_memory`` returns when a configuration fits with room
#: to spare. Named, because ``fit.py`` carries **two** verdict vocabularies — inference
#: says ``FITS``/``TIGHT``/``DOES NOT FIT``, training says
#: ``PLAUSIBLE``/``TIGHT``/``NOT FEASIBLE`` — and matching the inference one against a
#: training fit silently reports every candidate as infeasible. ``TIGHT`` is deliberately
#: excluded: under 1 GiB of headroom is one background process away from an OOM.
#: ``tests/test_scaling_candidates.py`` asserts this string is still what ``fit.py``
#: produces, so a change there fails loudly instead of emptying the table.
ACCEPTABLE_VERDICT = "PLAUSIBLE"
TIGHT_VERDICT = "TIGHT"


@dataclass
class CandidateFit:
    """One candidate on one device: the best configuration that fits, if any."""

    candidate: str
    parameters: int
    scale_class: str
    device: DeviceBudget
    best: TrainingConfig | None = None
    fit: TrainingFit | None = None
    #: Every configuration tried, so a NOT FEASIBLE verdict can be argued with.
    attempts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.best is not None

    @property
    def verdict(self) -> str:
        if self.fit is None:
            return "NO CONFIGURATION FITS"
        return self.fit.verdict

    @property
    def binding_term(self) -> str | None:
        """Which component dominates. Knowing it is the difference between a useful
        change and a random one: shrinking the batch does nothing if the optimizer state
        is what does not fit."""
        if self.fit is None:
            return None
        terms = {
            "base weights": self.fit.base_weights_gib,
            "gradients": self.fit.gradients_gib,
            "optimizer state": self.fit.optimizer_state_gib,
            "DeltaNet activations": self.fit.deltanet_activations_gib,
            "attention activations": self.fit.attention_activations_gib,
            "other activations": self.fit.non_attention_activations_gib,
            "logits + fp32 loss copies": self.fit.logits_gib,
        }
        return max(terms, key=lambda key: terms[key])

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "parameters": self.parameters,
            "scale_class": self.scale_class,
            "device": self.device.to_dict(),
            "feasible": self.feasible,
            "verdict": self.verdict,
            "best_config": self.best.to_dict() if self.best else None,
            "binding_term": self.binding_term,
            "estimate": self.fit.component_estimate() if self.fit else None,
            "total_gib": self.fit.total_gib if self.fit else None,
            "headroom_gib": self.fit.headroom_gib if self.fit else None,
            "attempts": self.attempts,
            "notes": self.notes,
        }


def evaluate_candidate(
    spec: HybridArchSpec,
    device: DeviceBudget,
    *,
    scale_class: str = "",
    configs: list[TrainingConfig] | None = None,
) -> CandidateFit:
    """Find the largest configuration of ``spec`` that fits on ``device``.

    "Largest" means most tokens per micro-step, which is what governs how long a run
    takes. Every configuration tried is recorded, including the ones that did not fit, so
    a NOT FEASIBLE verdict can be checked rather than believed.
    """
    configs = configs or default_sweep()
    parameters = count_parameters(spec).total
    result = CandidateFit(
        candidate=spec.name, parameters=parameters, scale_class=scale_class, device=device
    )

    for config in configs:
        fit = estimate_training_memory(
            spec,
            device.usable_gib,
            strategy=config.strategy,
            optimizer=config.optimizer,
            sequence_length=config.sequence_length,
            batch_size=config.batch_size,
            gradient_checkpointing=config.gradient_checkpointing,
            precision=config.precision,
            label=f"{spec.name} @ {device.label}",
        )
        result.attempts.append({
            "config": config.label,
            "tokens_per_micro_step": config.tokens_per_micro_step,
            "total_gib": round(fit.total_gib, 2),
            "verdict": fit.verdict,
        })
        # PLAUSIBLE only. A TIGHT configuration is one background process away from an
        # OOM, and recommending one is how a run dies eight hours in.
        if result.best is None and fit.verdict == ACCEPTABLE_VERDICT:
            result.best, result.fit = config, fit

    if result.best is None:
        tightest = min(result.attempts, key=lambda a: a["total_gib"])
        result.notes.append(
            f"nothing fits with headroom; the smallest configuration tried "
            f"({tightest['config']}) needs {tightest['total_gib']:.2f} GiB against "
            f"{device.usable_gib:.2f} GiB usable and is {tightest['verdict']}"
        )
        # Report the tightest attempt's estimate so the shortfall is quantified rather
        # than merely asserted.
        result.fit = estimate_training_memory(
            spec, device.usable_gib, strategy="full", optimizer="adamw_8bit",
            sequence_length=512, batch_size=1, gradient_checkpointing=True,
            precision="fp16", label=f"{spec.name} @ {device.label} (floor)",
        )
        result.best = None
    elif result.best.optimizer == "adamw_8bit":
        result.notes.append(
            "only fits with 8-bit optimizer moments; full fp32 AdamW state does not"
        )

    if result.fit is not None and result.fit.deltanet_activations_gib > result.fit.base_weights_gib:
        result.notes.append(
            f"DeltaNet activations ({result.fit.deltanet_activations_gib:.2f} GiB) exceed "
            f"the weights ({result.fit.base_weights_gib:.2f} GiB) — this architecture's "
            f"cost is dominated by activations, not parameters"
        )
    return result


def extrapolated_tokens_per_second(spec: HybridArchSpec, *, sequence_length: int = 1024) -> dict[str, Any]:
    """Scale Level 2's measured rate by the FLOP ratio. **Not a prediction.**

    One measured point cannot establish how throughput scales. This divides Level 2's
    2,089.2 tok/s by the ratio of forward FLOPs per token and reports the result as
    arithmetic, with its single anchor named. Memory bandwidth, kernel efficiency and
    occupancy all change with shape and none of them appear in a FLOP count, so treat the
    number as an order of magnitude and nothing more.
    """
    from ..architecture.flops import prefill_flops

    baseline = prefill_flops(LEVEL2_SPEC, sequence_length) / sequence_length
    candidate = prefill_flops(spec, sequence_length) / sequence_length
    ratio = candidate / baseline if baseline else None
    return {
        "flops_per_token": candidate,
        "flops_ratio_vs_level2": round(ratio, 3) if ratio else None,
        "extrapolated_tokens_per_second": (
            round(LEVEL2_TOKENS_PER_SECOND / ratio, 1) if ratio else None
        ),
        "anchor": {
            "spec": LEVEL2_SPEC.name,
            "measured_tokens_per_second": LEVEL2_TOKENS_PER_SECOND,
            "hardware": "Tesla T4",
            "config": "seq 1024, batch 4, gradient checkpointing on, fp16 autocast",
        },
        "status": "UNVALIDATED EXTRAPOLATION",
        "caveat": (
            "Derived from ONE measured point by FLOP ratio. It ignores memory bandwidth, "
            "kernel efficiency and occupancy, all of which change with shape. Treat as an "
            "order of magnitude. A second measured point would make this a measurement; "
            "until then it is arithmetic."
        ),
    }


@dataclass
class ScalingMatrix:
    """Every candidate against every device budget."""

    fits: list[CandidateFit] = field(default_factory=list)
    baseline: CandidateFit | None = None
    #: The estimator's number for the configuration Level 2 **actually ran**, on the
    #: device it actually ran on. The only row here that can be checked against reality,
    #: and the reason to trust or distrust the rest.
    anchor: TrainingFit | None = None
    findings: list[str] = field(default_factory=list)

    def for_device(self, label: str) -> list[CandidateFit]:
        return [f for f in self.fits if f.device.label == label]

    def for_candidate(self, name: str) -> list[CandidateFit]:
        return [f for f in self.fits if f.candidate == name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "anchor": (
                {
                    "config": LEVEL2_CONFIG.to_dict(),
                    "device": "Tesla T4 (14.56 GiB)",
                    "estimated_total_gib": round(self.anchor.total_gib, 2),
                    "verdict": self.anchor.verdict,
                    "components": self.anchor.component_estimate(),
                    "note": (
                        "the configuration Level 2 actually ran and completed. If this "
                        "row is wrong, every other row is wrong the same way."
                    ),
                }
                if self.anchor else None
            ),
            "candidates": [f.to_dict() for f in self.fits],
            "findings": self.findings,
            "estimator": (
                "qwen_distill.diagnostics.fit.estimate_training_memory — the same "
                "estimator used for the Level-2 OOM diagnosis, calibrated against six "
                "measured configurations. No second estimator is defined here."
            ),
            "parameter_counts": (
                "measured with architecture.params.count_parameters from each built spec. "
                "The 250M/350M/500M labels are scale classes, not achieved counts."
            ),
        }

    def render(self) -> str:
        rule = "=" * 92
        lines = [
            rule, "SCALING CANDIDATES — where the next size step fits", rule, "",
            "  Memory from qwen_distill.diagnostics.fit (the Level-2 OOM estimator).",
            "  Parameter counts MEASURED from each built spec; 250M/350M/500M are class",
            "  labels, not achieved counts.",
            "",
            "  'Best config' is the largest the estimator calls PLAUSIBLE. TIGHT ones are",
            "  excluded: under 1 GiB spare is one background process away from an OOM.",
            "",
        ]

        if self.anchor is not None:
            lines += [
                "  ANCHOR — the configuration Level 2 actually ran and completed:",
                f"    94,476,448 params, {LEVEL2_CONFIG.label}, Tesla T4 (13.56 GiB usable)",
                f"    estimated {self.anchor.total_gib:.2f} GiB -> {self.anchor.verdict}; "
                f"the run did not OOM, so the estimate is not an over-prediction",
                "    Every other row is this estimator extrapolated. If this row is wrong,",
                "    they are all wrong the same way.",
                "",
            ]
        if self.baseline is not None:
            b = self.baseline
            lines += [
                "  headroom at 94M: largest configuration tried that still fits a T4 -> "
                + (b.best.label if b.best else "none"),
                "",
            ]

        header = f"  {'candidate':<20} {'params':>13}  " + "".join(
            f"{d.label:>17}" for d in DEVICE_BUDGETS
        )
        lines += [header, "  " + "-" * (len(header) - 2)]

        seen: list[str] = []
        for entry in self.fits:
            if entry.candidate not in seen:
                seen.append(entry.candidate)
        for name in seen:
            entries = {e.device.label: e for e in self.for_candidate(name)}
            first = next(iter(entries.values()))
            cells = ""
            for device in DEVICE_BUDGETS:
                entry = entries.get(device.label)
                if entry is None or entry.best is None:
                    cells += f"{'no fit':>17}"
                else:
                    cells += (
                        f"{f'{entry.best.sequence_length}x{entry.best.batch_size}':>17}"
                    )
            lines.append(f"  {name:<20} {first.parameters:>13,}  {cells}")

        lines += [
            "",
            "  cells show the largest seq x batch the estimator calls PLAUSIBLE;",
            "  'no fit' means nothing tried cleared it",
            "",
        ]

        for device in DEVICE_BUDGETS:
            entries = self.for_device(device.label)
            if not entries:
                continue
            lines += [
                "-" * 92,
                f"{device.label} — {device.total_gib:.2f} GiB total "
                f"({device.usable_gib:.2f} usable after {device.reserved_gib:.1f} reserved)",
                f"  {device.examples}   [{device.source}]",
                "-" * 92,
            ]
            for entry in entries:
                if entry.best is None:
                    lines.append(f"  {entry.candidate:<20} NO CONFIGURATION FITS")
                else:
                    fit = entry.fit
                    lines.append(
                        f"  {entry.candidate:<20} {entry.best.label:<38} "
                        f"{fit.total_gib:>6.2f} GiB  {fit.headroom_gib:>5.2f} free   "
                        f"limited by {entry.binding_term}"
                    )
                for note in entry.notes:
                    lines.append(f"  {'':<20} ! {note}")

        if self.findings:
            lines += ["", "-" * 92, "FINDINGS", "-" * 92]
            lines += [f"  ! {f}" for f in self.findings]

        lines += [
            "", "-" * 92,
            "  These are FEASIBILITY estimates, not measurements. The estimator is",
            "  calibrated against six measured configurations at ~100M and has never been",
            "  checked above that. Validate with scripts/benchmark_memory.py on the real",
            "  device before committing GPU hours.",
            rule,
        ]
        return "\n".join(lines)


def build_matrix(
    *,
    devices: tuple[DeviceBudget, ...] = DEVICE_BUDGETS,
    precision: str = "fp16",
) -> ScalingMatrix:
    """Evaluate every candidate against every device budget."""
    matrix = ScalingMatrix()
    configs = default_sweep(precision=precision)

    t4 = next((d for d in devices if d.nominal_gb == 16), devices[0])
    matrix.baseline = evaluate_candidate(
        LEVEL2_SPEC, t4, scale_class="94M (measured)", configs=configs
    )
    matrix.anchor = estimate_training_memory(
        LEVEL2_SPEC, t4.usable_gib, strategy=LEVEL2_CONFIG.strategy,
        optimizer=LEVEL2_CONFIG.optimizer,
        sequence_length=LEVEL2_CONFIG.sequence_length,
        batch_size=LEVEL2_CONFIG.batch_size,
        gradient_checkpointing=LEVEL2_CONFIG.gradient_checkpointing,
        precision=LEVEL2_CONFIG.precision, label="level2 as run",
    )

    for scale in SCALE_CLASSES:
        for hidden, layers in scale.shapes:
            spec = scaled_spec(hidden, layers, name=_candidate_name(scale, hidden, layers))
            for device in devices:
                matrix.fits.append(
                    evaluate_candidate(spec, device, scale_class=scale.label, configs=configs)
                )

    infeasible_on_t4 = [
        f for f in matrix.fits if f.device.nominal_gb == 16 and not f.feasible
    ]
    if infeasible_on_t4:
        matrix.findings.append(
            f"{len(infeasible_on_t4)} of {len(SCALE_CLASSES) * 2} candidates do not fit a "
            f"T4 in any configuration tried: "
            + ", ".join(f.candidate for f in infeasible_on_t4)
        )
    dominated = [
        f for f in matrix.fits
        if f.fit is not None and f.fit.deltanet_activations_gib > f.fit.base_weights_gib
    ]
    if dominated:
        matrix.findings.append(
            f"{len(dominated)} of {len(matrix.fits)} configurations are dominated by "
            f"DeltaNet activations rather than weights — at this architecture, sequence "
            f"length and batch size buy memory pressure faster than parameters do"
        )
    largest = max(c.tokens_per_micro_step for c in configs)
    saturated = [
        f for f in matrix.fits
        if f.best is not None and f.best.tokens_per_micro_step == largest
    ]
    if saturated:
        matrix.findings.append(
            f"{len(saturated)} configuration(s) fit at the largest shape tried "
            f"({largest:,} tokens per micro-step). The true maximum is higher; the sweep "
            f"is capped, not the hardware."
        )
    matrix.findings.append(
        "the estimator has been validated at ~100M only. Above that these are "
        "extrapolations of a calibrated model, not measurements."
    )
    return matrix
