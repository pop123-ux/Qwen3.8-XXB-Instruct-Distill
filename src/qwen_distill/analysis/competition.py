"""The competitive field: who we have to beat inside a 16 GB card, and by how much.

The project's objective is not "make a 27B teacher fit in 16 GB". It is **be the strongest
model anyone can actually run in 16 GB**. That changes what has to be modelled. Parameter
count stops being a target and becomes an optimisation variable; the fixed quantity is the
deployment envelope, and the thing being maximised is measured capability inside it.

Which means the field has to be in the repository, not in someone's head. A candidate is
only interesting relative to what a user could otherwise download and run on the same card.

**Every number here carries where it came from.** That is the whole design. The failure this
module exists to prevent is declaring victory over a competitor's score we never verified,
using an evaluation protocol we never matched — a mistake that is invisible in a results
table and fatal to the claim. So:

* :class:`Provenance` distinguishes what a vendor published, what we computed, what we
  measured, and what was simply told to us.
* :func:`compare` **refuses to return a verdict** when either side is unverified or the
  protocols differ or are unknown. It returns ``INCOMPARABLE`` and says why.
* A competitor whose KV geometry we do not know reports ``None`` for KV cache and a
  partial footprint, never a silent zero. Gemma 3 alternates sliding-window and global
  attention; guessing uniform global attention would overstate its cache several-fold and
  hand us a flattering, false comparison.

Nothing here runs a benchmark. It records the target, computes the envelope, and polices
the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..architecture.memory import DTYPE_BYTES, GIB, QUANT_BYTES_PER_PARAM

# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
#: Published by the model's authors. Not verified by us, and typically produced under an
#: evaluation protocol we have not reproduced.
VENDOR_REPORTED = "vendor_reported"
#: Derived here from stated architecture facts. Only as good as those facts.
COMPUTED = "computed"
#: We ran it, in this repository, and the artifact is committed.
MEASURED = "measured"
#: Supplied to us. Source not checked. Usable as an aspiration, never as a bar.
UNVERIFIED = "unverified"
#: Seen in two or more independent secondary sources, and where possible cross-checked
#: arithmetically — but the primary source was not reached. Stronger than UNVERIFIED,
#: weaker than reading the model card. Still not something we produced.
CORROBORATED = "corroborated"

#: Provenance levels that may be used as a pass/fail bar in a claim of superiority.
CITABLE = frozenset({MEASURED})

PROVENANCE_ORDER = (MEASURED, COMPUTED, VENDOR_REPORTED, CORROBORATED, UNVERIFIED)

#: What a benchmark is for, so "reasoning performance" and "coding performance" are
#: answerable rather than impressionistic. A benchmark may belong to more than one.
CAPABILITIES = (
    "knowledge",
    "reasoning",
    "coding",
    "instruction_following",
    "long_context",
    "tool_use",
    "agentic",
    "multilingual",
)


class IncomparableError(ValueError):
    """A comparison that cannot be made honestly, with the reason it cannot."""


@dataclass(frozen=True)
class Score:
    """One benchmark number and everything needed to judge whether it means anything."""

    benchmark: str
    value: float
    provenance: str
    source: str
    #: The evaluation protocol: shots, CoT, decoding, harness version. Two numbers on the
    #: same benchmark under different protocols are different quantities.
    protocol: str | None = None
    retrieved: str | None = None
    notes: str = ""

    @property
    def citable(self) -> bool:
        return self.provenance in CITABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark, "value": self.value,
            "provenance": self.provenance, "source": self.source,
            "protocol": self.protocol, "retrieved": self.retrieved, "notes": self.notes,
        }


@dataclass(frozen=True)
class KVGeometry:
    """What a model's KV cache costs per token. ``None`` anywhere means unknown.

    ``full_attention_layers`` is deliberately separate from ``layers``: hybrid and
    sliding-window models pay cache on only some of them, and that is exactly the
    difference a competitive comparison turns on.
    """

    layers: int
    num_key_value_heads: int
    head_dim: int
    #: Layers that keep an unbounded cache. Defaults to all of them.
    full_attention_layers: int | None = None
    #: Layers with a bounded sliding window, and the window size.
    sliding_window_layers: int = 0
    sliding_window: int | None = None
    provenance: str = UNVERIFIED
    source: str = ""

    def bytes_at(self, context: int, *, dtype: str = "fp16", batch: int = 1) -> int:
        per_token_per_layer = 2 * self.num_key_value_heads * self.head_dim
        unbounded = self.layers if self.full_attention_layers is None else self.full_attention_layers
        tokens = per_token_per_layer * unbounded * context
        if self.sliding_window_layers and self.sliding_window:
            tokens += (
                per_token_per_layer
                * self.sliding_window_layers
                * min(context, self.sliding_window)
            )
        return int(tokens * batch * DTYPE_BYTES[dtype])


@dataclass(frozen=True)
class Competitor:
    """A model a user could download and run instead of ours."""

    name: str
    #: Total parameters. ``None`` when we do not know it, which blocks the weight estimate
    #: rather than substituting a guess.
    parameters: int | None = None
    #: Parameters active per token. Equal to ``parameters`` for a dense model; smaller for
    #: an MoE, where it drives compute but *not* the weight memory that has to fit.
    active_parameters: int | None = None
    kv: KVGeometry | None = None
    #: A full architecture spec, when the competitor is in the Qwen3.5/3.8 hybrid family.
    #: When present the envelope is computed with the same estimator we use on ourselves —
    #: KV on one layer in four, plus the DeltaNet recurrent and conv state — rather than
    #: from a parameter count. Typed loosely to keep this module free of that import.
    spec: Any = None
    max_context: int | None = None
    scores: dict[str, Score] = field(default_factory=dict)
    #: Decode throughput, if anyone has measured it. Always with the hardware it was on.
    tokens_per_second: float | None = None
    throughput_hardware: str | None = None
    parameters_provenance: str = UNVERIFIED
    source: str = ""
    notes: str = ""

    @property
    def is_moe(self) -> bool:
        return (
            self.active_parameters is not None
            and self.parameters is not None
            and self.active_parameters < self.parameters
        )

    def weight_gib(self, quant: str) -> float | None:
        if self.parameters is None:
            return None
        return self.parameters * QUANT_BYTES_PER_PARAM[quant] / GIB

    def kv_gib(self, context: int, *, dtype: str = "fp16", batch: int = 1) -> float | None:
        if self.kv is None:
            return None
        return self.kv.bytes_at(context, dtype=dtype, batch=batch) / GIB


@dataclass
class Envelope:
    """What a model costs in a fixed VRAM budget, and what is missing from the estimate."""

    name: str
    quant: str
    context: int
    weight_gib: float | None
    kv_gib: float | None
    overhead_gib: float
    budget_gib: float
    unknowns: list[str] = field(default_factory=list)

    @property
    def total_gib(self) -> float | None:
        if self.weight_gib is None:
            return None
        return self.weight_gib + (self.kv_gib or 0.0) + self.overhead_gib

    @property
    def verdict(self) -> str:
        """``FITS`` / ``TIGHT`` / ``DOES NOT FIT`` / ``UNKNOWN``.

        ``UNKNOWN`` when something the total depends on is missing — a partial sum that
        happens to be under budget is not a fit, and must not be reported as one.
        """
        total = self.total_gib
        if total is None or self.unknowns:
            return "UNKNOWN"
        if total > self.budget_gib:
            return "DOES NOT FIT"
        if self.budget_gib - total < 1.5:
            return "TIGHT"
        return "FITS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "quant": self.quant, "context": self.context,
            "weight_gib": self.weight_gib, "kv_gib": self.kv_gib,
            "overhead_gib": self.overhead_gib, "total_gib": self.total_gib,
            "budget_gib": self.budget_gib, "verdict": self.verdict,
            "unknowns": self.unknowns,
        }


def envelope(
    competitor: Competitor,
    *,
    budget_gib: float,
    quant: str = "q4_k_m",
    context: int = 32768,
    kv_dtype: str = "fp16",
    batch: int = 1,
    overhead_gib: float = 1.0,
) -> Envelope:
    """What this model costs on a card of ``budget_gib``, with the gaps named."""
    unknowns: list[str] = []
    if competitor.parameters is None:
        unknowns.append("parameter count unknown")
    elif competitor.parameters_provenance == UNVERIFIED:
        unknowns.append("parameter count unverified")
    if competitor.spec is None and competitor.kv is None:
        unknowns.append("KV geometry unknown, so the cache is not counted")
    elif competitor.kv is not None and competitor.kv.provenance == UNVERIFIED:
        unknowns.append("KV geometry unverified")
    if competitor.max_context is not None and context > competitor.max_context:
        unknowns.append(
            f"asked for {context} context but the model supports {competitor.max_context}"
        )
    if competitor.spec is not None:
        # Same estimator we hold ourselves to, so the comparison is like for like.
        from ..architecture.memory import DeploymentConfig, estimate_memory

        estimate = estimate_memory(
            competitor.spec,
            DeploymentConfig(
                context_length=context, batch_size=batch, weight_quant=quant,
                kv_cache_dtype=kv_dtype,
            ),
        )
        return Envelope(
            name=competitor.name, quant=quant, context=context,
            weight_gib=estimate.weights / GIB,
            kv_gib=(estimate.kv_cache + estimate.recurrent_state + estimate.conv_state) / GIB,
            overhead_gib=overhead_gib, budget_gib=budget_gib, unknowns=unknowns,
        )
    return Envelope(
        name=competitor.name, quant=quant, context=context,
        weight_gib=competitor.weight_gib(quant),
        kv_gib=competitor.kv_gib(context, dtype=kv_dtype, batch=batch),
        overhead_gib=overhead_gib, budget_gib=budget_gib, unknowns=unknowns,
    )


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
@dataclass
class Comparison:
    """Whether we beat one competitor on one benchmark — or why that cannot be said."""

    benchmark: str
    ours: Score | None
    theirs: Score | None
    verdict: str
    margin: float | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "ours": self.ours.to_dict() if self.ours else None,
            "theirs": self.theirs.to_dict() if self.theirs else None,
            "verdict": self.verdict, "margin": self.margin, "reasons": self.reasons,
        }


def compare(ours: Score | None, theirs: Score | None, *, benchmark: str) -> Comparison:
    """Compare two scores, refusing rather than guessing.

    A margin is only ever reported alongside a verdict of ``AHEAD``/``BEHIND``/``TIED``,
    and those are only reached when both sides are things we can actually stand behind:
    our own measurement, against a number whose protocol we know and match.
    """
    reasons: list[str] = []
    if ours is None:
        reasons.append("we have not measured this benchmark")
    if theirs is None:
        reasons.append("no competitor score recorded for this benchmark")
    if reasons:
        return Comparison(benchmark, ours, theirs, "INCOMPARABLE", None, reasons)

    if not ours.citable:
        reasons.append(
            f"our score is {ours.provenance}, not measured here — a claim of superiority "
            "needs a number this repository produced"
        )
    if theirs.provenance == UNVERIFIED:
        reasons.append(
            f"the competitor score is {UNVERIFIED} ({theirs.source}); verify it against the "
            "primary source before treating it as a bar"
        )
    if ours.protocol is None or theirs.protocol is None:
        reasons.append("at least one protocol is unrecorded, so the two may not be the same quantity")
    elif ours.protocol != theirs.protocol:
        reasons.append(
            f"protocols differ ({ours.protocol!r} vs {theirs.protocol!r}); the numbers are "
            "not comparable without re-running one of them"
        )
    if reasons:
        return Comparison(benchmark, ours, theirs, "INCOMPARABLE", None, reasons)

    margin = ours.value - theirs.value
    verdict = "AHEAD" if margin > 0 else "BEHIND" if margin < 0 else "TIED"
    return Comparison(benchmark, ours, theirs, verdict, margin, [])


def scoreboard(
    ours: dict[str, Score], competitor: Competitor
) -> list[Comparison]:
    """One comparison per benchmark either side records."""
    benchmarks = sorted(set(ours) | set(competitor.scores))
    return [
        compare(ours.get(name), competitor.scores.get(name), benchmark=name)
        for name in benchmarks
    ]


def gap_to_target(ours: dict[str, Score], competitor: Competitor) -> dict[str, Any]:
    """How far we are from beating this competitor, and what still blocks saying so."""
    comparisons = scoreboard(ours, competitor)
    ahead = [c for c in comparisons if c.verdict == "AHEAD"]
    behind = [c for c in comparisons if c.verdict == "BEHIND"]
    blocked = [c for c in comparisons if c.verdict == "INCOMPARABLE"]
    return {
        "competitor": competitor.name,
        "n_benchmarks": len(comparisons),
        "ahead": [c.benchmark for c in ahead],
        "behind": [c.benchmark for c in behind],
        "incomparable": [c.benchmark for c in blocked],
        "verdict": (
            "NOT YET COMPARABLE" if blocked
            else "AHEAD" if ahead and not behind
            else "BEHIND" if behind and not ahead
            else "MIXED"
        ),
        "comparisons": [c.to_dict() for c in comparisons],
    }


def with_scores(competitor: Competitor, scores: list[Score]) -> Competitor:
    """A copy carrying the given scores, keyed by benchmark."""
    return replace(competitor, scores={s.benchmark: s for s in scores})


# ---------------------------------------------------------------------------
# the benchmarks the objective is stated in
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkSpec:
    """A benchmark on the target board: what it measures and what running it needs."""

    name: str
    capabilities: tuple[str, ...]
    summary: str
    #: What this repository would have to build to produce a number. None of it exists.
    requires: str
    implemented: bool = False


TARGET_BENCHMARKS: dict[str, BenchmarkSpec] = {
    "mmlu_pro": BenchmarkSpec(
        "mmlu_pro", ("knowledge", "reasoning"),
        "multiple-choice across 14 domains, 10 options, CoT-friendly",
        "a multiple-choice harness with CoT extraction and per-domain aggregation",
    ),
    "gpqa_diamond": BenchmarkSpec(
        "gpqa_diamond", ("reasoning",),
        "graduate-level physics/chemistry/biology, deliberately Google-proof",
        "the gated GPQA dataset plus a multiple-choice harness",
    ),
    "ifeval": BenchmarkSpec(
        "ifeval", ("instruction_following",),
        "verifiable instruction constraints, scored programmatically",
        "the constraint verifiers; no model judge needed, so this is the cheapest to build",
    ),
    "livecodebench_v6": BenchmarkSpec(
        "livecodebench_v6", ("coding",),
        "competitive-programming problems with execution-based grading",
        "a sandboxed code executor with time and memory limits",
    ),
    "longbench_v2": BenchmarkSpec(
        "longbench_v2", ("long_context",),
        "long-document understanding, tens to hundreds of thousands of tokens",
        "a long-context runner; also the benchmark that most constrains the KV budget",
    ),
    "bfcl_v4": BenchmarkSpec(
        "bfcl_v4", ("tool_use",),
        "function-calling correctness against declared tool schemas",
        "a tool-schema harness and an AST/execution checker",
    ),
    "tau2_bench": BenchmarkSpec(
        "tau2_bench", ("agentic", "tool_use"),
        "multi-turn agentic tasks against simulated environments and policies",
        "environment simulators and a user simulator; the most expensive of these to build",
    ),
}

#: Named because the objective asks for "multilingual performance where relevant" and the
#: target board has no multilingual benchmark on it. Recorded as an open gap rather than
#: quietly dropped or quietly invented.
MISSING_CAPABILITIES = {
    "multilingual": (
        "no multilingual benchmark is on the target board. If multilingual capability is "
        "part of the objective it needs a chosen benchmark (MMMLU, Belebele, Flores-200 "
        "and MGSM are the usual candidates) before it can be optimised for; until then it "
        "is unmeasured, not passing."
    ),
}


# ---------------------------------------------------------------------------
# the field
# ---------------------------------------------------------------------------
#: Where a registry fact came from. Two very different things both look like "knowledge":
#: something a human handed us this session, and something recalled from model training
#: data. Neither has been checked against a model card *in this repository*, so both are
#: UNVERIFIED and neither can serve as a bar. The distinction is kept in `source` so the
#: verification pass knows what it is re-checking.
FROM_USER = "user-supplied 2026-08-27; not checked against a primary source"
#: Independent secondary sources reached by web search on 2026-08-27. The primary source
#: (huggingface.co) is blocked by this environment's egress proxy and was not reached; the
#: sandbox policy is not something to work around, so these stay short of vendor-reported.
FROM_SEARCH = (
    "corroborated by independent web sources 2026-08-27; the Hugging Face model card is "
    "blocked by the egress proxy here and was NOT read"
)
FROM_RECALL = (
    "recalled from model training data; NOT checked against the model card in this "
    "session and possibly wrong or outdated"
)


def qwen35_9b_spec():
    """Qwen3.5-9B's architecture, which is the *same hybrid family as our teacher*.

    Reported layout: hidden 4096, 32 layers as ``8 x (3 x DeltaNet -> 1 x gated
    attention)`` — period 4, exactly the teacher's — gated attention 16 Q / 4 KV at
    head_dim 256, DeltaNet 32 value / 16 key heads at 128, FFN 12288, rope dim 64,
    262,144 context.

    The vocabulary and embedding tying are **not** stated in any source reached here; they
    are inferred. With the teacher's 248,320-entry vocabulary and untied embeddings this
    spec computes to 8.95B parameters, which is what "9B" means — and no other combination
    tried lands as close (tied: 7.94B; a 151,936 vocabulary: 8.16B). That is corroboration
    by arithmetic, not a reading of the config, and it is recorded as such.
    """
    from ..architecture.spec import HybridArchSpec

    return HybridArchSpec(
        name="qwen3.5-9b", hidden_size=4096, num_hidden_layers=32, intermediate_size=12288,
        vocab_size=248320, tie_word_embeddings=False,
        num_attention_heads=16, num_key_value_heads=4, head_dim=256,
        partial_rotary_factor=0.25,
        linear_num_value_heads=32, linear_num_key_heads=16,
        linear_key_head_dim=128, linear_value_head_dim=128,
        full_attention_interval=4, max_position_embeddings=262144,
        provenance=FROM_SEARCH,
    )


def _qwen35_9b() -> Competitor:
    """The first concrete target — and, it turns out, a close relative.

    Six of the seven scores below were corroborated by an independent search; LongBench v2
    was not, and one search summary returned a conflicting LiveCodeBench v6 figure (82.7)
    that a second search contradicted with 65.6. That conflict is left recorded rather than
    resolved by preference: it is exactly why the primary source and the evaluation
    protocol both matter.
    """
    corroborated = {
        "mmlu_pro": 82.5, "gpqa_diamond": 81.7, "ifeval": 91.5,
        "livecodebench_v6": 65.6, "bfcl_v4": 66.1, "tau2_bench": 79.1,
    }
    scores = [
        Score(
            name, value, CORROBORATED, FROM_SEARCH,
            notes=(
                "a second search summary returned 82.7 for this benchmark; unresolved"
                if name == "livecodebench_v6" else ""
            ),
        )
        for name, value in corroborated.items()
    ]
    scores.append(
        Score("longbench_v2", 55.2, UNVERIFIED, FROM_USER,
              notes="not corroborated by search; the only score still resting on one source")
    )
    spec = qwen35_9b_spec()
    return with_scores(
        Competitor(
            name="Qwen3.5-9B",
            parameters=8_950_000_000,
            active_parameters=8_950_000_000,
            spec=spec,
            max_context=262144,
            parameters_provenance=CORROBORATED,
            source=FROM_SEARCH,
            notes=(
                "PRIMARY TARGET, and the same architecture family as the teacher: period-4 "
                "hybrid, head_dim 256, DeltaNet at 128. Its head ratios differ from the "
                "teacher's, though — 4 query heads per KV head against the teacher's 6, and "
                "2 DeltaNet value heads per key head against 3 — so a direct teacher->this "
                "transfer is refused by materialize.py rather than being silently regrouped."
            ),
        ),
        scores,
    )


def _qwen3_14b() -> Competitor:
    return Competitor(
        name="Qwen3-14B",
        parameters=14_800_000_000,
        active_parameters=14_800_000_000,
        kv=KVGeometry(
            layers=40, num_key_value_heads=8, head_dim=128,
            provenance=UNVERIFIED, source=FROM_RECALL,
        ),
        max_context=32768,
        parameters_provenance=UNVERIFIED,
        source=FROM_RECALL,
        notes=(
            "Dense. 32K native context, longer via YaRN. Every figure needs checking "
            "against the model card before use."
        ),
    )


def _gemma3_27b() -> Competitor:
    return Competitor(
        name="Gemma-3-27B",
        parameters=27_000_000_000,
        active_parameters=27_000_000_000,
        kv=KVGeometry(
            layers=62, num_key_value_heads=16, head_dim=128,
            # Interleaved local/global attention: only a minority of layers keep an
            # unbounded cache, which is precisely what makes a 27B model plausible on a
            # 16 GB card at long context. The ratio must be checked, not assumed.
            full_attention_layers=None,
            provenance=UNVERIFIED, source=FROM_RECALL,
        ),
        max_context=131072,
        parameters_provenance=UNVERIFIED,
        source=FROM_RECALL,
        notes=(
            "Interleaves sliding-window and global attention, so a uniform-global KV "
            "estimate overstates its cache substantially. The local:global ratio and window "
            "size are NOT filled in here because guessing them would flatter us: an "
            "overstated competitor cache is a competitor that looks like it does not fit. "
            "Read them off the config before comparing."
        ),
    )


def reference_field() -> dict[str, Competitor]:
    """The models a 16 GB user could run instead of ours.

    Deliberately not exhaustive and deliberately not authoritative: it names the field and
    forces the verification, and newer open-weight releases should be added as they appear.
    """
    return {c.name: c for c in (_qwen35_9b(), _qwen3_14b(), _gemma3_27b())}


def verification_backlog(field: dict[str, Competitor] | None = None) -> list[str]:
    """Everything that must be checked against a primary source before any claim.

    This list being non-empty is the honest state of the competitive picture today. It is
    returned rather than printed so a script can fail on it.
    """
    field = reference_field() if field is None else field
    backlog: list[str] = []
    for competitor in field.values():
        if competitor.parameters is None:
            backlog.append(f"{competitor.name}: parameter count unknown")
        elif competitor.parameters_provenance == UNVERIFIED:
            backlog.append(f"{competitor.name}: parameter count unverified ({competitor.source})")
        if competitor.kv is None:
            backlog.append(f"{competitor.name}: KV geometry unknown — cache cannot be estimated")
        elif competitor.kv.provenance == UNVERIFIED:
            backlog.append(f"{competitor.name}: KV geometry unverified ({competitor.kv.source})")
        unverified = [s.benchmark for s in competitor.scores.values() if s.provenance == UNVERIFIED]
        if unverified:
            backlog.append(
                f"{competitor.name}: {len(unverified)} score(s) unverified "
                f"({', '.join(unverified)})"
            )
        missing_protocol = [
            s.benchmark for s in competitor.scores.values() if s.protocol is None
        ]
        if missing_protocol:
            backlog.append(
                f"{competitor.name}: {len(missing_protocol)} score(s) have no recorded "
                "evaluation protocol, so we cannot match it"
            )
    for benchmark in TARGET_BENCHMARKS.values():
        if not benchmark.implemented:
            backlog.append(f"benchmark {benchmark.name}: not implemented — {benchmark.requires}")
    for gap in MISSING_CAPABILITIES.values():
        backlog.append(gap)
    return backlog
