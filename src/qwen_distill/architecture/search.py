"""Constrained architecture search over the hybrid DeltaNet/attention design space.

The project's central question is *"what is the largest capable model that fits a
16 GB consumer GPU with a genuinely large context?"*. This module answers the
tractable half of that question analytically: it enumerates architectures, prunes
those that cannot meet the deployment envelope, and ranks the survivors.

**On the ranking objective.** There is no way to know a candidate's benchmark score
without training it. What we can compute is *capacity*, and the most defensible
public proxy for capacity is the non-embedding parameter count: embedding and
output-head parameters are a lookup table whose cost scales with vocabulary rather
than with modelled function complexity, and scaling-law work consistently relates
loss to non-embedding parameters. So candidates are ranked by non-embedding
parameters under the constraint, and the ranking is explicitly a *hypothesis
generator*, not a result. Nothing here substitutes for training the top candidates
and measuring them (see ``docs/EVALUATION_PLAN.md``).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from itertools import product

from .flops import bandwidth_bound_tokens_per_second, decode_flops_per_token
from .memory import DeploymentConfig, MemoryEstimate, estimate_memory, max_context_within
from .params import ParamBreakdown, count_parameters, format_params
from .spec import HybridArchSpec


@dataclass(frozen=True)
class SearchConstraints:
    """The deployment envelope a candidate must satisfy to survive pruning."""

    vram_gib: float = 16.0
    #: Context the candidate must support *while* fitting ``vram_gib``.
    required_context: int = 32768
    #: Reserve left unused, for driver/display/desktop compositor on a real machine.
    reserved_gib: float = 1.0
    deployment: DeploymentConfig = DeploymentConfig()
    #: Reject architectures outside a sane aspect ratio; extreme depth/width ratios
    #: train poorly and parallelise badly.
    min_hidden_per_layer: float = 40.0
    max_hidden_per_layer: float = 260.0
    #: Minimum bandwidth-ceiling throughput to be worth deploying interactively.
    min_tokens_per_second: float = 12.0
    reference_bandwidth_gb_s: float = 448.0

    @property
    def usable_gib(self) -> float:
        return self.vram_gib - self.reserved_gib


@dataclass(frozen=True)
class Candidate:
    """An architecture together with its analytical deployment profile."""

    spec: HybridArchSpec
    params: ParamBreakdown
    memory: MemoryEstimate
    max_context: int
    decode_gflops: float
    tokens_per_second_ceiling: float
    rejected_reason: str | None = None

    @property
    def feasible(self) -> bool:
        return self.rejected_reason is None

    @property
    def teacher_param_ratio(self) -> float:
        """Set by :func:`evaluate_candidates` relative to a reference spec."""
        return self.params.total / TEACHER_TOTAL_PARAMS

    def summary_row(self) -> dict[str, object]:
        return {
            "name": self.spec.name,
            "total_params": self.params.total,
            "total_params_h": format_params(self.params.total),
            "non_embedding_params": self.params.non_embedding,
            "non_embedding_h": format_params(self.params.non_embedding),
            "hidden_size": self.spec.hidden_size,
            "layers": self.spec.num_hidden_layers,
            "intermediate": self.spec.intermediate_size,
            "full_attn_interval": self.spec.full_attention_interval,
            "n_full_attn": self.spec.num_full_attention_layers,
            "tied_embeddings": self.spec.tie_word_embeddings,
            "weights_gib": round(self.memory.weights / (1024 ** 3), 2),
            "total_gib": round(self.memory.total_gib, 2),
            "max_context": self.max_context,
            "decode_gflops": round(self.decode_gflops, 1),
            "tok_s_ceiling": round(self.tokens_per_second_ceiling, 1),
            "feasible": self.feasible,
            "rejected_reason": self.rejected_reason,
        }


#: Total parameters of the published teacher spec, used for ratio reporting.
TEACHER_TOTAL_PARAMS = count_parameters(HybridArchSpec(name="teacher")).total


def evaluate_candidate(spec: HybridArchSpec, constraints: SearchConstraints) -> Candidate:
    """Compute the deployment profile of ``spec`` and decide whether it survives."""
    params = count_parameters(spec)
    deployment = replace(constraints.deployment, context_length=constraints.required_context)
    memory = estimate_memory(spec, deployment)
    max_ctx = max_context_within(spec, constraints.usable_gib, deployment)
    decode = decode_flops_per_token(spec, constraints.required_context).total / 1e9
    tok_s = bandwidth_bound_tokens_per_second(
        spec, constraints.reference_bandwidth_gb_s, deployment.weight_quant
    )

    reason: str | None = None
    aspect = spec.hidden_size / spec.num_hidden_layers
    if not memory.fits_in(constraints.usable_gib):
        reason = (
            f"needs {memory.total_gib:.2f} GiB at {constraints.required_context} ctx, "
            f"budget {constraints.usable_gib:.2f} GiB"
        )
    elif max_ctx < constraints.required_context:
        reason = f"max context {max_ctx} < required {constraints.required_context}"
    elif not constraints.min_hidden_per_layer <= aspect <= constraints.max_hidden_per_layer:
        reason = f"aspect ratio hidden/layers = {aspect:.1f} outside sane range"
    elif tok_s < constraints.min_tokens_per_second:
        reason = f"bandwidth ceiling {tok_s:.1f} tok/s below {constraints.min_tokens_per_second}"

    return Candidate(
        spec=spec,
        params=params,
        memory=memory,
        max_context=max_ctx,
        decode_gflops=decode,
        tokens_per_second_ceiling=tok_s,
        rejected_reason=reason,
    )


def generate_grid(
    hidden_sizes: Iterable[int],
    layer_counts: Iterable[int],
    ffn_multipliers: Iterable[float],
    full_attention_intervals: Iterable[int],
    *,
    vocab_size: int = 248320,
    tie_word_embeddings: Iterable[bool] = (False,),
    num_attention_heads: int | None = None,
    num_key_value_heads: int = 4,
    head_dim: int = 256,
    linear_num_key_heads: int = 16,
    linear_value_head_multiple: int = 3,
    max_position_embeddings: int = 262144,
    ffn_round_to: int = 128,
) -> Iterator[HybridArchSpec]:
    """Enumerate specs over a grid, skipping structurally invalid combinations.

    Defaults follow the teacher's ratios so that a candidate differs from the teacher
    in as few dimensions as possible:

    * ``num_attention_heads`` defaults to ``hidden_size // 213`` rounded to a multiple
      of ``num_key_value_heads`` (the teacher runs 24 query heads at hidden 5120);
    * ``linear_num_value_heads`` is ``linear_value_head_multiple x linear_num_key_heads``
      (the teacher runs 48 value / 16 key heads);
    * ``intermediate_size`` is ``ffn_multiplier * hidden_size`` rounded to ``ffn_round_to``
      (the teacher runs 17408 / 5120 = 3.4x).
    """
    for hidden, layers, ffn_mult, interval, tie in product(
        hidden_sizes, layer_counts, ffn_multipliers, full_attention_intervals, tie_word_embeddings
    ):
        if num_attention_heads is None:
            heads = max(
                num_key_value_heads,
                round(hidden / 213 / num_key_value_heads) * num_key_value_heads,
            )
        else:
            heads = num_attention_heads
        inter = int(round(hidden * ffn_mult / ffn_round_to) * ffn_round_to)
        if inter <= 0:
            continue
        name = (
            f"h{hidden}-l{layers}-ffn{inter}-fa{interval}" + ("-tied" if tie else "")
        )
        try:
            yield HybridArchSpec(
                name=name,
                hidden_size=hidden,
                num_hidden_layers=layers,
                intermediate_size=inter,
                vocab_size=vocab_size,
                tie_word_embeddings=tie,
                num_attention_heads=heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                linear_num_key_heads=linear_num_key_heads,
                linear_num_value_heads=linear_num_key_heads * linear_value_head_multiple,
                full_attention_interval=interval,
                max_position_embeddings=max_position_embeddings,
                provenance="generate_grid",
            )
        except ValueError:
            # Structurally invalid combination (e.g. head divisibility); skip it.
            continue


def search(
    specs: Iterable[HybridArchSpec],
    constraints: SearchConstraints | None = None,
    *,
    keep_infeasible: bool = False,
) -> list[Candidate]:
    """Evaluate and rank ``specs``.

    Feasible candidates are sorted by non-embedding parameters descending — the
    capacity proxy described in the module docstring. Ties break toward the higher
    throughput ceiling.
    """
    constraints = constraints or SearchConstraints()
    evaluated = [evaluate_candidate(spec, constraints) for spec in specs]
    feasible = [c for c in evaluated if c.feasible]
    feasible.sort(key=lambda c: (c.params.non_embedding, c.tokens_per_second_ceiling), reverse=True)
    if keep_infeasible:
        infeasible = [c for c in evaluated if not c.feasible]
        infeasible.sort(key=lambda c: c.memory.total_gib)
        return feasible + infeasible
    return feasible
