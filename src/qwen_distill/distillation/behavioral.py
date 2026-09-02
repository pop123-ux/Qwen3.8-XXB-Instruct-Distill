"""Beyond layer matching: distilling what a block *computes*, not where it sits.

Conventional depth-reduced distillation matches hidden states at mapped positions: student
layer ``s`` is trained so that ``h_s`` resembles ``h_t`` at some chosen teacher layer
``m(s)``. That objective has a defect that gets worse the more layers are removed. It asks
the student to *pass through the same points*, when what was actually deleted is *work* —
the teacher's 64 layers each transform the residual stream, and 16 of those transformations
have no student layer to perform them. Pointwise matching never says who took over.

The alternative implemented here is to match the **residual contribution** instead of the
position. A block's contribution is::

    delta_l = h_{l+1} - h_l

and contributions telescope: the total work done by a *span* of teacher layers ``[a, b)`` is
exactly ``h_b - h_a``, whatever happened in between. So a student layer can be asked to
reproduce the combined contribution of every teacher layer it replaced, including the ones
that were removed, and the target costs nothing extra to compute.

That gives a clean pair of objectives over the same tensors:

=================  ====================================  ==================================
term               target                                what it assumes
=================  ====================================  ==================================
``pointwise``      ``h_s[l] ~ h_t[m(l)]``                the student should visit the
                                                         teacher's intermediate states
``delta``          ``h_s[l+1]-h_s[l] ~ h_t[b]-h_t[a]``   the student should do the teacher's
                                                         work, in whatever coordinates
=================  ====================================  ==================================

They are not variations on one idea; they make different predictions and can be run against
each other. ``pointwise`` is the A1 control precisely because it is the conventional choice.

The spans tile the teacher's depth: every teacher layer is charged to exactly one student
layer, so the removed layers' work is attributed rather than dropped. :func:`layer_spans`
constructs the tiling and asserts it is complete.

Direction versus magnitude
--------------------------
``delta`` is measured two ways, and both are reported. MSE penalises magnitude error, which
matters because a student that does the right thing at half strength drifts. Cosine
penalises direction error, which matters because a student that pushes the residual stream
the wrong way cannot be fixed by rescaling. A single blended number would hide which of the
two is failing, so :class:`BehavioralLossOutput` carries them separately.

What is not implemented, and is said so rather than approximated
----------------------------------------------------------------
``deltanet_state`` — matching the recurrent state tensor itself — is **not available**. The
student's DeltaNet state has 16 key heads and 48 value heads; the teacher's has its own
shape, and the two are not comparable without a projection that would itself be an
untested modelling choice. The block-output term below measures the same layers'
*behaviour* honestly, at the hidden-size interface where the shapes do agree. Requesting
the state term raises rather than silently substituting the proxy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .objectives import IMPLEMENTED, NOT_IMPLEMENTED, ObjectiveUnavailable

if TYPE_CHECKING:  # pragma: no cover
    from torch import Tensor

MatchMode = Literal["pointwise", "delta"]


# ---------------------------------------------------------------------------
# which teacher layers each student layer is responsible for
# ---------------------------------------------------------------------------
def layer_spans(mapping: dict[int, int], teacher_layers: int) -> dict[int, tuple[int, int]]:
    """Tile the teacher's depth so every teacher layer is charged to one student layer.

    ``mapping`` is ``student_layer -> teacher_layer`` from
    :func:`qwen_distill.architecture.moe_init.map_layers`. Student layer ``s`` is made
    responsible for the teacher layers from its own anchor up to the next student layer's
    anchor; the last student layer absorbs the remaining depth. The returned spans are
    half-open ``(start, end)`` **layer-boundary** indices into a ``hidden_states`` tuple of
    length ``teacher_layers + 1``, so the span's total contribution is
    ``hidden_states[end] - hidden_states[start]``.
    """
    if not mapping:
        raise ValueError("an empty mapping cannot be tiled")
    anchors = [mapping[s] for s in sorted(mapping)]
    if anchors != sorted(anchors):
        raise ValueError("the mapping reorders the teacher's depth; spans would overlap")
    spans: dict[int, tuple[int, int]] = {}
    students = sorted(mapping)
    for i, s in enumerate(students):
        start = anchors[i]
        end = anchors[i + 1] if i + 1 < len(anchors) else teacher_layers
        spans[s] = (start, end)
    # The first student layer also absorbs anything before its anchor.
    first = students[0]
    spans[first] = (0, spans[first][1])
    covered = sum(end - start for start, end in spans.values())
    if covered != teacher_layers:
        raise ValueError(f"spans cover {covered} teacher layers, not {teacher_layers}")
    return spans


# ---------------------------------------------------------------------------
# the losses
# ---------------------------------------------------------------------------
@dataclass
class BehavioralLossOutput:
    """A behavioural term and the diagnostics needed to tell *how* it is failing."""

    total: Any
    magnitude: Any
    direction: Any
    mode: str
    n_pairs: int
    per_layer: dict[int, float] = field(default_factory=dict)
    student_norm: float = 0.0
    teacher_norm: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": float(self.total), "magnitude": float(self.magnitude),
            "direction": float(self.direction), "mode": self.mode,
            "n_pairs": self.n_pairs, "per_layer": self.per_layer,
            "student_norm": self.student_norm, "teacher_norm": self.teacher_norm,
            "norm_ratio": self.student_norm / self.teacher_norm if self.teacher_norm else None,
        }


def _normalise(x: Tensor) -> Tensor:
    """Scale a hidden-state tensor to unit RMS per token.

    Layers deep in a residual stream have much larger activations than early ones, so an
    unnormalised MSE is dominated by whichever pairs happen to sit deepest. Normalising per
    token makes the per-layer terms comparable, which is what allows ``per_layer`` to be
    read as "which layers are struggling" rather than "which layers are deep".
    """
    return x / (x.float().pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)


def _plan_pairs(
    student_hidden, teacher_hidden, mapping: dict[int, int], mode: str,
    teacher_layers: int | None,
):
    """Resolve which pairs are supervised, and the teacher spans, before any arithmetic.

    Shared by the whole-batch and the chunked objective so that both supervise exactly the
    same set of pairs. The chunked form needs the pair count up front — it divides each
    chunk's terms by it — and deriving that count twice, in two places, is precisely how
    the two forms would come to disagree.
    """
    if len(student_hidden) < 2:
        raise ValueError("need at least one student layer's worth of hidden states")
    n_teacher = teacher_layers if teacher_layers is not None else len(teacher_hidden) - 1
    if len(teacher_hidden) != n_teacher + 1:
        raise ValueError(
            f"teacher_hidden has {len(teacher_hidden)} entries; expected {n_teacher + 1} "
            f"for {n_teacher} layers (pass output_hidden_states=True)"
        )
    spans = layer_spans(mapping, n_teacher) if mode == "delta" else {}
    pairs = tuple(
        s for s in sorted(mapping)
        if not (mode == "delta" and s + 1 >= len(student_hidden))
    )
    if not pairs:
        raise ValueError("no student/teacher pairs were produced by the mapping")
    return pairs, spans


def _pair_tensors(student_hidden, teacher_hidden, mapping, spans, mode, s):
    """The two tensors one supervised pair compares."""
    if mode == "delta":
        start, end = spans[s]
        return (student_hidden[s + 1] - student_hidden[s],
                teacher_hidden[end] - teacher_hidden[start])
    return student_hidden[s + 1], teacher_hidden[mapping[s] + 1]


def _pair_term(student, teacher, s: int, *, normalise: bool, mask):
    """One pair's magnitude and direction terms, and the norms reported beside them.

    Both :func:`behavioral_loss` and :func:`behavioral_loss_chunked` call *this*, so the
    two cannot drift into computing different things. They differ only in when the terms
    are summed and when the gradient is taken — never in what a pair's term is.
    """
    import torch.nn.functional as F

    def reduce(x: Tensor) -> Tensor:
        if mask is None:
            return x.reshape(-1, x.shape[-1])
        return x[mask.bool()]

    a, b = reduce(student).float(), reduce(teacher).float()
    if a.shape != b.shape:
        raise ValueError(
            f"student layer {s}: shape {tuple(a.shape)} against teacher "
            f"{tuple(b.shape)} — widths must match (this student does not reduce width)"
        )
    s_norm, t_norm = float(a.detach().norm()), float(b.detach().norm())
    if normalise:
        a, b = _normalise(a), _normalise(b)
    magnitude = F.mse_loss(a, b)
    direction = 1.0 - F.cosine_similarity(a, b, dim=-1).mean()
    return magnitude, direction, s_norm, t_norm


def behavioral_loss(
    student_hidden: tuple[Tensor, ...] | list[Tensor],
    teacher_hidden: tuple[Tensor, ...] | list[Tensor],
    mapping: dict[int, int],
    *,
    mode: MatchMode = "delta",
    teacher_layers: int | None = None,
    direction_weight: float = 1.0,
    normalise: bool = True,
    mask: Tensor | None = None,
) -> BehavioralLossOutput:
    """Match student blocks to teacher blocks, either by position or by contribution.

    ``student_hidden`` and ``teacher_hidden`` are ``output_hidden_states=True`` tuples of
    length ``n_layers + 1``: entry ``i`` is the residual stream *before* layer ``i``.

    ``mode="pointwise"`` is conventional layer matching. ``mode="delta"`` is the behavioural
    objective: student layer ``l``'s contribution against the summed contribution of the
    teacher span it replaced.

    Every pair's loss tensors are held live until the caller's ``backward()``, which is
    what makes the peak memory scale with the number of pairs.
    :func:`behavioral_loss_chunked` computes the same objective without that; this
    function remains the reference the chunked form is validated against.
    """
    import torch

    pairs, spans = _plan_pairs(student_hidden, teacher_hidden, mapping, mode, teacher_layers)

    mags, dirs, per_layer = [], [], {}
    s_norms, t_norms = [], []
    for s in pairs:
        student, teacher = _pair_tensors(
            student_hidden, teacher_hidden, mapping, spans, mode, s)
        magnitude, direction, s_norm, t_norm = _pair_term(
            student, teacher, s, normalise=normalise, mask=mask)
        s_norms.append(s_norm)
        t_norms.append(t_norm)
        mags.append(magnitude)
        dirs.append(direction)
        per_layer[s] = float(magnitude.detach())

    magnitude = torch.stack(mags).mean()
    direction = torch.stack(dirs).mean()
    return BehavioralLossOutput(
        total=magnitude + direction_weight * direction,
        magnitude=magnitude, direction=direction, mode=mode, n_pairs=len(mags),
        per_layer=per_layer,
        student_norm=sum(s_norms) / len(s_norms), teacher_norm=sum(t_norms) / len(t_norms),
    )


@dataclass
class ChunkedBehavioralLoss:
    """What :func:`behavioral_loss_chunked` produced: the terms, and the gradient it holds.

    The gradient with respect to the student's hidden states has already been computed and
    is sitting in :attr:`grads`; nothing has yet been propagated into the student. Calling
    :meth:`backward` does that, in one traversal of the student's graph.
    """

    output: BehavioralLossOutput
    #: The student hidden-state tensors the gradient belongs to, in layer order.
    sources: list[Any] = field(default_factory=list)
    #: ``d(objective)/d(source)``, already scaled by ``loss_scale``.
    grads: list[Any] = field(default_factory=list)
    n_chunks: int = 0
    chunk_pairs: int = 0

    def backward(self) -> None:
        """Propagate the held gradient into the student, once."""
        import torch

        live = [(t, g) for t, g in zip(self.sources, self.grads, strict=True) if t.requires_grad]
        if not live:
            return
        torch.autograd.backward([t for t, _ in live], [g for _, g in live])

    def to_dict(self) -> dict[str, Any]:
        return self.output.to_dict() | {
            "n_chunks": self.n_chunks, "chunk_pairs": self.chunk_pairs,
        }


def behavioral_loss_chunked(
    student_hidden: tuple[Tensor, ...] | list[Tensor],
    teacher_hidden: tuple[Tensor, ...] | list[Tensor],
    mapping: dict[int, int],
    *,
    mode: MatchMode = "delta",
    teacher_layers: int | None = None,
    direction_weight: float = 1.0,
    normalise: bool = True,
    mask: Tensor | None = None,
    chunk_pairs: int = 4,
    loss_scale: float = 1.0,
    backward: Any = None,
) -> ChunkedBehavioralLoss:
    """The same objective as :func:`behavioral_loss`, without holding every pair at once.

    **The objective is unchanged.** The same pairs are supervised over the same positions
    with the same normalisation, the same per-pair terms and the same reduction. What
    changes is *when* the gradient is taken.

    :func:`behavioral_loss` builds all ``n`` pairs' loss tensors and returns a scalar whose
    ``backward()`` needs every one of them still live. ``mse_loss`` saves both of its
    normalised fp32 inputs, so the peak scales with ``n`` — at 48 pairs and 1536 positions
    that is roughly 4 GiB, and it is what put Run 003 over its memory gate.

    Here the objective is written as a sum over pairs, which it already was::

        L = (1/n) * sum_s [ magnitude_s + direction_weight * direction_s ]

    Sums split. A chunk of pairs contributes ``(1/n) * sum over that chunk``, and because
    every term carries the *same* ``1/n`` — not a per-chunk average, which would weight a
    ragged final chunk wrongly — the chunk losses sum to exactly ``L`` and each pair's
    gradient carries exactly the coefficient it has in ``L``. Each chunk's gradient is
    taken as soon as the chunk is built, so only ``chunk_pairs`` pairs' tensors are ever
    live at once.

    The gradient lands on detached stand-ins for the student hidden states rather than
    flowing straight into the student, so the student's own graph is traversed **once**,
    by :meth:`ChunkedBehavioralLoss.backward`, and not once per chunk. ``detach()`` shares
    storage, so the stand-ins cost nothing; what is held between the chunks and that final
    traversal is one gradient tensor per supervised layer, in the hidden states' own dtype.

    ``loss_scale`` multiplies every chunk — gradient accumulation's ``1/steps``, and a
    ``GradScaler`` factor when one is enabled. ``backward`` is the callable that takes a
    chunk's scalar loss, defaulting to ``Tensor.backward``; a scaled run passes
    ``lambda t: scaler.scale(t).backward()``.

    ``per_layer``, ``student_norm`` and ``teacher_norm`` are reported exactly as the
    unchunked form reports them.
    """
    import torch

    if chunk_pairs < 1:
        raise ValueError(f"chunk_pairs must be at least 1, got {chunk_pairs}")
    pairs, spans = _plan_pairs(student_hidden, teacher_hidden, mapping, mode, teacher_layers)
    n_pairs = len(pairs)
    if backward is None:
        def backward(tensor):  # noqa: E306 - the documented default
            tensor.backward()

    # Detached stand-ins. `detach()` shares storage with the student's own tensor, so this
    # allocates nothing; it only redirects where the loss's gradient lands.
    touched = set()
    for s in pairs:
        touched.add(s + 1)
        if mode == "delta":
            touched.add(s)
    order = sorted(touched)
    view = list(student_hidden)
    for i in order:
        leaf = student_hidden[i].detach()
        leaf.requires_grad_(True)
        view[i] = leaf

    mag_sum = dir_sum = 0.0
    per_layer, s_norms, t_norms = {}, [], []
    n_chunks = 0
    for start in range(0, n_pairs, chunk_pairs):
        terms = []
        for s in pairs[start:start + chunk_pairs]:
            student, teacher = _pair_tensors(view, teacher_hidden, mapping, spans, mode, s)
            magnitude, direction, s_norm, t_norm = _pair_term(
                student, teacher, s, normalise=normalise, mask=mask)
            terms.append(magnitude + direction_weight * direction)
            mag_sum += float(magnitude.detach())
            dir_sum += float(direction.detach())
            per_layer[s] = float(magnitude.detach())
            s_norms.append(s_norm)
            t_norms.append(t_norm)
        # 1/n_pairs, never 1/len(chunk): the divisor is the objective's, not the chunk's.
        chunk_loss = torch.stack(terms).sum() / n_pairs
        if loss_scale != 1.0:
            chunk_loss = chunk_loss * loss_scale
        backward(chunk_loss)
        n_chunks += 1
        # Drop this chunk's graph before the next one allocates its own. Without this the
        # chunking saves nothing: Python would hold the tensors until the loop rebinds.
        del terms, chunk_loss, student, teacher, magnitude, direction

    magnitude = mag_sum / n_pairs
    direction = dir_sum / n_pairs
    output = BehavioralLossOutput(
        total=magnitude + direction_weight * direction,
        magnitude=magnitude, direction=direction, mode=mode, n_pairs=n_pairs,
        per_layer=per_layer,
        student_norm=sum(s_norms) / len(s_norms), teacher_norm=sum(t_norms) / len(t_norms),
    )
    sources, grads = [], []
    for i in order:
        leaf = view[i]
        sources.append(student_hidden[i])
        grads.append(leaf.grad if leaf.grad is not None else torch.zeros_like(leaf))
    return ChunkedBehavioralLoss(
        output=output, sources=sources, grads=grads,
        n_chunks=n_chunks, chunk_pairs=chunk_pairs,
    )


def attention_behavior_loss(
    student_attentions: tuple[Tensor, ...] | list[Tensor],
    teacher_attentions: tuple[Tensor, ...] | list[Tensor],
    pairs: list[tuple[int, int]],
    *,
    mask: Tensor | None = None,
) -> BehavioralLossOutput:
    """Match *where the model looks*, marginalised over heads.

    Student and teacher have different head counts (24 against the teacher's), so per-head
    correspondence does not exist and inventing one would be an untested modelling choice.
    Averaging over heads gives a per-query distribution over keys that both models define
    identically, and comparing those with a KL is a statement about attention behaviour that
    survives the head-count change.

    ``pairs`` are ``(student_index, teacher_index)`` into the *attention layers only* —
    the student's 12 and the teacher's 16 — not into all layers.
    """
    import torch

    if not pairs:
        raise ValueError("no attention layer pairs given")
    terms, per_layer = [], {}
    for s_idx, t_idx in pairs:
        s = student_attentions[s_idx].float().mean(1)       # (batch, q, k)
        t = teacher_attentions[t_idx].float().mean(1)
        if s.shape != t.shape:
            raise ValueError(
                f"attention pair ({s_idx}, {t_idx}): {tuple(s.shape)} against "
                f"{tuple(t.shape)} — both must be scored on the same sequence"
            )
        kl = (t * (t.clamp_min(1e-9).log() - s.clamp_min(1e-9).log())).sum(-1)
        if mask is not None:
            m = mask.bool()
            kl = kl[m] if kl.shape[:2] == m.shape else kl.reshape(-1)[m.reshape(-1)]
        value = kl.mean()
        terms.append(value)
        per_layer[s_idx] = float(value)
    total = torch.stack(terms).mean()
    return BehavioralLossOutput(
        total=total, magnitude=total, direction=total * 0.0,
        mode="attention_kl", n_pairs=len(pairs), per_layer=per_layer,
    )


# ---------------------------------------------------------------------------
# composable loss configuration
# ---------------------------------------------------------------------------
CE = "ce"
LOGIT_KD = "logit_kd"
HIDDEN_POINTWISE = "hidden_pointwise"
HIDDEN_DELTA = "hidden_delta"
ATTENTION = "attention"
DELTANET_STATE = "deltanet_state"
ROUTER_BALANCE = "router_balance"
MTP = "mtp"
REASONING = "reasoning"


@dataclass(frozen=True)
class LossTerm:
    """One composable term: what it does, whether it can run, and what it needs."""

    name: str
    status: str
    description: str
    #: Forward flags the term needs switched on to produce its inputs.
    requires: tuple[str, ...] = ()
    blocking_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.status == IMPLEMENTED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"available": self.available}


LOSS_TERMS: dict[str, LossTerm] = {
    CE: LossTerm(CE, IMPLEMENTED, "cross-entropy against the reference tokens"),
    LOGIT_KD: LossTerm(
        LOGIT_KD, IMPLEMENTED,
        "KL against the teacher's output distribution, with exact tail mass",
    ),
    HIDDEN_POINTWISE: LossTerm(
        HIDDEN_POINTWISE, IMPLEMENTED,
        "conventional layer matching: student hidden states against mapped teacher hidden "
        "states. The A1 control.",
        requires=("output_hidden_states",),
    ),
    HIDDEN_DELTA: LossTerm(
        HIDDEN_DELTA, IMPLEMENTED,
        "behavioural matching: each student layer's residual contribution against the summed "
        "contribution of the teacher span it replaced",
        requires=("output_hidden_states",),
    ),
    ATTENTION: LossTerm(
        ATTENTION, IMPLEMENTED,
        "head-marginalised attention distribution KL on the 12 full-attention layers",
        requires=("output_attentions",),
    ),
    DELTANET_STATE: LossTerm(
        DELTANET_STATE, NOT_IMPLEMENTED,
        "direct matching of the DeltaNet recurrent state tensor",
        blocking_reason=(
            "the student's recurrent state (16 key heads, 48 value heads, head dim 128) and "
            "the teacher's are different shapes, and any comparison needs a projection that "
            "is itself an untested modelling choice. Use hidden_delta, which measures the "
            "same DeltaNet layers' behaviour at the hidden-size interface where the shapes "
            "genuinely agree."
        ),
    ),
    ROUTER_BALANCE: LossTerm(
        ROUTER_BALANCE, IMPLEMENTED,
        "the architecture's own load-balancing auxiliary loss, coefficient 0.001, obtained by "
        "passing output_router_logits=True",
        requires=("output_router_logits",),
    ),
    MTP: LossTerm(
        MTP, NOT_IMPLEMENTED,
        "multi-token prediction against the teacher's MTP head",
        blocking_reason=(
            "the runtime builds no MTP head for this architecture, so the student has no "
            "tensors to train and the teacher's mtp.* weights are discarded on load. A "
            "reported MTP result would be fabricated. The field is kept as the extension "
            "point; see MTP_STATUS in architecture.moe_student."
        ),
    ),
    REASONING: LossTerm(
        REASONING, IMPLEMENTED,
        "reasoning-mode preservation: KD applied separately inside and outside the "
        "<think>...</think> span, so thinking behaviour is distilled rather than averaged away",
        requires=("reasoning_spans",),
    ),
}


@dataclass
class CompositeLossConfig:
    """Weights for the composable terms, validated against what can actually run.

    Weights default to zero. A term is off unless a weight is given, so a configuration
    cannot accidentally include an objective nobody chose, and an experiment's loss is
    exactly what its config says.
    """

    weights: dict[str, float] = field(default_factory=dict)
    #: ``delta`` matching's direction term, relative to its magnitude term.
    direction_weight: float = 1.0
    kd_temperature: float = 1.0
    kd_tail: str = "bucket"
    normalise_hidden: bool = True
    #: Enabling both hidden-matching terms is the A4 cell of the layer-matching factorial
    #: and is legitimate *when chosen*. It is refused by default because arriving at it by
    #: accident — merging two configs, copying an arm and adding a term — produces a run
    #: whose result cannot be attributed to either objective.
    allow_combined_hidden: bool = False

    def __post_init__(self) -> None:
        unknown = sorted(set(self.weights) - set(LOSS_TERMS))
        if unknown:
            raise ValueError(f"unknown loss terms {unknown}; have {sorted(LOSS_TERMS)}")
        negative = sorted(k for k, v in self.weights.items() if v < 0)
        if negative:
            raise ValueError(f"negative loss weights would reward divergence: {negative}")

    @property
    def active(self) -> dict[str, float]:
        return {k: v for k, v in self.weights.items() if v > 0}

    def validate(self) -> None:
        """Raise if any active term cannot run. Never degrade silently."""
        if not self.active:
            raise ValueError("no loss term has a positive weight; there is nothing to train")
        blocked = [(k, LOSS_TERMS[k].blocking_reason)
                   for k in self.active if not LOSS_TERMS[k].available]
        if blocked:
            detail = "; ".join(f"{name}: {why}" for name, why in blocked)
            raise ObjectiveUnavailable(f"requested unavailable loss terms — {detail}")
        both_hidden = HIDDEN_POINTWISE in self.active and HIDDEN_DELTA in self.active
        if both_hidden and not self.allow_combined_hidden:
            raise ValueError(
                "hidden_pointwise and hidden_delta are the two factors of the layer-matching "
                "ablation; enabling both without allow_combined_hidden=True makes a result "
                "unattributable. Pick one, or set the flag — that combination is arm A4."
            )

    def forward_flags(self) -> dict[str, bool]:
        """The ``model(...)`` keyword flags the active terms require."""
        flags: dict[str, bool] = {}
        for name in self.active:
            for requirement in LOSS_TERMS[name].requires:
                if requirement != "reasoning_spans":
                    flags[requirement] = True
        return flags

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(sorted(self.weights.items())),
            "active": dict(sorted(self.active.items())),
            "direction_weight": self.direction_weight,
            "kd_temperature": self.kd_temperature,
            "kd_tail": self.kd_tail,
            "normalise_hidden": self.normalise_hidden,
            "allow_combined_hidden": self.allow_combined_hidden,
            "forward_flags": self.forward_flags(),
        }


def describe_loss_terms() -> str:
    lines = ["composable loss terms:", ""]
    for term in LOSS_TERMS.values():
        flag = "available" if term.available else "UNAVAILABLE"
        lines.append(f"  {term.name:<18} [{flag}] {term.description}")
        if term.blocking_reason:
            lines.append(f"  {'':<18}  blocked: {term.blocking_reason}")
    return "\n".join(lines)
