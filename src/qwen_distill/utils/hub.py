"""Turn Hugging Face access failures into actionable diagnoses.

Reaching the teacher checkpoint is the project's main bottleneck, so when it fails the
message needs to say *which* kind of failure it was and what to do about it. A raw
``httpx.ProxyError`` traceback does not.

The distinctions that matter, because the remedy differs for each:

* **blocked egress** — a proxy or firewall refused the connection; nothing about the
  repo id is wrong;
* **gated or private** — the repo exists but needs authentication or an accepted licence;
* **not found** — wrong repo id, or the revision does not exist;
* **offline** — no network at all.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Substrings that identify each failure class, checked against the exception text.
_PROXY_MARKERS = ("proxyerror", "connect tunnel", "403 forbidden", "proxy", "407")
_AUTH_MARKERS = ("gated", "401", "unauthorized", "authentication", "access to model", "awaiting")
_NOTFOUND_MARKERS = ("404", "not found", "repositorynotfound", "revisionnotfound")
_OFFLINE_MARKERS = ("connection", "timed out", "timeout", "name resolution", "dns", "unreachable")
#: `transformers` collapses "could not reach the Hub" and "no such repo" into one
#: generic OSError, so this phrasing genuinely cannot be resolved further from the
#: message alone. Say so rather than guessing.
_AMBIGUOUS_MARKERS = ("can't load the configuration", "can't load tokenizer", "is not a local folder")


@dataclass(frozen=True)
class HubDiagnosis:
    """A classified checkpoint-access failure, with a remedy."""

    kind: str
    summary: str
    remedy: list[str]
    original: str

    def render(self) -> str:
        lines = [f"Could not reach the checkpoint: {self.summary}", "", "What to try:"]
        lines += [f"  - {step}" for step in self.remedy]
        lines += ["", f"Underlying error: {self.original}"]
        return "\n".join(lines)


def diagnose_hub_error(exc: BaseException, model: str) -> HubDiagnosis:
    """Classify a checkpoint-access failure and suggest what to do next."""
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    local_hint = (
        f"if you have the files locally, pass the directory instead of the repo id "
        f"(e.g. --path /models/{model.split('/')[-1]})"
    )

    if any(marker in lowered for marker in _PROXY_MARKERS):
        return HubDiagnosis(
            kind="egress_blocked",
            summary="a proxy or firewall refused the connection to huggingface.co",
            remedy=[
                "this is a network policy issue, not a problem with the repo id",
                "check HTTPS_PROXY / HTTP_PROXY and any corporate firewall rules",
                "download the metadata on an unrestricted machine and copy it across",
                local_hint,
            ],
            original=text,
        )
    if any(marker in lowered for marker in _AUTH_MARKERS):
        return HubDiagnosis(
            kind="gated_or_private",
            summary="the repository exists but requires authentication or licence acceptance",
            remedy=[
                f"open https://huggingface.co/{model} and accept the licence if prompted",
                "authenticate with `hf auth login` (or set HF_TOKEN)",
                local_hint,
            ],
            original=text,
        )
    if any(marker in lowered for marker in _NOTFOUND_MARKERS):
        return HubDiagnosis(
            kind="not_found",
            summary=f"'{model}' was not found on the Hub",
            remedy=[
                "check the spelling and capitalisation of the repo id",
                "check that the --revision exists, if you passed one",
                "confirm the model has actually been published under that name",
                local_hint,
            ],
            original=text,
        )
    if any(marker in lowered for marker in _AMBIGUOUS_MARKERS):
        return HubDiagnosis(
            kind="unreachable_or_not_found",
            summary=(
                "transformers could not load it, and its error does not distinguish "
                "'the Hub was unreachable' from 'the repo does not exist'"
            ),
            remedy=[
                "check network access to huggingface.co first — a blocked proxy produces "
                "this exact message",
                "then check the repo id spelling and that the model is published",
                "if it is gated, accept the licence and authenticate (`hf auth login`)",
                local_hint,
            ],
            original=text,
        )
    if any(marker in lowered for marker in _OFFLINE_MARKERS):
        return HubDiagnosis(
            kind="offline",
            summary="no network connection to huggingface.co",
            remedy=["check connectivity and DNS, then retry", local_hint],
            original=text,
        )
    return HubDiagnosis(
        kind="unknown",
        summary="the checkpoint could not be loaded",
        remedy=[
            "check the repo id, your network, and your authentication",
            local_hint,
        ],
        original=text,
    )


class HubAccessError(RuntimeError):
    """Raised when a checkpoint cannot be reached, carrying a classified diagnosis."""

    def __init__(self, diagnosis: HubDiagnosis) -> None:
        super().__init__(diagnosis.summary)
        self.diagnosis = diagnosis
