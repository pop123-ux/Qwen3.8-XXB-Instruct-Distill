"""Tests for checkpoint-access failure diagnosis.

Reaching the teacher is this project's main bottleneck, so a failure must say which
kind it was. These pin the classification of the failures actually observed in the
wild, including the one `transformers` reports ambiguously.
"""

from __future__ import annotations

import pytest

from qwen_distill.utils.hub import HubAccessError, diagnose_hub_error

MODEL = "Qwen/Qwen3.8-27B"


class ProxyError(Exception):
    """Stands in for httpx.ProxyError without importing httpx."""


def test_proxy_403_is_classified_as_blocked_egress():
    """The failure this project actually hits."""
    diagnosis = diagnose_hub_error(ProxyError("403 Forbidden"), MODEL)
    assert diagnosis.kind == "egress_blocked"
    assert "proxy or firewall" in diagnosis.summary
    assert any("network policy" in step for step in diagnosis.remedy)


def test_connect_tunnel_failure_is_blocked_egress():
    assert diagnose_hub_error(
        OSError("CONNECT tunnel failed, response 403"), MODEL
    ).kind == "egress_blocked"


def test_gated_repo_is_classified_as_auth():
    diagnosis = diagnose_hub_error(
        OSError("401 Client Error: Unauthorized. Access to model is gated."), MODEL
    )
    assert diagnosis.kind == "gated_or_private"
    assert any("licence" in step or "auth" in step for step in diagnosis.remedy)


def test_missing_repo_is_classified_as_not_found():
    diagnosis = diagnose_hub_error(OSError("404 Client Error: Not Found"), MODEL)
    assert diagnosis.kind == "not_found"
    assert any("spelling" in step for step in diagnosis.remedy)


def test_transformers_generic_error_is_reported_as_ambiguous():
    """`transformers` collapses unreachable and nonexistent into one message.

    Claiming either one specifically would be a guess, so the diagnosis must say the
    error cannot distinguish them.
    """
    diagnosis = diagnose_hub_error(
        OSError(
            "Can't load the configuration of 'Qwen/Qwen3.8-27B'. If you were trying to "
            "load it from 'https://huggingface.co/models', make sure ..."
        ),
        MODEL,
    )
    assert diagnosis.kind == "unreachable_or_not_found"
    assert "does not distinguish" in diagnosis.summary


def test_offline_is_classified_as_offline():
    assert diagnose_hub_error(
        OSError("Failed to establish a new connection: name resolution failed"), MODEL
    ).kind == "offline"


def test_unrecognised_error_still_produces_a_remedy():
    diagnosis = diagnose_hub_error(RuntimeError("something entirely new"), MODEL)
    assert diagnosis.kind == "unknown"
    assert diagnosis.remedy


@pytest.mark.parametrize(
    "exc",
    [
        ProxyError("403 Forbidden"),
        OSError("404 Not Found"),
        RuntimeError("mystery"),
    ],
)
def test_every_diagnosis_suggests_the_local_path_fallback(exc):
    """A local checkpoint sidesteps every network failure, so always mention it."""
    assert any("--path" in step for step in diagnose_hub_error(exc, MODEL).remedy)


def test_render_includes_summary_remedy_and_original():
    text = diagnose_hub_error(ProxyError("403 Forbidden"), MODEL).render()
    assert "Could not reach the checkpoint" in text
    assert "What to try:" in text
    assert "Underlying error:" in text
    assert "403 Forbidden" in text


def test_hub_access_error_carries_the_diagnosis():
    diagnosis = diagnose_hub_error(ProxyError("403 Forbidden"), MODEL)
    error = HubAccessError(diagnosis)
    assert error.diagnosis is diagnosis
    assert isinstance(error, RuntimeError)
