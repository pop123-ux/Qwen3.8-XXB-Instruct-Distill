"""Tests that offline mode genuinely never touches the network.

The claim these back is load-bearing: a report produced from a local metadata directory
is only meaningful if it came *purely* from the supplied files. If a tool silently
fetched a newer config from the Hub, the report would describe something other than
what the contributor supplied.

Rather than trust the environment variables, these tests **sever the network** by
replacing ``socket.socket`` with a class that raises on construction, then run the
offline code paths. Any attempted connection fails the test.
"""

from __future__ import annotations

import json
import os
import socket

import pytest
from conftest import requires_stack

from qwen_distill.teacher.metadata import load_metadata, validate_metadata
from qwen_distill.utils.offline import OFFLINE_ENV, looks_local, offline_for, offline_mode


class NetworkAccessAttempted(AssertionError):
    """Raised if code under test tries to open a socket."""


@pytest.fixture
def no_network(monkeypatch):
    """Sever the network for the duration of a test."""

    def forbidden(*args, **kwargs):
        raise NetworkAccessAttempted("code under test attempted to open a socket")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    return forbidden


@pytest.fixture
def metadata_dir(tmp_path):
    root = tmp_path / "meta"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_text", "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 128, "num_hidden_layers": 4, "intermediate_size": 256,
        "vocab_size": 512, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 32, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "linear_key_head_dim": 16, "linear_value_head_dim": 16,
        "tie_word_embeddings": False, "max_position_embeddings": 4096,
        "full_attention_interval": 4,
    }), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": "{{ reasoning_effort }}<think></think>",
    }), encoding="utf-8")
    return root


# --- the guard itself --------------------------------------------------
def test_offline_mode_sets_and_restores_env():
    for key in OFFLINE_ENV:
        os.environ.pop(key, None)
    with offline_mode() as active:
        assert active
        for key, value in OFFLINE_ENV.items():
            assert os.environ[key] == value
    for key in OFFLINE_ENV:
        assert key not in os.environ


def test_offline_mode_restores_preexisting_values(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "preexisting")
    with offline_mode():
        assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "preexisting"


def test_offline_mode_restores_env_after_an_exception():
    for key in OFFLINE_ENV:
        os.environ.pop(key, None)
    with pytest.raises(RuntimeError), offline_mode():
        raise RuntimeError("boom")
    for key in OFFLINE_ENV:
        assert key not in os.environ


def test_offline_mode_can_be_disabled():
    for key in OFFLINE_ENV:
        os.environ.pop(key, None)
    with offline_mode(enabled=False) as active:
        assert not active
        assert "HF_HUB_OFFLINE" not in os.environ


def test_looks_local_distinguishes_paths_from_repo_ids(tmp_path):
    assert looks_local(tmp_path)
    assert not looks_local("Qwen/Qwen3.8-27B")
    assert not looks_local(tmp_path / "does-not-exist")


def test_offline_for_activates_only_on_local_paths(tmp_path):
    with offline_for(tmp_path) as active:
        assert active
    with offline_for("Qwen/Qwen3.8-27B") as active:
        assert not active


# --- the load-bearing claims -------------------------------------------
def test_metadata_loading_never_opens_a_socket(no_network, metadata_dir):
    """Pure-filesystem ingestion. Must work with the network severed."""
    metadata = load_metadata(metadata_dir)
    files, fields = validate_metadata(metadata)
    assert metadata.config["model_type"] == "qwen3_5_text"
    assert any(f.name == "hidden_size" and f.status == "FOUND" for f in fields)
    assert files


@requires_stack
def test_config_resolution_from_local_dir_never_opens_a_socket(no_network, metadata_dir):
    """AutoConfig against a local directory must not phone home."""
    from qwen_distill.teacher.loader import verify_loader

    with offline_for(metadata_dir):
        report = verify_loader(str(metadata_dir))
    assert report.errors == [], report.errors
    assert report.model_type == "qwen3_5_text"
    assert report.resolved_model_class == "Qwen3_5ForCausalLM"


@requires_stack
def test_inspect_local_never_opens_a_socket(no_network, metadata_dir):
    from qwen_distill.teacher.inspect import inspect_local

    with offline_mode():
        report = inspect_local(metadata_dir, config_only=True)
    assert report.model_type == "qwen3_5_text"
    assert report.spec is not None


def test_network_guard_actually_bites(no_network):
    """Confirms the fixture works — otherwise the tests above prove nothing."""
    with pytest.raises(NetworkAccessAttempted):
        socket.socket()
