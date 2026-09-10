"""Tests for config/settings.py.

Intended function: every tuning value is read from the environment with a
recorded default, so nothing that governs behaviour is a literal buried at a
call site.
"""
from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from visionagent.config.settings import _jwt_secret, load_settings

TUNABLES = [
    ("RETRIEVAL_TOP_K", "retrieval_top_k", "7", 7),
    ("RERANK_TOP_N", "rerank_top_n", "9", 9),
    ("VECTOR_SIMILARITY_WEIGHT", "vector_similarity_weight", "0.25", 0.25),
    ("SUFFICIENT_THRESHOLD", "sufficient_threshold", "0.8", 0.8),
    ("TOOL_TIMEOUT_S", "tool_timeout_s", "45", 45.0),
    ("PARSE_TIMEOUT_S", "parse_timeout_s", "600", 600.0),
    ("MAX_UPLOAD_TOTAL_BYTES", "max_upload_total_bytes", "12345", 12345),
    ("MAX_CONCURRENT_UPLOADS", "max_concurrent_uploads", "3", 3),
    ("MAX_UPLOAD_WORKERS", "max_upload_workers", "4", 4),
    ("MAX_UPLOAD_ATTEMPTS", "max_upload_attempts", "5", 5),
    ("UPLOAD_RETRY_BACKOFF_S", "upload_retry_backoff_s", "0.5", 0.5),
    ("GRAPH_ENTITY_TOP_K", "graph_entity_top_k", "3", 3),
    ("GRAPH_RELATION_TOP_K", "graph_relation_top_k", "4", 4),
]

DEFAULTS = {
    "retrieval_top_k": 5, "rerank_top_n": 5, "vector_similarity_weight": 0.6,
    "sufficient_threshold": 0.5, "tool_timeout_s": 120.0, "parse_timeout_s": 1800.0,
    "max_upload_total_bytes": 500 * 1024 * 1024,
    "max_concurrent_uploads": 2,
    "max_upload_workers": 2,
    "max_upload_attempts": 3,
    "upload_retry_backoff_s": 5.0,
    "graph_entity_top_k": 5,
    "graph_relation_top_k": 5,
}


@pytest.mark.parametrize("env,attr,raw,expected", TUNABLES,
                         ids=[t[0] for t in TUNABLES])
def test_each_tunable_is_read_from_the_environment(monkeypatch, env, attr, raw, expected):
    monkeypatch.setenv(env, raw)
    assert getattr(load_settings(), attr) == expected


@pytest.mark.parametrize("attr,expected", sorted(DEFAULTS.items()))
def test_each_tunable_has_the_recorded_default(monkeypatch, attr, expected):
    """Configuration defaults remain part of the deployment contract."""
    for env, _, _, _ in TUNABLES:
        monkeypatch.delenv(env, raising=False)
    assert getattr(load_settings(), attr) == expected


def test_graph_vectors_default_to_the_same_width_as_chunk_vectors(monkeypatch):
    """One embedding model serves both graph and chunk indices by default."""
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    monkeypatch.delenv("GRAPH_EMBEDDING_DIM", raising=False)
    s = load_settings()
    assert s.graph_embedding_dimensions == s.embedding_dimensions == 1024


def test_graph_width_is_still_overridable(monkeypatch):
    """The stored vectors have a fixed width, so an existing index built at a
    different one must still be readable without re-embedding."""
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    monkeypatch.setenv("GRAPH_EMBEDDING_DIM", "1536")
    assert load_settings().graph_embedding_dimensions == 1536


def test_an_empty_value_falls_back_to_the_default(monkeypatch):
    """An unset variable in a .env file arrives as "" rather than absent."""
    monkeypatch.setenv("RERANK_TOP_N", "")
    assert load_settings().rerank_top_n == 5


@pytest.mark.parametrize(
    "name",
    [
        "MAX_UPLOAD_FILES",
        "MAX_UPLOAD_BYTES",
        "MAX_UPLOAD_TOTAL_BYTES",
        "MAX_CONCURRENT_UPLOADS",
        "MAX_UPLOAD_WORKERS",
        "MAX_UPLOAD_ATTEMPTS",
    ],
)
def test_upload_resource_limits_must_be_positive(monkeypatch, name):
    monkeypatch.setenv(name, "0")
    with pytest.raises(RuntimeError, match=f"{name} must be a positive integer"):
        load_settings()


@pytest.mark.parametrize("name", ["PARSE_TIMEOUT_S", "UPLOAD_RETRY_BACKOFF_S"])
def test_upload_time_budgets_must_be_positive(monkeypatch, name):
    monkeypatch.setenv(name, "0")
    with pytest.raises(RuntimeError, match=f"{name} must be positive"):
        load_settings()


def test_graphrag_config_reads_through_to_settings(monkeypatch):
    """Graph retrieval reads its tunables from the canonical settings object."""
    monkeypatch.setenv("GRAPH_ENTITY_TOP_K", "11")
    # Reached through sys.modules: the package binds the name `config` to the
    # GraphRAGConfig instance, which shadows the submodule of the same name.
    import importlib
    import sys

    importlib.import_module("visionagent.service.executer.tools.graphrag.config")
    graph_config = sys.modules["visionagent.service.executer.tools.graphrag.config"]

    monkeypatch.setattr(graph_config, "settings", load_settings())
    assert graph_config.config.ENTITY_TOP_K == 11


def test_development_jwt_secret_is_atomically_shared_across_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    workers = 16
    barrier = Barrier(workers)

    def load_after_barrier() -> str:
        barrier.wait()
        return _jwt_secret("development", tmp_path)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        values = list(pool.map(lambda _index: load_after_barrier(), range(workers)))

    assert len(set(values)) == 1
    keyfile = tmp_path / "jwt_secret.dev"
    assert keyfile.read_text(encoding="utf-8") == values[0]
    assert stat.S_IMODE(keyfile.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".jwt_secret.*.tmp"))


def test_development_jwt_secret_rejects_corrupt_existing_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    keyfile = tmp_path / "jwt_secret.dev"
    keyfile.write_text("short", encoding="utf-8")

    with pytest.raises(RuntimeError, match="corrupt or too weak"):
        _jwt_secret("development", tmp_path)

    assert keyfile.read_text(encoding="utf-8") == "short"
    assert stat.S_IMODE(keyfile.stat().st_mode) == 0o600


def test_development_jwt_secret_never_follows_a_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    outside = tmp_path / "outside"
    outside.write_text("x" * 64, encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    os.symlink(outside, state / "jwt_secret.dev")

    with pytest.raises(RuntimeError, match="cannot be opened safely"):
        _jwt_secret("development", state)

    assert outside.read_text(encoding="utf-8") == "x" * 64
