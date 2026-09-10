"""Fail-closed semantics at the vendored Elasticsearch wire adapter."""

from __future__ import annotations

from typing import Any

import pytest

from visionagent.vendor.ragflow.rag.utils import es_conn


def _connection(client: Any) -> Any:
    """Construct the decorated adapter class without opening a real client."""
    closure = es_conn.ESConnection.__closure__ or ()
    adapter_class = next(
        cell.cell_contents
        for cell in closure
        if isinstance(cell.cell_contents, type)
    )
    adapter = object.__new__(adapter_class)
    adapter.es = client
    return adapter


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (ConnectionError("cluster unavailable"), "cluster unavailable"),
        (TimeoutError("bulk Timeout"), "bulk Timeout"),
    ],
)
def test_persistent_bulk_transport_failure_never_reports_success(
    failure: Exception,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingClient:
        calls = 0

        def bulk(self, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            raise failure

    client = FailingClient()
    monkeypatch.setattr(es_conn.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="bulk insert failed") as caught:
        _connection(client).insert([{"id": "chunk-a", "content": "body"}], "tenant")

    assert client.calls == es_conn.ATTEMPT_TIME
    assert caught.value.__cause__ is failure
    assert message in str(caught.value.__cause__)


def test_bulk_count_is_derived_only_from_a_complete_identity_checked_response() -> None:
    class PartialClient:
        @staticmethod
        def bulk(**_kwargs: Any) -> dict[str, Any]:
            return {
                "errors": True,
                "items": [
                    {"index": {"_id": "chunk-a", "status": 201}},
                    {
                        "index": {
                            "_id": "chunk-b",
                            "status": 429,
                            "error": {"type": "rejected_execution_exception"},
                        }
                    },
                ],
            }

    failures = _connection(PartialClient()).insert(
        [{"id": "chunk-a"}, {"id": "chunk-b"}], "tenant"
    )

    assert len(failures) == 1
    assert failures[0].startswith("chunk-b:")


@pytest.mark.parametrize(
    "response",
    [
        {"errors": False, "items": []},
        {
            "errors": False,
            "items": [{"index": {"_id": "wrong-id", "status": 201}}],
        },
        {
            "errors": False,
            "items": [{"index": {"_id": "chunk-a"}}],
        },
    ],
)
def test_ambiguous_bulk_response_is_an_error(
    response: dict[str, Any],
) -> None:
    class Client:
        @staticmethod
        def bulk(**_kwargs: Any) -> dict[str, Any]:
            return response

    with pytest.raises(RuntimeError, match="bulk insert failed"):
        _connection(Client()).insert([{"id": "chunk-a"}], "tenant")


def test_delete_refuses_partial_or_ambiguous_completion() -> None:
    class Client:
        @staticmethod
        def delete_by_query(**_kwargs: Any) -> dict[str, Any]:
            return {
                "deleted": 1,
                "total": 2,
                "timed_out": False,
                "version_conflicts": 1,
                "failures": [],
                "_shards": {"failed": 0},
            }

    with pytest.raises(RuntimeError, match="did not fully delete"):
        _connection(Client()).delete({"doc_id": "document"}, "tenant")


def test_delete_returns_a_structurally_verified_count() -> None:
    class Client:
        @staticmethod
        def delete_by_query(**_kwargs: Any) -> dict[str, Any]:
            return {
                "deleted": 2,
                "total": 2,
                "timed_out": False,
                "version_conflicts": 0,
                "failures": [],
                "_shards": {"failed": 0},
            }

    assert _connection(Client()).delete({"doc_id": "document"}, "tenant") == 2
