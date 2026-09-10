"""The OpenAPI request contract consumed by the frontend."""
from __future__ import annotations

from test.pipeline.conftest import SESSION_ID


def test_the_request_body_is_unchanged(client):
    schema = client.app.openapi()
    body = (schema["paths"]["/ai_search/"]["post"]["requestBody"]
            ["content"]["application/json"]["schema"])
    ref = body.get("$ref", "")
    name = ref.rsplit("/", 1)[-1] if ref else None
    assert name == "ChatRequest", f"the body model changed: {body}"

    props = schema["components"]["schemas"]["ChatRequest"]["properties"]
    assert {"message", "web_search", "deep_research"} <= set(props)
    # A bool on the wire, resolved to a mode server-side. If this became a
    # string the frontend would break silently.
    assert "boolean" in str(props["web_search"]), props["web_search"]


def test_session_id_is_still_a_query_parameter(client):
    schema = client.app.openapi()
    params = schema["paths"]["/ai_search/"]["post"].get("parameters", [])
    assert any(p["name"] == "session_id" and p["in"] == "query" for p in params), params


def test_the_response_is_an_event_stream_with_buffering_off(client):
    r = client.post(f"/ai_search/?session_id={SESSION_ID}", json={"message": "q"})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    # nginx buffers SSE by default, which holds every frame until the turn ends.
    assert r.headers["x-accel-buffering"] == "no"
