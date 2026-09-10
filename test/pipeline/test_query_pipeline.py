"""One turn, over HTTP, through the real route and the real trust boundary.

This is what the one-way dependency buys. The test imports the router and
nothing else from the package: no slot, no store, no provider. If any layer
started reaching sideways, this file would have to import it too.
"""
from __future__ import annotations

import json

import pytest

from test.pipeline.conftest import SESSION_ID, frames

PATH = f"/ai_search/?session_id={SESSION_ID}"


def post(client, message="how wide is the lane?", **body):
    return client.post(PATH, json={"message": message, **body})


# ------------------------------------------------------------------ the turn
def test_a_turn_streams_progress_then_evidence_then_the_answer(client):
    r = post(client)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = [e for e, _ in frames(r.text)]
    assert "step" in events, f"no progress frames: {set(events)}"
    assert "end" in events, "the stream never terminated"

    # Evidence has to arrive before the answer text, or the citations render
    # against nothing. "message" carries both the legacy progress frames and
    # the answer, so the answer is identified by its content rather than its
    # event name.
    pairs = frames(r.text)
    first_evidence = next(i for i, (e, _) in enumerate(pairs) if e == "evidence")
    first_answer = next(
        i for i, (e, d) in enumerate(pairs)
        if e == "message" and "the answer" in d
    )
    assert first_evidence < first_answer


def test_the_answer_reaches_the_client(client):
    r = post(client)
    payloads = [json.loads(d) for e, d in frames(r.text) if e == "message"]
    assert any("the answer" in str(p.get("content", "")) for p in payloads)


# --------------------------------------------------------------- the boundary
def test_a_foreign_session_is_refused_before_the_stream_starts(client):
    """404 before any header, not a 200 carrying an SSE error frame."""
    r = client.post("/ai_search/?session_id=someone-elses",
                    json={"message": "q"})
    assert r.status_code == 404
    assert "text/event-stream" not in r.headers.get("content-type", "")


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_a_blank_question_is_refused(client, message):
    assert post(client, message).status_code == 422


def test_an_over_long_question_is_refused(client):
    from visionagent.config.settings import settings

    assert post(client, "x" * (settings.max_question_chars + 1)).status_code == 422


# --------------------------------------------------------------- web search
def test_force_runs_web_search(client, executer):
    from visionagent.models import ToolName

    post(client, web_search=True)
    assert ToolName.WEB_SEARCH in executer.ran


def test_auto_does_not_force_web_search(client, executer):
    """OFF on the wire means AUTO, not a prohibition: the planner decides, and
    this planner did not ask for it."""
    from visionagent.models import ToolName

    post(client, web_search=False)
    assert ToolName.WEB_SEARCH not in executer.ran


def test_a_denied_tool_is_never_invoked(client, executer):
    """Policy refuses web search; asking for it changes nothing.

    Overridden through FastAPI, not by patching the module attribute: the
    route captured the dependency at registration, so a patched name turned
    the lambda's argument into a query parameter and the request failed with
    422 -- and an assertion on `executer.ran` alone passed vacuously.
    """
    from visionagent.api import deps
    from visionagent.models import AuthorizationScope, ToolName

    client.app.dependency_overrides[deps.authorization_scope] = lambda: AuthorizationScope(
        allowed_tools=frozenset(ToolName) - {ToolName.WEB_SEARCH}
    )
    r = post(client, web_search=True)
    assert r.status_code == 200, r.text
    assert ToolName.RAG in executer.ran, "the turn ran"
    assert ToolName.WEB_SEARCH not in executer.ran


def test_the_executer_is_told_what_is_authorized(client, executer):
    from visionagent.models import ToolName

    post(client)
    assert executer.allowed_seen, "the executer was never given a scope"
    assert ToolName.RAG in executer.allowed_seen[0]
