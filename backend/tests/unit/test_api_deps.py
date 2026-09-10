"""Tests for api/deps.py.

Intended function:
    current_user_id -> exactly the authenticated user's id, as a str
    owned_session   -> exactly the session id, but only when the caller owns
                       it; otherwise the same 404 verify_session_owner raises
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from visionagent.api import deps


def test_current_user_id_returns_the_id_as_a_string(monkeypatch):
    """Downstream indexes Elasticsearch by this value, and an int index name
    is a different index from a str one."""
    monkeypatch.setattr(deps, "require_auth", lambda c: 7)
    assert deps.current_user_id(credentials=object()) == "7"


def test_current_user_id_rejects_missing_identity_instead_of_returning_string_none(
    monkeypatch,
):
    def reject(_credentials):
        raise HTTPException(status_code=401, detail="Authentication required")

    monkeypatch.setattr(deps, "require_auth", reject)
    with pytest.raises(HTTPException) as exc:
        deps.current_user_id(credentials=None)
    assert exc.value.status_code == 401


def test_owned_session_returns_the_session_when_the_caller_owns_it(monkeypatch):
    seen: list[tuple] = []
    monkeypatch.setattr(deps, "verify_session_owner",
                        lambda s, u: seen.append((s, u)))
    assert deps.owned_session(session_id="abc", user_id="u1") == "abc"
    assert seen == [("abc", "u1")]


def test_owned_session_propagates_the_ownership_failure(monkeypatch):
    """404, not 403: confirming a session exists but belongs to someone else
    is itself a disclosure."""
    def deny(session_id, user_id):
        raise HTTPException(status_code=404, detail="Session not found")

    monkeypatch.setattr(deps, "verify_session_owner", deny)
    with pytest.raises(HTTPException) as exc:
        deps.owned_session(session_id="someone-elses", user_id="attacker")
    assert exc.value.status_code == 404


def test_the_check_is_a_parameter_so_a_handler_cannot_forget_it():
    """verify_session_owner was called by hand at six sites. As a dependency
    the parameter IS the check."""
    import typing

    hints = typing.get_type_hints(deps.owned_session, include_extras=True)
    assert "user_id" in hints, "the ownership check must depend on the authenticated user"


def _chat(message="q", web_search=False, deep_research=False):
    from visionagent.models import ChatRequest

    return ChatRequest(message=message, web_search=web_search,
                       deep_research=deep_research)


def _scope():
    from visionagent.models import AuthorizationScope, ToolName

    return AuthorizationScope(allowed_tools=frozenset(ToolName))


def test_the_job_is_minted_only_after_the_ownership_check(monkeypatch):
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        deps,
        "verify_session_owner",
        lambda session_id, user_id: seen.append((session_id, user_id)),
    )

    job = deps.authorized_query_job(
        user_id="653", scope=_scope(), request=_chat(), session_id="sessA"
    )

    assert seen == [("sessA", "653")]
    assert job.identity.user_id == "653"
    assert job.session_id == "sessA"
    assert job.identity.run_id


def test_no_job_is_minted_for_a_foreign_session(monkeypatch):
    def deny(_session_id, _user_id):
        raise HTTPException(status_code=404, detail="Session not found")

    monkeypatch.setattr(deps, "verify_session_owner", deny)
    with pytest.raises(HTTPException) as exc:
        deps.authorized_query_job(
            user_id="653", scope=_scope(), request=_chat(), session_id="sessB"
        )
    assert exc.value.status_code == 404


def test_the_job_carries_no_credential():
    """It outlives the handler and reaches the logs, so anything in it that
    could re-authenticate would be a long-lived credential with no owner."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(deps, "verify_session_owner", lambda *a: None)
    try:
        job = deps.authorized_query_job(
            user_id="653", scope=_scope(), request=_chat(), session_id="sessA"
        )
    finally:
        monkeypatch.undo()

    dumped = job.model_dump_json()
    for secret in ("password", "token", "salting", "credential", "authorization_header"):
        assert secret not in dumped.lower()


@pytest.mark.parametrize(
    "asked,allowed,expected_effective",
    [
        (True, True, "force"),
        (False, True, "auto"),
        (True, False, "disabled"),
        (False, False, "disabled"),
    ],
    ids=["asked+allowed", "not-asked+allowed", "asked+denied", "not-asked+denied"],
)
def test_web_search_is_resolved_against_policy_not_the_request(
    monkeypatch, asked, allowed, expected_effective
):
    """A client may ask. Whether it happens is the server's decision, and both
    halves are kept: `requested` says what was asked, `effective` what will
    happen, so "asked and refused" is distinguishable from "never asked"."""
    from visionagent.models import AuthorizationScope, ToolName

    monkeypatch.setattr(deps, "verify_session_owner", lambda *a: None)
    tools = frozenset(ToolName) if allowed else frozenset(ToolName) - {ToolName.WEB_SEARCH}

    job = deps.authorized_query_job(
        user_id="653",
        scope=AuthorizationScope(allowed_tools=tools),
        request=_chat(web_search=asked),
        session_id="sessA",
    )

    assert job.effective.web_search.value == expected_effective
    assert job.requested.web_search.value == ("force" if asked else "auto")


@pytest.mark.parametrize("message", ["   ", "\n\t ", "\u00a0"])
def test_a_whitespace_only_question_is_refused(monkeypatch, message):
    """ChatRequest only requires a string, so a question of spaces reached the
    pipeline, was embedded, and produced an answer to nothing."""
    monkeypatch.setattr(deps, "verify_session_owner", lambda *a: None)
    with pytest.raises(HTTPException) as exc:
        deps.authorized_query_job(
            user_id="653", scope=_scope(), request=_chat(message), session_id="sessA"
        )
    assert exc.value.status_code == 422


def test_a_question_past_the_limit_is_refused(monkeypatch):
    from visionagent.config.settings import settings

    monkeypatch.setattr(deps, "verify_session_owner", lambda *a: None)
    with pytest.raises(HTTPException) as exc:
        deps.authorized_query_job(
            user_id="653", scope=_scope(),
            request=_chat("x" * (settings.max_question_chars + 1)),
            session_id="sessA",
        )
    assert exc.value.status_code == 422


def test_ai_search_rejects_a_foreign_session_before_starting_the_stream(monkeypatch):
    from visionagent.api.routes import ai_search_rt

    app = FastAPI()
    app.include_router(ai_search_rt.router)
    app.dependency_overrides[deps.current_user_id] = lambda: "attacker"

    def deny(_session_id, _user_id):
        raise HTTPException(status_code=404, detail="Session not found")

    pipeline_started = False

    def forbidden_pipeline():
        nonlocal pipeline_started
        pipeline_started = True
        raise AssertionError("pipeline must not start before authorization")

    monkeypatch.setattr(deps, "verify_session_owner", deny)
    app.dependency_overrides[ai_search_rt.get_pipeline] = forbidden_pipeline

    response = TestClient(app).post(
        "/ai_search/?session_id=victim-session",
        json={"message": "steal context"},
    )

    assert response.status_code == 404
    assert pipeline_started is False


def test_ai_search_passes_the_authorized_context_into_the_stream(monkeypatch):
    from visionagent.api.routes import ai_search_rt

    app = FastAPI()
    app.include_router(ai_search_rt.router)
    app.dependency_overrides[deps.current_user_id] = lambda: "653"
    monkeypatch.setattr(deps, "verify_session_owner", lambda _session, _user: None)

    observed = []

    class Pipeline:
        async def stream(self, run):
            observed.append(run)
            if False:
                yield None

    # A dependency override, not a module attribute: FastAPI resolves the
    # dependency when the route is registered, so patching the name afterwards
    # leaves the real pipeline in place.
    app.dependency_overrides[ai_search_rt.get_pipeline] = lambda: Pipeline()
    response = TestClient(app).post(
        "/ai_search/?session_id=sessA",
        json={"message": "hello"},
    )

    assert response.status_code == 200
    assert len(observed) == 1
    assert observed[0].context.user_id == "653"
    assert observed[0].context.session_id == "sessA"
    assert observed[0].context.run_id
