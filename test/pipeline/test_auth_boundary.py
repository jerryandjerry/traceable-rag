"""The trust boundary, over HTTP: a token alone is not enough.

A signed token proves the caller held the password at some point in the last
two days. It does not prove the account still exists, or that the password
has not changed since. Both are checked on every request against the users
table, and both are what these assert -- through real routes, with only the
token decoder and the table faked.
"""
from __future__ import annotations

from test.pipeline.conftest import SESSION_ID, USER_ID

CHAT = f"/ai_search/?session_id={SESSION_ID}"


def test_a_live_user_with_a_current_token_is_let_in(client):
    r = client.get("/get_sessions/")
    assert r.status_code == 200
    assert r.json()["user_id"] == USER_ID


def test_a_deleted_users_token_stops_working(client, users):
    """The row is gone, so the token is refused now -- not when it expires."""
    users.exists = False
    assert client.get("/get_sessions/").status_code == 401
    assert client.post(CHAT, json={"message": "still here?"}).status_code == 401


def test_a_token_issued_before_a_password_change_is_refused(client, users):
    """The row's auth_version moved; every token carrying the old one dies."""
    users.version = 1
    stale = {"X-Test-Version": "0"}
    fresh = {"X-Test-Version": "1"}
    assert client.get("/get_sessions/", headers=stale).status_code == 401
    assert client.get("/get_sessions/", headers=fresh).status_code == 200


def test_no_job_is_minted_for_a_revoked_token(client, users, executer):
    users.version = 5
    r = client.post(CHAT, json={"message": "q"}, headers={"X-Test-Version": "4"})
    assert r.status_code == 401
    assert executer.ran == [], "the pipeline must not have run"


def test_changing_the_password_bumps_the_version_in_the_same_commit(monkeypatch):
    """The revocation only works if the bump cannot be skipped."""
    from visionagent.pipeline import account

    class _User:
        password_hash = "$old$"
        auth_version = 3

    class _Db:
        committed = False

        def query(self, _m):
            return self

        def filter(self, *_a):
            return self

        def first(self):
            return user

        def commit(self):
            self.committed = True

        def close(self):
            pass

    user, db = _User(), _Db()
    monkeypatch.setattr(account, "get_db", lambda: iter([db]))
    monkeypatch.setattr(account, "verify_password", lambda p, h: True)
    monkeypatch.setattr("visionagent.utils.password.hash_password", lambda p: "$new$")

    account.change_password(7, "current-pw", "a-longer-new-password")

    assert user.password_hash == "$new$"
    assert user.auth_version == 4
    assert db.committed


def test_policy_comes_from_the_server_not_the_request(client, executer, monkeypatch):
    """ALLOWED_TOOLS is resolved at start-up; a request body cannot widen it."""
    from visionagent.api import deps
    from visionagent.models import AuthorizationScope, ToolName

    client.app.dependency_overrides[deps.authorization_scope] = (
        lambda: AuthorizationScope(allowed_tools=frozenset({ToolName.RAG}))
    )
    r = client.post(CHAT, json={"message": "q", "web_search": True,
                                "allowed_tools": ["web_search"]})
    assert r.status_code == 200
    assert executer.allowed_seen and executer.allowed_seen[0] == frozenset({ToolName.RAG})
    assert ToolName.WEB_SEARCH not in executer.ran


def test_an_unknown_tool_name_in_policy_refuses_to_start():
    """Fail closed at import, not "no tools" or "all tools" at request time."""
    import pytest

    from visionagent.api.deps import _parse_allowed_tools

    with pytest.raises(RuntimeError, match="does not exist"):
        _parse_allowed_tools(("RAG", "teleport"))


def test_a_token_signed_with_another_secret_is_a_401_not_a_500():
    """fastapi_jwt converts only its own exceptions; authlib's BadSignatureError
    escaped it and every request carrying a stale or forged token was a 500
    with a traceback instead of "log in again"."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from visionagent.api.routes import user_rt

    app = FastAPI()
    app.include_router(user_rt.router)
    forged = "eyJhbGciOiJIUzI1NiJ9.eyJzdWJqZWN0Ijp7InVzZXJfaWQiOjF9fQ.not-the-right-signature"
    r = TestClient(app).get("/me", headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401, (r.status_code, r.text[:200])
