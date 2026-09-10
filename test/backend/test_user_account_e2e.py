"""Account management, end to end against the live API.

Intended function:
    POST /me/password  -> the old password stops working and the new one starts,
                          but only when the CURRENT password is supplied
    DELETE /me         -> the account and everything it owns are gone, and only
                          with the password plus an explicit confirmation

Needs no LLM or embedding provider: none of these paths retrieve anything.
"""
from __future__ import annotations

import uuid

import pytest
import requests

PASSWORD = "PytestPass123!"


@pytest.fixture
def fresh_user(live_backend):
    """A throwaway account, and a client already logged into it."""
    name = f"pytest_acct_{uuid.uuid4().hex[:10]}"
    requests.post(f"{live_backend}/register",
                  json={"username": name, "password": PASSWORD}, timeout=60)
    s = requests.Session()
    token = requests.post(f"{live_backend}/login",
                          json={"username": name, "password": PASSWORD},
                          timeout=60).json()["access_token"]
    s.headers["Authorization"] = f"Bearer {token}"
    return name, s


def login(base: str, name: str, password: str) -> int:
    return requests.post(f"{base}/login",
                         json={"username": name, "password": password},
                         timeout=60).status_code


# =========================================================== change password
def test_password_change_swaps_which_password_works(live_backend, fresh_user):
    name, s = fresh_user
    new = "PytestNewPass456!"

    r = s.post(f"{live_backend}/me/password",
               json={"current_password": PASSWORD, "new_password": new}, timeout=60)
    assert r.status_code == 200, r.text

    assert login(live_backend, name, new) == 200, "the new password must work"
    assert login(live_backend, name, PASSWORD) == 401, "the old one must not"


def test_password_change_requires_the_current_password(live_backend, fresh_user):
    """Being logged in is not enough. A session left open on a shared machine
    must not be enough to lock the real owner out of their own account."""
    name, s = fresh_user
    r = s.post(f"{live_backend}/me/password",
               json={"current_password": "not-it", "new_password": "PytestOther789!"},
               timeout=60)
    assert r.status_code == 400
    assert login(live_backend, name, PASSWORD) == 200, "the password must be unchanged"


def test_password_change_rejects_a_short_password(live_backend, fresh_user):
    name, s = fresh_user
    r = s.post(f"{live_backend}/me/password",
               json={"current_password": PASSWORD, "new_password": "short"}, timeout=60)
    assert r.status_code == 400
    assert login(live_backend, name, PASSWORD) == 200


def test_password_change_rejects_reusing_the_same_password(live_backend, fresh_user):
    _, s = fresh_user
    r = s.post(f"{live_backend}/me/password",
               json={"current_password": PASSWORD, "new_password": PASSWORD}, timeout=60)
    assert r.status_code == 400


def test_password_change_needs_authentication(live_backend):
    r = requests.post(f"{live_backend}/me/password",
                      json={"current_password": "x", "new_password": "PytestPass999!"},
                      timeout=60)
    assert r.status_code in (401, 403)


def test_one_user_cannot_change_anothers_password(live_backend, fresh_user):
    """There is no user id in the request at all -- the caller can only ever
    change their own. This pins that the endpoint has no such parameter."""
    victim_name, _ = fresh_user
    _, attacker = fresh_user_session(live_backend)

    attacker.post(f"{live_backend}/me/password",
                  json={"current_password": PASSWORD, "new_password": "PytestAtk111!"},
                  timeout=60)
    assert login(live_backend, victim_name, PASSWORD) == 200, (
        "the victim's password must be untouched"
    )


def fresh_user_session(base: str):
    name = f"pytest_acct_{uuid.uuid4().hex[:10]}"
    requests.post(f"{base}/register", json={"username": name, "password": PASSWORD}, timeout=60)
    s = requests.Session()
    token = requests.post(f"{base}/login",
                          json={"username": name, "password": PASSWORD},
                          timeout=60).json()["access_token"]
    s.headers["Authorization"] = f"Bearer {token}"
    return name, s


# ============================================================ delete account
def test_delete_removes_the_account_and_its_sessions(live_backend, fresh_user):
    name, s = fresh_user
    session_id = s.post(f"{live_backend}/create_session/", timeout=60).json()["session_id"]
    assert s.get(f"{live_backend}/get_sessions/", timeout=60).json()["sessions"]

    r = s.delete(f"{live_backend}/me",
                 json={"password": PASSWORD, "confirm": "DELETE"}, timeout=300)
    assert r.status_code == 200, r.text
    assert r.json()["removed"]["sessions"] >= 1

    assert login(live_backend, name, PASSWORD) == 401, "the account must be gone"
    # And the session must not be readable by anyone afterwards.
    _other_name, other = fresh_user_session(live_backend)
    assert other.get(f"{live_backend}/get_messages/?session_id={session_id}",
                     timeout=60).status_code == 404


def test_delete_requires_the_password(live_backend, fresh_user):
    name, s = fresh_user
    r = s.delete(f"{live_backend}/me",
                 json={"password": "not-it", "confirm": "DELETE"}, timeout=60)
    assert r.status_code == 400
    assert login(live_backend, name, PASSWORD) == 200, "the account must survive"


def test_delete_requires_an_explicit_confirmation(live_backend, fresh_user):
    """Irreversible and it removes documents, history, graph and index. A
    mistyped request must not be able to do it."""
    name, s = fresh_user
    for confirm in ("", "delete", "yes"):
        r = s.delete(f"{live_backend}/me",
                     json={"password": PASSWORD, "confirm": confirm}, timeout=60)
        assert r.status_code == 400, f"confirm={confirm!r} was accepted"
    assert login(live_backend, name, PASSWORD) == 200


def test_delete_needs_authentication(live_backend):
    r = requests.request("DELETE", f"{live_backend}/me",
                         json={"password": "x", "confirm": "DELETE"}, timeout=60)
    assert r.status_code in (401, 403)


def test_a_deleted_users_token_stops_working(live_backend, fresh_user):
    """The token is still cryptographically valid after deletion; every
    endpoint that resolves a user must stop honouring it."""
    _, s = fresh_user
    s.delete(f"{live_backend}/me",
             json={"password": PASSWORD, "confirm": "DELETE"}, timeout=300)
    # A deleted account invalidates an otherwise unexpired stateless token.
    for path in ("/get_sessions/", "/get_files/", "/me"):
        r = s.get(f"{live_backend}{path}", timeout=60)
        assert r.status_code == 401, (path, r.status_code, r.text[:200])
    r = s.post(f"{live_backend}/create_session/", timeout=60)
    assert r.status_code == 401, "a deleted user must not be able to start a new session"
