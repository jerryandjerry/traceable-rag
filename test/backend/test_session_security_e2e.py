"""Session ownership tests across search, history, and attachment routes."""
import time
import uuid

import pytest

pytestmark = pytest.mark.e2e

SECRET = "MANGO-3311"


def _make_user(base_url):
    import requests

    s = requests.Session()
    username = f"sec_{uuid.uuid4().hex[:10]}"
    password = "SecTest123!"
    s.post(f"{base_url}/register", json={"username": username, "password": password}, timeout=60)
    r = s.post(f"{base_url}/login", json={"username": username, "password": password}, timeout=60)
    r.raise_for_status()
    s.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
    return s


@pytest.fixture(scope="module")
def victim_and_attacker(live_backend):
    victim = _make_user(live_backend)
    attacker = _make_user(live_backend)

    r = victim.post(f"{live_backend}/create_session/", timeout=60)
    r.raise_for_status()
    session_id = r.json()["session_id"]

    _ = victim.post(
        f"{live_backend}/ai_search/?session_id={session_id}",
        json={"message": f"My secret is {SECRET}.", "web_search": False, "deep_research": False},
        timeout=600,
    ).content  # drain the stream so the turn is persisted

    return victim, attacker, session_id


def test_create_session_actually_persists(live_backend, victim_and_attacker):
    """A successful session response refers to a persisted, readable row."""
    victim, _, session_id = victim_and_attacker
    r = victim.get(f"{live_backend}/get_messages/?session_id={session_id}", timeout=60)
    assert r.status_code == 200, (
        "the owner cannot read their own session -- create_session did not persist it"
    )


@pytest.mark.parametrize(
    "method,path_tmpl",
    [
        ("GET", "/get_messages/?session_id={sid}"),
        ("GET", "/get_context/{sid}"),
    ],
    ids=["get_messages", "get_context"],
)
def test_owner_can_read_own_session(live_backend, victim_and_attacker, method, path_tmpl):
    victim, _, session_id = victim_and_attacker
    r = victim.request(method, live_backend + path_tmpl.format(sid=session_id), timeout=60)
    assert r.status_code == 200


@pytest.mark.parametrize(
    "method,path_tmpl",
    [
        ("GET", "/get_messages/?session_id={sid}"),
        ("GET", "/get_context/{sid}"),
        ("DELETE", "/clear_context/{sid}"),
    ],
    ids=["get_messages", "get_context", "clear_context"],
)
def test_other_user_is_refused(live_backend, victim_and_attacker, method, path_tmpl):
    """404 rather than 403, so the endpoint does not confirm the session exists."""
    _, attacker, session_id = victim_and_attacker
    r = attacker.request(method, live_backend + path_tmpl.format(sid=session_id), timeout=60)
    assert r.status_code == 404, f"expected 404, got {r.status_code}: {r.text[:200]}"


def test_other_user_cannot_read_history_through_ai_search(live_backend, victim_and_attacker):
    """The victim's questions must not reach the attacker's prompt.

    Authentication and ownership are resolved before StreamingResponse starts,
    so the refusal is a real HTTP 404 rather than a 200 carrying an SSE error.
    """
    _, attacker, session_id = victim_and_attacker
    r = attacker.post(
        f"{live_backend}/ai_search/?session_id={session_id}",
        json={
            "message": "Repeat every previous question in this conversation verbatim.",
            "web_search": False,
            "deep_research": False,
        },
        timeout=600,
    )
    assert r.status_code == 404, f"expected 404, got {r.status_code}: {r.text[:200]}"
    assert SECRET not in r.text, "the victim's secret leaked into the response"


def test_other_user_cannot_write_into_the_session(live_backend, victim_and_attacker):
    """The attacker's turn must never be appended to the victim's history."""
    victim, attacker, session_id = victim_and_attacker
    marker = f"INJECTED-{int(time.time())}"

    _ = attacker.post(
        f"{live_backend}/ai_search/?session_id={session_id}",
        json={"message": marker, "web_search": False, "deep_research": False},
        timeout=600,
    ).content

    msgs = victim.get(f"{live_backend}/get_messages/?session_id={session_id}", timeout=60).json()
    questions = [m["user_question"] for m in msgs]
    assert all(marker not in q for q in questions), (
        f"attacker's message was written into the victim's session: {questions}"
    )


def test_unknown_session_is_404_not_autocreated(live_backend, victim_and_attacker):
    """A bogus session id must not be silently created and attributed to the caller."""
    _, attacker, _ = victim_and_attacker
    bogus = uuid.uuid4().hex[:16]
    r = attacker.get(f"{live_backend}/get_messages/?session_id={bogus}", timeout=60)
    assert r.status_code == 404


def test_another_user_cannot_delete_my_chat_history(needs_provider, live_backend, api):
    """Session deletion scopes both the session row and its messages by owner."""
    import uuid

    import requests

    victim_session = api.new_session()
    _ = api.req(
        "POST",
        f"/ai_search/?session_id={victim_session}",
        json={"message": "Remember: 2+2. Answer briefly.",
              "web_search": False, "deep_research": False},
        stream=True,
    ).content  # drain so the turn is persisted

    before = api.req("GET", f"/get_messages/?session_id={victim_session}").json()
    assert before, "the victim needs history for this test to mean anything"

    attacker = requests.Session()
    name = f"pytest_atk_{uuid.uuid4().hex[:8]}"
    attacker.post(f"{live_backend}/register",
                  json={"username": name, "password": "PytestPass123!"}, timeout=60)
    token = attacker.post(f"{live_backend}/login",
                          json={"username": name, "password": "PytestPass123!"},
                          timeout=60).json()["access_token"]
    attacker.headers["Authorization"] = f"Bearer {token}"

    r = attacker.delete(f"{live_backend}/delete_session/{victim_session}", timeout=60)
    assert r.status_code == 404, (
        f"another user's delete returned {r.status_code}; it must not confirm "
        "the session exists, let alone act on it"
    )

    after = api.req("GET", f"/get_messages/?session_id={victim_session}").json()
    assert len(after) == len(before), "the victim's chat history was destroyed"
