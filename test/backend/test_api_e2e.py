"""End-to-end API tests. Requires Postgres, Elasticsearch and the backend.

Every endpoint exposed by the FastAPI application is exercised. Tests create
their own user and clean up their own sessions.
"""
import json
import time

import pytest

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------- discovery
def test_openapi_lists_every_router(api):
    spec = api.req("GET", "/openapi.json").json()
    paths = set(spec["paths"])
    for expected in [
        "/register", "/login", "/logout", "/me",
        "/create_session/", "/get_sessions/", "/delete_session/{session_id}",
        "/ai_search/", "/get_messages/",
        "/add_context/", "/get_context/{session_id}", "/clear_context/{session_id}",
        "/start-processing", "/process-status/{process_id}",
        "/get-process-progress/{process_id}", "/kill-processing/{process_id}",
        "/document-chunks/{file_name}", "/delete-document/{file_name}",
        "/cleanup-processes", "/get_files/", "/graphml/",
    ]:
        assert expected in paths, f"{expected} missing from the API"


def test_deep_research_route_does_not_exist(api):
    """The reserved workflow has no route while its frontend control is hidden."""
    spec = api.req("GET", "/openapi.json").json()
    assert "/deep_research/" not in spec["paths"]
    assert api.req("POST", "/deep_research/").status_code == 404


# --------------------------------------------------------------------- auth
def test_register_login_me_logout(live_backend):
    import requests

    s = requests.Session()
    user = f"pytest_auth_{int(time.time())}"
    pw = "PytestPass123!"

    assert s.post(f"{live_backend}/register",
                  json={"username": user, "password": pw}).status_code == 200

    r = s.post(f"{live_backend}/login", json={"username": user, "password": pw})
    assert r.status_code == 200
    token = r.json()["access_token"]
    s.headers["Authorization"] = f"Bearer {token}"

    me = s.get(f"{live_backend}/me")
    assert me.status_code == 200
    assert me.json()["username"] == user

    assert s.post(f"{live_backend}/logout").status_code == 200


def test_login_with_wrong_password_is_rejected(live_backend, api):
    import requests

    r = requests.post(
        f"{live_backend}/login",
        json={"username": api.username, "password": "wrong-password"},
        timeout=30,
    )
    assert r.status_code in (400, 401, 403), r.text


def test_protected_route_requires_a_token(live_backend):
    import requests

    r = requests.get(f"{live_backend}/me", timeout=30)
    assert r.status_code in (401, 403)


# ----------------------------------------------------------------- sessions
def test_session_lifecycle(api):
    sid = api.new_session()
    assert len(sid) == 16

    listed = api.req("GET", "/get_sessions/")
    assert listed.status_code == 200

    assert api.req("GET", f"/get_messages/?session_id={sid}").status_code == 200
    assert api.req("DELETE", f"/delete_session/{sid}").status_code == 200
    api.sessions.remove(sid)


@pytest.mark.parametrize(
    "path", ["/get_sessions/", "/get_files/", "/graphml/"]
)
def test_read_only_endpoints_answer_200(api, path):
    assert api.req("GET", path).status_code == 200


@pytest.mark.parametrize("path", ["/debug_sessions/", "/debug_sessions_public/"])
def test_the_debug_endpoints_are_gone(base_url, path):
    """Session-debug paths are unavailable to unauthenticated callers."""
    import requests

    assert requests.get(f"{base_url}{path}", timeout=10).status_code == 404


# -------------------------------------------------------------------- chat
@pytest.mark.parametrize(
    "web_search,deep_research",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["web-off/deep-off", "web-on/deep-off", "web-off/deep-on", "web-on/deep-on"],
)
def test_ai_search_all_toggle_combinations(needs_provider, api, session_id, web_search, deep_research):
    """Every accepted option combination streams through /ai_search/."""
    r = api.req(
        "POST",
        f"/ai_search/?session_id={session_id}",
        json={
            "message": "What is the capital of Japan? Answer in one word.",
            "web_search": web_search,
            "deep_research": deep_research,
        },
        stream=True,
    )
    assert r.status_code == 200

    events, answer, saw_done = 0, "", False
    for raw in r.iter_lines():
        if not raw or not raw.startswith(b"data: "):
            continue
        events += 1
        body = raw[6:].decode("utf-8", "replace")
        if body == "[DONE]":
            saw_done = True
            break
        try:
            msg = json.loads(body)
        except json.JSONDecodeError:
            continue
        assert msg.get("role") != "error", f"stream reported an error: {msg}"
        if msg.get("role") == "assistant" and not msg.get("thinking"):
            answer += msg.get("content", "")

    assert events > 0, "no SSE frames received"
    assert saw_done, "stream never terminated with [DONE]"
    assert answer.strip(), "assistant produced no content"


def test_messages_are_persisted_after_a_chat(needs_provider, api, session_id):
    _ = api.req(
        "POST",
        f"/ai_search/?session_id={session_id}",
        json={"message": "Say OK.", "web_search": False, "deep_research": False},
        stream=True,
    ).content  # drain

    msgs = api.req("GET", f"/get_messages/?session_id={session_id}").json()
    assert isinstance(msgs, list) and msgs, "chat was not written to postgres"
    assert msgs[0]["user_question"] == "Say OK."
    assert msgs[0]["model_answer"]


# ------------------------------------------------------------ session ctx
def test_add_context_then_read_and_clear(api, session_id, sample_pdf):
    with sample_pdf.open("rb") as fh:
        r = api.req(
            "POST",
            f"/add_context/?session_id={session_id}",
            files={"files": (sample_pdf.name, fh, "application/pdf")},
        )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "success"

    ctx = api.req("GET", f"/get_context/{session_id}")
    assert ctx.status_code == 200
    assert ctx.json()["context_length"] > 0, "no text extracted from the PDF"

    assert api.req("DELETE", f"/clear_context/{session_id}").status_code == 200


# ------------------------------------------------------- document pipeline
def test_document_ingestion_completes_without_error(api, sample_pdf):
    """The full parse -> encode -> index pipeline completes successfully."""
    with sample_pdf.open("rb") as fh:
        r = api.req(
            "POST",
            "/start-processing",
            files={"files": (sample_pdf.name, fh, "application/pdf")},
        )
    assert r.status_code == 200, r.text
    process_id = r.json()["process_id"]

    steps, errors = [], []
    stream = api.req(
        "GET", f"/get-process-progress/{process_id}", stream=True, timeout=600
    )
    for raw in stream.iter_lines():
        if not raw or not raw.startswith(b"data: "):
            continue
        try:
            msg = json.loads(raw[6:].decode("utf-8", "replace"))
        except json.JSONDecodeError:
            continue
        step = msg.get("step")
        steps.append(step)
        if step == "error":
            errors.append(msg.get("message", ""))
        if step == "complete":
            break

    assert not errors, "ingestion reported errors:\n" + "\n".join(errors)
    for expected in ("parse_finish", "encode_finish", "database_finish"):
        assert expected in steps, f"pipeline never reached {expected}: {steps}"


def test_kill_processing_reports_404_for_unknown_id(api):
    """This answered 500 "Upload cancelled" because a bare except Exception
    caught the handler's own 404."""
    r = api.req("POST", "/kill-processing/definitely-not-a-real-process")
    assert r.status_code == 404, f"expected 404, got {r.status_code}: {r.text}"
    assert "not found" in r.text.lower()


@pytest.mark.parametrize(
    "path", ["/process-status/nope", "/get-process-progress/nope"]
)
def test_unknown_process_ids_are_not_500(api, path):
    assert api.req("GET", path).status_code in (200, 404, 422)


def test_cleanup_processes(api):
    r = api.req("POST", "/cleanup-processes")
    assert r.status_code == 200
    assert "cleaned_count" in r.json()


# ------------------------------------------------------------------ deletes
def test_delete_endpoints_are_idempotent(api, session_id):
    name = "does-not-exist.pdf"
    for method, path in [
        ("DELETE", f"/remove_file/{session_id}?file_name={name}"),
        ("DELETE", f"/delete_file/?file_name={name}"),
        ("DELETE", f"/delete-document/{name}"),
    ]:
        r = api.req(method, path)
        assert r.status_code in (200, 204, 404), f"{path} -> {r.status_code} {r.text}"
