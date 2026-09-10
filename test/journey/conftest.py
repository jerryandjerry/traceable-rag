"""Journey fixtures. The repo-root test/conftest.py supplies base_url etc."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def sample_doc() -> dict:
    """The document the journey ingests, and the questions that interrogate it.

    `Doc1.pdf` is the pinned Apache-2.0 RAGFlow benchmark fixture. An answer
    containing the right phrase is not by itself proof that retrieval ran, so
    the assertions check **provenance**: every citation must be a
    knowledge_base chunk carrying this document's name. A model answering from
    memory produces no citations at all.
    """
    path = HERE.parent / "fixtures" / "ragflow" / "Doc1.pdf"
    if not path.exists():
        pytest.skip(f"{path} is missing")

    return {
        "pdf": path.name,
        "path": path,
        "title": "Purpose of RAGFlow",
        "signature": "Purpose of RAGFlow",
        "questions": [
            {
                "question": "What is RAGFlow designed to turn raw documents into?",
                "must_contain": ["reliable context"],
                "answer_should_contain": ["reliable", "context"],
                "fact": "turn raw documents into reliable context",
            },
            {
                "question": "At question time, what does RAGFlow retrieve and send "
                            "to the model?",
                "must_contain": ["relevant passages", "context"],
                "answer_should_contain": ["passages", "context"],
                "fact": "retrieves the most relevant passages and sends them to the model as context",
            },
            {
                "question": "What does RAGFlow say retrieval context reduces, and "
                            "what does source-linked answering improve?",
                "must_contain": ["hallucinations", "traceability"],
                "answer_should_contain": ["hallucinations", "traceability"],
                "fact": "reduces hallucinations and improves traceability",
            },
        ],
    }


@pytest.fixture(scope="session")
def clean_user(live_backend: str):
    """A brand-new account: no documents, no sessions, its own search index.

    This is what "a clean database" means here. Dropping the real database
    would take the developer's own data with it; a fresh user is isolated by
    construction -- the Elasticsearch index is named after the user id, the
    graph files are per user, and every table is scoped by it. Emptiness is
    asserted rather than assumed, and the account is deleted afterwards.
    """
    requests = pytest.importorskip("requests")

    class Client:
        def __init__(self) -> None:
            self.root = live_backend
            self.s = requests.Session()
            self.username = f"journey_{uuid.uuid4().hex[:10]}"
            self.password = "PytestPass123!"

        def req(self, method: str, path: str, **kw):
            kw.setdefault("timeout", 300)
            return self.s.request(method, self.root + path, **kw)

    c = Client()
    c.req("POST", "/register", json={"username": c.username, "password": c.password}).raise_for_status()
    token = c.req("POST", "/login", json={"username": c.username, "password": c.password})
    token.raise_for_status()
    c.s.headers["Authorization"] = f"Bearer {token.json()['access_token']}"
    c.user_id = c.req("GET", "/me").json()["user_id"]

    yield c

    # Removes the account, its documents, chat history, graph and search index.
    c.req("DELETE", "/me", json={"password": c.password, "confirm": "DELETE"}, timeout=600)


def sse(response) -> list[dict]:
    """Decode an SSE stream into frames, dropping keepalives and [DONE]."""
    out = []
    for raw in response.iter_lines():
        if not raw or not raw.startswith(b"data: "):
            continue
        body = raw[6:].decode("utf-8", "replace")
        if body == "[DONE]":
            break
        try:
            out.append(json.loads(body))
        except json.JSONDecodeError:
            continue
    return out


@pytest.fixture(scope="session")
def parsed(clean_user, sample_doc):
    """Upload the document and follow the parse to completion."""
    with sample_doc["path"].open("rb") as fh:
        r = clean_user.req(
            "POST", "/start-processing",
            files={"files": (sample_doc["path"].name, fh, "application/pdf")},
        )
    r.raise_for_status()
    process_id = r.json()["process_id"]

    started = time.perf_counter()
    frames = sse(clean_user.req(
        "GET", f"/get-process-progress/{process_id}", stream=True, timeout=1800))
    errors = [f.get("message", "") for f in frames if f.get("step") == "error"]
    if errors:
        pytest.fail("parsing failed:\n" + "\n".join(errors))

    return {
        "frames": frames,
        "seconds": time.perf_counter() - started,
        "filename": sample_doc["path"].name,
    }
