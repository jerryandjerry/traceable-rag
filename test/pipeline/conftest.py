"""In-process HTTP tests: every router, the real trust boundary, fake stores.

The tier between unit tests and the live-stack suite. One FastAPI app mounts
all six routers and is driven over HTTP, so the routes, the job issuance and
the pipelines run exactly as they do in production.

What is faked is only what would otherwise need a network or a database: the
bearer-token decoder, the two repositories the trust boundary reads, and the
slots. `require_auth`, `verify_session_owner`, `authorized_query_job` and
`delete_account_command` are deliberately NOT overridden -- they are the trust
boundary these tests exist to exercise. Overriding them would leave user
existence, token revocation, session ownership, policy resolution, option
mapping and job minting untested while appearing to test the endpoint.

Needs no Postgres, no Elasticsearch and no provider.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI, Request

from visionagent.models import (
    Answer,
    Evaluation,
    Evidence,
    Intent,
    Plan,
    RetrievedChunk,
    Scenario,
    SourceType,
    ToolCall,
    ToolName,
    ToolResult,
)


# ----------------------------------------------------------- deterministic slots
class FakeIntent:
    name = "fake"

    def __init__(self, scenario: str = "professional") -> None:
        self._scenario = scenario

    async def analyze_chat_scenario(self, question: str) -> str:
        return self._scenario

    async def analyze_query_intent(self, queries: list[str]) -> Intent:
        return Intent(scenario=Scenario.KNOWLEDGE, intents=["kb(filter)"],
                      keywords_high=["k"], keywords_low=["k"])


class FakePlanner:
    """Schedules exactly what it is allowed to, so a test can assert on policy."""

    name = "fake"

    def __init__(self, tools: list[ToolName] | None = None) -> None:
        self._tools = tools or [ToolName.RAG]

    async def agent_plan(self, queries, intent, session_id=None, chat_id=None,
                         available=None, web_search=None) -> Plan:
        from visionagent.models import WebSearchMode

        calls = [ToolCall(tool_name=t, query=list(queries))
                 for t in self._tools if available is None or t in available]
        if web_search is WebSearchMode.FORCE and ToolName.WEB_SEARCH not in [
            c.tool_name for c in calls
        ]:
            calls.append(ToolCall(tool_name=ToolName.WEB_SEARCH, query=list(queries)))
        if web_search is WebSearchMode.DISABLED:
            calls = [c for c in calls if c.tool_name is not ToolName.WEB_SEARCH]
        return Plan(calls=calls)


class FakeExecuter:
    """Records what it was asked to run, and returns one chunk per call."""

    name = "fake"

    def __init__(self) -> None:
        self.ran: list[ToolName] = []
        self.allowed_seen: list[Any] = []
        self.contexts: list[Any] = []

    def names(self) -> list[ToolName]:
        return list(ToolName)

    async def run(self, plan, *, context, allowed_tools, observer=None):
        self.allowed_seen.append(allowed_tools)
        self.contexts.append(context)
        out = []
        for call in plan.calls:
            if call.tool_name not in allowed_tools:
                out.append(ToolResult(tool_name=call.tool_name,
                                      error="POLICY_DENIED: not authorized"))
                continue
            self.ran.append(call.tool_name)
            label = f"{call.tool_name.value} said something"
            if observer is not None:
                with observer.step(f"running {call.tool_name.value}"):
                    observer.emit("chunk", label, chunk_id=call.tool_name.value)
            out.append(ToolResult(
                tool_name=call.tool_name,
                chunks=[RetrievedChunk(id=call.tool_name.value, content=label,
                                       source_type=SourceType.KNOWLEDGE_BASE,
                                       score=0.9, doc_name="doc.pdf")],
            ))
        return out


class FakeReranker:
    name = "fake"

    async def score(self, *, query: str, texts: list[str]) -> list[float]:
        return [0.9] * len(texts)


class FakeEvaluator:
    name = "fake"

    async def evaluate_context_sufficiency(self, chunks, question) -> Evaluation:
        return Evaluation(sufficient_score=0.9, reasons="enough")

    def is_sufficient(self, evaluation: Evaluation) -> bool:
        return evaluation.sufficient_score > 0.5

    async def reflection(self, user_query, context_list, evaluation,
                         web_search=None, available=None) -> list[str] | None:
        return None


class FakeAnswer:
    name = "fake"

    def __init__(self) -> None:
        self.media_seen: list[Any] = []

    async def get_chat_completion(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[str]:
        self.media_seen.append(kwargs.get("media"))
        yield 'event: message\ndata: {"role": "assistant", "content": "the answer"}\n\n'
        yield "event: end\ndata: [DONE]\n\n"

    async def casual_chat_completion(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[str]:
        yield 'event: message\ndata: {"role": "assistant", "content": "hello back"}\n\n'
        yield "event: end\ndata: [DONE]\n\n"


@pytest.fixture
def executer() -> FakeExecuter:
    return FakeExecuter()


@pytest.fixture
def pipeline(executer: FakeExecuter, monkeypatch):
    import visionagent.pipeline.query as query_workflow
    from visionagent.pipeline.query import QueryPipeline

    # This tier has no database; answer assembly still receives an empty,
    # deterministic history through the production boundary.
    monkeypatch.setattr(query_workflow, "get_user_history_questions", lambda _session_id: [])
    return QueryPipeline(
        intent=FakeIntent(),
        planner=FakePlanner(),
        executer=executer,
        reranker=FakeReranker(),
        evaluator=FakeEvaluator(),
        answer=FakeAnswer(),
    )


USER_ID = "42"
SESSION_ID = "sessA"


# ------------------------------------------------------------------ fake stores
class FakeUsers:
    """The users table as the trust boundary sees it.

    `exists` is what a deleted account flips; `version` is what a password
    change bumps; `password_ok` is what verify_password answers.
    """

    def __init__(self) -> None:
        self.exists = True
        self.version = 0
        self.password_ok = True

    def auth_version(self, user_id: str) -> int | None:
        return self.version if self.exists else None

    def password_hash(self, user_id: str) -> str | None:
        return "$hashed$" if self.exists else None

    def accepts_upload_work(self, user_id: str) -> bool:
        return self.exists


def _Session(session_id: str, user_id: str = USER_ID):
    from visionagent.models import SessionResponse

    return SessionResponse(session_id=session_id, session_name="", user_id=user_id,
                           created_at="2026-09-02 00:00:00", updated_at="2026-09-02 00:00:00")


class FakeSessions:
    """The sessions table: one session, owned by USER_ID."""

    def __init__(self) -> None:
        self.owned = {SESSION_ID: USER_ID}
        self.created: list[tuple[str, str]] = []
        self.deleted: list[tuple[str, str]] = []

    def owns(self, session_id: str, user_id: str) -> bool:
        return self.owned.get(session_id) == str(user_id)

    def owner_of(self, session_id: str) -> str | None:
        return self.owned.get(session_id)

    def list_for_user(self, user_id: str):
        return [_Session(s, u) for s, u in self.owned.items() if u == str(user_id)]

    def list_messages(self, user_id: str, session_id: str):
        return []

    def create(self, *, session_id: str, user_id: str, name: str = "") -> None:
        self.owned[session_id] = str(user_id)
        self.created.append((session_id, str(user_id)))

    def delete(self, *, session_id: str, user_id: str) -> bool:
        if self.owned.get(session_id) != str(user_id):
            return False
        del self.owned[session_id]
        self.deleted.append((session_id, str(user_id)))
        return True


@pytest.fixture
def users(monkeypatch) -> FakeUsers:
    from visionagent.api import deps, security
    from visionagent.pipeline import context as context_workflow

    fake = FakeUsers()
    monkeypatch.setattr(security, "UserRepository", lambda: fake)
    monkeypatch.setattr(deps, "UserRepository", lambda: fake)
    monkeypatch.setattr(context_workflow, "UserRepository", lambda: fake)
    monkeypatch.setattr(deps, "verify_password", lambda p, h: fake.password_ok)
    return fake


@pytest.fixture
def sessions(monkeypatch) -> FakeSessions:
    from visionagent.api import deps, security
    from visionagent.api.routes import history_rt
    from visionagent.pipeline import context as context_workflow
    from visionagent.pipeline import session as session_workflow

    fake = FakeSessions()
    for module in (security, deps, history_rt, context_workflow, session_workflow):
        monkeypatch.setattr(module, "SessionRepository", lambda: fake)
    return fake


@pytest.fixture
def deletions(monkeypatch) -> list:
    """What the account route handed the deletion workflow."""
    from visionagent.api.routes import user_rt

    seen: list = []

    def record(command):
        seen.append(command)
        return {"sessions": 0, "messages": 0, "documents": 0, "chunks": 0}

    monkeypatch.setattr(user_rt, "delete_account", record)
    return seen


@pytest.fixture
def app(monkeypatch, pipeline, users, sessions):
    """Every real router, with only the network edges faked.

    The bearer token is decoded for real in production; here the decoder
    returns a fixed subject, and `X-Test-User` / `X-Test-Version` headers let
    a test present a different identity or a stale token. Everything after
    the decoder -- existence, revocation, ownership, policy, option mapping,
    minting -- runs unchanged.
    """
    from visionagent.api import security
    from visionagent.api.routes import (
        add_context_rt,
        ai_search_rt,
        file_upload_rt,
        graphml_rt,
        history_rt,
        user_rt,
    )

    app = FastAPI()
    for module in (user_rt, history_rt, ai_search_rt, add_context_rt, file_upload_rt, graphml_rt):
        app.include_router(module.router)
    app.state.pipeline = pipeline
    app.dependency_overrides[ai_search_rt.get_pipeline] = lambda: pipeline

    def decoded_token(request: Request) -> dict[str, Any]:
        return {
            "user_id": request.headers.get("x-test-user", USER_ID),
            "user_name": "tester",
            "auth_version": int(request.headers.get("x-test-version", "0")),
        }

    app.dependency_overrides[security.access_security] = decoded_token
    return app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


def frames(text: str) -> list[tuple[str, str]]:
    """The SSE stream as (event, data) pairs."""
    out: list[tuple[str, str]] = []
    event = None
    for line in text.splitlines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: ") and event is not None:
            out.append((event, line[6:]))
    return out


__all__ = ["Answer", "Evidence", "frames"]
