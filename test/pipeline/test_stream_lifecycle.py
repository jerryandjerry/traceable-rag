"""Turn-stream lifecycle, cancellation, ordering, and error-boundary tests."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from test.pipeline.conftest import (
    SESSION_ID,
    USER_ID,
    FakeAnswer,
    FakeEvaluator,
    FakeExecuter,
    FakeIntent,
    FakePlanner,
    FakeReranker,
    frames,
)
from visionagent.models import (
    AuthorizationScope,
    JobIdentity,
    QueryJob,
    StepStatus,
    ToolName,
    TurnOptions,
    WebSearchMode,
)
from visionagent.pipeline.query import QueryPipeline


def _job(question="q", *, web_search=WebSearchMode.AUTO) -> QueryJob:
    return QueryJob(
        identity=JobIdentity(run_id="r1", user_id=USER_ID),
        authorization=AuthorizationScope(allowed_tools=frozenset(ToolName)),
        session_id=SESSION_ID,
        question=question,
        effective=TurnOptions(web_search=web_search),
    )


def _pipeline(**overrides: Any) -> QueryPipeline:
    kw: dict[str, Any] = {
        "intent": FakeIntent(),
        "planner": FakePlanner(),
        "executer": FakeExecuter(),
        "reranker": FakeReranker(),
        "evaluator": FakeEvaluator(),
        "answer": FakeAnswer(),
    }
    kw.update(overrides)
    return QueryPipeline(**kw)


@pytest.fixture(autouse=True)
def _no_db_history(monkeypatch):
    import visionagent.pipeline.query as q

    monkeypatch.setattr(q, "get_user_history_questions", lambda sid: [])


# --------------------------------------------------------------- cancellation
class HangingExecuter(FakeExecuter):
    """A tool that never answers, and records whether it was cancelled."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.cancelled = False

    async def run(self, plan, *, context, allowed_tools, observer=None):
        self.entered.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return []


def test_closing_the_stream_cancels_the_turn():
    """The client disconnected; the tools must stop, not run to completion
    for nobody. Before, one task stayed alive after aclose()."""

    async def go():
        executer = HangingExecuter()
        p = _pipeline(executer=executer)
        stream = p.stream(_job())
        first = await stream.__anext__()
        assert first[0] == "step"
        await executer.entered.wait()

        await stream.aclose()
        await asyncio.sleep(0)

        leftover = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        return executer.cancelled, leftover

    cancelled, leftover = asyncio.run(go())
    assert cancelled, "the hanging tool was never cancelled"
    assert leftover == [], f"tasks survived the stream: {leftover}"


def test_the_whole_turn_has_a_deadline(monkeypatch):
    """Per-tool timeouts bound one call; two rounds of four can still add up."""
    import dataclasses

    from visionagent.config.settings import settings

    monkeypatch.setattr(
        "visionagent.pipeline.query.settings",
        dataclasses.replace(settings, turn_timeout_s=0.05),
    )

    class Slow(FakeExecuter):
        async def run(self, plan, *, context, allowed_tools, observer=None):
            await asyncio.sleep(10)
            return []

    async def go():
        with pytest.raises(TimeoutError):
            async for _ in _pipeline(executer=Slow()).stream(_job()):
                pass

    asyncio.run(go())


# ------------------------------------------------------ the loop stays free
def test_a_slow_async_slot_does_not_stall_the_event_loop():
    """Provider waits yield to the server's event loop."""

    class SlowIntent(FakeIntent):
        async def analyze_chat_scenario(self, question):
            await asyncio.sleep(0.2)
            return "professional"

        async def analyze_query_intent(self, queries):
            await asyncio.sleep(0.2)
            return await super().analyze_query_intent(queries)

    async def go():
        worst = 0.0
        stop = asyncio.Event()

        async def heartbeat():
            nonlocal worst
            while not stop.is_set():
                t0 = time.perf_counter()
                await asyncio.sleep(0.01)
                worst = max(worst, time.perf_counter() - t0 - 0.01)

        beat = asyncio.create_task(heartbeat())
        async for _ in _pipeline(intent=SlowIntent()).stream(_job()):
            pass
        stop.set()
        await beat
        return worst

    assert asyncio.run(go()) < 0.1, "the heartbeat was held while a slot ran"


# ------------------------------------------------------------- trace frames
def test_a_step_is_reported_running_then_done_as_two_snapshots():
    """Queued trace snapshots preserve both the running and terminal states."""

    async def go():
        return [v async for k, v in _pipeline().stream(_job()) if k == "step"]

    steps = asyncio.run(go())
    by_id: dict[str, list[StepStatus]] = {}
    for s in steps:
        by_id.setdefault(s.id, []).append(s.status)
    first = steps[0].id
    assert by_id[first] == [StepStatus.RUNNING, StepStatus.DONE], by_id[first]
    for step_id, statuses in by_id.items():
        if len(statuses) == 2:
            assert statuses == [StepStatus.RUNNING, StepStatus.DONE], (step_id, statuses)


# ------------------------------------------------------------- the answer
def test_the_final_state_carries_the_answer():
    """AgentState.answer was declared and never assigned."""

    async def go():
        return [v async for k, v in _pipeline().stream(_job()) if k == "run"]

    (run,) = asyncio.run(go())
    assert run.answer is not None
    assert run.answer.content == "the answer"
    assert [c.id for c in run.answer.references] == [c.id for c in run.ranked]


def test_auto_searches_the_web_on_the_casual_path_when_the_answer_is_current():
    """AUTO follows intent, while FORCE and DISABLED remain authoritative."""

    async def go(scenario: str, mode: WebSearchMode):
        executer = FakeExecuter()
        p = _pipeline(intent=FakeIntent(scenario), executer=executer)
        async for _ in p.stream(_job("weather in Chicago?", web_search=mode)):
            pass
        return executer.ran

    assert asyncio.run(go("casual_web", WebSearchMode.AUTO)) == [ToolName.WEB_SEARCH]
    assert asyncio.run(go("casual", WebSearchMode.AUTO)) == []
    assert asyncio.run(go("casual_web", WebSearchMode.DISABLED)) == []
    assert asyncio.run(go("casual", WebSearchMode.FORCE)) == [ToolName.WEB_SEARCH]


# ---------------------------------------------------- persistence and errors
def test_the_answer_is_saved_before_the_client_is_told_it_is_done(monkeypatch):
    """A client that closes on [DONE] stops the generator being advanced, so
    a write after the terminal frame may never run."""
    from visionagent.service.answer import chat

    order: list[str] = []

    class _Llm:
        async def stream(self, **kw):
            yield ("hello", "")

    async def name(*args):
        order.append("name")

    monkeypatch.setattr(chat, "_llm", lambda: _Llm())
    monkeypatch.setattr(chat, "write_chat_to_db", lambda *a: order.append("write"))
    monkeypatch.setattr(chat, "check_and_update_session_name", name)

    async def collect():
        out = []
        async for frame in chat.casual_chat_completion(
            "s", "q", "u", "prompt", run_id="run-save-order"
        ):
            if "[DONE]" in frame:
                order.append("done")
            out.append(frame)
        return out

    asyncio.run(collect())

    assert order == ["write", "name", "done"]


def test_a_failed_save_ends_the_stream_with_an_error_not_done(monkeypatch):
    from visionagent.service.answer import chat

    class _Llm:
        async def stream(self, **kw):
            yield ("hello", "")

    def explode(*a):
        raise RuntimeError("postgres: connection refused at db-host:5432")

    monkeypatch.setattr(chat, "_llm", lambda: _Llm())
    monkeypatch.setattr(chat, "write_chat_to_db", explode)

    async def collect():
        return "".join([
            frame async for frame in chat.casual_chat_completion(
                "s", "q", "u", "prompt", run_id="run-save-failure"
            )
        ])

    body = asyncio.run(collect())
    events = frames(body)
    assert ("end", "[DONE]") not in events
    error = [json.loads(d) for e, d in events if e == "error"]
    assert error and error[0]["role"] == "error"
    assert error[0]["run_id"] == "run-save-failure"
    assert "run-save-failure" in error[0]["content"]
    assert "db-host" not in error[0]["content"], "internal detail reached the client"


def test_an_answer_provider_failure_carries_the_run_reference(monkeypatch):
    from visionagent.service.answer import chat

    class _Llm:
        async def stream(self, **kw):
            if False:
                yield ("", "")
            raise RuntimeError("provider failed at secret-host")

    monkeypatch.setattr(chat, "_llm", lambda: _Llm())

    async def collect():
        return "".join([
            frame async for frame in chat.casual_chat_completion(
                "s", "q", "u", "prompt", run_id="run-provider-failure"
            )
        ])

    error = [json.loads(data) for event, data in frames(asyncio.run(collect())) if event == "error"]
    assert error == [{
        "role": "error",
        "content": "The casual answer could not be completed. Reference: run-provider-failure",
        "run_id": "run-provider-failure",
    }]


def test_a_failing_turn_streams_a_reference_not_the_exception(client, monkeypatch):
    """The frontend appends the error frame to the answer as if the assistant
    had said it; provider hosts and SQL must not be what it says."""
    from visionagent.api.routes import ai_search_rt

    class Boom:
        async def stream(self, job):
            raise RuntimeError("DashScope 401 at https://secret-host/v1 with key sk-...")
            yield  # pragma: no cover

    client.app.dependency_overrides[ai_search_rt.get_pipeline] = lambda: Boom()
    r = client.post(f"/ai_search/?session_id={SESSION_ID}", json={"message": "q"})
    assert r.status_code == 200
    error = [json.loads(d) for e, d in frames(r.text) if e == "error"]
    assert len(error) == 1
    assert "secret-host" not in error[0]["content"] and "sk-" not in error[0]["content"]
    assert "Reference:" in error[0]["content"]
    assert error[0]["run_id"] in error[0]["content"]


def test_session_creation_returns_conflict_when_account_deletion_has_started(
    client, monkeypatch
):
    from visionagent.api.routes import ai_search_rt
    from visionagent.pipeline.session import AccountWriteUnavailable

    def gated(_user_id: str) -> str:
        raise AccountWriteUnavailable("gate closed")

    monkeypatch.setattr(ai_search_rt, "create_session_workflow", gated)

    response = client.post("/create_session/")

    assert response.status_code == 409
    assert "not created" in response.json()["detail"]


# ------------------------------------------------------ two users, one path
def test_two_identities_through_the_production_dependency_path(app, executer):
    """Two callers, concurrently, through the real route and the real job
    issuance: each turn's tools see their own user and session."""
    import httpx

    from test.pipeline.conftest import FakeSessions

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            a = c.post("/ai_search/?session_id=sessA", json={"message": "A"},
                       headers={"X-Test-User": "42"})
            b = c.post("/ai_search/?session_id=sessB", json={"message": "B"},
                       headers={"X-Test-User": "99"})
            return await asyncio.gather(a, b)

    # sessB belongs to user 99
    from visionagent.api import security

    sessions: FakeSessions = security.SessionRepository()  # the fixture's fake
    sessions.owned["sessB"] = "99"

    ra, rb = asyncio.run(go())
    assert ra.status_code == 200 and rb.status_code == 200
    seen = {(c.user_id, c.session_id) for c in executer.contexts}
    assert seen == {("42", "sessA"), ("99", "sessB")}
    assert len({c.run_id for c in executer.contexts}) == 2


def test_the_pipeline_is_built_once_under_concurrent_cold_start(monkeypatch):
    """FastAPI resolves sync dependencies in worker threads; an unguarded lazy
    global was constructed twice."""
    import threading

    from fastapi import FastAPI

    from visionagent.api.routes import ai_search_rt

    built: list[int] = []

    class _Pipeline:
        def __init__(self):
            time.sleep(0.05)
            built.append(1)

    monkeypatch.setattr(ai_search_rt, "QueryPipeline", _Pipeline)
    app = FastAPI()
    request = type("R", (), {"app": app})()

    got: list[object] = []
    threads = [threading.Thread(target=lambda: got.append(ai_search_rt.get_pipeline(request)))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(built) == 1
    assert len({id(p) for p in got}) == 1


def test_the_answer_is_final_before_the_terminal_frame_is_forwarded():
    """A client that closes on [DONE] never resumes the generator, so the
    assignment after the loop never ran and the state ended with no answer."""
    from visionagent.models import AgentState
    from visionagent.pipeline.trace import Tracer

    async def go():
        p = _pipeline()
        run = AgentState(job=_job())
        async for kind, payload in p._answer(run, Tracer()):
            if kind == "frame" and "[DONE]" in payload:
                return run.answer
        return None

    answer = asyncio.run(go())
    assert answer is not None and answer.content == "the answer"


def test_the_deadline_covers_the_casual_path_too(monkeypatch):
    """The turn deadline covers classification and answering as well as retrieval."""
    import dataclasses

    from visionagent.config.settings import settings

    monkeypatch.setattr("visionagent.pipeline.query.settings",
                        dataclasses.replace(settings, turn_timeout_s=0.05))

    class SlowIntent(FakeIntent):
        async def analyze_chat_scenario(self, question):
            await asyncio.sleep(0.3)
            return "casual"

    async def go():
        with pytest.raises(TimeoutError):
            async for _ in _pipeline(intent=SlowIntent()).stream(_job()):
                pass

    asyncio.run(go())
