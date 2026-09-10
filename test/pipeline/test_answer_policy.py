"""No provider is reached outside the turn's authorization.

These tests drive the real route, web tool, and executer with a provider spy.
Media exists only when the authorized web tool ran.
"""
from __future__ import annotations

import pytest

from test.pipeline.conftest import (
    SESSION_ID,
    FakeAnswer,
    FakeEvaluator,
    FakeIntent,
    FakePlanner,
    FakeReranker,
)
from visionagent.models import AuthorizationScope, ToolName

PATH = f"/ai_search/?session_id={SESSION_ID}"


class ProviderSpy:
    """The web-search provider: every call is recorded."""

    name = "spy"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search(self, query, *, num=10):
        from visionagent.providers.websearch import WebSearchResults

        self.calls.append("search")
        return WebSearchResults(results=[], related_questions=[])

    async def images(self, query, *, num=5):
        from visionagent.providers.websearch import ImageResult

        self.calls.append("images")
        return [ImageResult(title="pic", image_url="https://i/1.png",
                            thumbnail_url="https://i/t.png", link="https://p", source="s")]

    async def videos(self, query, *, num=5):
        from visionagent.providers.websearch import VideoResult

        self.calls.append("videos")
        return [VideoResult(title="clip", link="https://v", thumbnail_url="https://v/t.png")]


@pytest.fixture
def spy(monkeypatch) -> ProviderSpy:
    """Every path to the provider goes through build_web_search; spy on it."""
    from visionagent.service.executer.tools import web
    from visionagent.service.executer.tools.web import snippets

    spy = ProviderSpy()
    monkeypatch.setattr(web, "build_web_search", lambda *a, **k: spy)
    monkeypatch.setattr(snippets, "build_web_search", lambda *a, **k: spy)
    # The snippet re-rank embeds and scores; not needed to prove the boundary.
    async def snippets_for(q):
        return ([
            {"url": "https://example.org", "title": "A",
             "content": "found on the web"}
        ], [])

    monkeypatch.setattr(web, "store_and_query_snippets", snippets_for)
    return spy


@pytest.fixture
def answer() -> FakeAnswer:
    return FakeAnswer()


@pytest.fixture
def real_web_pipeline(monkeypatch, answer):
    """Fake slots, but a real executer running the real web tool."""
    from visionagent.models import RetrievedChunk, SourceType, ToolResult
    from visionagent.pipeline.query import QueryPipeline
    from visionagent.service.executer.concurrent import ConcurrentExecuter
    from visionagent.service.executer.tools import ToolRegistry
    from visionagent.service.executer.tools.web import web_search_answer

    async def rag(query, *, context, emit=None):
        return ToolResult(tool_name=ToolName.RAG, chunks=[RetrievedChunk(
            id="kb1", content="from the corpus", source_type=SourceType.KNOWLEDGE_BASE,
            score=0.9, doc_name="doc.pdf")])

    registry = ToolRegistry({ToolName.RAG: rag, ToolName.WEB_SEARCH: web_search_answer})
    return QueryPipeline(
        intent=FakeIntent(), planner=FakePlanner([ToolName.RAG]),
        executer=ConcurrentExecuter(registry=registry, tool_timeout_s=5),
        reranker=FakeReranker(), evaluator=FakeEvaluator(), answer=answer,
    )


@pytest.fixture
def client(app, real_web_pipeline):
    from fastapi.testclient import TestClient

    from visionagent.api.routes import ai_search_rt

    app.dependency_overrides[ai_search_rt.get_pipeline] = lambda: real_web_pipeline
    return TestClient(app)


def _deny_web(client):
    from visionagent.api import deps

    client.app.dependency_overrides[deps.authorization_scope] = lambda: AuthorizationScope(
        allowed_tools=frozenset(ToolName) - {ToolName.WEB_SEARCH}
    )


def test_disabled_never_reaches_the_provider(client, spy, answer):
    """Policy refuses web search. Asking for it, and then answering, must not
    send the question anywhere public -- not for results, not for pictures."""
    _deny_web(client)
    r = client.post(PATH, json={"message": "what does the corpus say?", "web_search": True})
    assert r.status_code == 200, r.text
    assert spy.calls == [], f"the provider was reached: {spy.calls}"
    assert answer.media_seen == [{"images": [], "videos": []}]


def test_auto_without_a_planned_search_never_reaches_the_provider(client, spy, answer):
    """AUTO, and the planner chose the knowledge base only."""
    r = client.post(PATH, json={"message": "what does the corpus say?", "web_search": False})
    assert r.status_code == 200, r.text
    assert spy.calls == []
    assert answer.media_seen == [{"images": [], "videos": []}]


def test_force_runs_the_search_and_its_media_through_the_executer(client, spy, answer):
    """When allowed and asked for, one authorized tool run produces the results
    and the media together; the answer slot receives them, it does not fetch."""
    r = client.post(PATH, json={"message": "what does the corpus say?", "web_search": True})
    assert r.status_code == 200, r.text
    assert sorted(spy.calls) == ["images", "videos"], spy.calls
    (media,) = answer.media_seen
    assert media["images"][0]["imageUrl"] == "https://i/1.png"
    assert media["videos"][0]["link"] == "https://v"
    assert "searching online" in r.text, "the web step must appear in the trace"


def test_the_answer_slot_owns_no_provider():
    """The bypass was one import. Its absence is the whole guarantee."""
    from pathlib import Path

    from visionagent.service.answer import chat

    src = Path(chat.__file__).read_text(encoding="utf-8")
    assert "build_web_search" not in src
    assert "providers.websearch" not in src
