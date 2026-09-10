"""Tests for the tools/ slot and the registry.

Intended function of a retrieval tool:
    retrieve(query) -> exactly one ToolResult, its chunks carrying the tool's
    own source_type, scores clamped into [0, 1], and a provider failure
    reported as `error` rather than raised -- because the pipeline gathers
    tools concurrently and one failing must not end the round.

Intended function of the registry:
    run(name, queries) -> exactly one merged ToolResult: every query's chunks
    in order, related questions de-duplicated, and errors collected rather
    than swallowed. This is what the four duplicated if/elif blocks each did
    by hand.
"""
from __future__ import annotations

import asyncio

import pytest

from visionagent.models import RetrievedChunk, SourceType, ToolContext, ToolName, ToolResult
from visionagent.service.executer.tools import ToolRegistry, build_registry


def run(coro):
    return asyncio.run(coro)


CONTEXT = ToolContext(run_id="run-1", user_id="user-1", session_id="session-1")


class StubTool:
    """A tool with no provider behind it.

    A tool is a callable now -- the registry holds the functions themselves
    (`rag`, `graphrag`, `web_search_answer`, `direct_llm_answer`) rather than
    objects wrapping them -- so this defines __call__, not .retrieve().
    """

    def __init__(self, name: ToolName, per_query: dict[str, ToolResult] | None = None,
                 boom: bool = False) -> None:
        self._name = name
        self._per_query = per_query or {}
        self._boom = boom
        self.seen: list[str] = []
        self.contexts: list[ToolContext] = []

    async def __call__(self, query: str, *, context: ToolContext, emit=None) -> ToolResult:
        self.seen.append(query)
        self.contexts.append(context)
        if self._boom:
            raise RuntimeError("provider exploded")
        return self._per_query.get(query, ToolResult(tool_name=self._name))


def chunk(cid: str, body: str = "body") -> RetrievedChunk:
    return RetrievedChunk(id=cid, content=body, source_type=SourceType.KNOWLEDGE_BASE)


# ================================================================== registry
def test_registry_dispatches_by_name_and_rejects_unknown():
    r = build_registry()
    assert callable(r.get(ToolName.RAG))
    assert callable(r.get("web_search"))
    with pytest.raises(KeyError, match="no tool registered"):
        ToolRegistry().get(ToolName.RAG)


def test_registry_has_reports_membership_without_raising():
    r = build_registry()
    assert r.has("RAG") is True
    assert r.has("deep_research") is False, "an unknown name must not raise"


def test_registry_ships_exactly_the_four_tools():
    """Registry names are the exact tool vocabulary exposed to the planner."""
    assert [n.value for n in build_registry().names()] == [
        "RAG", "GraphRAG", "web_search", "LLM"
    ]


def test_planner_tool_list_is_generated_from_the_registry():
    """Hardcoding it is how a prompt drifts from the tools that exist."""
    described = build_registry().describe()
    for name in ("RAG", "GraphRAG", "web_search", "LLM"):
        assert f"- {name}:" in described


def test_run_merges_every_query_in_order():
    """The four duplicated blocks each looped the queries and extended a list."""
    tool = StubTool(ToolName.RAG, {
        "q1": ToolResult(tool_name=ToolName.RAG, chunks=[chunk("a"), chunk("b")]),
        "q2": ToolResult(tool_name=ToolName.RAG, chunks=[chunk("c")]),
    })
    out = run(ToolRegistry({ToolName.RAG: tool}).run(
        ToolName.RAG, queries=["q1", "q2"], context=CONTEXT, allowed_tools=frozenset(ToolName)))
    assert [c.id for c in out.chunks] == ["a", "b", "c"]
    assert tool.seen == ["q1", "q2"]
    assert tool.contexts == [CONTEXT, CONTEXT]
    assert all(context is CONTEXT for context in tool.contexts)
    assert out.error is None


def test_run_deduplicates_related_questions_across_queries():
    """Merge every query's related questions in first-seen order."""
    tool = StubTool(ToolName.WEB_SEARCH, {
        "q1": ToolResult(tool_name=ToolName.WEB_SEARCH, related_questions=["how wide?", "why?"]),
        "q2": ToolResult(tool_name=ToolName.WEB_SEARCH, related_questions=["why?", "when?"]),
    })
    out = run(ToolRegistry({ToolName.WEB_SEARCH: tool}).run(
        ToolName.WEB_SEARCH, queries=["q1", "q2"], context=CONTEXT, allowed_tools=frozenset(ToolName)))
    assert out.related_questions == ["how wide?", "why?", "when?"]


def test_run_reports_a_raising_tool_instead_of_propagating():
    """Tools are gathered concurrently; one failing must not end the round."""
    reg = ToolRegistry({ToolName.RAG: StubTool(ToolName.RAG, boom=True)})
    out = run(reg.run(ToolName.RAG, queries=["q1"], context=CONTEXT, allowed_tools=frozenset(ToolName)))
    assert out.chunks == []
    assert out.error and "provider exploded" in out.error


def test_run_keeps_the_chunks_of_queries_that_did_succeed():
    class Flaky(StubTool):
        async def __call__(self, query: str, *, context: ToolContext, emit=None) -> ToolResult:
            if query == "bad":
                raise RuntimeError("timeout")
            return ToolResult(tool_name=ToolName.RAG, chunks=[chunk(query)])

    reg = ToolRegistry({ToolName.RAG: Flaky(ToolName.RAG)})
    out = run(reg.run(ToolName.RAG, queries=["good", "bad"], context=CONTEXT, allowed_tools=frozenset(ToolName)))
    assert [c.id for c in out.chunks] == ["good"]
    assert out.error and "timeout" in out.error


def test_run_of_no_queries_returns_an_empty_result():
    reg = ToolRegistry({ToolName.RAG: StubTool(ToolName.RAG)})
    out = run(reg.run(ToolName.RAG, queries=[], context=CONTEXT, allowed_tools=frozenset(ToolName)))
    assert out == ToolResult(tool_name=ToolName.RAG)


def test_registry_refuses_to_run_without_an_explicit_security_context():
    reg = ToolRegistry({ToolName.RAG: StubTool(ToolName.RAG)})
    with pytest.raises(TypeError, match="context"):
        reg.run(ToolName.RAG, queries=["q"], allowed_tools=frozenset(ToolName))


def test_registering_a_new_tool_is_one_call():
    """One registry operation makes a tool discoverable and callable."""
    reg = ToolRegistry()
    reg.register(ToolName.LLM, StubTool(ToolName.LLM))
    assert list(reg.names()) == [ToolName.LLM]
    assert callable(reg.get(ToolName.LLM))


def test_the_production_registry_cannot_be_changed_while_turns_are_running():
    """build_registry() returns a frozen registry.

    One module-wide registry is shared by every concurrent turn, so a
    register() call after start-up would change which tools exist underneath
    requests already in flight.
    """
    reg = build_registry()
    before = list(reg.names())
    with pytest.raises(RuntimeError, match="frozen"):
        reg.register(ToolName.LLM, StubTool(ToolName.LLM))
    assert list(reg.names()) == before


def test_queries_run_concurrently_not_one_after_another():
    """Measured, not asserted structurally: three 0.2s queries must finish in
    well under their 0.6s sum."""
    import time

    class Slow(StubTool):
        async def __call__(self, query: str, *, context: ToolContext, emit=None) -> ToolResult:
            await asyncio.sleep(0.2)
            return ToolResult(tool_name=ToolName.RAG, chunks=[chunk(query)])

    reg = ToolRegistry({ToolName.RAG: Slow(ToolName.RAG)})
    start = time.perf_counter()
    out = run(reg.run(ToolName.RAG, queries=["a", "b", "c"], context=CONTEXT, allowed_tools=frozenset(ToolName)))
    elapsed = time.perf_counter() - start

    assert len(out.chunks) == 3
    assert elapsed < 0.4, f"queries were serialised: {elapsed:.2f}s for 3x0.2s"


# ===================================================================== tools
@pytest.mark.parametrize("name", ["RAG", "GraphRAG", "web_search", "LLM"])
def test_shipped_tools_satisfy_the_protocol(name: str):
    tool = build_registry().get(name)
    # A Protocol with only __call__ cannot be runtime-checked, so assert the
    # shape the registry actually relies on: an awaitable called with an
    # explicit, keyword-only security context.
    import inspect

    assert inspect.iscoroutinefunction(tool), f"{name} is not async"
    params = list(inspect.signature(tool).parameters)
    assert params[:3] == ["query", "context", "emit"], f"{name} takes {params}"
    assert inspect.signature(tool).parameters["context"].kind is inspect.Parameter.KEYWORD_ONLY


def test_rag_maps_a_hit_to_exactly_one_chunk(monkeypatch):
    import visionagent.service.executer.tools.rag as agent
    from visionagent.service.executer.tools.rag import rag

    # Patch the bound reference on the module under test.
    async def retrieve(idx, q):
        return [{
            "chunk_id": "c1", "content_with_weight": "Sidewalk widths.",
            "docnm": "curb.pdf", "page_num": 4, "similarity": 0.83,
            "ref_images": ["REF64"],
        }]

    monkeypatch.setattr(agent, "retrieve_content", retrieve)
    context = ToolContext(run_id="r", user_id="user7", session_id="s")
    out = run(rag("how wide?", context=context))
    assert out.tool_name is ToolName.RAG
    assert out.chunks == [RetrievedChunk(
        id="c1", content="Sidewalk widths.", source_type=SourceType.KNOWLEDGE_BASE,
        score=0.83, doc_name="curb.pdf", page_num=4, images=["REF64"],
    )]


def test_rag_storage_boundary_keeps_page_images_encoded(monkeypatch):
    import visionagent.database.elasticsearch.retrieval as retrieval

    class Dealer:
        @staticmethod
        def retrieval(**_kwargs):
            return {
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "content_with_weight": "Sidewalk widths.",
                        "ref_images": ["REF64", object()],
                    }
                ]
            }

    monkeypatch.setattr(retrieval, "dealer", Dealer())

    rows = retrieval._retrieve_content_sync("tenant", "width", [0.1])

    assert rows[0]["ref_images"] == ["REF64"]


def test_rag_awaits_embedding_then_closes_before_threaded_es(monkeypatch):
    import threading

    import visionagent.database.elasticsearch.retrieval as retrieval

    events: list[str] = []
    worker_threads: list[int] = []

    class Embedder:
        async def aembed(self, text):
            events.append(f"embed:{text}")
            await asyncio.sleep(0)
            return [0.25, 0.75]

        async def aclose(self):
            events.append("close")

    def search(index_name, question, query_vector):
        worker_threads.append(threading.get_ident())
        events.append("search")
        assert (index_name, question, query_vector) == (
            "tenant-7", "street width", [0.25, 0.75]
        )
        return [{"chunk_id": "c1"}]

    monkeypatch.setattr(retrieval, "build_embedder", lambda: Embedder())
    monkeypatch.setattr(retrieval, "_retrieve_content_sync", search)

    async def scenario():
        loop_thread = threading.get_ident()
        result = await retrieval.retrieve_content("tenant-7", "street width")
        return loop_thread, result

    loop_thread, result = run(scenario())
    assert result == [{"chunk_id": "c1"}]
    assert events == ["embed:street width", "close", "search"]
    assert worker_threads and worker_threads[0] != loop_thread


def test_rag_embedding_cancellation_reaches_provider_and_closes(monkeypatch):
    import visionagent.database.elasticsearch.retrieval as retrieval

    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()
        closed = asyncio.Event()
        never = asyncio.Event()
        search_called = False

        class Embedder:
            async def aembed(self, text):
                started.set()
                try:
                    await never.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            async def aclose(self):
                closed.set()

        def search(*args):
            nonlocal search_called
            search_called = True

        monkeypatch.setattr(retrieval, "build_embedder", lambda: Embedder())
        monkeypatch.setattr(retrieval, "_retrieve_content_sync", search)

        task = asyncio.create_task(retrieval.retrieve_content("tenant", "q"))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert cancelled.is_set()
        assert closed.is_set()
        assert not search_called

    run(scenario())


def test_dealer_uses_precomputed_query_vector_without_sync_provider_call(monkeypatch):
    from visionagent.vendor.ragflow.rag.nlp import search_v2

    def forbidden_sync_embedding(text):
        raise AssertionError("request-time embedding escaped into ES worker")

    monkeypatch.setattr(search_v2, "generate_embedding", forbidden_sync_embedding)
    dealer = search_v2.Dealer(dataStore=None)
    dense = dealer.get_vector(
        "street width",
        None,
        embedding_data=[0.25, 0.75],
    )
    assert dense.embedding_data == [0.25, 0.75]


class _NoMediaProvider:
    name = "stub"

    async def images(self, q, *, num=5):
        return []

    async def videos(self, q, *, num=5):
        return []


def test_web_results_carry_the_web_source_type_and_their_url(monkeypatch):
    import visionagent.service.executer.tools.web as agent
    from visionagent.service.executer.tools.web import web_search_answer

    async def snippets(q):
        return (
            [{"url": "https://example.org/a", "title": "A", "content": "body a"}],
            ["a related question"],
        )

    monkeypatch.setattr(agent, "store_and_query_snippets", snippets)
    monkeypatch.setattr(agent, "build_web_search", lambda *a, **k: _NoMediaProvider())
    out = run(web_search_answer("q", context=CONTEXT))
    assert out.chunks[0].source_type is SourceType.WEB_SEARCH
    assert out.chunks[0].url == "https://example.org/a"
    assert out.chunks[0].doc_name == "A"
    assert out.related_questions == ["a related question"]


def test_web_chunk_identity_is_a_deterministic_url_digest():
    import hashlib

    from visionagent.service.executer.tools.web import _web_chunk_id

    url = "https://example.org/same-page"
    expected = "web_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    first = _web_chunk_id({"url": url, "title": "first", "content": "one"})
    second = _web_chunk_id({"url": url, "title": "changed", "content": "two"})
    assert first == second == expected


def test_web_media_rides_on_the_tool_result(monkeypatch):
    """Images and videos are the web tool's output, so they exist only when
    the tool ran -- which is only when policy allowed the search."""
    import visionagent.service.executer.tools.web as agent
    from visionagent.providers.websearch import ImageResult, VideoResult
    from visionagent.service.executer.tools.web import web_search_answer

    class Provider(_NoMediaProvider):
        async def images(self, q, *, num=5):
            return [ImageResult(title="t", image_url="https://i/1.png",
                                thumbnail_url="https://i/t.png", link="https://p", source="s")]

        async def videos(self, q, *, num=5):
            return [VideoResult(title="v", link="https://v", thumbnail_url="https://v/t.png")]

    async def no_snippets(q):
        return [], []

    monkeypatch.setattr(agent, "store_and_query_snippets", no_snippets)
    monkeypatch.setattr(agent, "build_web_search", lambda *a, **k: Provider())
    out = run(web_search_answer("q", context=CONTEXT))
    assert out.images == [{"title": "t", "imageUrl": "https://i/1.png",
                           "thumbnailUrl": "https://i/t.png", "link": "https://p", "source": "s"}]
    assert out.videos == [{"title": "v", "link": "https://v", "imageUrl": "https://v/t.png"}]


def test_web_skips_snippets_with_no_body(monkeypatch):
    """An empty chunk fails RetrievedChunk's min_length and would abort the
    whole tool rather than losing one useless snippet."""
    import visionagent.service.executer.tools.web as agent
    from visionagent.service.executer.tools.web import web_search_answer

    async def snippets(q):
        return ([
            {"url": "u1", "title": "A", "content": "   "},
            {"url": "u2", "title": "B", "content": "real"},
        ], [])

    monkeypatch.setattr(agent, "store_and_query_snippets", snippets)
    monkeypatch.setattr(agent, "build_web_search", lambda *a, **k: _NoMediaProvider())
    out = run(web_search_answer("q", context=CONTEXT))
    assert [c.doc_name for c in out.chunks] == ["B"]


def test_graph_hits_cite_as_knowledge_base(monkeypatch):
    """Graph results resolve to chunks of the user's own documents, so they
    must cite the same way as RAG hits, not as a separate source."""
    import visionagent.service.executer.tools.graphrag as gs

    monkeypatch.setattr(gs, "extract_keywords_advanced", lambda q: (["a"], ["b"]))
    async def search_entities(k, s):
        return [{"chunk_ids": ["g1"]}]

    async def search_relationships(k, s):
        return []

    monkeypatch.setattr(gs, "search_entities", search_entities)
    monkeypatch.setattr(gs, "search_relationships", search_relationships)
    monkeypatch.setattr(gs, "extract_chunk_ids_from_graph_results", lambda r: ["g1"])
    seen_users: list[str] = []

    class Graph:
        async def aclose(self):
            return None

    def graph_service(user_id):
        seen_users.append(user_id)
        return Graph()

    def retrieve_chunks(ids, user_id):
        seen_users.append(user_id)
        return [
            {
                "chunk_id": "g1",
                "content_with_weight": "text",
                "docnm": "d.pdf",
                "sim": 0.5,
                "ref_images": ["REF64"],
            }
        ]

    monkeypatch.setattr(gs, "retrieve_chunks_from_es", retrieve_chunks)
    monkeypatch.setattr(gs, "GraphRepository", graph_service)

    context = ToolContext(run_id="r", user_id="7", session_id="s")
    out = run(gs.graphrag("q", context=context))
    assert out.chunks[0].source_type is SourceType.KNOWLEDGE_BASE
    assert out.chunks[0].score == 0.5
    assert out.chunks[0].images == ["REF64"]
    assert seen_users == ["7", "7"]


def test_a_failing_provider_becomes_an_error_field_not_an_exception(monkeypatch):
    """A tool that raises must mark its own step failed and let the round carry
    on with partial results, without exposing provider details."""
    import visionagent.service.executer.tools.graphrag as gs

    def boom(uid):
        raise ConnectionError("es unreachable")

    monkeypatch.setattr(gs, "GraphRepository", boom)

    out = run(gs.graphrag("q", context=ToolContext(
        run_id="r", user_id="7", session_id="s")))
    assert out.chunks == []
    assert out.error == "graphrag failed"
    assert "es unreachable" not in out.error


def test_graph_embedding_failure_sets_error_and_repository_closes(monkeypatch):
    import visionagent.service.executer.tools.graphrag as gs

    closed = False

    class Graph:
        async def query_entities(self, text, top_k):
            raise RuntimeError("embedding provider down")

        async def aclose(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr(gs, "GraphRepository", lambda uid: Graph())
    monkeypatch.setattr(gs, "extract_keywords_advanced", lambda q: ([], ["entity"]))

    out = run(gs.graphrag("q", context=CONTEXT))
    assert out.error == "graphrag failed"
    assert out.chunks == []
    assert closed


def test_graph_es_failure_sets_error_instead_of_looking_empty(monkeypatch):
    import visionagent.service.executer.tools.graphrag as gs

    class Graph:
        async def query_entities(self, text, top_k):
            return [{"entity_name": "Road", "score": 0.9}]

        async def get_nodes(self, names):
            return {"Road": {"source_id": "chunk-1"}}

        async def aclose(self):
            return None

    def failed_es(ids, user_id):
        raise OSError("Elasticsearch unavailable")

    monkeypatch.setattr(gs, "GraphRepository", lambda uid: Graph())
    monkeypatch.setattr(gs, "extract_keywords_advanced", lambda q: ([], ["road"]))
    monkeypatch.setattr(gs, "retrieve_chunks_from_es", failed_es)

    out = run(gs.graphrag("q", context=CONTEXT))
    assert out.error == "graphrag failed"
    assert out.chunks == []


def test_graph_chunk_reader_propagates_es_and_keeps_images_encoded(monkeypatch):
    import visionagent.database.elasticsearch.chunks as chunks

    class Client:
        def __init__(self, response=None, error=None):
            self.response = response
            self.error = error

        def mget(self, **kwargs):
            if self.error is not None:
                raise self.error
            return self.response

    class Connection:
        def __init__(self, client):
            self.es = client

    monkeypatch.setattr(
        chunks,
        "ESConnection",
        lambda: Connection(Client(error=OSError("down"))),
    )
    with pytest.raises(OSError, match="down"):
        chunks.retrieve_chunks_from_es(["c1"], "tenant")

    response = {"docs": [{
        "found": True,
        "_source": {"ref_images": ["REF64", object()]},
    }]}
    monkeypatch.setattr(
        chunks,
        "ESConnection",
        lambda: Connection(Client(response=response)),
    )
    result = chunks.retrieve_chunks_from_es(["c1"], "tenant")
    assert result[0]["ref_images"] == ["REF64"]


def test_graph_chunk_reader_returns_empty_for_actual_not_found(monkeypatch):
    import visionagent.database.elasticsearch.chunks as chunks

    class Connection:
        class es:
            @staticmethod
            def mget(**kwargs):
                return {"docs": [{"found": False}]}

    monkeypatch.setattr(chunks, "ESConnection", Connection)
    assert chunks.retrieve_chunks_from_es(["missing"], "tenant") == []


def test_graph_cancellation_reaches_async_query_and_repository_closes(monkeypatch):
    import visionagent.service.executer.tools.graphrag as gs

    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()
        closed = asyncio.Event()
        never = asyncio.Event()

        class Graph:
            async def query_entities(self, text, top_k):
                started.set()
                try:
                    await never.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            async def aclose(self):
                closed.set()

        graph = Graph()
        monkeypatch.setattr(gs, "GraphRepository", lambda uid: graph)
        monkeypatch.setattr(
            gs, "extract_keywords_advanced", lambda q: ([], ["entity"])
        )

        task = asyncio.create_task(gs.graphrag("q", context=CONTEXT))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert cancelled.is_set()
        assert closed.is_set()

    run(scenario())


def test_graph_service_closes_only_the_repository_it_constructs(monkeypatch):
    import visionagent.service.vectorstore.graphstore.service as service_module

    class Repository:
        working_dir = "/tmp"
        node_vdb = None
        edge_vdb = None
        knowledge_graph = None

        def __init__(self):
            self.close_calls = 0

        async def aclose(self):
            self.close_calls += 1

    owned = Repository()
    monkeypatch.setattr(service_module, "GraphRepository", lambda uid: owned)
    service = service_module.GraphRAGService("tenant")
    assert run(service.process_chunks_for_graphrag([], "empty.pdf"))["success"] is False
    assert owned.close_calls == 1

    injected = Repository()
    service = service_module.GraphRAGService("tenant", repository=injected)
    assert run(service.process_chunks_for_graphrag([], "empty.pdf"))["success"] is False
    assert injected.close_calls == 0


def test_graph_service_never_reuses_an_llm_across_event_loops(monkeypatch):
    import visionagent.service.vectorstore.graphstore.service as service_module

    clients = []

    class Client:
        def __init__(self):
            self.loop = asyncio.get_running_loop()
            self.closed = False
            clients.append(self)

        async def complete(self, **kwargs):
            assert asyncio.get_running_loop() is self.loop
            return "ok"

        async def aclose(self):
            assert asyncio.get_running_loop() is self.loop
            self.closed = True

    class Repository:
        working_dir = "/tmp"
        node_vdb = None
        edge_vdb = None
        knowledge_graph = None

        async def aclose(self):
            return None

    monkeypatch.setattr(service_module, "_llm", Client)

    async def one_item():
        service = service_module.GraphRAGService(
            "tenant", repository=Repository()
        )
        assert await service._call_llm("prompt") == "ok"
        await service.aclose()

    run(one_item())
    run(one_item())
    assert len(clients) == 2
    assert all(client.closed for client in clients)


def test_graph_service_closes_repository_and_preserves_primary_failure(monkeypatch):
    import visionagent.service.vectorstore.graphstore.service as service_module

    class Client:
        async def aclose(self):
            raise RuntimeError("llm close failed")

    class Repository:
        working_dir = "/tmp"
        node_vdb = None
        edge_vdb = None
        knowledge_graph = None

        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    repository = Repository()
    monkeypatch.setattr(service_module, "GraphRepository", lambda uid: repository)
    service = service_module.GraphRAGService("tenant")
    service._llm_client = Client()

    async def fail(*args, **kwargs):
        raise ValueError("processing failed first")

    monkeypatch.setattr(service, "_process_chunks_for_graphrag", fail)

    with pytest.raises(ValueError, match="processing failed first"):
        run(service.process_chunks_for_graphrag([], "file.pdf"))
    assert repository.closed


def test_async_tenant_lock_wait_is_cancellable_and_releases_descriptor(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    import visionagent.database.graph.repository as repository

    monkeypatch.setattr(
        repository,
        "settings",
        SimpleNamespace(graph_dir=tmp_path),
    )

    async def scenario():
        waiter_entered = asyncio.Event()
        async with repository.async_tenant_lock("tenant"):
            async def wait_for_same_lock():
                async with repository.async_tenant_lock("tenant"):
                    waiter_entered.set()

            waiter = asyncio.create_task(wait_for_same_lock())
            await asyncio.sleep(0.08)
            assert not waiter_entered.is_set()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter

        # If cancellation leaked the waiter's descriptor or lock state, this
        # bounded acquisition fails after the original owner exits.
        async with repository.async_tenant_lock("tenant", timeout_s=0.2):
            pass

    run(scenario())


def test_out_of_range_provider_scores_are_clamped(monkeypatch):
    import visionagent.service.executer.tools.rag as agent
    from visionagent.service.executer.tools.rag import rag

    # Patch on the module under test: agent.py binds these names at import, so
    # patching the module they came from leaves the bound reference untouched.
    async def retrieve(idx, q):
        return [
            {"chunk_id": "hi", "content_with_weight": "x", "similarity": 4.2},
            {"chunk_id": "lo", "content_with_weight": "x", "similarity": -1.0},
        ]

    monkeypatch.setattr(agent, "retrieve_content", retrieve)
    out = run(rag("q", context=ToolContext(run_id="r", user_id="u", session_id="s")))
    assert [(c.id, c.score) for c in out.chunks] == [("hi", 1.0), ("lo", 0.0)]


def test_concurrent_rag_calls_cannot_cross_tenant_indexes(monkeypatch):
    """Overlapping RAG calls carry separate tenant identities into storage."""
    import visionagent.service.executer.tools.rag as agent
    from visionagent.service.executer.tools.rag import rag

    seen: list[tuple[str, str]] = []
    both_arrived = asyncio.Event()

    async def retrieve(index_name: str, query: str):
        seen.append((query, index_name))
        if len(seen) == 2:
            both_arrived.set()
        await asyncio.wait_for(both_arrived.wait(), timeout=2)
        return [{
            "chunk_id": f"{index_name}-{query}",
            "content_with_weight": f"evidence for {index_name}",
        }]

    monkeypatch.setattr(agent, "retrieve_content", retrieve)
    first = ToolContext(run_id="run-a", user_id="653", session_id="sessA")
    second = ToolContext(run_id="run-b", user_id="999", session_id="sessB")

    async def concurrent_calls():
        return await asyncio.gather(
            rag("question-a", context=first),
            rag("question-b", context=second),
        )

    results = run(concurrent_calls())
    assert sorted(seen) == [("question-a", "653"), ("question-b", "999")]
    assert [result.chunks[0].id for result in results] == [
        "653-question-a", "999-question-b",
    ]


def test_direct_llm_loads_only_the_explicit_session_context(monkeypatch):
    import visionagent.service.executer.tools.llm_direct as agent
    from visionagent.database.session_context import session_context_manager
    from visionagent.service.executer.tools.llm_direct import direct_llm_answer

    seen_sessions: list[str] = []
    monkeypatch.setattr(
        session_context_manager,
        "get_session_context",
        lambda session_id: seen_sessions.append(session_id) or "attached text",
    )

    class FakeLLM:
        async def complete(self, **kwargs):
            assert "attached text" in kwargs["prompt"]
            return "answer"

    monkeypatch.setattr(agent, "_llm", lambda: FakeLLM())
    out = run(direct_llm_answer("q", context=CONTEXT))

    assert seen_sessions == ["session-1"]
    assert out.chunks[0].content == "answer"


# ------------------------------------------------------- authorization
def test_the_registry_refuses_a_tool_policy_did_not_authorize():
    """Defence in depth behind the planner.

    The planner is handed only authorized tools, so a plan naming a denied one
    means a planner defect or a plan built somewhere else. Enforcing at the
    gateway means policy does not depend on every upstream step being correct.
    """
    called: list[str] = []

    async def spy(q, *, context, emit=None):
        called.append(q)
        return ToolResult(tool_name=ToolName.WEB_SEARCH)

    registry = ToolRegistry({ToolName.WEB_SEARCH: spy})
    ctx = ToolContext(run_id="r", user_id="u", session_id="s")

    result = asyncio.run(
        registry.run(
            ToolName.WEB_SEARCH, queries=["q"], context=ctx,
            allowed_tools=frozenset({ToolName.RAG}),
        )
    )

    assert called == [], "a denied tool must not be invoked at all"
    assert result.error and result.error.startswith("POLICY_DENIED")
    assert result.chunks == []


def test_an_authorized_tool_still_runs():
    async def spy(q, *, context, emit=None):
        return ToolResult(tool_name=ToolName.RAG,
                          chunks=[RetrievedChunk(id="c1", content="x",
                                                 source_type=SourceType.KNOWLEDGE_BASE)])

    registry = ToolRegistry({ToolName.RAG: spy})
    ctx = ToolContext(run_id="r", user_id="u", session_id="s")

    result = asyncio.run(
        registry.run(ToolName.RAG, queries=["q"], context=ctx,
                     allowed_tools=frozenset({ToolName.RAG}))
    )
    assert result.error is None
    assert [c.id for c in result.chunks] == ["c1"]
