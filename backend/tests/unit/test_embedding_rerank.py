"""Tests for the embedding/ and rerank/ slots. No network."""
from __future__ import annotations

import asyncio

import pytest

from visionagent.models import RetrievedChunk, SourceType
from visionagent.providers.embedding import Embedder, build_embedder
from visionagent.service.rerank import Reranker, build_reranker


class FakeEmbedder:
    name = "fake"
    dimensions = 4

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0, 0.0, 1.0] for t in texts]

    async def aembed(self, text: str) -> list[float]:
        return self.embed(text)

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.embed_batch(texts)

    async def aclose(self) -> None:
        return None


class FakeReranker:
    """Scores by position in a fixed relevance order, to expose misalignment."""

    name = "fake"

    def __init__(self, ranking: list[str]) -> None:
        self.ranking = ranking

    def score_sync(self, *, query: str, texts: list[str]) -> list[float]:
        return [
            1.0 - (self.ranking.index(t) / max(len(self.ranking), 1))
            if t in self.ranking else 0.0
            for t in texts
        ]

    async def score(self, *, query: str, texts: list[str]) -> list[float]:
        return self.score_sync(query=query, texts=texts)

    async def rerank(
        self, *, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        scores = await self.score(query=query, texts=[c.content for c in chunks])
        rescored = [c.model_copy(update={"score": s}) for c, s in zip(chunks, scores, strict=True)]
        return sorted(rescored, key=lambda c: c.score, reverse=True)


def chunk(cid: str, body: str) -> RetrievedChunk:
    return RetrievedChunk(id=cid, content=body, source_type=SourceType.KNOWLEDGE_BASE)


def run(awaitable):
    return asyncio.run(awaitable)


# ----------------------------------------------------------------- protocols
def test_real_implementations_satisfy_their_protocols():
    assert isinstance(build_embedder(), Embedder)
    assert isinstance(build_reranker(), Reranker)


def test_fakes_satisfy_their_protocols():
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakeReranker([]), Reranker)


@pytest.mark.parametrize(
    "builder,err",
    [(build_embedder, "unknown embedding provider"), (build_reranker, "unknown rerank provider")],
)
def test_factories_reject_unknown_providers(builder, err: str):
    with pytest.raises(ValueError, match=err):
        builder("nope")


def test_embedder_exposes_its_width():
    """The ES mapping keys off vector length (q_<dim>_vec), so a replacement of
    a different width writes to a different field and silently stops matching."""
    e = build_embedder()
    assert isinstance(e.dimensions, int) and e.dimensions > 0


def test_embed_batch_preserves_input_order():
    assert FakeEmbedder().embed_batch(["a", "bbb", "cc"]) == [
        [1.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]
    ]


def test_embed_batch_handles_empty_input():
    assert FakeEmbedder().embed_batch([]) == []


# -------------------------------------------------------------------- rerank
def test_score_is_aligned_to_input_order_not_rank():
    """The regression this slot exists for.

    The provider returns candidates re-sorted by relevance. Reading scores off
    that sorted list and zipping them against the original order gave every
    chunk another chunk's score -- and since the scores arrived descending, the
    caller's own sort became a no-op, so "top N" meant "first N in gather
    order".
    """
    texts = ["least relevant", "most relevant", "middling"]
    r = FakeReranker(ranking=["most relevant", "middling", "least relevant"])
    scores = run(r.score(query="q", texts=texts))
    assert scores[1] > scores[2] > scores[0], f"scores not aligned to input: {scores}"


def test_rerank_keeps_each_score_attached_to_its_own_chunk():
    chunks = [chunk("c1", "least relevant"), chunk("c2", "most relevant"),
              chunk("c3", "middling")]
    r = FakeReranker(ranking=["most relevant", "middling", "least relevant"])
    out = run(r.rerank(query="q", chunks=chunks))

    assert [c.id for c in out] == ["c2", "c3", "c1"]
    by_id = {c.id: c for c in out}
    assert by_id["c2"].content == "most relevant"
    assert by_id["c1"].content == "least relevant"


def test_rerank_output_scores_stay_within_the_contract_bounds():
    """RetrievedChunk.score is bounded [0, 1]; a provider returning a logit
    would otherwise raise a ValidationError deep in the pipeline."""
    out = run(
        FakeReranker(ranking=["a"]).rerank(
            query="q", chunks=[chunk("c1", "a")]
        )
    )
    assert 0.0 <= out[0].score <= 1.0


def test_rerank_handles_empty_input():
    assert run(FakeReranker([]).rerank(query="q", chunks=[])) == []
    assert run(build_reranker().score(query="q", texts=[])) == []


def test_vendored_shim_still_returns_the_legacy_shape(monkeypatch):
    """search_v2 is vendored and does numpy arithmetic on the result, so the
    shim must keep returning (ndarray, placeholder)."""
    import numpy as np

    from visionagent.vendor.ragflow.rag.nlp import model as shim

    monkeypatch.setattr(shim, "_RERANKER", FakeReranker(ranking=["b", "a"]))
    scores, placeholder = shim.rerank_similarity("q", ["a", "b"])
    assert isinstance(scores, np.ndarray) and placeholder is None
    assert scores[1] > scores[0]


# ==========================================================================
# EXACT BEHAVIOUR
#
# Intended function of the embedding component:
#   embed(text)          -> exactly one vector of exactly `dimensions` floats
#   embed_batch(texts)   -> exactly len(texts) vectors, in the INPUT order,
#                           regardless of the order the provider replies in
#   embed_batch([])      -> exactly [] and no provider call
#   aembed[_batch](...)  -> the same contract through native async hosted I/O
#   aclose()             -> release an opened async transport, idempotently
#
# Intended function of the rerank component:
#   await score(query, texts) -> exactly len(texts) floats, the i-th belonging
#                           to texts[i], with 0.0 for anything the provider omits
#   rerank(query, chunks)-> exactly the same chunks, each carrying its own
#                           score, sorted highest first
# ==========================================================================
class StubEmbeddingItem:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


def _stub_embedder(embedder, items, calls=None):
    def create(**kw):
        if calls is not None:
            calls.append(kw)
        return type("R", (), {"data": items})()

    embedder._client = type(
        "Client", (), {"embeddings": type("E", (), {"create": staticmethod(create)})()}
    )()
    return embedder


def test_embed_batch_reorders_provider_output_back_to_input_order():
    """The API may return items out of order; `index` is authoritative. Reading
    them positionally would attach the wrong vector to the wrong chunk."""
    from visionagent.providers.embedding.online import OnlineEmbedder

    e = _stub_embedder(
        OnlineEmbedder(dimensions=1),
        # deliberately shuffled
        [StubEmbeddingItem(2, [3.0]), StubEmbeddingItem(0, [1.0]), StubEmbeddingItem(1, [2.0])],
    )
    assert e.embed_batch(["a", "b", "c"]) == [[1.0], [2.0], [3.0]]


def test_embed_returns_exactly_the_first_vector():
    from visionagent.providers.embedding.online import OnlineEmbedder

    e = _stub_embedder(
        OnlineEmbedder(dimensions=3),
        [StubEmbeddingItem(0, [0.1, 0.2, 0.3])],
    )
    assert e.embed("sidewalk") == [0.1, 0.2, 0.3]


def test_embed_batch_of_nothing_makes_no_provider_call():
    from visionagent.providers.embedding.online import OnlineEmbedder

    calls: list[dict] = []
    e = _stub_embedder(OnlineEmbedder(), [], calls)
    assert e.embed_batch([]) == []
    assert calls == []


def test_embed_batch_sends_the_configured_width_and_model():
    from visionagent.providers.embedding.online import OnlineEmbedder

    calls: list[dict] = []
    e = _stub_embedder(
        OnlineEmbedder(model="text-embedding-v3", dimensions=8),
        [StubEmbeddingItem(0, [0.0] * 8)], calls,
    )
    e.embed_batch(["x"])
    assert calls[0]["model"] == "text-embedding-v3"
    assert calls[0]["dimensions"] == 8
    assert calls[0]["input"] == ["x"]


def test_embedding_failure_becomes_an_embeddingerror():
    from visionagent.providers.embedding import EmbeddingError
    from visionagent.providers.embedding.online import OnlineEmbedder

    e = OnlineEmbedder()

    def boom(**kw):
        raise TimeoutError("provider down")

    e._client = type("Client", (), {"embeddings": type(
        "E", (), {"create": staticmethod(boom)})()})()
    with pytest.raises(EmbeddingError, match="online embedding failed"):
        e.embed("x")


@pytest.mark.parametrize(
    "items,match",
    [
        ([StubEmbeddingItem(0, [1.0]), StubEmbeddingItem(0, [2.0])], "duplicate"),
        ([StubEmbeddingItem(1, [1.0])], "expected"),
        ([StubEmbeddingItem(0, [1.0, 2.0])], "width 2, expected 1"),
    ],
)
def test_online_embedder_rejects_incomplete_or_malformed_batches(items, match):
    from visionagent.providers.embedding import EmbeddingError
    from visionagent.providers.embedding.online import OnlineEmbedder

    e = _stub_embedder(OnlineEmbedder(dimensions=1), items)
    with pytest.raises(EmbeddingError, match=match):
        e.embed_batch(["x"])


def test_online_embedder_constructs_only_the_transport_that_is_used():
    from visionagent.providers.embedding.online import OnlineEmbedder

    e = OnlineEmbedder(dimensions=1)
    assert e._client is None and e._async_client is None

    _stub_embedder(e, [StubEmbeddingItem(0, [1.0])])
    assert e.embed("x") == [1.0]
    assert e._async_client is None


def test_online_async_embedding_is_cancellable_and_client_is_closed():
    from visionagent.providers.embedding.online import OnlineEmbedder

    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()
        closed = asyncio.Event()
        never = asyncio.Event()

        async def create(**kwargs):
            started.set()
            try:
                await never.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def close():
            closed.set()

        e = OnlineEmbedder(dimensions=1)
        e._async_client = type(
            "Client",
            (),
            {
                "embeddings": type(
                    "Embeddings", (), {"create": staticmethod(create)}
                )(),
                "close": staticmethod(close),
            },
        )()

        task = asyncio.create_task(e.aembed("x"))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await e.aclose()

        assert cancelled.is_set()
        assert closed.is_set()
        assert e._async_client is None

    run(scenario())


def test_graph_query_and_ingest_propagate_embedding_failure_without_partial_rows(
    monkeypatch, tmp_path
):
    import visionagent.database.graph.lightrag_storage as graph_storage

    class FailedEmbedder:
        async def aembed(self, text):
            raise RuntimeError("provider down")

        async def aclose(self):
            return None

    monkeypatch.setattr(
        graph_storage, "build_embedder", lambda **kwargs: FailedEmbedder()
    )

    ingest_store = graph_storage.nanoVectorDB(
        2, str(tmp_path / "new.json"), "new"
    )
    with pytest.raises(RuntimeError, match="provider down"):
        run(ingest_store.upsert({"entity-1": {"content": "road"}}))
    assert ingest_store.ids == []
    assert ingest_store.embeddings == []
    assert ingest_store.metadatas == []

    query_store = graph_storage.nanoVectorDB(
        2, str(tmp_path / "existing.json"), "existing"
    )
    query_store.ids = ["entity-1"]
    query_store.embeddings = [[1.0, 0.0]]
    query_store.metadatas = [{"entity_name": "Road"}]
    with pytest.raises(RuntimeError, match="provider down"):
        run(query_store.query("road", top_k=1))


def test_graph_repository_closes_both_vector_store_transports():
    from visionagent.database.graph import GraphRepository

    closed: list[str] = []

    class Store:
        def __init__(self, name):
            self.name = name

        async def aclose(self):
            closed.append(self.name)

    repository = object.__new__(GraphRepository)
    repository.node_vdb = Store("nodes")
    repository.edge_vdb = Store("edges")
    run(repository.aclose())
    assert sorted(closed) == ["edges", "nodes"]


# ------------------------------------------------------------------- rerank
class StubRanked:
    def __init__(self, node, score: float) -> None:
        self.node = node
        self.score = score


def _stub_reranker(reranker, order: list[int], scores: list[float]):
    """Return provider indices in rank order, as DashScope does."""
    results = [
        type("Result", (), {"index": i, "relevance_score": score})()
        for i, score in zip(order, scores, strict=True)
    ]
    response = type(
        "Response",
        (),
        {"status_code": 200, "output": type("Output", (), {"results": results})()},
    )()

    async def async_call(**kwargs):
        return response

    def sync_call(**kwargs):
        return response

    reranker._async_call = async_call
    reranker._sync_call = sync_call
    return reranker


def test_score_returns_exact_values_aligned_to_input(monkeypatch):
    """Provider ranks texts[1] first, then [2], then [0]. The returned list
    must still read [score_of_0, score_of_1, score_of_2]."""
    from visionagent.providers.rerank.dashscope.provider import DashScopeReranker

    r = _stub_reranker(DashScopeReranker(), order=[1, 2, 0], scores=[0.9, 0.5, 0.1])
    assert run(r.score(query="q", texts=["a", "b", "c"])) == [0.1, 0.9, 0.5]


def test_texts_the_provider_omits_score_exactly_zero(monkeypatch):
    """Missing entries must default rather than shift the alignment."""
    from visionagent.providers.rerank.dashscope.provider import DashScopeReranker

    r = _stub_reranker(DashScopeReranker(), order=[2, 0], scores=[0.7, 0.2])
    assert run(r.score(query="q", texts=["a", "b", "c"])) == [0.2, 0.0, 0.7]


def test_rerank_returns_exact_chunk_order_and_exact_scores(monkeypatch):
    """Sorting and cutting is rerank_results() -- shared by every provider
    rather than reimplemented by each -- so it is driven here."""
    import visionagent.service.rerank as agent
    from visionagent.providers.rerank.dashscope.provider import DashScopeReranker

    r = _stub_reranker(DashScopeReranker(), order=[1, 2, 0], scores=[0.9, 0.5, 0.1])
    monkeypatch.setattr(agent, "build_reranker", lambda *a, **k: r)
    out = run(agent.rerank_results(
        [chunk("c0", "a"), chunk("c1", "b"), chunk("c2", "c")], "q", top_n=5))
    assert [(c.id, c.score) for c in out] == [("c1", 0.9), ("c2", 0.5), ("c0", 0.1)]


def test_rerank_clamps_provider_scores_into_the_contract_bounds(monkeypatch):
    """RetrievedChunk.score is [0, 1]; a provider returning a logit would
    otherwise raise a ValidationError deep in the pipeline."""
    import visionagent.service.rerank as agent
    from visionagent.providers.rerank.dashscope.provider import DashScopeReranker

    r = _stub_reranker(DashScopeReranker(), order=[0, 1], scores=[7.5, -3.0])
    monkeypatch.setattr(agent, "build_reranker", lambda *a, **k: r)
    out = run(
        agent.rerank_results(
            [chunk("c0", "a"), chunk("c1", "b")], "q", top_n=5
        )
    )

    # 7.5 clamps to 1.0 and survives. -3.0 clamps to 0.0 and is then dropped by
    # rerank_results' own "relevancy smaller than 0.1" floor, which predates
    # the slot and is preserved.
    assert [(c.id, c.score) for c in out] == [("c0", 1.0)]


def test_embed_batch_splits_at_the_providers_limit_and_keeps_order():
    """DashScope rejects a batch over 10 outright. A caller should not have to
    know that, and every caller would otherwise rediscover it the hard way."""
    from visionagent.providers.embedding.online import OnlineEmbedder

    sent: list[list[str]] = []

    def create(**kw):
        batch = list(kw["input"])
        sent.append(batch)
        # reply shuffled, to prove `index` is what puts it back in order
        items = [StubEmbeddingItem(i, [float(t)]) for i, t in enumerate(batch)]
        return type("R", (), {"data": list(reversed(items))})()

    e = OnlineEmbedder(dimensions=1)
    e._client = type("C", (), {"embeddings": type("E", (), {"create": staticmethod(create)})()})()

    got = e.embed_batch([str(i) for i in range(25)])
    assert [len(b) for b in sent] == [10, 10, 5], "batches must respect MAX_BATCH"
    assert got == [[float(i)] for i in range(25)], "order must survive the split"


def test_the_reranker_passes_the_key_directly_to_the_async_sdk():
    """Authentication must not depend on a process-wide SDK global."""
    from visionagent.providers.rerank.dashscope.provider import DashScopeReranker

    captured: dict[str, object] = {}

    async def call(**kwargs):
        captured.update(kwargs)
        return type(
            "Response",
            (),
            {"status_code": 200, "output": type("Output", (), {"results": []})()},
        )()

    reranker = DashScopeReranker(api_key="sk-test-key")
    reranker._async_call = call
    run(reranker.score(query="q", texts=["a"]))
    assert captured["api_key"] == "sk-test-key"


def test_reranker_cancellation_reaches_async_transport():
    from visionagent.providers.rerank.dashscope.provider import DashScopeReranker

    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()
        closed = asyncio.Event()
        never = asyncio.Event()

        async def call(**kwargs):
            started.set()
            try:
                await never.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                closed.set()

        reranker = DashScopeReranker(
            api_key="sk-test-key",
            session_factory=Session,
        )
        reranker._async_call = call
        task = asyncio.create_task(reranker.score(query="q", texts=["a"]))
        await asyncio.wait_for(started.wait(), timeout=1)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        assert closed.is_set()

    run(scenario())


# ------------------------------------------------------ offline embeddings


def _fake_st(monkeypatch, width=768, recorder=None):
    """Stand in for the optional local-embedding runtime."""
    import sys
    import types

    class Model:
        def __init__(self, path, device=None):
            self.path, self.device = path, device

        def get_embedding_dimension(self):
            return width

        def encode(self, texts, **kw):
            if recorder is not None:
                recorder.append({"texts": list(texts), **kw})
            return [[float(i)] * width for i, _ in enumerate(texts)]

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = Model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)

    torch = types.ModuleType("torch")
    torch.set_num_threads = lambda _count: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)

    import visionagent.providers.embedding.offline as offline

    monkeypatch.setattr(offline, "_MODELS", {})
    return offline


def test_the_offline_embedder_reports_the_models_own_width(monkeypatch, tmp_path):
    """`dimensions` is read off the weights, not configured. Elasticsearch names
    the field after the vector it is handed (`q_<dim>_vec`), so a number that
    disagreed with the model would write where nothing queries."""
    offline = _fake_st(monkeypatch, width=768)
    e = offline.OfflineEmbedder(model_path=tmp_path)
    assert e.dimensions == 768
    assert len(e.embed("a tall building")) == 768


def test_a_dimension_the_model_cannot_produce_is_refused(monkeypatch, tmp_path):
    """Silently returning 768 to a caller that asked for 1024 is the failure
    base.py warns about: the documents land in q_768_vec and the 1024 queries
    never see them again."""
    from visionagent.providers.embedding import EmbeddingError

    offline = _fake_st(monkeypatch, width=768)
    with pytest.raises(EmbeddingError, match="produces 768-wide vectors, not 1024"):
        offline.OfflineEmbedder(model_path=tmp_path, dimensions=1024)


def test_the_model_is_loaded_once_per_path(monkeypatch, tmp_path):
    """build_embedder() is called per use in places -- snippets.py builds one
    for every web search -- so the 300M-parameter model must not be re-read."""
    offline = _fake_st(monkeypatch)
    a = offline.OfflineEmbedder(model_path=tmp_path)
    b = offline.OfflineEmbedder(model_path=tmp_path)
    assert a._model is b._model


def test_offline_embedding_names_no_task_prefix(monkeypatch, tmp_path):
    """The generic embedder must not assume a model-specific prompt name."""
    seen: list[dict] = []
    offline = _fake_st(monkeypatch, recorder=seen)
    offline.OfflineEmbedder(model_path=tmp_path).embed("a passage")
    assert "prompt_name" not in seen[0]


def test_offline_embed_batch_of_nothing_does_not_touch_the_model(monkeypatch, tmp_path):
    seen: list[dict] = []
    offline = _fake_st(monkeypatch, recorder=seen)
    assert offline.OfflineEmbedder(model_path=tmp_path).embed_batch([]) == []
    assert seen == []


def test_a_missing_model_directory_says_so(monkeypatch, tmp_path):
    from visionagent.providers.embedding import EmbeddingError

    offline = _fake_st(monkeypatch)
    with pytest.raises(EmbeddingError, match="no embedding model at"):
        offline.OfflineEmbedder(model_path=tmp_path / "absent")


def test_the_factory_still_answers_to_the_old_provider_name():
    """EMBEDDING_PROVIDER=dashscope predates there being a second
    implementation; a deployment still setting it must keep booting."""
    from visionagent.providers.embedding import build_embedder

    assert build_embedder("dashscope").name == "online"
    assert build_embedder("online").name == "online"
