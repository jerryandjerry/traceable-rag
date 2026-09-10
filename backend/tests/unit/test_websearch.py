"""Tests for the websearch/ slot.

Intended function:
    A provider turns whatever its backend returns into `WebSearchResults`.
    Nothing above the slot can tell which backend ran, so swapping DuckDuckGo
    for Serper must change the pipeline's input shape not at all.

No network: every provider's backend is stubbed.
"""
from __future__ import annotations

import asyncio
from urllib.parse import parse_qs

import httpx
import pytest
from pydantic import ValidationError

from visionagent.providers.websearch import (
    WebResult,
    WebSearchError,
    WebSearchProvider,
    WebSearchResults,
    build_web_search,
)


def run(awaitable):
    return asyncio.run(awaitable)


# ================================================================== the slot
def test_ddgs_is_the_default_provider(monkeypatch):
    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    assert build_web_search().name == "ddgs"


def test_the_env_var_selects_the_provider(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "serper")
    assert build_web_search().name == "serper"


def test_duckduckgo_is_accepted_as_an_alias(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "duckduckgo")
    assert build_web_search().name == "ddgs"


def test_an_unknown_provider_names_the_known_ones():
    with pytest.raises(ValueError, match=r"unknown web search provider.*ddgs.*serper"):
        build_web_search("bing")


def test_both_providers_satisfy_the_protocol():
    for name in ("ddgs", "serper"):
        assert isinstance(build_web_search(name), WebSearchProvider), name


# ============================================================ the shared shape
def test_a_result_must_carry_content():
    """An empty snippet is not evidence; it must never reach the prompt."""
    with pytest.raises(ValidationError):
        WebResult(title="t", url="u", content="")


def test_the_shape_forbids_provider_specific_extras():
    """`href`/`body`/`link`/`snippet` must be translated, not passed through."""
    with pytest.raises(ValidationError):
        WebResult(content="x", href="http://example.com")


def test_results_default_to_empty_rather_than_none():
    empty = WebSearchResults()
    assert empty.results == [] and empty.related_questions == []


# ==================================================================== ddgs
def _ddg_html(*hits: tuple[str, str, str]) -> bytes:
    rows = "".join(
        f'<div class="body"><h2>{title}</h2><a href="{url}">{body}</a></div>'
        for title, url, body in hits
    )
    return f"<html><body>{rows}</body></html>".encode()


def _transport(body: bytes, *, status: int = 200, calls: list | None = None):
    async def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        return httpx.Response(status, content=body, request=request)

    return httpx.MockTransport(handler)


def test_ddgs_translates_href_and_body_into_url_and_content():
    provider = build_web_search(
        "ddgs",
        transport=_transport(
            _ddg_html(("Lane Width", "https://nacto.org/x", "Lane widths..."))
        ),
    )
    out = run(provider.search("lane width"))
    assert out.results == [
        WebResult(title="Lane Width", url="https://nacto.org/x", content="Lane widths...")
    ]


def test_ddgs_drops_hits_with_no_body():
    """A hit with no snippet carries no evidence, and WebResult would reject
    it -- so it must be filtered, not allowed to raise mid-search."""
    body = _ddg_html(
        ("a", "u1", "   "),
        ("b", "u2", "real content"),
        ("c", "u3", ""),
    )
    out = run(build_web_search("ddgs", transport=_transport(body)).search("q"))
    assert [r.content for r in out.results] == ["real content"]


def test_ddgs_drops_ad_redirect_records_like_the_ddgs_backend():
    body = _ddg_html(
        ("ad", "https://duckduckgo.com/y.js?ad_domain=x", "sponsored"),
        ("organic", "https://example.com", "real content"),
    )
    out = run(build_web_search("ddgs", transport=_transport(body)).search("q"))
    assert [(result.title, result.url) for result in out.results] == [
        ("organic", "https://example.com")
    ]


def test_ddgs_tolerates_missing_title_and_href(monkeypatch):
    import visionagent.providers.websearch.duckduckgo as ddg

    monkeypatch.setattr(
        ddg, "_parse_text_hits", lambda body: [{"body": "content only"}]
    )
    provider = build_web_search("ddgs", transport=_transport(b"<html/>"))
    assert run(provider.search("q")).results[0] == WebResult(
        title="", url="", content="content only")


def test_ddgs_returns_no_related_questions():
    """A capability DuckDuckGo does not have. Empty, never an error."""
    provider = build_web_search(
        "ddgs", transport=_transport(_ddg_html(("", "", "x")))
    )
    assert run(provider.search("q")).related_questions == []


def test_ddgs_sends_query_and_region_and_respects_requested_count():
    calls: list[httpx.Request] = []
    body = _ddg_html(("a", "u1", "one"), ("b", "u2", "two"))
    provider = build_web_search(
        "ddgs", region="uk-en", transport=_transport(body, calls=calls)
    )
    out = run(provider.search("lane width", num=1))
    form = parse_qs(calls[0].content.decode())
    assert form["q"] == ["lane width"] and form["l"] == ["uk-en"]
    assert [result.content for result in out.results] == ["one"]


def test_ddgs_returning_nothing_is_empty_not_an_error():
    provider = build_web_search("ddgs", transport=_transport(b"<html></html>"))
    assert run(provider.search("q")).results == []


def test_a_ddgs_transport_failure_becomes_websearcherror():
    provider = build_web_search("ddgs", transport=_transport(b"", status=429))
    with pytest.raises(WebSearchError, match=r"ddgs search failed:.*429"):
        run(provider.search("q"))


# =================================================================== serper
def test_serper_translates_its_own_field_names(monkeypatch):
    import visionagent.providers.websearch.serper_client as ws

    async def search(q, num=10):
        return {"organic": [
            {"title": "T", "link": "https://e.com", "snippet": "S"},
        ]}

    monkeypatch.setattr(ws, "serper_search", search)
    out = run(build_web_search("serper").search("q"))
    assert out.results == [WebResult(title="T", url="https://e.com", content="S")]


def test_serper_keeps_its_related_questions(monkeypatch):
    """The capability ddgs lacks; serper must not lose it."""
    import visionagent.providers.websearch.serper_client as ws

    async def search(q, num=10):
        return {
            "organic": [{"title": "T", "link": "u", "snippet": "S"}],
            "peopleAlsoAsk": [{"question": "How wide?"}, {"question": "Why?"}],
        }

    monkeypatch.setattr(ws, "serper_search", search)
    assert run(build_web_search("serper").search("q")).related_questions == [
        "How wide?", "Why?"]


def test_a_serper_failure_becomes_websearcherror(monkeypatch):
    import visionagent.providers.websearch.serper_client as ws

    async def boom(q, num=10):
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(ws, "serper_search", boom)
    with pytest.raises(WebSearchError, match="serper search failed: 401"):
        run(build_web_search("serper").search("q"))


# ==================================================== the re-ranking wrapper
def test_snippets_fall_back_to_provider_order_when_embedding_fails(monkeypatch):
    """Reranking failure must not discard successful provider results."""
    import visionagent.service.executer.tools.web.snippets as pws

    monkeypatch.setattr(pws, "build_web_search", lambda: _StubProvider())
    monkeypatch.setattr(pws, "_rerank", _boom)

    snippets, related = run(pws.store_and_query_snippets("q", top_k=2))
    assert [s["content"] for s in snippets] == ["one", "two"]
    assert related == ["r?"]


def test_no_results_short_circuits_before_reranking(monkeypatch):
    """An empty search must not call the embedding provider."""
    import visionagent.service.executer.tools.web.snippets as pws

    monkeypatch.setattr(pws, "build_web_search", lambda: _StubProvider(empty=True))
    monkeypatch.setattr(pws, "_rerank", _boom)
    assert run(pws.store_and_query_snippets("q")) == ([], [])


def test_snippet_rerank_batches_question_and_results_once_and_keeps_order(monkeypatch):
    import visionagent.service.executer.tools.web.snippets as pws

    calls: list[list[str]] = []
    closed = False

    class FakeEmbedder:
        async def aembed_batch(self, texts):
            calls.append(list(texts))
            return [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]

        async def aclose(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr(pws, "build_embedder", lambda: FakeEmbedder())
    results = [
        WebResult(content="one"),
        WebResult(content="two"),
        WebResult(content="three"),
    ]
    out = run(pws._rerank("q", results, 2))
    assert calls == [["q", "one", "two", "three"]]
    assert [item["content"] for item in out] == ["two", "three"]
    assert closed


def test_snippet_rerank_closes_embedder_when_embedding_fails(monkeypatch):
    import visionagent.service.executer.tools.web.snippets as pws

    closed = False

    class FakeEmbedder:
        async def aembed_batch(self, texts):
            raise RuntimeError("provider failed")

        async def aclose(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr(pws, "build_embedder", lambda: FakeEmbedder())
    with pytest.raises(RuntimeError, match="provider failed"):
        run(pws._rerank("q", [WebResult(content="one")], 1))
    assert closed


def test_snippet_rerank_cancellation_reaches_embedding_and_closes(monkeypatch):
    import visionagent.service.executer.tools.web.snippets as pws

    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()
        closed = asyncio.Event()
        never = asyncio.Event()

        class FakeEmbedder:
            async def aembed_batch(self, texts):
                started.set()
                try:
                    await never.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            async def aclose(self):
                closed.set()

        monkeypatch.setattr(pws, "build_embedder", lambda: FakeEmbedder())
        task = asyncio.create_task(
            pws._rerank("q", [WebResult(content="one")], 1)
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert cancelled.is_set()
        assert closed.is_set()

    run(scenario())


class _StubProvider:
    name = "stub"

    def __init__(self, empty: bool = False) -> None:
        self.empty = empty

    async def search(self, query: str, *, num: int = 10) -> WebSearchResults:
        if self.empty:
            return WebSearchResults()
        return WebSearchResults(
            results=[WebResult(content=c) for c in ("one", "two", "three")],
            related_questions=["r?"],
        )


async def _boom(*a, **k):
    raise RuntimeError("embedder unavailable")


# ============================================================ images / videos
def test_ddgs_images_map_onto_the_wire_field_names(monkeypatch):
    """chat/component/result.tsx reads thumbnailUrl and link; ddgs calls those
    thumbnail and url. The translation happens in the provider."""
    import visionagent.providers.websearch.duckduckgo as ddg

    async def media(self, kind, query, num):
        return [{"title": "T", "image": "i.png", "thumbnail": "t.png",
                 "url": "https://e.com", "source": "e.com"}]

    monkeypatch.setattr(ddg.DuckDuckGoSearch, "_media", media)
    from visionagent.providers.websearch import ImageResult

    assert run(build_web_search("ddgs").images("q")) == [ImageResult(
        title="T", image_url="i.png", thumbnail_url="t.png",
        link="https://e.com", source="e.com")]


def test_a_media_backend_failure_yields_nothing_rather_than_raising():
    """Images are decoration around an answer. DuckDuckGo's video backend
    answers "No results found" for queries that plainly have videos, and that
    must not take the chat response down with it."""
    p = build_web_search("ddgs", transport=_transport(b"", status=503))
    assert run(p.images("q")) == []
    assert run(p.videos("q")) == []


def test_serper_images_and_videos_keep_their_wire_shape(monkeypatch):
    import visionagent.providers.websearch.serper_client as ws

    async def images(q, hl="en", num=5):
        return {"images": [
            {"title": "T", "imageUrl": "i", "thumbnailUrl": "t",
             "link": "l", "source": "s"}
        ]}

    async def videos(q, hl="en", num=5):
        return {"videos": [
            {"title": "V", "link": "vl", "imageUrl": "vi"}
        ]}

    monkeypatch.setattr(ws, "serper_images", images, raising=False)
    monkeypatch.setattr(ws, "serper_videos", videos, raising=False)

    p = build_web_search("serper")
    img = run(p.images("q"))[0]
    assert (img.title, img.image_url, img.thumbnail_url, img.link) == ("T", "i", "t", "l")
    vid = run(p.videos("q"))[0]
    assert (vid.title, vid.link, vid.thumbnail_url) == ("V", "vl", "vi")


def test_both_providers_still_satisfy_the_extended_protocol():
    from visionagent.providers.websearch import WebSearchProvider

    for name in ("ddgs", "serper"):
        p = build_web_search(name)
        assert isinstance(p, WebSearchProvider)
        assert callable(p.images) and callable(p.videos)


# ===================================================== async transport/cancel
def test_duckduckgo_transport_is_async_and_response_closes_on_cancel():
    """A cancelled turn must stop the HTTP body, not leave a thread running."""

    async def scenario():
        started = asyncio.Event()
        never = asyncio.Event()

        class BlockingStream(httpx.AsyncByteStream):
            def __init__(self) -> None:
                self.closed = False

            async def __aiter__(self):
                started.set()
                await never.wait()
                yield b"<html/>"  # pragma: no cover - cancellation wins

            async def aclose(self) -> None:
                self.closed = True

        class Transport(httpx.AsyncBaseTransport):
            def __init__(self) -> None:
                self.stream = BlockingStream()
                self.closed = False

            async def handle_async_request(self, request):
                return httpx.Response(200, stream=self.stream, request=request)

            async def aclose(self) -> None:
                self.closed = True

        transport = Transport()
        provider = build_web_search("ddgs", transport=transport)
        task = asyncio.create_task(provider.search("q"))
        await asyncio.wait_for(started.wait(), timeout=1)

        # Reaching this coroutine while the response is stalled proves the
        # network wait yielded to the event loop.
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert transport.stream.closed
        assert transport.closed

    run(scenario())


def test_serper_http_client_is_async_and_preserves_request_shape(monkeypatch):
    from visionagent.providers.websearch import serper_client

    monkeypatch.setattr(
        serper_client,
        "settings",
        type("Settings", (), {"serper_api_key": "test-key"})(),
    )
    calls: list[httpx.Request] = []
    transport = _transport(
        b'{"organic":[{"title":"T","link":"u","snippet":"S"}]}',
        calls=calls,
    )
    payload = run(
        serper_client.make_request("street design", "en", "/search", 7,
                                   transport=transport)
    )

    assert payload["organic"][0]["snippet"] == "S"
    assert calls[0].headers["X-API-KEY"] == "test-key"
    assert calls[0].url.path == "/search"
    assert calls[0].read().decode().count("street design") == 1
