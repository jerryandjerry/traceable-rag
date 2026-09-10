"""DuckDuckGo search through a native-async HTTP transport.

The provider id remains ``ddgs`` for configuration compatibility. Direct
``httpx`` requests let asyncio cancellation close the in-flight response.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

import httpx
from lxml import html as lxml_html

from visionagent.providers.websearch._transport import read_response
from visionagent.providers.websearch.base import (
    ImageResult,
    VideoResult,
    WebResult,
    WebSearchError,
    WebSearchResults,
)

logger = logging.getLogger(__name__)

_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
}


def _parse_text_hits(body: bytes) -> list[dict[str, str]]:
    """Parse DuckDuckGo's HTML result shape; deliberately CPU-only."""
    tree = lxml_html.fromstring(body)
    hits: list[dict[str, str]] = []
    for item in tree.xpath("//div[contains(@class, 'body')]"):
        title = " ".join("".join(item.xpath(".//h2//text()")).split())
        hrefs = item.xpath("./a/@href")
        content = " ".join("".join(item.xpath("./a//text()")).split())
        href = str(hrefs[0]) if hrefs else ""
        # Match ddgs' Duckduckgo.post_extract_results filter: these are its
        # ad/redirect records, not organic evidence.
        if href.startswith("https://duckduckgo.com/y.js?"):
            continue
        hits.append({"title": title, "href": href, "body": content})
    return hits


def _extract_vqd(body: bytes) -> str:
    """Extract DuckDuckGo's request token without performing any I/O."""
    for marker, offset, delimiter in (
        (b'vqd="', 5, b'"'),
        (b"vqd=", 4, b"&"),
        (b"vqd='", 5, b"'"),
    ):
        start = body.find(marker)
        if start >= 0:
            value_start = start + offset
            value_end = body.find(delimiter, value_start)
            if value_end >= 0:
                return body[value_start:value_end].decode("utf-8")
    raise ValueError("DuckDuckGo request token was not present")


class DuckDuckGoSearch:
    """Implements ``visionagent.providers.websearch.base.WebSearchProvider``."""

    name = "ddgs"

    def __init__(
        self,
        *,
        region: str = "us-en",
        timeout: float = 20,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.region = region
        self.timeout = timeout
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=_HEADERS,
            timeout=self.timeout,
            follow_redirects=True,
            transport=self._transport,
            trust_env=True,
        )

    async def search(self, query: str, *, num: int = 10) -> WebSearchResults:
        try:
            async with self._client() as client:
                body = await read_response(
                    client,
                    "POST",
                    "https://html.duckduckgo.com/html/",
                    data={"q": query, "b": "", "l": self.region},
                )
            hits = await asyncio.to_thread(_parse_text_hits, body)
        except Exception as exc:  # noqa: BLE001 - transport/parser errors vary
            raise WebSearchError(f"ddgs search failed: {exc}") from exc

        return WebSearchResults(
            results=[
                WebResult(
                    title=hit.get("title") or "",
                    url=hit.get("href") or "",
                    content=content,
                )
                for hit in hits[:num]
                if (content := (hit.get("body") or "").strip())
            ],
            related_questions=[],
        )

    async def images(self, query: str, *, num: int = 5) -> list[ImageResult]:
        hits = await self._media("images", query, num)
        return [
            ImageResult(
                title=hit.get("title") or "",
                image_url=hit.get("image") or "",
                thumbnail_url=hit.get("thumbnail") or "",
                link=hit.get("url") or "",
                source=hit.get("source") or "",
            )
            for hit in hits
        ]

    async def videos(self, query: str, *, num: int = 5) -> list[VideoResult]:
        hits = await self._media("videos", query, num)
        return [
            VideoResult(
                title=hit.get("title") or "",
                link=hit.get("content") or "",
                thumbnail_url=(hit.get("images") or {}).get("medium") or "",
            )
            for hit in hits
        ]

    async def _media(
        self, kind: Literal["images", "videos"], query: str, num: int
    ) -> list[dict[str, Any]]:
        """Best-effort media fetch using the cancellable HTTP client."""
        try:
            async with self._client() as client:
                landing = await read_response(
                    client, "GET", "https://duckduckgo.com/", params={"q": query}
                )
                token = _extract_vqd(landing)
                endpoint = "i.js" if kind == "images" else "v.js"
                params: dict[str, str] = {
                    "l": self.region,
                    "o": "json",
                    "q": query,
                    "vqd": token,
                    "p": "1" if kind == "images" else "-1",
                }
                if kind == "images":
                    params["ct"] = "AT"
                else:
                    params["f"] = ",,,"
                body = await read_response(
                    client,
                    "GET",
                    f"https://duckduckgo.com/{endpoint}",
                    params=params,
                    headers={"Referer": "https://duckduckgo.com/"},
                )
            payload = await asyncio.to_thread(json.loads, body)
            results = payload.get("results") if isinstance(payload, dict) else None
            return list(results or [])[:num]
        except Exception as exc:  # noqa: BLE001 - media is decorative
            logger.info(
                "web media search returned nothing",
                extra={
                    "provider": "duckduckgo",
                    "media_kind": kind,
                    "exception_type": type(exc).__name__,
                },
            )
            return []
