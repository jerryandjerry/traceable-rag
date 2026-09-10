"""Serper (Google), via SERPER_API_KEY."""
from __future__ import annotations

import logging
from typing import Any

from visionagent.providers.websearch.base import (
    ImageResult,
    VideoResult,
    WebResult,
    WebSearchError,
    WebSearchResults,
)

logger = logging.getLogger(__name__)


class SerperSearch:
    """Implements `visionagent.providers.websearch.base.WebSearchProvider`."""

    name = "serper"

    async def search(self, query: str, *, num: int = 10) -> WebSearchResults:
        from visionagent.providers.websearch.serper_client import (
            process_search_results,
            serper_search,
        )

        try:
            raw = await serper_search(query, num=num)
        except Exception as exc:  # noqa: BLE001 - requests raises several types
            raise WebSearchError(f"serper search failed: {exc}") from exc

        snippets, related = process_search_results(raw)
        return WebSearchResults(
            results=[
                WebResult(
                    title=s.get("title") or "",
                    url=s.get("url") or "",
                    content=content,
                )
                for s in snippets or []
                if (content := (s.get("content") or "").strip())
            ],
            related_questions=list(related or []),
        )

    async def images(self, query: str, *, num: int = 5) -> list[ImageResult]:
        raw = await self._media("serper_images", query, num)
        return [
            ImageResult(
                title=i.get("title") or "",
                image_url=i.get("imageUrl") or "",
                thumbnail_url=i.get("thumbnailUrl") or "",
                link=i.get("link") or "",
                source=i.get("source") or "",
            )
            for i in (raw.get("images") or [])
        ]

    async def videos(self, query: str, *, num: int = 5) -> list[VideoResult]:
        raw = await self._media("serper_videos", query, num)
        return [
            VideoResult(
                title=v.get("title") or "",
                link=v.get("link") or "",
                thumbnail_url=v.get("imageUrl") or "",
            )
            for v in (raw.get("videos") or [])
        ]

    async def _media(self, fn_name: str, query: str, num: int) -> dict[str, Any]:
        """See DuckDuckGoSearch._media: decoration must not fail the answer."""
        try:
            import visionagent.providers.websearch.serper_client as ws

            result = await getattr(ws, fn_name)(q=query, hl="en", num=num)
            return dict(result or {})
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "web media search returned nothing",
                extra={
                    "provider": "serper",
                    "media_kind": fn_name,
                    "exception_type": type(exc).__name__,
                },
            )
            return {}
