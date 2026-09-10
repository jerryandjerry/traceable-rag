"""The web-search tool.

The websearch slot selects the provider; this tool consumes its shared contract.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from visionagent.models import RetrievedChunk, SourceType, ToolContext, ToolName, ToolResult
from visionagent.providers.websearch import build_web_search
from visionagent.service.executer.tools.base import Emitter
from visionagent.service.executer.tools.web.snippets import store_and_query_snippets

logger = logging.getLogger(__name__)


def _web_chunk_id(item: dict[str, Any]) -> str:
    """A deterministic identity for cross-process result de-duplication."""
    identity = item.get("url") or "\0".join(
        (item.get("title") or "", item.get("content") or "")
    )
    return f"web_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"


async def _media(query: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Image and video hits for the query, in the shape the chat page reads.

    Fetching media inside the web tool keeps it behind the turn's web-search
    policy. It is best-effort: a media failure must not lose text results.
    """
    try:
        provider = build_web_search()
        image_results, video_results = await asyncio.gather(
            provider.images(query),
            provider.videos(query),
        )
        images = [
            {"title": i.title, "imageUrl": i.image_url, "thumbnailUrl": i.thumbnail_url,
             "link": i.link, "source": i.source}
            for i in image_results
        ]
        videos = [
            {"title": v.title, "link": v.link, "imageUrl": v.thumbnail_url}
            for v in video_results
        ]
        return images, videos
    except Exception:  # noqa: BLE001
        logger.warning("web media lookup failed", exc_info=True)
        return [], []


# web serach
async def web_search_answer(
    query: str, *, context: ToolContext, emit: Emitter | None = None
) -> ToolResult:
    """Web search, then a per-query snippet re-rank. See websearch/ for the
    provider; this function does not know which engine ran. The uniform context
    is accepted for tracing and authorization consistency but identity fields
    are never sent to the public search provider."""
    try:
        (web_results, related_questions), (images, videos) = await asyncio.gather(
            store_and_query_snippets(query),
            _media(query),
        )
    except Exception:  # noqa: BLE001
        logger.exception("web search failed")
        return ToolResult(tool_name=ToolName.WEB_SEARCH, error="web search failed")

    return ToolResult(
        tool_name=ToolName.WEB_SEARCH,
        images=images,
        videos=videos,
        chunks=[
            RetrievedChunk(
                # Stable across processes: the same URL found twice
                # de-duplicates, unlike Python's randomized hash().
                id=_web_chunk_id(item),
                content=content,
                source_type=SourceType.WEB_SEARCH,
                score=1.0,
                doc_name=item.get("title"),
                url=item.get("url"),
            )
            for item in web_results or []
            if (content := (item.get("content") or "").strip())
        ],
        related_questions=list(related_questions or []),
    )
