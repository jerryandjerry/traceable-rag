"""Native-async Serper HTTP client and response normalization helpers."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from visionagent.config.settings import settings
from visionagent.providers.websearch._transport import read_response

_BASE_URL = "https://google.serper.dev"


async def serper_search(q: str = "architecture", hl: str = "en", num: int = 5) -> Any:
    return await make_request(q, hl, "/search", num)


async def serper_images(q: str = "apple inc", hl: str = "en", num: int = 5) -> Any:
    return await make_request(q, hl, "/images", num)


async def serper_videos(q: str = "apple inc", hl: str = "en", num: int = 5) -> Any:
    return await make_request(q, hl, "/videos", num)


async def make_request(
    q: str,
    hl: str,
    endpoint: str,
    num: int = 10,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Any:
    """POST one Serper request; cancellation closes its streamed response."""
    api_key = settings.serper_api_key
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is not configured")

    async with httpx.AsyncClient(
        base_url=_BASE_URL,
        timeout=20,
        transport=transport,
        trust_env=True,
    ) as client:
        body = await read_response(
            client,
            "POST",
            endpoint,
            json={"q": q, "hl": hl, "num": num},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        )
    return await asyncio.to_thread(json.loads, body)


def process_search_results(search_results: Any) -> tuple[list[dict[str, str]], list[str]]:
    """Normalize Serper organic hits and related questions."""
    snippets: list[dict[str, str]] = []
    questions: list[str] = []

    if isinstance(search_results, dict):
        for result in search_results.get("organic") or []:
            if not isinstance(result, dict):
                continue
            snippets.append(
                {
                    "title": str(result.get("title") or ""),
                    "url": str(result.get("link") or ""),
                    "content": str(result.get("snippet") or ""),
                }
            )

        for question_data in search_results.get("peopleAlsoAsk") or []:
            if isinstance(question_data, dict) and question_data.get("question"):
                questions.append(str(question_data["question"]))

    return snippets, questions
