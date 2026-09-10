"""Construct the configured web search provider."""
from __future__ import annotations

import os
from typing import Any

from visionagent.providers.websearch.base import WebSearchProvider

_PROVIDERS = {"ddgs", "serper"}


def build_web_search(provider: str | None = None, **kwargs: Any) -> WebSearchProvider:
    name = (provider or os.getenv("WEB_SEARCH_PROVIDER") or "ddgs").lower()
    if name in ("ddgs", "duckduckgo"):
        from visionagent.providers.websearch.duckduckgo import DuckDuckGoSearch

        return DuckDuckGoSearch(**kwargs)
    if name == "serper":
        from visionagent.providers.websearch.serper import SerperSearch

        return SerperSearch(**kwargs)
    raise ValueError(f"unknown web search provider {name!r}; known: {sorted(_PROVIDERS)}")
