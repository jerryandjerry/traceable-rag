"""Construct the configured reranker."""
from __future__ import annotations

import os
from typing import Any

from visionagent.service.rerank.base import Reranker

_PROVIDERS = {"dashscope"}


def build_reranker(provider: str | None = None, **kwargs: Any) -> Reranker:
    name = (provider or os.getenv("RERANK_PROVIDER") or "dashscope").lower()
    if name == "dashscope":
        from visionagent.providers.rerank import DashScopeReranker

        return DashScopeReranker(**kwargs)
    raise ValueError(f"unknown rerank provider {name!r}; known: {sorted(_PROVIDERS)}")
