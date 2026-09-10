"""Construct the configured embedder."""
from __future__ import annotations

import os
from typing import Any

from visionagent.providers.embedding.base import Embedder

_PROVIDERS = {"online", "offline"}


def build_embedder(provider: str | None = None, **kwargs: Any) -> Embedder:
    name = (provider or os.getenv("EMBEDDING_PROVIDER") or "online").lower()
    # Configuration alias for the hosted online adapter.
    if name == "dashscope":
        name = "online"
    if name == "online":
        from visionagent.providers.embedding.online import OnlineEmbedder

        return OnlineEmbedder(**kwargs)
    if name == "offline":
        from visionagent.providers.embedding.offline import OfflineEmbedder

        return OfflineEmbedder(**kwargs)
    raise ValueError(f"unknown embedding provider {name!r}; known: {sorted(_PROVIDERS)}")
