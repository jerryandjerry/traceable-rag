"""Construct the configured chunk store."""
from __future__ import annotations

import os
from typing import Any

from visionagent.service.vectorstore.base import ChunkStore

_STORES = {"elasticsearch"}


def build_chunkstore(name: str | None = None, **kwargs: Any) -> ChunkStore:
    chosen = (name or os.getenv("CHUNKSTORE") or "elasticsearch").lower()
    if chosen in ("elasticsearch", "es"):
        from visionagent.service.vectorstore.elasticsearch.store import ElasticsearchChunkStore

        return ElasticsearchChunkStore(**kwargs)
    raise ValueError(f"unknown chunk store {chosen!r}; known: {sorted(_STORES)}")
