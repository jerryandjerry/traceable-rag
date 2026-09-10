"""The indexing slot: text in, an index written.

It embeds chunks, produces the analysed fields Elasticsearch's BM25 half needs,
extracts entities and relations with the model, and writes all of it through
database/. Retrieval is not here: the executer's tools read database/ directly,
and a store that both writes and answers queries is how the tools ended up
importing this slot in the first place.

The graph service is a lazy compatibility export.  Importing the document
shaper for an offline contract test must not initialize graph, database, or LLM
configuration.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from visionagent.service.vectorstore.base import ChunkStore, ChunkStoreError
from visionagent.service.vectorstore.factory import build_chunkstore

if TYPE_CHECKING:
    from visionagent.service.vectorstore.graphstore.service import GraphRAGService


def __getattr__(name: str) -> Any:
    if name == "GraphRAGService":
        from visionagent.service.vectorstore.graphstore.service import GraphRAGService

        globals()[name] = GraphRAGService
        return GraphRAGService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ChunkStore", "ChunkStoreError", "GraphRAGService", "build_chunkstore"]
