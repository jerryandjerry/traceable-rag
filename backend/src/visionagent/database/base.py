"""Storage contracts split by read/write capability.

Separate protocols enforce least privilege, and explicit tenant arguments keep
request identity out of ambient process state.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from visionagent.models import ParsedChunk, RetrievedChunk


@runtime_checkable
class ChunkReader(Protocol):
    """Search and fetch stored chunks. No method here mutates anything."""

    def search(self, *, index_name: str, query: str,
               top_k: int = 5) -> list[RetrievedChunk]: ...

    def get_by_ids(self, *, index_name: str,
                   ids: list[str]) -> list[dict[str, Any]]: ...


@runtime_checkable
class ChunkWriter(Protocol):
    """Write and remove stored chunks."""

    def index(self, *, index_name: str, documents: list[dict[str, Any]]) -> int: ...

    def delete_document(self, *, index_name: str, doc_name: str) -> int: ...

    def delete_index(self, *, index_name: str) -> int:
        """Everything one tenant owns. Account deletion needs it, and a
        half-deleted account leaves documents searchable by nobody but still
        stored."""
        ...


@runtime_checkable
class GraphReader(Protocol):
    """Query one tenant's graph. Scoped by construction, not by argument: a
    repository is opened for a user and cannot be asked about another."""

    async def query_entities(self, text: str, top_k: int) -> list[dict[str, Any]]: ...

    async def query_relations(self, text: str, top_k: int) -> list[dict[str, Any]]: ...

    async def get_nodes(self, names: list[str]) -> dict[str, Any]: ...

    async def get_edges(self, pairs: list[dict[str, str]]) -> dict[Any, Any]: ...


@runtime_checkable
class GraphWriter(Protocol):
    """Persist one tenant's graph."""

    def save(self) -> None:
        """All three files, each replaced atomically."""
        ...


__all__ = [
    "ChunkReader", "ChunkWriter", "GraphReader", "GraphWriter", "ParsedChunk",
]
