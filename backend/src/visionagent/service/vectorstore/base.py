"""The chunk-indexing slot: parsed chunks in, a searchable index written.

Everything Elasticsearch-specific about a *document* lives behind this: the
field names, the `*_ltks` analyzed fields produced by analyzer/, and the
`q_<dim>_vec` dense vector. The connection, the search and the deletes are
`database/elasticsearch`, reached through `ChunkWriter`.

There is no read method: retrieval tools read through ``database/`` adapters,
while this slot defines only index construction and deletion.

The analyzer belongs here rather than in the store because analysis is part of
deciding what a document *is*, not of putting it somewhere.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from visionagent.models import ParsedChunk

Progress = Callable[[int, int, str], None]
"""(done, total, chunk_id) as each chunk is prepared. Embedding is the slow
part of indexing, and the upload indicator reads one tick per chunk."""


class ChunkStoreError(RuntimeError):
    """The store rejected the operation."""


@runtime_checkable
class ChunkStore(Protocol):
    """Turns parsed chunks into stored, searchable documents."""

    name: str

    def index(self, *, chunks: list[ParsedChunk], index_name: str, doc_name: str,
              progress: Progress | None = None) -> int:
        """Write chunks and return how many were accepted.

        `progress` is called once per chunk as it is prepared, before the
        write. Optional: a caller with nothing to report passes nothing.
        """
        ...

    def delete_document(self, *, doc_name: str, index_name: str) -> int:
        """Remove every chunk of one document; returns how many were removed."""
        ...

    def delete_index(self, *, index_name: str) -> int:
        """Remove everything a user owns. Returns how many chunks went.

        On the Protocol rather than reached for through the implementation:
        account deletion needs it, and a half-deleted account leaves documents
        searchable by nobody but still stored, which is worse than not offering
        deletion at all.
        """
        ...
