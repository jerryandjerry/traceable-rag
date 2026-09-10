"""Elasticsearch chunk store.

The field names here are the persistent index contract. Python model names are
deliberately different -- ``RetrievedChunk.content`` maps to
``content_with_weight`` -- so changing this mapping requires an index migration.

Pinned by tests/unit/test_golden.py::test_es_document_field_names_are_unchanged.
"""
from __future__ import annotations

import datetime
from typing import Any

import xxhash

from visionagent.database.base import ChunkWriter
from visionagent.models import ParsedChunk
from visionagent.providers.embedding import Embedder, build_embedder
from visionagent.service.vectorstore.base import ChunkStoreError, Progress
from visionagent.service.vectorstore.elasticsearch.analyzer import Analyzer, build_analyzer


class ElasticsearchChunkStore:
    """Implements `visionagent.service.vectorstore.base.ChunkStore`."""

    name = "elasticsearch"

    def __init__(
        self,
        *,
        analyzer: Analyzer | None = None,
        embedder: Embedder | None = None,
        connection: Any | None = None,
        store: ChunkWriter | None = None,
    ) -> None:
        # One analyzer, used by index() and search() both. That is what makes
        # index-time and query-time analysis impossible to diverge.
        self._analyzer = analyzer or build_analyzer()
        self._embedder = embedder or build_embedder()
        # The writer, not a connection: this slot shapes documents and hands
        # them over. It has no way to read one back.
        if store is None:
            # Keep the database adapter out of pure document-shaping imports.
            # It loads application settings because its read path needs
            # retrieval tuning; an injected writer does not.
            from visionagent.database.elasticsearch.store import ElasticsearchStore

            store = ElasticsearchStore(connection)
        self._store = store

    def document_id(self, *, doc_name: str, index_name: str) -> str:
        """Return the stable identifier for a document within one tenant.

        Computed rather than asked of the writer: the id is part of the
        document's shape, which is this slot's, and ChunkWriter deliberately
        exposes nothing but writes.
        """
        return xxhash.xxh64((doc_name + index_name).encode("utf-8")).hexdigest()

    def to_document(self, chunk: ParsedChunk, *, index_name: str, doc_name: str) -> dict[str, Any]:
        """ParsedChunk to the exact Elasticsearch document shape.

        Analysis happens here, not in the parser: the analyzed fields exist only
        because ES's BM25 half reads them, so producing them is the store's job.
        """
        content = chunk.content
        # Ingest owns persisted identity because it knows the durable document
        # item and occurrence. Recomputing from content here caused identical
        # passages in two documents to overwrite one another.
        if not chunk.id:
            raise ValueError("persisted chunks require a pipeline-assigned id")
        chunk_id = chunk.id
        now = datetime.datetime.now()
        vector = self._embedder.embed(content)

        return {
            "id": chunk_id,
            "chunk_id": chunk_id,
            "content_with_weight": content,
            "content_ltks": chunk.content_tokens or self._analyzer.analyze(content),
            "content_sm_ltks": chunk.content_tokens_fine
            or self._analyzer.analyze_fine(content),
            "important_kwd": [],
            "important_tks": [],
            "question_kwd": [],
            "question_tks": [],
            "create_time": str(now).replace("T", " ")[:19],
            "create_timestamp_flt": now.timestamp(),
            "page_num": chunk.page_num,
            "top_int": chunk.top_offset,
            "kb_id": index_name,
            "docnm_tks": chunk.doc_name_tokens or self._analyzer.analyze(doc_name),
            "doc_id": self.document_id(doc_name=doc_name, index_name=index_name),
            "docnm": doc_name,
            "ref_images": chunk.ref_images,
            "image": chunk.image or "",
            f"q_{len(vector)}_vec": vector,
        }

    # -------------------------------------------------------------- protocol
    def index(self, *, chunks: list[ParsedChunk], index_name: str, doc_name: str,
              progress: Progress | None = None) -> int:
        if not chunks:
            return 0
        docs = []
        for i, c in enumerate(chunks, start=1):
            docs.append(self.to_document(c, index_name=index_name, doc_name=doc_name))
            if progress is not None:
                progress(i, len(chunks), docs[-1]["id"])
        try:
            return self._store.index(index_name=index_name, documents=docs)
        except Exception as exc:  # noqa: BLE001
            raise ChunkStoreError(f"{self.name} insert failed: {exc}") from exc

    def delete_index(self, *, index_name: str) -> int:
        """Every chunk in one user's index."""
        try:
            return self._store.delete_index(index_name=index_name)
        except Exception as exc:  # noqa: BLE001
            raise ChunkStoreError(f"{self.name} index delete failed: {exc}") from exc

    def delete_document(self, *, doc_name: str, index_name: str) -> int:
        try:
            return self._store.delete_document(
                index_name=index_name, doc_name=doc_name
            )
        except Exception as exc:  # noqa: BLE001
            raise ChunkStoreError(f"{self.name} delete failed: {exc}") from exc
