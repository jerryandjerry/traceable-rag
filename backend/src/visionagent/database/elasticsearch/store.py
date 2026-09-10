"""The Elasticsearch client, and the reads and deletes that go through it.

Implements `database.base.ChunkReader` and `ChunkWriter` structurally. What a
stored document looks like is the vectorstore slot's decision; getting one in
or out is this.
"""
from __future__ import annotations

from typing import Any

import xxhash

from visionagent.config.settings import settings
from visionagent.exceptions.database import DatabaseError
from visionagent.models import RetrievedChunk, SourceType


class ElasticsearchStore:
    """One connection, opened lazily and shared."""

    name = "elasticsearch"

    def __init__(self, connection: Any | None = None) -> None:
        self._conn = connection
        self._dealer: Any | None = None

    @property
    def conn(self) -> Any:
        if self._conn is None:
            from visionagent.vendor.ragflow.rag.utils.es_conn import ESConnection

            self._conn = ESConnection()
        return self._conn

    @property
    def dealer(self) -> Any:
        if self._dealer is None:
            from visionagent.vendor.ragflow.rag.nlp.search_v2 import Dealer

            self._dealer = Dealer(dataStore=self.conn)
        return self._dealer

    def document_id(self, *, doc_name: str, index_name: str) -> str:
        """Stable per (document, tenant)."""
        return xxhash.xxh64((doc_name + index_name).encode("utf-8")).hexdigest()

    # ---------------------------------------------------------------- reading
    def search(self, *, index_name: str, query: str,
               top_k: int | None = None) -> list[RetrievedChunk]:
        """Hybrid BM25 + kNN, through the vendored engine.

        The engine embeds the query itself, which is why this layer reaches a
        provider at all.
        """
        try:
            results = self.dealer.retrieval(
                question=query,
                embd_mdl=None,
                tenant_ids=index_name,
                kb_ids=None,
                vector_similarity_weight=settings.vector_similarity_weight,
                page=1,
                page_size=top_k or settings.retrieval_top_k,
            )
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"{self.name} search failed: {exc}") from exc

        if not results:
            return []
        return [self.to_chunk(hit) for hit in results.get("chunks", [])]

    def get_by_ids(self, *, index_name: str, ids: list[str]) -> list[dict[str, Any]]:
        from visionagent.database.elasticsearch.chunks import retrieve_chunks_from_es

        return retrieve_chunks_from_es(ids, index_name)

    def to_chunk(self, hit: dict[str, Any]) -> RetrievedChunk:
        """Map one hit to the public retrieval fields, omitting index-only data."""
        return RetrievedChunk(
            id=str(hit.get("chunk_id") or hit.get("id") or ""),
            content=hit.get("content_with_weight") or "",
            source_type=SourceType.KNOWLEDGE_BASE,
            score=max(0.0, min(1.0, float(hit.get("similarity") or 0.0))),
            doc_name=hit.get("docnm"),
            page_num=int(hit["page_num"]) if hit.get("page_num") is not None else None,
            images=list(hit.get("ref_images") or []),
        )

    # ---------------------------------------------------------------- writing
    def index(self, *, index_name: str, documents: list[dict[str, Any]]) -> int:
        """Write documents already shaped by the vectorstore slot."""
        if not documents:
            return 0
        try:
            failures = self.conn.insert(documents=documents, indexName=index_name)
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"{self.name} insert failed: {exc}") from exc
        return len(documents) - len(failures or [])

    def _exists(self, index_name: str) -> bool:
        try:
            return bool(self.conn.es.indices.exists(index=index_name))
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"{self.name} unreachable: {exc}") from exc

    def delete_index(self, *, index_name: str) -> int:
        """Every chunk in one tenant's index.

        A tenant with no index yet has nothing to delete, and that is 0, not
        an error: account deletion must be safe to run twice.
        """
        if not self._exists(index_name):
            return 0
        try:
            return int(self.conn.delete({"kb_id": index_name}, index_name) or 0)
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"{self.name} index delete failed: {exc}") from exc

    def delete_document(self, *, index_name: str, doc_name: str) -> int:
        if not self._exists(index_name):
            return 0
        doc_id = self.document_id(doc_name=doc_name, index_name=index_name)
        try:
            return int(self.conn.delete({"doc_id": doc_id}, index_name) or 0)
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"{self.name} delete failed: {exc}") from exc
