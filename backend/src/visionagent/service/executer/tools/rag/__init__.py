"""Elasticsearch hybrid BM25 + kNN over the tenant's documents."""
from __future__ import annotations

import logging
from typing import Any

from visionagent.database.elasticsearch.retrieval import retrieve_content
from visionagent.models import RetrievedChunk, SourceType, ToolContext, ToolName, ToolResult
from visionagent.service.executer.tools.base import Emitter

logger = logging.getLogger(__name__)


# rag search
async def rag(
    query: str, *, context: ToolContext, emit: Emitter | None = None
) -> ToolResult:
    """Knowledge-base retrieval: Elasticsearch hybrid BM25 + kNN."""
    indexNames = context.user_id
    try:
        # retrieve_content awaits hosted embeddings natively, then isolates the
        # synchronous Elasticsearch client in its own narrow worker bridge.
        rag_results = await retrieve_content(indexNames, query)
    except Exception:  # noqa: BLE001
        logger.exception("RAG retrieval failed")
        return ToolResult(tool_name=ToolName.RAG, error="rag failed")

    return ToolResult(
        tool_name=ToolName.RAG,
        chunks=[_chunk_from_hit(h, SourceType.KNOWLEDGE_BASE) for h in (rag_results or [])],
    )
def _chunk_from_hit(hit: dict[str, Any], source_type: SourceType) -> RetrievedChunk:
    """One Elasticsearch/graph hit as a RetrievedChunk."""
    return RetrievedChunk(
        id=str(hit.get("chunk_id") or hit.get("id") or ""),
        content=hit.get("content_with_weight") or "",
        source_type=source_type,
        score=max(0.0, min(1.0, float(hit.get("similarity") or hit.get("sim") or 0.0))),
        doc_name=hit.get("docnm") or hit.get("docnm_kwd"),
        page_num=int(hit["page_num"]) if hit.get("page_num") is not None else None,
        images=[i for i in (hit.get("ref_images") or []) if isinstance(i, str)],
    )
