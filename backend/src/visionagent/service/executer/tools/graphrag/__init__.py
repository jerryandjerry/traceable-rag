"""GraphRAG retrieval for Traceable RAG.

Entity and relationship matches follow LightRAG's hybrid approach; their source
IDs are resolved to tenant-owned Elasticsearch chunks for the shared tool result.
"""

import asyncio
import logging
from typing import Any

from visionagent.database.elasticsearch import retrieve_chunks_from_es
from visionagent.database.graph import GRAPH_FIELD_SEP, GraphRepository
from visionagent.models import (
    RetrievedChunk,
    SourceType,
    ToolContext,
    ToolName,
    ToolResult,
)
from visionagent.service.executer.tools.graphrag.config import config
from visionagent.utils.keyword_extraction import extract_keywords_advanced

logger = logging.getLogger(__name__)


async def graphrag(
    query: str, *, context: ToolContext, emit: object | None = None
) -> ToolResult:
    """Resolve graph matches to tenant-owned text chunks."""
    logger.debug("GraphRAG tool invoked")

    user_id = context.user_id

    if not query or not query.strip():
        logger.debug("GraphRAG received an empty query")
        return ToolResult(tool_name=ToolName.GRAPHRAG)

    graph: GraphRepository | None = None
    try:
        # This path opens the tenant's graph read-only. Construction loads the
        # vector files and GraphML, so keep it off the event loop.
        graph = await asyncio.to_thread(GraphRepository, user_id)

        hl_keywords, ll_keywords = extract_keywords_advanced(query)
        logger.debug(
            "GraphRAG extracted keywords high_level=%d low_level=%d",
            len(hl_keywords),
            len(ll_keywords),
        )

        if not hl_keywords and not ll_keywords:
            logger.debug("GraphRAG extracted no keywords")
            return ToolResult(tool_name=ToolName.GRAPHRAG)

        all_graph_results = []

        if ll_keywords:
            entity_results = await search_entities(ll_keywords, graph)
            all_graph_results.extend(entity_results)
            logger.debug("GraphRAG entity matches count=%d", len(entity_results))

        if hl_keywords:
            relation_results = await search_relationships(hl_keywords, graph)
            all_graph_results.extend(relation_results)
            logger.debug("GraphRAG relationship matches count=%d", len(relation_results))

        chunk_ids = extract_chunk_ids_from_graph_results(all_graph_results)
        logger.debug("GraphRAG extracted chunk identifiers count=%d", len(chunk_ids))

        es_chunks = []
        extracted_data = []
        if chunk_ids:
            # Synchronous Elasticsearch client: same reason as above.
            es_chunks = await asyncio.to_thread(
                retrieve_chunks_from_es, chunk_ids, user_id
            )

            # Elasticsearch already returns base64 strings, which are the
            # RetrievedChunk contract. Keep them encoded through the typed
            # boundary so answer serialization can forward page previews.
            for chunk in es_chunks:
                filter_key = ['content_ltks', 'content_sm_ltks', 'image', 'q_1024_vec']
                for key in filter_key:
                    chunk.pop(key, None) # None prevents KeyError if key is missing
                
                extracted_data.append(chunk)
        
        return ToolResult(
        tool_name=ToolName.GRAPHRAG,
        chunks=[
            RetrievedChunk(
                id=str(c.get("chunk_id") or c.get("id") or ""),
                content=c.get("content_with_weight") or "",
                source_type=SourceType.KNOWLEDGE_BASE,
                score=max(0.0, min(1.0, float(c.get("similarity") or c.get("sim") or 0.0))),
                doc_name=c.get("docnm") or c.get("docnm_kwd"),
                images=[i for i in (c.get("ref_images") or []) if isinstance(i, str)],
            )
            for c in extracted_data
            if (c.get("content_with_weight") or "").strip()
        ],
    ) 
        
    except Exception:
        # Reported, not swallowed: an empty result with no error reads as
        # "searched and found nothing", so a broken graph looked like an empty
        # one and the step showed as succeeded.
        logger.exception("GraphRAG tool failed")
        return ToolResult(tool_name=ToolName.GRAPHRAG, error="graphrag failed")
    finally:
        if graph is not None:
            try:
                await graph.aclose()
            except Exception:  # noqa: BLE001 -- cleanup must not mask the result
                logger.exception("GraphRAG repository close failed")


async def search_entities(
    keywords: list[str], graph: GraphRepository
) -> list[dict[str, Any]]:
    """Return entity matches for detail-level keywords."""
    try:
        query_text = ", ".join(keywords)
        entity_results = await graph.query_entities(
            query_text, top_k=config.ENTITY_TOP_K
        )
        
        if not entity_results:
            return []
        
        entity_names = [r["entity_name"] for r in entity_results]
        nodes_dict = await graph.get_nodes(entity_names)

        results = []
        for i, entity_name in enumerate(entity_names):
            node_data = nodes_dict.get(entity_name)
            if node_data:
                results.append({
                    "type": "entity",
                    "entity_name": entity_name,
                    "entity_type": node_data.get("entity_type", ""),
                    "description": node_data.get("description", ""),
                    "source_id": node_data.get("source_id", ""),
                    "docnm": node_data.get("docnm", ""),
                    "score": entity_results[i].get("score", 0.0)
                })
        
        return results
        
    except Exception:
        logger.exception("GraphRAG entity search failed")
        raise


async def search_relationships(
    keywords: list[str], graph: GraphRepository
) -> list[dict[str, Any]]:
    """Return relationship matches for concept-level keywords."""
    try:
        query_text = ", ".join(keywords)
        relation_results = await graph.query_relations(
            query_text, top_k=config.RELATIONSHIP_TOP_K
        )
        
        if not relation_results:
            return []
        
        edge_pairs = [{"src": r["src_id"], "tgt": r["tgt_id"]} for r in relation_results]
        edge_data_dict = await graph.get_edges(edge_pairs)

        results = []
        for _i, rel_result in enumerate(relation_results):
            edge_key = (rel_result["src_id"], rel_result["tgt_id"])
            edge_data = edge_data_dict.get(edge_key, {})
            
            results.append({
                "type": "relationship",
                "source_entity": rel_result["src_id"],
                "target_entity": rel_result["tgt_id"],
                "description": edge_data.get("description", ""),
                "keywords": edge_data.get("keywords", ""),
                "weight": edge_data.get("weight", 1.0),
                "source_id": edge_data.get("source_id", ""),
                "docnm": edge_data.get("docnm", ""),
                "score": rel_result.get("score", 0.0)
            })
        
        return results
        
    except Exception:
        logger.exception("GraphRAG relationship search failed")
        raise


def extract_chunk_ids_from_graph_results(results: list[dict[str, Any]]) -> list[str]:
    """Return unique source chunk IDs in graph-rank order.

    First-seen order is load-bearing because the downstream reranker applies a
    top-n cut; set-based deduplication would make that subset nondeterministic.
    """
    seen: dict[str, None] = {}

    for result in results:
        source_id = result.get('source_id', '')
        if source_id:
            for chunk_id in source_id.split(GRAPH_FIELD_SEP):
                chunk_id = chunk_id.strip()
                if chunk_id:
                    seen.setdefault(chunk_id)

    unique_chunk_ids = list(seen)
    logger.debug("GraphRAG de-duplicated chunk identifiers count=%d", len(unique_chunk_ids))
    return unique_chunk_ids
