

import asyncio
import logging
from typing import Any

from visionagent.config.settings import settings
from visionagent.providers.embedding import build_embedder
from visionagent.vendor.ragflow.rag.nlp.search_v2 import Dealer
from visionagent.vendor.ragflow.rag.utils.es_conn import ESConnection

logger = logging.getLogger(__name__)

es_connection = ESConnection()

# ESConnection implements DocStoreConnection structurally, but the vendored
# base is untyped so mypy cannot see the relationship.
dealer = Dealer(dataStore=es_connection)  # type: ignore[arg-type]

async def retrieve_content(indexNames: str, question: str) -> Any:
    """Embed natively, then bridge the synchronous Elasticsearch client.

    Hosted embedding HTTP must remain cancellable with the request, so it is
    awaited before entering the worker. The Elasticsearch dependency exposes
    only a synchronous client; that bounded remainder is the intentional
    thread bridge.
    """
    # OfflineEmbedder loads a local model in its constructor. Factory creation
    # therefore belongs off-loop even though OnlineEmbedder is lazy and cheap.
    embedder = await asyncio.to_thread(build_embedder)
    try:
        query_vector = await embedder.aembed(question)
    finally:
        await embedder.aclose()

    return await asyncio.to_thread(
        _retrieve_content_sync,
        indexNames,
        question,
        query_vector,
    )


def _retrieve_content_sync(
    indexNames: str,
    question: str,
    query_vector: list[float],
) -> Any:
    """Synchronous Elasticsearch search and local result normalization."""

    # The vendored retrieval result wraps its chunks under ``results["chunks"]``.
    results = dealer.retrieval(question = question,
                               embd_mdl = None,
                               tenant_ids = indexNames,
                               kb_ids = None,
                               vector_similarity_weight=settings.vector_similarity_weight,
                               page = 1,
                               page_size = settings.retrieval_top_k,
                               query_vector=query_vector,
    )
    if not results:
        return None
    else:
        logger.debug("RAG retrieval returned chunks=%d", len(results["chunks"]))
    
    extracted_data = []
    for _i, chunk in enumerate(results['chunks'], start=1):
        stored_images = chunk.get("ref_images", [])
        chunk["ref_images"] = (
            [value for value in stored_images if isinstance(value, str)]
            if isinstance(stored_images, list)
            else []
        )
        filter_key = ['content_ltks', 'content_sm_ltks', 'image', 'q_1024_vec']
        for key in filter_key:
            chunk.pop(key, None) # None prevents KeyError if key is missing
        
        extracted_data.append(chunk)

    if extracted_data is not None:
        logger.debug("RAG retrieval normalized chunks=%d", len(extracted_data))
    return extracted_data
