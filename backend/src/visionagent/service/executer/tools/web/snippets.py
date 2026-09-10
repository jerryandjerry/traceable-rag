"""Fetch web snippets, then keep the ones closest to the question.

The search and hosted embedding calls are native async operations. Ranking is
a bounded in-memory NumPy cosine calculation over the small provider result
set; no ephemeral vector database is created for each query.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from visionagent.config.settings import settings
from visionagent.providers.embedding import build_embedder
from visionagent.providers.websearch import WebResult, build_web_search

logger = logging.getLogger(__name__)


async def _rerank(
    question: str,
    results: list[WebResult],
    top_k: int,
) -> list[dict[str, Any]]:
    """Rank a bounded provider result set by embedding cosine similarity."""
    # OfflineEmbedder loads a local model in its constructor. Keep factory
    # creation off-loop; OnlineEmbedder itself is lazy and performs no I/O here.
    embedder = await asyncio.to_thread(build_embedder)
    try:
        vectors = await embedder.aembed_batch(
            [question, *(result.content for result in results)]
        )
    finally:
        try:
            await embedder.aclose()
        except Exception:  # noqa: BLE001 - cleanup cannot discard search hits
            logger.warning("web snippet embedder cleanup failed", exc_info=True)

    if len(vectors) != len(results) + 1:
        raise ValueError("embedding provider returned an incomplete batch")

    query_vector = np.asarray(vectors[0], dtype=np.float32)
    document_vectors = np.asarray(vectors[1:], dtype=np.float32)
    if document_vectors.ndim != 2 or query_vector.ndim != 1:
        raise ValueError("embedding provider returned an invalid vector shape")
    if document_vectors.shape[1] != query_vector.shape[0]:
        raise ValueError("query and result embedding widths differ")

    query_norm = float(np.linalg.norm(query_vector))
    document_norms = np.linalg.norm(document_vectors, axis=1)
    if query_norm == 0:
        raise ValueError("embedding provider returned a zero query vector")
    denominator = document_norms * query_norm
    scores = np.divide(
        document_vectors @ query_vector,
        denominator,
        out=np.full(len(results), -np.inf, dtype=np.float32),
        where=denominator != 0,
    )
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [
        {
            "title": results[int(index)].title,
            "url": results[int(index)].url,
            "content": results[int(index)].content,
        }
        for index in order
    ]


async def store_and_query_snippets(
    question: str, top_k: int | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Search the web and return ``(snippets, related_questions)``."""
    top_k = top_k or settings.web_search_top_k
    found = await build_web_search().search(
        question, num=settings.web_search_results
    )
    if not found.results:
        return [], list(found.related_questions)

    try:
        snippets = await _rerank(question, found.results, top_k)
    except Exception:  # noqa: BLE001 - reranking is an optional refinement
        logger.warning(
            "web snippet re-ranking failed; using provider order",
            exc_info=True,
        )
        snippets = [
            {"title": result.title, "url": result.url, "content": result.content}
            for result in found.results[:top_k]
        ]

    return snippets, list(found.related_questions)
