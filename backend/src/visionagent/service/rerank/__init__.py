"""The rerank slot: score every candidate, sort, cut to top_n.

`rerank_results` is the step and is the same whichever provider answers.
`providers/rerank` holds the scoring call underneath it, which is the only
thing a replacement has to implement.
"""
from __future__ import annotations

import logging

from visionagent.config.settings import settings
from visionagent.models import RetrievedChunk
from visionagent.service.rerank.base import Reranker, RerankError
from visionagent.service.rerank.factory import build_reranker

__all__ = ["RerankError", "Reranker", "build_reranker", "rerank_results"]

logger = logging.getLogger(__name__)


async def rerank_results(
    all_results: list[RetrievedChunk],
    query: str,
    top_n: int | None = None,
    *,
    reranker: Reranker | None = None,
) -> list[RetrievedChunk]:
    """Attach aligned scores, sort candidates, and return the best ``top_n``."""
    top_n = settings.rerank_top_n if top_n is None else top_n
    if not all_results:
        return []

    scores = await (reranker or build_reranker()).score(
        query=query, texts=[c.content for c in all_results]
    )
    rescored = [
        c.model_copy(update={"score": max(0.0, min(1.0, s))})
        for c, s in zip(all_results, scores, strict=True)
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)

    # Apply the relevance floor before the top-n cut.
    selected = [c for c in rescored if c.score >= 0.1][:top_n]
    logger.debug(
        "reranking completed candidates=%d selected=%d",
        len(rescored),
        len(selected),
    )
    return selected
