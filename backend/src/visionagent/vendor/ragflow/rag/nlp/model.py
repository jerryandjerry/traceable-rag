"""Compatibility shim over the embedding/ and rerank/ slots.

This file is **not** upstream RAGFlow -- it was added by this project and only
ever contained a DashScope embedding call and a DashScope rerank call, both of
which are now slots. It survives because genuinely vendored code
(`rag/nlp/search_v2.py`) imports these two names, and rewriting anything beyond
imports inside vendor/ is not allowed.

First-party callers use `visionagent.providers.embedding` and
`visionagent.providers.rerank`
directly. This shim goes away in Step 9, when chunkstore/ owns the search path.
"""
from __future__ import annotations

import logging

import numpy as np

from visionagent.providers.embedding import build_embedder
from visionagent.providers.rerank import DashScopeReranker

_EMBEDDER = None
_RERANKER = None
logger = logging.getLogger(__name__)


def _embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = build_embedder()
    return _EMBEDDER


def _reranker():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = DashScopeReranker()
    return _RERANKER


def rerank_similarity(query, texts):
    """Scores aligned to the order of `texts`, plus a positional placeholder.

    Returns a numpy array because search_v2 does arithmetic on the result.
    """
    if not texts:
        return np.array([]), None
    return np.array(
        _reranker().score_sync(query=query, texts=list(texts)), dtype=float
    ), None


def generate_embedding(text: str, api_key=None, base_url=None, model_name=None,
                       dimensions=None, encoding_format="float"):
    """Returns the vector, or None on failure.

    None rather than raising, because both historical callers test for it.
    """
    from visionagent.providers.embedding import EmbeddingError

    try:
        return _embedder().embed(text)
    except EmbeddingError:
        logger.warning("embedding request failed", exc_info=True)
        return None
