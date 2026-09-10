"""The reranking-provider slot."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from visionagent.exceptions.rerank import RerankError

__all__ = ["RerankError", "Reranker"]


@runtime_checkable
class Reranker(Protocol):
    """Re-score candidates against the query.

    Providers return one score per input string in the same order.
    ``rerank_results`` owns attachment, sorting, filtering, and truncation so
    every provider follows the same policy.
    """

    name: str

    async def score(self, *, query: str, texts: list[str]) -> list[float]:
        """Relevance per text, aligned to the input order.

        This is an async provider boundary: cancellation reaches the transport
        request instead of abandoning a blocking SDK call in a worker thread.
        Sorting and cutting to top-n is rerank_results() in this package,
        shared by every provider rather than reimplemented by each.
        """
        ...
