"""DashScope reranking through the SDK's native asynchronous transport."""
from __future__ import annotations

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import aiohttp
from dashscope import AioTextReRank, TextReRank

from visionagent.config.settings import settings
from visionagent.exceptions.rerank import RerankError


class DashScopeReranker:
    """Implements ``visionagent.service.rerank.base.Reranker``.

    Query turns use :meth:`score`, backed by DashScope's aiohttp transport, so
    timeout and task cancellation reach the socket. ``score_sync`` exists only
    for the temporary RAGFlow compatibility shim whose caller is synchronous;
    first-party query orchestration never calls it.
    """

    name = "dashscope"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        session_factory: Callable[[], aiohttp.ClientSession] | None = None,
    ) -> None:
        self._api_key = api_key or settings.dashscope_api_key
        self._model = model or settings.rerank_model
        # Instance seams keep transport tests off the network without mutating
        # process-wide SDK globals.
        self._async_call = AioTextReRank.call
        self._sync_call = TextReRank.call
        self._session_factory = session_factory or self._new_session

    @staticmethod
    def _new_session() -> aiohttp.ClientSession:
        return aiohttp.ClientSession(trust_env=True)

    async def score(self, *, query: str, texts: list[str]) -> list[float]:
        """Return one score per input text using the cancellable SDK call."""
        if not texts:
            return []

        try:
            # The SDK otherwise uses a process-level session pool. Supplying a
            # per-operation session gives this layer deterministic ownership:
            # both its response context and transport close when the turn is
            # cancelled, without requiring lifecycle hooks on the singleton
            # QueryPipeline.
            async with self._session_factory() as session:
                response = await self._async_call(
                    model=self._model,
                    query=query,
                    documents=texts,
                    top_n=len(texts),
                    api_key=self._api_key,
                    session=session,
                    request_timeout=settings.tool_timeout_s,
                )
            return self._aligned_scores(response, count=len(texts))
        except RerankError:
            raise
        except Exception as exc:  # noqa: BLE001 -- provider errors are opaque
            raise RerankError(f"{self.name} rerank failed: {exc}") from exc

    def score_sync(self, *, query: str, texts: list[str]) -> list[float]:
        """Compatibility bridge for the still-synchronous vendored search.

        QueryPipeline does not use this method. Keeping the bridge explicit
        avoids hiding blocking network I/O behind an async-looking contract.
        """
        if not texts:
            return []
        try:
            response = self._sync_call(
                model=self._model,
                query=query,
                documents=texts,
                top_n=len(texts),
                api_key=self._api_key,
            )
            return self._aligned_scores(response, count=len(texts))
        except RerankError:
            raise
        except Exception as exc:  # noqa: BLE001 -- provider errors are opaque
            raise RerankError(f"{self.name} rerank failed: {exc}") from exc

    def _aligned_scores(self, response: Any, *, count: int) -> list[float]:
        if getattr(response, "status_code", None) != HTTPStatus.OK:
            status = getattr(response, "status_code", "unknown")
            raise RerankError(f"{self.name} rerank failed with status {status}")

        output = getattr(response, "output", None)
        results = getattr(output, "results", None) or []
        scores = [0.0] * count
        for result in results:
            index = int(result.index)
            if 0 <= index < count:
                scores[index] = float(result.relevance_score)
        return scores
