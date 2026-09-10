"""Embeddings from a hosted, OpenAI-compatible endpoint.

It uses the OpenAI SDK's embeddings API against a configurable base URL.
DashScope is what the defaults and tests cover; another endpoint/model must
implement that compatible embeddings contract.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from openai import AsyncOpenAI, OpenAI

from visionagent.config.settings import settings
from visionagent.providers.embedding.base import EmbeddingError


class OnlineEmbedder:
    """Implements `visionagent.providers.embedding.base.Embedder`."""

    name = "online"

    # DashScope rejects a batch larger than this with
    # "InternalError.Algo.InvalidParameter: batch size is invalid". Splitting
    # here rather than at call sites: a caller should not have to know the
    # provider's limit, and every caller would otherwise have to rediscover it.
    MAX_BATCH = 10

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions
        self._api_key = api_key or settings.dashscope_api_key
        self._base_url = base_url or settings.dashscope_base_url

        # Create each transport on first use; callers may use only the sync or
        # async interface and close resources through their owning adapter.
        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def _get_async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._async_client

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self.MAX_BATCH):
            out.extend(self._embed_one_batch(texts[start:start + self.MAX_BATCH]))
        return out

    def _embed_one_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            completion = self._get_client().embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
                encoding_format="float",
            )
        except Exception as exc:  # noqa: BLE001 -- provider errors are opaque
            raise EmbeddingError(f"{self.name} embedding failed: {exc}") from exc

        return self._validated_vectors(completion.data, expected=len(texts))

    def _validated_vectors(
        self, items: Iterable[Any], *, expected: int
    ) -> list[list[float]]:
        """Validate the indexed provider response and restore input order."""
        by_index: dict[int, list[float]] = {}
        try:
            for item in items:
                index = item.index
                if not isinstance(index, int) or index in by_index:
                    raise EmbeddingError(
                        f"{self.name} embedding returned duplicate or invalid index"
                    )
                vector = [float(value) for value in item.embedding]
                if len(vector) != self.dimensions:
                    raise EmbeddingError(
                        f"{self.name} embedding returned width {len(vector)}, "
                        f"expected {self.dimensions}"
                    )
                by_index[index] = vector
        except EmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001 -- response shape is provider data
            raise EmbeddingError(
                f"{self.name} embedding returned an invalid response"
            ) from exc

        required = set(range(expected))
        if set(by_index) != required:
            raise EmbeddingError(
                f"{self.name} embedding returned indexes {sorted(by_index)}, "
                f"expected {sorted(required)}"
            )
        return [by_index[index] for index in range(expected)]

    async def aembed(self, text: str) -> list[float]:
        return (await self.aembed_batch([text]))[0]

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed through AsyncOpenAI so turn cancellation reaches HTTP."""
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self.MAX_BATCH):
            batch = texts[start:start + self.MAX_BATCH]
            try:
                completion = await self._get_async_client().embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                    encoding_format="float",
                )
            except Exception as exc:  # noqa: BLE001 -- provider errors are opaque
                raise EmbeddingError(f"{self.name} embedding failed: {exc}") from exc
            out.extend(self._validated_vectors(completion.data, expected=len(batch)))
        return out

    async def aclose(self) -> None:
        client = self._async_client
        self._async_client = None
        if client is not None:
            await client.close()
