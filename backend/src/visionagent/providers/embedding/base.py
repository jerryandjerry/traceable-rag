"""The embedding-provider slot."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


class EmbeddingError(RuntimeError):
    """The provider failed, or returned an unusable vector."""


@runtime_checkable
class Embedder(Protocol):
    """Text to vector.

    `dimensions` is on the interface because it is not a private detail: the
    Elasticsearch mapping keys off the vector's length (`q_<dim>_vec`), so a
    replacement with a different width writes to a different field and silently
    stops matching existing documents.
    """

    name: str
    dimensions: int

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    async def aembed(self, text: str) -> list[float]:
        """Async query/runtime path; hosted providers use native async I/O."""
        ...

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Async batched query/runtime path, preserving input order."""
        ...

    async def aclose(self) -> None:
        """Release async transport resources owned by this instance."""
        ...
