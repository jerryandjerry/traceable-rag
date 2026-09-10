"""LLM provider contract for text, streaming, and typed JSON completions.

``complete_json`` validates provider output at the boundary and raises
``LLMError`` for invalid JSON or schema violations.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


class LLMError(RuntimeError):
    """The provider failed, or returned something that is not usable."""


@runtime_checkable
class LLM(Protocol):
    """Async text in, text out.

    Implementations own their transport, credentials, and cancellation cleanup.
    A cancelled caller must be able to stop the underlying network request or
    subprocess; abandoning a blocking call in a worker thread is not enough.
    """

    name: str

    async def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        json_object: bool = False,
    ) -> str:
        """One-shot completion. Returns the assistant text, never None.

        `json_object` asks the provider to constrain output to a JSON object
        where it supports that; providers without the feature ignore it.
        """
        ...

    def stream(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yield `(content, reasoning)` deltas.

        Two channels because the reasoning models return `reasoning_content`
        separately, and the frontend renders it in its own collapsible block.
        Either element may be empty for a given delta.
        """
        ...

    async def complete_json(
        self,
        *,
        prompt: str,
        schema: type[M],
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> M:
        """Completion validated into `schema`.

        Raises `LLMError` if the response cannot be parsed or does not
        validate, rather than returning a partially-populated object.
        """
        ...

    async def aclose(self) -> None:
        """Release provider transports owned by this client."""
        ...
