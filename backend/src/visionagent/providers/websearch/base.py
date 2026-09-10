"""The web search slot.

The web tool consumes these provider-neutral results and returns a
``ToolResult`` without exposing DuckDuckGo or Serper response shapes.

Providers differ in what they can offer. Serper returns "people also ask" as
related questions; DuckDuckGo has no equivalent and returns none. That is a
missing capability, not an error, so `related_questions` is allowed to be empty
rather than each caller special-casing the provider.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class WebSearchError(RuntimeError):
    """The search itself failed. A provider returning nothing is not an error."""


class WebResult(BaseModel):
    """One hit, normalised away from any provider's field names.

    Serper calls these `title`/`link`/`snippet`, DuckDuckGo calls them
    `title`/`href`/`body`. Neither name reaches the pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = ""
    url: str = ""
    content: str = Field(min_length=1)


class WebSearchResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[WebResult] = Field(default_factory=list)
    related_questions: list[str] = Field(default_factory=list)
    """Empty from providers that do not offer them."""


class ImageResult(BaseModel):
    """One image hit with full image, preview, and source-page URLs."""

    model_config = ConfigDict(extra="forbid")

    title: str = ""
    image_url: str = ""
    thumbnail_url: str = ""
    link: str = ""
    source: str = ""


class VideoResult(BaseModel):
    """One video hit; ``thumbnail_url`` is mapped to the tool's image preview."""

    model_config = ConfigDict(extra="forbid")

    title: str = ""
    link: str = ""
    thumbnail_url: str = ""


@runtime_checkable
class WebSearchProvider(Protocol):
    """Whatever actually performs the search.

    Network operations are async so request cancellation reaches the provider
    transport. Implementations may offload bounded parsing work, but they must
    not hide blocking HTTP calls in worker threads.
    """

    name: str

    async def search(self, query: str, *, num: int = 10) -> WebSearchResults:
        """Return normalized text results."""
        ...

    async def images(self, query: str, *, num: int = 5) -> list[ImageResult]:
        """Empty when the provider offers no image search."""
        ...

    async def videos(self, query: str, *, num: int = 5) -> list[VideoResult]:
        """Empty when the provider offers no video search."""
        ...
