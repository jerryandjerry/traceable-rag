"""Contracts for the two things that move through retrieval: parsed and found.

`ParsedChunk` is what any `DocumentParser` must return. Parser adapters validate
that representation once; the ingest pipeline and indexing slots then carry
this model without converting it back to a dict.

`RetrievedChunk` is what every retrieval tool returns. Source-specific wire and
storage dictionaries are normalized at their owning boundaries.

The Elasticsearch boundary maps ``content`` to ``content_with_weight`` and the
other typed fields to their persisted names. The golden suite pins that storage
contract.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SourceType(StrEnum):
    """Where a retrieved chunk came from; values are part of the API contract."""

    KNOWLEDGE_BASE = "knowledge_base"
    WEB_SEARCH = "web_search"
    CURRENT_CONTEXT = "current_context"


class ParsedChunk(BaseModel):
    """One chunk produced by a document parser, before indexing."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    """Stable identity assigned by the ingest pipeline before persistence.

    It is a length-prefixed SHA-256 digest of tenant, normalized document
    name, durable part identity, parser ordinal, and content. Parsers cannot
    assign it because they do not own tenant or durable-item identity.
    """

    content: str = Field(min_length=1)
    """The chunk text. Stored as `content_with_weight`."""

    content_tokens: str = ""
    """Analyzer output. Stored as `content_ltks`, searched by BM25."""

    content_tokens_fine: str = ""
    """Fine-grained analyzer output. Stored as `content_sm_ltks`."""

    doc_name_tokens: str = ""
    """Analyzed document name. Stored as `docnm_tks`."""

    page_nums: list[int] = Field(default_factory=list)
    """Every page this chunk spans; DeepDoc's ``page_num_int`` is a list."""

    top_offsets: list[int] = Field(default_factory=list)
    """Vertical offsets, one per page. deepdoc's `top_int`."""

    @property
    def page_num(self) -> int:
        """First page, which is what the Elasticsearch document stores."""
        return self.page_nums[0] if self.page_nums else 0

    @property
    def top_offset(self) -> int:
        return self.top_offsets[0] if self.top_offsets else 0

    ref_images: list[str] = Field(default_factory=list)
    """Base64 page images referenced by this chunk."""

    image: str | None = None
    """Base64 image of the chunk itself, when the parser produced one."""


class Citation(BaseModel):
    """A reference the answer points at, as `[doc][cite_XXX]` / `[web][cite_XXX]`."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    source_type: SourceType
    doc_name: str | None = None
    page_num: int | None = Field(default=None, ge=0)
    url: str | None = None


class RetrievedChunk(BaseModel):
    """One chunk returned by a retrieval tool, before reranking."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_type: SourceType

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    """Relevance in [0, 1], set by retrieval and replaced by reranking."""

    doc_name: str | None = None
    page_num: int | None = Field(default=None, ge=0)
    url: str | None = None
    """Set for web results, absent for knowledge-base results."""

    images: list[str] = Field(default_factory=list)
