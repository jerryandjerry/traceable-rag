"""Contracts for the GraphRAG entity/relationship store."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Entity(BaseModel):
    """A node extracted from document chunks."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    entity_type: str = ""
    description: str = ""
    doc_names: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    """An edge between two entities."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    description: str = ""
    keywords: str = ""
    weight: float = 1.0
    doc_names: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class GraphHit(BaseModel):
    """A graph search result, before its chunks are fetched from the store."""

    model_config = ConfigDict(extra="forbid")

    entity: Entity | None = None
    relation: Relation | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
