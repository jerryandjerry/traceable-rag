"""Contracts for what the pipeline streams back."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from visionagent.models.chunk import Citation, RetrievedChunk


class AnswerChunk(BaseModel):
    """One streamed fragment of the answer.

    `think` carries reasoning-model output, which the frontend renders in a
    separate collapsible block from `content`.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    think: str = ""


class Answer(BaseModel):
    """A completed answer and everything shown alongside it."""

    model_config = ConfigDict(extra="forbid")

    content: str
    think: str = ""
    citations: list[Citation] = Field(default_factory=list)
    references: list[RetrievedChunk] = Field(default_factory=list)
    related_questions: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    videos: list[str] = Field(default_factory=list)
