"""Contracts for the query pipeline: intent, plan, tool results, evaluation."""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from visionagent.models.chunk import RetrievedChunk


class ToolName(StrEnum):
    """The tools the planner may schedule.

    Values are the stable plan/wire names used by the planner and the frozen
    tool registry.
    """

    RAG = "RAG"
    GRAPHRAG = "GraphRAG"
    WEB_SEARCH = "web_search"
    LLM = "LLM"


class Scenario(StrEnum):
    """Whether the question needs retrieval at all."""

    CASUAL = "casual"
    KNOWLEDGE = "knowledge"


class Intent(BaseModel):
    """What the user is asking for, before any retrieval happens."""

    model_config = ConfigDict(extra="forbid")

    scenario: Scenario
    intents: list[str] = Field(default_factory=list)
    keywords_high: list[str] = Field(default_factory=list)
    """Entity-level keywords, used to enter the graph."""

    keywords_low: list[str] = Field(default_factory=list)
    """Broader terms, used for chunk-level matching."""


class ToolCall(BaseModel):
    """One scheduled retrieval, before it runs."""

    model_config = ConfigDict(extra="forbid")

    tool_name: ToolName
    query: list[str] = Field(min_length=1)


class Plan(BaseModel):
    """The set of retrievals to run for one round."""

    model_config = ConfigDict(extra="forbid")

    calls: list[ToolCall] = Field(default_factory=list)

    def uses(self, tool: ToolName) -> bool:
        return any(c.tool_name == tool for c in self.calls)


class ToolResult(BaseModel):
    """What one retrieval produced.

    `error` is a field rather than an exception because the pipeline gathers
    tools concurrently: one failing tool marks its step FAILED and the round
    continues with partial results.
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: ToolName
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    related_questions: list[str] = Field(default_factory=list)
    images: list[dict[str, Any]] = Field(default_factory=list)
    """Image hits from a web search, in the wire shape the frontend reads.
    Only the authorized web tool fills these, so media follows the same policy
    as text search."""

    videos: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    timed_out: bool = False


class Evaluation(BaseModel):
    """Whether the gathered context is enough to answer.

    The field names must match both the evaluator output and reflection input;
    this model turns contract drift into a validation error.
    """

    model_config = ConfigDict(extra="forbid")

    sufficient_score: float = Field(ge=0.0, le=1.0)
    reasons: str = ""
    comments: str = ""
    """What is missing, when the context is judged insufficient."""
