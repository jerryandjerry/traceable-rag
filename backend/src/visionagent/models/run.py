"""Orchestration state held by ``pipeline`` and hidden from components.

Components take narrow typed arguments so their access and replacement
boundaries remain explicit.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from visionagent.models.answer import Answer
from visionagent.models.chunk import ParsedChunk, RetrievedChunk
from visionagent.models.context import QueryJob, ToolContext
from visionagent.models.query import Evaluation, Intent, Plan, ToolResult


class AgentState(BaseModel):
    """Everything one turn of chat knows: the question, what was decided about
    it, what was retrieved, and what came back.

    The pipeline reads and writes this; the slots never see it. A slot takes
    exactly the arguments it needs and returns exactly what it produces, which
    is what makes it replaceable -- a component handed the whole state can read
    and write anything, and then swappability is a claim nobody can check.
    """

    model_config = ConfigDict(extra="forbid")

    job: QueryJob = Field(frozen=True)
    """The authorized ticket the API issued. Frozen, and the binding is frozen
    too, so nothing downstream can swap in a different identity or a different
    set of allowed tools mid-turn. Everything below it is what this run has
    worked out so far."""

    intent: Intent | None = None
    plan: Plan | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    ranked: list[RetrievedChunk] = Field(default_factory=list)
    evaluation: Evaluation | None = None
    round: int = Field(default=0, ge=0)
    """Retrieval rounds completed. 0 is the initial search, 1 the first
    reflection round, and so on.

    The ceiling is MAX_ROUNDS (default 2, counting the initial search), enforced
    by the pipeline's loop rather than by this bound: the setting is meant to be
    raisable without the contract fighting it."""

    answer: Answer | None = None

    @property
    def context(self) -> ToolContext:
        """The identity a tool may see. Derived, never stored, so it cannot
        drift from the ticket it came from."""
        return self.job.context

    @property
    def question(self) -> str:
        return self.job.question

    @property
    def run_id(self) -> str:
        return self.job.identity.run_id

    @property
    def user_id(self) -> str:
        return self.job.identity.user_id

    @property
    def session_id(self) -> str:
        return self.job.session_id


class IngestRun(BaseModel):
    """One document, from upload through to indexed and graphed."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    file_name: str = Field(min_length=1)

    chunks: list[ParsedChunk] = Field(default_factory=list)
    indexed_count: int = Field(default=0, ge=0)
    entity_count: int = Field(default=0, ge=0)
    relation_count: int = Field(default=0, ge=0)
    error: str | None = None
