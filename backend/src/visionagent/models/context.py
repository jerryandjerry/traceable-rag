"""Immutable execution scope passed explicitly across trust boundaries.

Identity is data, not ambient process state.  The authenticated route creates
this scope only after session ownership has been verified; the query pipeline
then passes the same value through the registry into every retrieval tool.

Keeping the scope explicit makes tenant selection visible in function
signatures, testable under concurrency, and impossible to overwrite from an
unrelated request sharing the same event loop or worker process.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from visionagent.models.query import ToolName


class ToolContext(BaseModel):
    """Security and tracing scope for one tool execution.

    ``frozen=True`` prevents a tool from changing the identity observed by a
    sibling tool.  ``extra='forbid'`` rejects misspelled or stale identity
    fields at the boundary rather than silently dropping them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class WebSearchMode(StrEnum):
    """What the turn is allowed and asked to do about web search.

    A boolean could not say why the answer was no. "The user did not ask" and
    "policy forbids it" both arrived as False, so the planner treated a denial
    as a preference and nothing could record that a request had been refused.
    """

    AUTO = "auto"
    """No instruction: intent and the planner decide."""

    FORCE = "force"
    """The user asked for it. Schedule web search this turn."""

    DISABLED = "disabled"
    """Policy refuses it. Never schedule it, and strip it from any plan."""


class AuthorizationScope(BaseModel):
    """What the server decided this caller may do, before the turn starts.

    Resolved from server policy, never from the request. A client may ask for
    web search; whether it is allowed is decided here and nowhere else.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_tools: frozenset[ToolName] = Field(default_factory=frozenset)
    can_upload: bool = True


class TurnOptions(BaseModel):
    """The switches on one turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    web_search: WebSearchMode = WebSearchMode.AUTO
    deep_research: bool = False


class JobIdentity(BaseModel):
    """Who the work is for, and which run it is.

    Both are server-established: run_id is generated, user_id comes from a
    verified token. Neither is ever taken from the request body.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)


class QueryJob(BaseModel):
    """One authorized question. The API issues it; the pipeline executes it.

    Frozen, and it holds no credential: not the bearer token, not the decoded
    claims, not the password. It outlives the request handler -- the pipeline
    keeps it for the whole turn and it reaches the logs -- so anything in here
    that could re-authenticate would be a credential with a long life and no
    owner.

    `requested` and `effective` are both kept. Collapsing them loses the
    difference between "the user did not ask for web search" and "the user
    asked and policy refused", which is exactly what an audit needs to know.
    The pipeline reads `effective` and never `requested`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: JobIdentity
    authorization: AuthorizationScope
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    requested: TurnOptions = TurnOptions()
    effective: TurnOptions = TurnOptions()

    @property
    def context(self) -> ToolContext:
        """The identity a tool may see: three strings, and nothing else."""
        return ToolContext(
            run_id=self.identity.run_id,
            user_id=self.identity.user_id,
            session_id=self.session_id,
        )


class IngestJob(BaseModel):
    """One authorized upload. Separate from QueryJob on purpose.

    An upload has no question and no web-search option, and the retrieval tool
    names in AuthorizationScope do not describe permission to index. Growing
    one job model to cover both would make half its fields optional and mean
    nothing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: JobIdentity
    authorization: AuthorizationScope
    session_id: str | None = None
    file_names: tuple[str, ...] = ()


class DeleteAccountCommand(BaseModel):
    """Authorization to erase one account and everything it owns.

    The API verifies the password before issuing this; the password does not
    travel in it. ``session_ids`` is an issue-time cleanup hint; the workflow
    unions it with an authoritative post-gate query before deleting rows.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: JobIdentity
    authorization: AuthorizationScope
    session_ids: tuple[str, ...] = ()
