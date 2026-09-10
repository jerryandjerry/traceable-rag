"""FastAPI dependencies: authentication and component injection.

`OwnedSession` and `AuthorizedQueryJob` make authentication, ownership, and
policy checks declarative handler requirements. The latter also mints the only
ticket under which a query turn may run.
"""
from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import Body, Depends, HTTPException, Query

from visionagent.api.security import access_security, require_auth, verify_session_owner
from visionagent.config.settings import settings
from visionagent.database.postgres.repositories import SessionRepository, UserRepository
from visionagent.models import (
    AuthorizationScope,
    ChatRequest,
    DeleteAccountCommand,
    DeleteAccountRequest,
    IngestJob,
    JobIdentity,
    QueryJob,
    ToolName,
    TurnOptions,
    WebSearchMode,
)
from visionagent.utils.password import verify_password


def _parse_allowed_tools(names: tuple[str, ...]) -> frozenset[ToolName]:
    """ALLOWED_TOOLS as ToolName members, or refuse to start.

    Failing closed at import: a misspelt tool name in the environment must not
    become "no tools" at request time (every question answered from nothing)
    or "all tools" (the misconfiguration ignored). Either would be discovered
    by a user rather than by whoever deployed it.
    """
    try:
        return frozenset(ToolName(n) for n in names)
    except ValueError as exc:
        raise RuntimeError(
            f"ALLOWED_TOOLS names a tool that does not exist: {exc}. "
            f"Valid names: {[t.value for t in ToolName]}"
        ) from exc


_ALLOWED_TOOLS = _parse_allowed_tools(settings.allowed_tools)


def current_user_id(credentials: Any = Depends(access_security)) -> str:
    """The authenticated user, or 401.

    `require_auth` also confirms the row still exists and the token's
    auth_version matches it, so a deleted account or a changed password ends
    the token here rather than two days later.
    """
    return str(require_auth(credentials))


CurrentUser = Annotated[str, Depends(current_user_id)]


def owned_session(
    user_id: CurrentUser,
    session_id: str = Query(...),
) -> str:
    """A session id the caller actually owns, or 404.

    404 rather than 403 on purpose: telling an attacker that a session exists
    but belongs to someone else is itself a disclosure.
    """
    verify_session_owner(session_id, user_id)
    return session_id


OwnedSession = Annotated[str, Depends(owned_session)]


def authorization_scope(user_id: CurrentUser) -> AuthorizationScope:
    """What server policy lets this caller do.

    One function, so that a per-user entitlement table later changes one place.
    Today the answer is the same for every authenticated caller and comes from
    ALLOWED_TOOLS, validated at start-up; the point is that it comes from the
    server and never from the request.
    """
    return AuthorizationScope(allowed_tools=_ALLOWED_TOOLS)


def authorized_query_job(
    user_id: CurrentUser,
    scope: Annotated[AuthorizationScope, Depends(authorization_scope)],
    request: ChatRequest = Body(..., description="User message"),
    session_id: str = Query(...),
) -> QueryJob:
    """Mint one immutable, authorized ticket for a turn.

    Everything a turn is allowed to do is decided here, before the handler body
    and before any response header, so an invalid or foreign session gets a real
    HTTP 401/404 rather than a 200 carrying an SSE error frame. The pipeline is
    never entered for a request that fails.

    The web-search bool on the wire is a preference. Whether it is honoured is
    resolved here against the scope, so the pipeline reads a decision rather
    than re-deriving one, and a client cannot grant itself a tool.
    """
    verify_session_owner(session_id, user_id)

    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=422, detail="message must not be blank")
    if len(question) > settings.max_question_chars:
        raise HTTPException(
            status_code=422,
            detail=f"message must be at most {settings.max_question_chars} characters",
        )

    requested = TurnOptions(
        web_search=WebSearchMode.FORCE if request.web_search else WebSearchMode.AUTO,
        deep_research=request.deep_research,
    )
    effective = TurnOptions(
        web_search=(
            WebSearchMode.DISABLED
            if ToolName.WEB_SEARCH not in scope.allowed_tools
            else requested.web_search
        ),
        # Accepted and recorded as requested, but not yet implemented, so the
        # effective answer is no. An option that is silently ignored cannot be
        # told apart from one that ran.
        deep_research=False,
    )

    return QueryJob(
        identity=JobIdentity(run_id=uuid.uuid4().hex, user_id=user_id),
        authorization=scope,
        session_id=session_id,
        question=question,
        requested=requested,
        effective=effective,
    )


AuthorizedQueryJob = Annotated[QueryJob, Depends(authorized_query_job)]


def authorized_ingest_job(
    user_id: CurrentUser,
    scope: Annotated[AuthorizationScope, Depends(authorization_scope)],
    session_id: str | None = Query(None),
) -> IngestJob:
    """Mint the ticket an upload runs under.

    Separate from QueryJob because an upload has no question and no web-search
    option, and the retrieval tool names in AuthorizationScope do not describe
    permission to index. One job model covering both would make half its fields
    optional and mean nothing.

    The file names are attached by the route once it has read the upload; what
    is decided here is who is uploading and whether they may.
    """
    if not scope.can_upload:
        raise HTTPException(status_code=403, detail="Uploading is not permitted")
    if session_id:
        verify_session_owner(session_id, user_id)
    return IngestJob(
        identity=JobIdentity(run_id=uuid.uuid4().hex, user_id=user_id),
        authorization=scope,
        session_id=session_id,
    )


AuthorizedIngestJob = Annotated[IngestJob, Depends(authorized_ingest_job)]


def delete_account_command(
    user_id: CurrentUser,
    scope: Annotated[AuthorizationScope, Depends(authorization_scope)],
    request: DeleteAccountRequest = Body(...),
) -> DeleteAccountCommand:
    """Re-authenticate, then mint the authorization to erase this account.

    The password is checked here, at the trust boundary, and does not travel
    further: the workflow receives a decision, not the credential that produced
    it. A workflow that re-checks a password is a workflow that can be called
    without one.

    The literal "DELETE" is required as well. This is irreversible and removes
    the user's documents, chat history, graph and search index -- a mistyped
    request must not be able to do it.

    session_ids are captured as a cleanup hint. The workflow closes the gate
    and then reads the authoritative set again before deleting any rows, so a
    session that committed between these two phases is not missed.
    """
    if request.confirm != "DELETE":
        raise HTTPException(
            status_code=400, detail='confirm must be the literal string "DELETE"'
        )
    stored = UserRepository().password_hash(user_id)
    if stored is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not verify_password(request.password, stored):
        raise HTTPException(status_code=400, detail="Password is incorrect")

    return DeleteAccountCommand(
        identity=JobIdentity(run_id=uuid.uuid4().hex, user_id=user_id),
        authorization=scope,
        session_ids=tuple(s.session_id for s in SessionRepository().list_for_user(user_id)),
    )


AuthorizedDeleteAccount = Annotated[DeleteAccountCommand, Depends(delete_account_command)]
