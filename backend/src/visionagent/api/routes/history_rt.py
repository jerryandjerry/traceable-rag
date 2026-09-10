import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from visionagent.api.observability import translated_http_error
from visionagent.api.security import access_security, get_current_user_id, verify_session_owner
from visionagent.database.postgres.repositories import (
    KnowledgeBaseRepository,
    SessionRepository,
)
from visionagent.models import FilestResponse, SessionListResponse
from visionagent.pipeline.ingest import (
    DocumentNameAmbiguous,
    DocumentNotFound,
    delete_document,
)
from visionagent.pipeline.session import delete_session as delete_session_workflow

router = APIRouter()

@router.get("/get_files/", response_model=list[FilestResponse])
async def get_documents_by_user_id(
    credentials: Any = Depends(access_security),
) -> Any:
    try:
        user_id = get_current_user_id(credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        # The query is scoped to the caller inside the repository. A route that
        # builds its own SELECT is one WHERE clause away from returning
        # someone else's rows.
        documents = KnowledgeBaseRepository().list_for_user(user_id)

        return documents

    except HTTPException:
        raise
    except Exception as e:
        raise translated_http_error(
            e,
            operation="list_documents",
            detail="Failed to retrieve documents",
        ) from e

@router.delete("/delete_file/", status_code=200)
async def delete_file_by_name(
    file_name: str = Query(..., description="File name to delete"),
    credentials: Any = Depends(access_security),
) -> dict[str, Any]:
    try:
        user_id = get_current_user_id(credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        # One call: Postgres, Elasticsearch and the graph, with the ordering
        # and partial-failure policy owned by the pipeline.
        try:
            await delete_document(user_id, file_name)
        except DocumentNotFound as e:
            raise HTTPException(status_code=404, detail="File not found") from e
        except DocumentNameAmbiguous as e:
            raise HTTPException(
                status_code=409,
                detail="Conflicting legacy document names require reindexing",
            ) from e
        except TimeoutError as e:
            raise HTTPException(status_code=409, detail="An upload is still running; try again") from e

        return {"message": "File deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise translated_http_error(
            e,
            operation="delete_file",
            detail="Failed to delete file",
            file_name=file_name,
        ) from e

@router.delete("/delete_session/{session_id}")
async def delete_session(
    session_id: str,
    credentials: Any = Depends(access_security),
) -> dict[str, Any]:
    try:
        user_id = get_current_user_id(credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        # Authorize before mutation. The workflow scopes both deletes to the
        # caller; this explicit check returns 404 for cross-tenant identifiers.
        verify_session_owner(session_id, user_id)

        try:
            deleted = await asyncio.to_thread(
                delete_session_workflow,
                user_id,
                session_id,
            )
        except TimeoutError as exc:
            raise HTTPException(
                status_code=409,
                detail="Another account operation is still running; try again",
            ) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"message": "Session deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise translated_http_error(
            e,
            operation="delete_session",
            detail="Failed to delete session",
            session_id=session_id,
        ) from e

@router.get("/get_messages/")
async def get_messages_by_session_id(
    session_id: str,
    credentials: Any = Depends(access_security),
) -> Any:
    try:
        user_id = get_current_user_id(credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        # messages has no user_id column, so ownership still comes from its
        # parent session even though the foreign key protects parent lifetime.
        verify_session_owner(session_id, user_id)

        # Ownership is also a predicate in this query, not only the check
        # above: two statements can disagree, one cannot.
        messages = SessionRepository().list_messages(user_id, session_id)

        return messages

    except HTTPException:
        raise
    except Exception as e:
        raise translated_http_error(
            e,
            operation="list_messages",
            detail="Failed to retrieve messages",
            session_id=session_id,
        ) from e
    
@router.get("/get_sessions/", response_model=SessionListResponse)
async def get_sessions_by_user_id(
    credentials: Any = Depends(access_security),
) -> dict[str, Any]:
    try:
        user_id = get_current_user_id(credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        sessions = SessionRepository().list_for_user(user_id)

        return {"user_id": user_id, "sessions": sessions}

    except HTTPException:
        raise
    except Exception as e:
        raise translated_http_error(
            e,
            operation="list_sessions",
            detail="Failed to retrieve sessions",
        ) from e
