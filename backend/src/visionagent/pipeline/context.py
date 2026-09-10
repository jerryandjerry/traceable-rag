"""Attached-context workflow: extract through the VLM slot, then persist.

This module owns orchestration and account-deletion fencing. The session
context store owns file and JSON persistence.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from visionagent.database.graph import async_tenant_lock
from visionagent.database.postgres.repositories import SessionRepository, UserRepository
from visionagent.database.session_context import session_context_manager
from visionagent.service.parsers import vlm_processor

logger = logging.getLogger(__name__)


class ContextWriteUnavailable(RuntimeError):
    """The account/session disappeared after HTTP authorization."""


class ContextCommitCancelled(asyncio.CancelledError):
    """Cancellation arrived while the atomic context record was committing."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__("context persistence finished during cancellation")
        self.result = result


@asynccontextmanager
async def context_write_scope(user_id: str, session_id: str) -> AsyncIterator[None]:
    """Fence persistent context writes against account deletion.

    The cancellation-safe async lock never abandons a blocking lock-acquisition
    thread. The durable deletion gate and session ownership are re-read
    *inside* it. If this scope wins first, deletion waits and erases its writes;
    if deletion wins first, no file or context JSON can be recreated.
    """
    async with async_tenant_lock(str(user_id), timeout_s=30.0):
        account_live = await asyncio.to_thread(
            UserRepository().accepts_upload_work, str(user_id)
        )
        owns_session = await asyncio.to_thread(
            SessionRepository().owns, session_id, str(user_id)
        )
        if not account_live or not owns_session:
            raise ContextWriteUnavailable("context write lost its account fence")
        yield


async def add_file_to_context(
    session_id: str,
    file_path: str,
    file_name: str,
) -> dict[str, Any]:
    """Extract a file's content with the VLM and attach it to the session."""
    try:
        result = await vlm_processor.process_file(file_path)
    except Exception as e:  # noqa: BLE001 - reported to the caller, not raised
        logger.exception("attached-context extraction failed")
        return {"success": False, "error": str(e), "file_name": file_name}

    if not result["success"]:
        return {"success": False, "error": result["error"], "file_name": file_name}

    # SessionContextManager performs JSON file reads and an atomic file write.
    # It is intentionally synchronous at the database boundary, so bridge it
    # explicitly rather than blocking the request's event loop.
    commit = asyncio.create_task(
        asyncio.to_thread(
            session_context_manager.add_content_to_context,
            session_id,
            file_path,
            file_name,
            content=result["content"],
            pages_processed=result.get("pages_processed", 0),
        )
    )
    try:
        return await asyncio.shield(commit)
    except asyncio.CancelledError:
        # The sync store may already have passed its atomic rename. Keep the
        # tenant lock held until its result is known so the route can retain
        # the matching raw file exactly when the context record committed.
        committed = await commit
        raise ContextCommitCancelled(committed) from None


def get_context(session_id: str) -> dict[str, Any]:
    """What is attached to a session: the files and how much text they add."""
    context = session_context_manager.get_session_context(session_id)
    files = session_context_manager.get_session_files(session_id)
    return {"context_length": len(context), "files_count": len(files), "files": files}


def clear_context(session_id: str) -> None:
    """Detach everything from a session."""
    session_context_manager.clear_session_context(session_id)


def remove_file_from_context(session_id: str, file_name: str) -> bool:
    """Detach one file; False if it was not attached."""
    return session_context_manager.remove_file_from_context(session_id, file_name)
