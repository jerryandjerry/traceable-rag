"""Chat sessions: create one, delete one.

Session writes are workflows; the HTTP layer delegates them here rather than
issuing database mutations directly.
"""
from __future__ import annotations

import uuid

from visionagent.database.graph import tenant_lock
from visionagent.database.postgres.repositories import (
    AccountWriteUnavailable as AccountWriteUnavailable,
)
from visionagent.database.postgres.repositories import SessionRepository
from visionagent.database.session_context import session_context_manager


def create_session(user_id: str) -> str:
    """Create an empty session for this user and return its id.

    Sixteen hex characters, the width of sessions.session_id. The name is the
    empty placeholder the answer slot fills in after the first message.
    """
    session_id = uuid.uuid4().hex[:16]
    SessionRepository().create(session_id=session_id, user_id=user_id, name="")
    return session_id


def delete_session(user_id: str, session_id: str) -> bool:
    """Remove one owned session and every attached-context file.

    File erasure precedes the database transaction. A filesystem failure leaves
    the owned session available for a retry; once the row is gone, the context
    paths are already gone. The tenant lock serializes this with add/remove/
    clear context and account deletion.
    """
    repository = SessionRepository()
    with tenant_lock(str(user_id), timeout_s=30.0):
        # Recheck ownership after taking the write lock. An add authorized
        # before this deletion either finishes first and is erased below, or
        # enters later and observes that the session no longer exists.
        if not repository.owns(session_id, user_id):
            return False

        # Reuse the store's descriptor-relative, symlink-safe erasure. Raw
        # attachment bytes disappear before the JSON retry record, and only
        # then may the owned PostgreSQL session row be removed.
        session_context_manager.clear_session_context(session_id)
        return repository.delete(session_id=session_id, user_id=user_id)
