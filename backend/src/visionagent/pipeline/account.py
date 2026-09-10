"""Account workflows, including ordered cross-store account deletion."""
import logging

from sqlalchemy.exc import SQLAlchemyError

from visionagent.database.postgres.engine import get_db
from visionagent.database.postgres.tables import User
from visionagent.exceptions.auth import AuthError
from visionagent.models import DeleteAccountCommand
from visionagent.pipeline.ingest import (
    cancel_user_uploads,
    cleanup_upload_jobs,
    discard_user_uploads_for_account_deletion,
    upload_staging_lock,
)
from visionagent.utils.password import verify_password

logger = logging.getLogger(__name__)

def authenticate(username: str, password: str) -> tuple[int, str]:
    """Verify a username and password.

    Returns (user_id, username) rather than a token. Minting a credential is
    the API's job: the token format, its lifetime and the cookie-or-header
    question are all HTTP decisions, and a function that verifies a password
    should not also decide them. The login route calls create_token on what
    this returns.

    Raises:
        AuthError: on a bad username or password, the same error for both so
            the response does not confirm which usernames exist.
    """
    db = next(get_db())
    try:
        user = db.query(User).filter(
            User.username == username,
            User.deletion_requested.is_(False),
        ).first()
        
        if not user:
            raise AuthError("Authentication failed")
        
        if not verify_password(password, user.password_hash):
            raise AuthError("Authentication failed")
        
        return int(user.id), str(user.username)
    
    except SQLAlchemyError as e:
        raise AuthError("Authentication failed") from e
    finally:
        db.close()

def register_user(username: str, password: str) -> None:
    """Register a user or raise ``AuthError`` if registration fails."""
    from visionagent.utils.password import hash_password
    db = next(get_db())
    try:
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            raise AuthError("Username already exists")
        
        password_hash = hash_password(password)

        new_user = User(username=username, password_hash=password_hash)
        db.add(new_user)
        db.commit()
        logger.info("user registration completed")
        
    except SQLAlchemyError as e:
        db.rollback()
        logger.exception("user registration failed")
        raise AuthError("Registration failed") from e
    finally:
        db.close()
        logger.debug("registration database session closed")

def change_password(user_id: int, current_password: str, new_password: str) -> None:
    """Change a user's password.

    The current password is required even though the caller is already
    authenticated: a stolen or forgotten-open session should not be enough to
    lock the real owner out of their own account.

    auth_version moves with the hash, in the same commit. Every token issued
    before this carries the old value and stops verifying, so changing the
    password is also how a user ends a session they no longer control.
    """
    from visionagent.utils.password import hash_password

    if len(new_password) < 8:
        raise AuthError("New password must be at least 8 characters")
    if new_password == current_password:
        raise AuthError("New password must differ from the current one")

    db = next(get_db())
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise AuthError("User not found")
        if not verify_password(current_password, user.password_hash):
            raise AuthError("Current password is incorrect")
        user.password_hash = hash_password(new_password)
        user.auth_version = int(user.auth_version or 0) + 1
        db.commit()
    finally:
        db.close()


def delete_account(command: DeleteAccountCommand) -> dict[str, int]:
    """Delete a user and everything they own, under an issued authorization.

    Takes a command rather than a password: by the time this runs the caller
    has already been proven, and a workflow that re-checks a credential is a
    workflow that can be called without one.

    Order: the stores outside Postgres first, the account row last. If the
    Elasticsearch index cannot be dropped, the user still exists and can run
    the deletion again; the reverse order left the row gone, the chunks
    searchable by nobody, and no account left to retry from. Every step
    tolerates "already gone", so a second run is a no-op that returns zeros
    rather than an error.

    Returns what was removed.
    """
    from visionagent.database.graph import tenant_lock

    uid = command.identity.user_id
    # This is both the byte-writer fence and the account-deletion coordinator.
    # It is acquired before touching the gate and held through success or
    # rollback, so concurrent deletion attempts cannot adopt the same gate and
    # then let one caller reopen it underneath the other. Global lock order is
    # staging -> tenant; upload workers release tenant before staging cleanup.
    with upload_staging_lock(uid, timeout_s=30.0):
        transitioned_before_tenant = _set_account_deletion_gate(uid, enabled=True)
        entered_tenant_lock = False
        try:
            # Upload creation holds a conflicting row lock, so every pre-gate
            # owner ticket is committed and visible here and every post-gate
            # ticket is refused. Since we already hold the staging fence, a
            # pre-gate writer cannot create bytes after this snapshot.
            cancel_user_uploads(uid, cleanup_staging=False)
            cleanup_upload_jobs(
                uid, require_all=True, staging_lock_held=True
            )

            # The worker takes this lock around every tenant write and checks
            # the gate only after acquiring it. If it was already inside,
            # cancellation makes it leave and deletion removes its writes
            # afterwards; if it was waiting, it sees the gate and never writes.
            with tenant_lock(uid, timeout_s=30.0):
                entered_tenant_lock = True
                # A previous deletion may have crashed after closing the gate.
                # Once we own both locks, adopt or re-close it for the complete
                # destructive section.
                owns_gate = _adopt_account_deletion_gate(uid)
                try:
                    cancel_user_uploads(uid, cleanup_staging=False)
                    # A crashed in-flight checkpoint is normally replayed and
                    # finalized. Account deletion is the explicit exception:
                    # owning the tenant lock proves no worker is mutating this
                    # user, and every partial effect is erased next.
                    discard_user_uploads_for_account_deletion(uid)
                    return _delete_stores(
                        uid,
                        list(command.session_ids),
                        staging_lock_held=True,
                    )
                except BaseException:
                    # Reopen while both locks are still held. A new upload or
                    # deletion cannot enter between failed cleanup and rollback.
                    if owns_gate:
                        _set_account_deletion_gate(uid, enabled=False)
                    raise
        except BaseException:
            # Before the tenant lock is entered, reopen only if this serialized
            # attempt performed the false->true transition. A crash-left closed
            # gate remains closed for an explicit retry.
            if not entered_tenant_lock and transitioned_before_tenant:
                _set_account_deletion_gate(uid, enabled=False)
            raise


def _set_account_deletion_gate(uid: str, *, enabled: bool) -> bool:
    """Transition the deletion gate; False means absent or already in that state."""
    from sqlalchemy import text

    db = next(get_db())
    try:
        result = db.execute(
            text(
                "UPDATE users SET deletion_requested = :enabled "
                "WHERE id = :uid "
                "AND deletion_requested IS DISTINCT FROM :enabled"
            ),
            {"enabled": enabled, "uid": int(uid)},
        )
        changed = bool(result.rowcount)
        db.commit()
        return changed
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _adopt_account_deletion_gate(uid: str) -> bool:
    """Ensure the gate is closed after acquiring the tenant lock.

    PostgreSQL reports the extant row even when its value was already true, so
    True means this lock holder now owns the deletion attempt and must reopen
    the gate on an in-lock failure. False means the account is already gone.
    """
    from sqlalchemy import text

    db = next(get_db())
    try:
        result = db.execute(
            text(
                "UPDATE users SET deletion_requested = TRUE "
                "WHERE id = :uid RETURNING id"
            ),
            {"uid": int(uid)},
        )
        owns_gate = result.fetchone() is not None
        db.commit()
        return owns_gate
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _current_session_ids(uid: str) -> list[str]:
    """Read the authoritative set only after the deletion gate is closed."""
    from sqlalchemy import text

    db = next(get_db())
    try:
        rows = db.execute(
            text(
                "SELECT session_id FROM sessions "
                "WHERE user_id = :uid ORDER BY session_id"
            ),
            {"uid": uid},
        ).fetchall()
        return [str(row.session_id) for row in rows]
    finally:
        db.close()


def _delete_stores(
    uid: str,
    issued_session_ids: list[str],
    *,
    staging_lock_held: bool = False,
) -> dict[str, int]:
    """The deletion itself, run while the tenant lock is held."""
    from sqlalchemy import text

    from visionagent.config.settings import settings
    from visionagent.database.session_context import SessionContextManager
    from visionagent.service.vectorstore import build_chunkstore

    removed = {"sessions": 0, "messages": 0, "documents": 0, "chunks": 0}

    # The command snapshot can miss a session that committed immediately
    # before the deletion gate. The post-gate query is authoritative; retain
    # the issued set too so a concurrently removed session's leftover files
    # are still erased.
    session_ids = sorted(set(issued_session_ids) | set(_current_session_ids(uid)))

    # Staging erasure is a precondition, before any user-visible store is
    # changed. If the shared volume refuses erasure, deletion remains retryable.
    cleanup_upload_jobs(
        uid,
        require_all=True,
        staging_lock_held=staging_lock_held,
    )

    # 1. The search index. Raised, not logged: the row survives below, so the
    #    caller learns the account is still there and can try again.
    removed["chunks"] = build_chunkstore().delete_index(index_name=uid)

    # 2. Files. unlink(missing_ok) and ignore_errors make these idempotent.
    for path in list(settings.graph_dir.glob(f"*_{uid}.json")) + \
            list(settings.graph_dir.glob(f"*_{uid}.graphml")):
        path.unlink(missing_ok=True)
    context_store = SessionContextManager(storage_dir=settings.storage_dir)
    for sid in session_ids:
        # The context store erases the complete owned raw tree first, without
        # following symlinks, and removes metadata only after that succeeds.
        # A filesystem failure therefore leaves both the account and a durable
        # retry handle instead of orphaning private bytes.
        context_store.clear_session_context(sid)

    # 3. Postgres, last, in one transaction.
    db = next(get_db())
    try:
        removed["messages"] = db.execute(
            text(
                "DELETE FROM messages AS message USING sessions AS session "
                "WHERE message.session_id = session.session_id "
                "AND session.user_id = :uid"
            ),
            {"uid": uid},
        ).rowcount or 0
        removed["sessions"] = db.execute(
            text("DELETE FROM sessions WHERE user_id = :uid"), {"uid": uid}
        ).rowcount or 0
        removed["documents"] = db.execute(
            text("DELETE FROM knowledgebases WHERE user_id = :uid"), {"uid": uid}
        ).rowcount or 0
        user = db.query(User).filter(User.id == int(uid)).first()
        if user is not None:
            db.delete(user)
        db.commit()
    finally:
        db.close()
    return removed
