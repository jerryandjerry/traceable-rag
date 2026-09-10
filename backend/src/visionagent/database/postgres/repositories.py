"""PostgreSQL repositories for user-owned application data.

Every query that returns a user's data takes the user id as a predicate. A
caller that has already checked ownership gets the same answer; a caller that
forgot cannot get someone else's rows.
"""
from __future__ import annotations

import unicodedata
from typing import Any

from sqlalchemy import text

from visionagent.database.postgres.engine import SessionLocal
from visionagent.models.api import FilestResponse, SessionResponse


class AccountWriteUnavailable(RuntimeError):
    """A durable user-owned write lost its account/deletion fence."""


class UserRepository:
    """The users table, read side.

    Two reads, both answering a question the HTTP boundary asks on its own:
    is this token still good, and did the caller prove the password. Neither
    mutates anything, which is what lets api/ call them directly.
    """

    def __init__(self, session_factory: Any = None) -> None:
        self._sessions = session_factory or SessionLocal

    def auth_version(self, user_id: str) -> int | None:
        """The row's revocation counter, or None if there is no such user."""
        db = self._sessions()
        try:
            row = db.execute(
                text(
                    "SELECT auth_version FROM users WHERE id = :uid"
                ),
                {"uid": int(user_id)},
            ).fetchone()
        finally:
            db.close()
        return int(row.auth_version) if row else None

    def accepts_upload_work(self, user_id: str) -> bool:
        """Whether durable work may still mutate this tenant's stores.

        Account deletion sets its gate before cancelling jobs or waiting for the
        tenant store lock. A worker checks the same gate only after acquiring
        that lock, so it cannot pass an old existence check and write after the
        deletion has completed.
        """
        db = self._sessions()
        try:
            row = db.execute(
                text(
                    "SELECT 1 FROM users "
                    "WHERE id = :uid AND deletion_requested = FALSE"
                ),
                {"uid": int(user_id)},
            ).fetchone()
        finally:
            db.close()
        return row is not None

    def password_hash(self, user_id: str) -> str | None:
        """What the stored password verifies against, or None for no such user."""
        db = self._sessions()
        try:
            row = db.execute(
                text("SELECT password_hash FROM users WHERE id = :uid"),
                {"uid": int(user_id)},
            ).fetchone()
        finally:
            db.close()
        return str(row.password_hash) if row else None


class SessionRepository:
    """Sessions and their messages."""

    def __init__(self, session_factory: Any = None) -> None:
        self._sessions = session_factory or SessionLocal

    def owner_of(self, session_id: str) -> str | None:
        """The user a session belongs to, or None if it does not exist."""
        db = self._sessions()
        try:
            row = db.execute(
                text("SELECT user_id FROM sessions WHERE session_id = :sid"),
                {"sid": session_id},
            ).fetchone()
        finally:
            db.close()
        return str(row.user_id) if row else None

    def owns(self, session_id: str, user_id: str) -> bool:
        return self.owner_of(session_id) == str(user_id)

    def list_for_user(self, user_id: str) -> list[SessionResponse]:
        """Every session this user owns, newest first."""
        db = self._sessions()
        try:
            rows = db.execute(
                text(
                    "SELECT session_id, session_name, user_id, created_at, updated_at "
                    "FROM sessions WHERE user_id = :uid ORDER BY updated_at DESC"
                ),
                {"uid": str(user_id)},
            ).fetchall()
        finally:
            db.close()
        return [
            SessionResponse(
                session_id=r.session_id,
                session_name=r.session_name,
                user_id=str(r.user_id),
                created_at=r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                updated_at=r.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
            )
            for r in rows
        ]

    def list_messages(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        """One session's messages, empty if it is not this user's.

        Ownership is part of the query, not a separate check that could race
        an unscoped read. The foreign key protects parent lifetime; the join
        protects tenant scope.
        """
        db = self._sessions()
        try:
            rows = db.execute(
                text(
                    "SELECT m.message_id, m.session_id, m.user_question, "
                    "m.model_answer, m.documents, m.recommended_questions, "
                    "m.think, m.created_at "
                    "FROM messages m JOIN sessions s ON s.session_id = m.session_id "
                    "WHERE m.session_id = :sid AND s.user_id = :uid "
                    "ORDER BY m.created_at"
                ),
                {"sid": session_id, "uid": str(user_id)},
            ).fetchall()
        finally:
            db.close()
        out: list[dict[str, Any]] = []
        for m in rows:
            questions = [
                q.strip() for q in (m.recommended_questions or "").strip('{}"').split(",")
            ]
            out.append({
                "message_id": m.message_id,
                "session_id": m.session_id,
                "user_question": m.user_question,
                "model_answer": m.model_answer,
                "documents": m.documents,
                "recommended_questions": questions,
                "think": m.think,
                "created_at": m.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            })
        return out

    def create(self, *, session_id: str, user_id: str, name: str = "") -> None:
        """Create a session while holding the account-deletion fence."""
        db = self._sessions()
        try:
            # Serialize session admission with account deletion exactly like
            # upload-job admission. If this commits first, deletion's current
            # session query sees it; if the gate commits first, creation fails.
            account = db.execute(
                text(
                    "SELECT deletion_requested FROM users "
                    "WHERE id = :uid FOR SHARE"
                ),
                {"uid": int(user_id)},
            ).fetchone()
            if account is None or bool(account.deletion_requested):
                raise AccountWriteUnavailable("account is unavailable for session creation")
            db.execute(
                text(
                    "INSERT INTO sessions (session_id, user_id, session_name) "
                    "VALUES (:sid, :uid, :name)"
                ),
                {"sid": session_id, "uid": str(user_id), "name": name},
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def ids_for_user(self, user_id: str) -> list[str]:
        """Current session ids, used by account erasure after its gate closes."""
        db = self._sessions()
        try:
            rows = db.execute(
                text(
                    "SELECT session_id FROM sessions "
                    "WHERE user_id = :uid ORDER BY session_id"
                ),
                {"uid": str(user_id)},
            ).fetchall()
            return [str(row.session_id) for row in rows]
        finally:
            db.close()

    def delete(self, *, session_id: str, user_id: str) -> bool:
        """Remove one session and its messages; False if it was not this user's.

        Both statements are scoped to the user. The explicit message delete
        preserves a useful count/order while the foreign-key cascade is the
        final integrity fence.
        """
        db = self._sessions()
        try:
            db.execute(
                text(
                    "DELETE FROM messages WHERE session_id = :sid AND session_id IN "
                    "(SELECT session_id FROM sessions WHERE session_id = :sid AND user_id = :uid)"
                ),
                {"sid": session_id, "uid": str(user_id)},
            )
            result = db.execute(
                text("DELETE FROM sessions WHERE session_id = :sid AND user_id = :uid"),
                {"sid": session_id, "uid": str(user_id)},
            )
            gone = int(getattr(result, "rowcount", 0) or 0)
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return bool(gone)


class KnowledgeBaseRepository:
    """The documents a user has ingested."""

    def __init__(self, session_factory: Any = None) -> None:
        self._sessions = session_factory or SessionLocal

    def list_for_user(self, user_id: str) -> list[FilestResponse]:
        """Every document this user owns. Scoped by the query, not the caller."""
        db = self._sessions()
        try:
            rows = db.execute(
                text(
                    "SELECT user_id, file_name, created_at, updated_at, "
                    "total_chunks, process_time, error FROM knowledgebases "
                    "WHERE user_id = :uid ORDER BY created_at"
                ),
                {"uid": str(user_id)},
            ).fetchall()
        finally:
            db.close()
        return [
            FilestResponse(
                user_id=str(r.user_id),
                file_name=str(r.file_name),
                created_at=r.created_at.isoformat(),
                updated_at=r.updated_at.isoformat(),
                total_chunks=r.total_chunks,
                process_time=r.process_time,
                error=r.error,
            )
            for r in rows
        ]

    def names_matching_canonical(
        self, user_id: str, canonical_name: str
    ) -> tuple[str, ...]:
        """Resolve an NFC identity while preserving exact legacy spelling.

        Historical rows predate NFC admission and the database uniqueness key
        is byte-exact.  Returning every match lets the workflow reject a
        legacy canonical collision rather than deleting an arbitrary row.
        """
        db = self._sessions()
        try:
            rows = db.execute(
                text(
                    "SELECT file_name FROM knowledgebases "
                    "WHERE user_id = :uid ORDER BY file_name"
                ),
                {"uid": str(user_id)},
            ).fetchall()
        finally:
            db.close()
        return tuple(
            str(row.file_name)
            for row in rows
            if unicodedata.normalize("NFC", str(row.file_name)) == canonical_name
        )

    def delete_exact(self, user_id: str, stored_name: str) -> bool:
        """Delete only the exact row whose external-store identity was used."""
        db = self._sessions()
        try:
            result = db.execute(
                text(
                    "DELETE FROM knowledgebases "
                    "WHERE user_id = :uid AND file_name = :file_name"
                ),
                {"uid": str(user_id), "file_name": stored_name},
            )
            removed = int(getattr(result, "rowcount", 0) or 0)
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return bool(removed)
