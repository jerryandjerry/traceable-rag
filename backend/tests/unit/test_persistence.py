"""Tests for persistence/.

Intended function:
    owner_of(session)      -> exactly the owning user id, or None if unknown
    owns(session, user)    -> exactly whether that user owns it
    create(...)            -> a row whose session_name is a string, never None
                             file, and is rooted at settings.storage_dir
                             rather than the current working directory
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from visionagent.database.postgres.repositories import (
    AccountWriteUnavailable,
    KnowledgeBaseRepository,
    SessionRepository,
    UserRepository,
)


class FakeRow:
    def __init__(self, user_id): self.user_id = user_id


class FakeDB:
    def __init__(self, row=None, *, account_live=True):
        self.row, self.executed, self.committed = row, [], False
        self.account_live = account_live
        self.current_row = row
        self.rolled_back = False

    def execute(self, statement, params=None):
        sql = str(statement)
        self.executed.append((sql, params))
        if "deletion_requested FROM users" in sql:
            self.current_row = (
                SimpleNamespace(deletion_requested=False)
                if self.account_live
                else None
            )
        else:
            self.current_row = self.row
        return self

    def fetchone(self): return self.current_row

    def commit(self): self.committed = True

    def rollback(self): self.rolled_back = True

    def close(self): pass


def repo(row=None) -> tuple[SessionRepository, FakeDB]:
    db = FakeDB(row)
    return SessionRepository(session_factory=lambda: db), db


def test_owner_of_returns_the_user_id_as_a_string():
    r, _ = repo(FakeRow(7))
    assert r.owner_of("s1") == "7"


def test_owner_of_an_unknown_session_is_none():
    r, _ = repo(None)
    assert r.owner_of("nope") is None


def test_owns_compares_across_int_and_str():
    """user_id arrives as a str from the token and as an int from Postgres."""
    r, _ = repo(FakeRow(7))
    assert r.owns("s1", "7") is True
    assert r.owns("s1", "8") is False


def test_an_unknown_session_is_owned_by_nobody():
    r, _ = repo(None)
    assert r.owns("nope", "7") is False


def test_create_always_passes_a_string_session_name():
    """The NOT NULL session_name column always receives a string."""
    r, db = repo()
    r.create(session_id="s1", user_id="7")
    _, params = db.executed[1]
    assert params["name"] == ""
    assert params["name"] is not None
    assert db.committed is True


def test_create_commits_the_transaction():
    r, db = repo()
    r.create(session_id="s1", user_id="7", name="Curb widths")
    assert db.committed and db.executed[1][1]["name"] == "Curb widths"


def test_create_refuses_an_account_whose_deletion_gate_is_closed():
    r, db = repo()
    db.account_live = False

    with pytest.raises(AccountWriteUnavailable):
        r.create(session_id="s1", user_id="7")

    assert len(db.executed) == 1
    assert "FOR SHARE" in db.executed[0][0]
    assert db.rolled_back


def test_existing_tokens_can_still_identify_a_crash_gated_account_for_retry():
    db = FakeDB(SimpleNamespace(auth_version=9))
    users = UserRepository(session_factory=lambda: db)

    assert users.auth_version("7") == 9
    assert "deletion_requested" not in db.executed[0][0]


def test_message_metadata_has_the_session_cascade_on_the_shared_base():
    from visionagent.database.postgres.tables import Base, Message

    assert {"sessions", "messages"}.issubset(Base.metadata.tables)
    (foreign_key,) = Message.__table__.c.session_id.foreign_keys
    assert foreign_key.target_fullname == "sessions.session_id"
    assert foreign_key.ondelete == "CASCADE"
    assert foreign_key.constraint.name == "fk_messages_session"


def test_existing_schema_upgrade_enforces_future_messages_without_deleting_history():
    from visionagent.database.postgres.engine import _SCHEMA_UPGRADES

    upgrades = "\n".join(_SCHEMA_UPGRADES)
    assert "FOREIGN KEY (session_id)" in upgrades
    assert "NOT VALID" in upgrades
    assert "DELETE FROM messages" not in upgrades
    assert "VALIDATE CONSTRAINT fk_messages_session" not in upgrades


def test_existing_schema_upgrade_and_orm_enforce_immutable_document_identity():
    from visionagent.database.postgres.engine import _SCHEMA_UPGRADES
    from visionagent.database.postgres.tables import KnowledgeBase

    upgrades = "\n".join(_SCHEMA_UPGRADES)
    assert "uq_kb_user_file" in upgrades
    assert "GROUP BY user_id, file_name HAVING COUNT(*) > 1" in upgrades
    assert "deduplicate before startup" in upgrades
    assert any(
        constraint.name == "uq_kb_user_file"
        for constraint in KnowledgeBase.__table__.constraints
    )


def test_existing_upload_schema_upgrade_adds_staging_idempotently():
    from visionagent.database.postgres.engine import _SCHEMA_UPGRADES
    from visionagent.database.postgres.tables import UploadJob

    upgrades = "\n".join(_SCHEMA_UPGRADES)
    expected_states = (
        "'staging', 'queued', 'processing', 'completed', 'failed', 'cancelled'"
    )
    assert expected_states in upgrades
    assert "upload_jobs_status_check" in upgrades
    assert "ck_upload_jobs_status" in upgrades
    assert "pg_get_constraintdef" in upgrades
    assert "POSITION('staging'" in upgrades
    assert "ALTER COLUMN status SET DEFAULT 'staging'" in upgrades
    assert str(UploadJob.__table__.c.status.server_default.arg) == "staging"


def test_knowledgebase_lookup_preserves_exact_legacy_spelling() -> None:
    rows = [
        SimpleNamespace(file_name="cafe\N{COMBINING ACUTE ACCENT}.pdf"),
        SimpleNamespace(file_name="other.pdf"),
    ]

    class _Result:
        def fetchall(self):
            return rows

    class _Db:
        def execute(self, _statement, params=None):
            assert params == {"uid": "7"}
            return _Result()

        def close(self):
            pass

    documents = KnowledgeBaseRepository(session_factory=_Db)
    assert documents.names_matching_canonical(
        "7", "caf\N{LATIN SMALL LETTER E WITH ACUTE}.pdf"
    ) == ("cafe\N{COMBINING ACUTE ACCENT}.pdf",)


def test_knowledgebase_exact_delete_uses_stored_identity() -> None:
    calls: list[tuple[str, object]] = []

    class _Result:
        rowcount = 1

    class _Db:
        committed = False

        def execute(self, statement, params=None):
            calls.append((str(statement), params))
            return _Result()

        def commit(self):
            self.committed = True

        def rollback(self):
            raise AssertionError("successful delete must not roll back")

        def close(self):
            assert self.committed

    legacy_name = "cafe\N{COMBINING ACUTE ACCENT}.pdf"
    assert KnowledgeBaseRepository(session_factory=_Db).delete_exact(
        "7", legacy_name
    )
    assert calls[0][1] == {"uid": "7", "file_name": legacy_name}


class _ChatResult:
    def __init__(self, *, row=None, rowcount=0):
        self._row = row
        self.rowcount = rowcount

    def fetchone(self):
        return self._row


class _ChatDB:
    def __init__(self, *, row=None, rowcount=0):
        self.row = row
        self.rowcount = rowcount
        self.executed: list[tuple[str, object]] = []
        self.committed = False
        self.rolled_back = False

    def execute(self, statement, params=None):
        self.executed.append((str(statement), params))
        return _ChatResult(row=self.row, rowcount=self.rowcount)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        pass


def _chat_db(monkeypatch, db: _ChatDB):
    from visionagent.service.answer import chat

    def get_db():
        yield db

    monkeypatch.setattr(chat, "get_db", get_db)
    return chat


def test_chat_insert_rechecks_account_and_session_in_one_statement(monkeypatch):
    db = _ChatDB(row=SimpleNamespace(message_id="m1"))
    chat = _chat_db(monkeypatch, db)

    chat.write_chat_to_db("s1", "7", "question", "answer", [], [], "")

    sql, params = db.executed[0]
    assert "INSERT INTO messages" in sql
    assert "JOIN users AS account" in sql
    assert "session.user_id = :user_id" in sql
    assert "account.deletion_requested = FALSE" in sql
    assert params["user_id"] == "7"
    assert db.committed


def test_chat_insert_fails_closed_if_deletion_or_session_removal_wins(monkeypatch):
    db = _ChatDB(row=None)
    chat = _chat_db(monkeypatch, db)

    with pytest.raises(RuntimeError, match="chat persistence failed") as exc:
        chat.write_chat_to_db("gone", "7", "question", "answer", [], [], "")

    assert isinstance(exc.value.__cause__, AccountWriteUnavailable)
    assert db.rolled_back
    assert not db.committed


def test_session_naming_never_recreates_a_missing_or_deleting_session(monkeypatch):
    db = _ChatDB(rowcount=0)
    chat = _chat_db(monkeypatch, db)

    chat._store_session_name("gone", "7", "name")

    assert len(db.executed) == 1
    assert "UPDATE sessions" in db.executed[0][0]
    assert "INSERT INTO sessions" not in db.executed[0][0]
    assert db.committed


def test_context_is_rooted_at_settings_not_the_working_directory():
    """The context store location is independent of the process working directory."""
    from visionagent.config.settings import settings
    from visionagent.database.session_context import session_context_manager

    assert session_context_manager.context_dir == str(settings.storage_dir / "session_context")
    assert settings.storage_dir.parent == settings.state_dir
