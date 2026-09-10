"""Account deletion, over HTTP: authorization is issued once, then the
workflow runs external stores first and Postgres last, and runs twice safely.

The one workflow that spans four stores. What this tier checks without a
stack: the password is verified at the trust boundary and never reaches the
workflow; the command carries what the cleanup needs; the index is dropped
before the row so a failed drop leaves an account to retry from; and a second
deletion is a no-op rather than an error.
"""
from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from test.pipeline.conftest import SESSION_ID, USER_ID
from visionagent.models import (
    AuthorizationScope,
    DeleteAccountCommand,
    IngestJob,
    JobIdentity,
    ToolName,
)
from visionagent.pipeline import account

PATH = "/me"


# ------------------------------------------------------------ the boundary
def test_a_wrong_password_authorizes_nothing(client, users, deletions):
    users.password_ok = False
    r = client.request("DELETE", PATH, json={"password": "wrong", "confirm": "DELETE"})
    assert r.status_code == 400
    assert deletions == []


def test_a_missing_confirmation_authorizes_nothing(client, users, deletions):
    for confirm in ("", "delete", "yes"):
        r = client.request("DELETE", PATH, json={"password": "pw", "confirm": confirm})
        assert r.status_code == 400, confirm
    assert deletions == []


def test_a_deleted_user_cannot_authorize_a_second_deletion(client, users, deletions):
    users.exists = False
    r = client.request("DELETE", PATH, json={"password": "pw", "confirm": "DELETE"})
    assert r.status_code == 401
    assert deletions == []


def test_the_command_carries_the_decision_not_the_password(client, users, deletions):
    r = client.request("DELETE", PATH, json={"password": "correct-horse", "confirm": "DELETE"})
    assert r.status_code == 200, r.text

    (command,) = deletions
    assert isinstance(command, DeleteAccountCommand)
    assert command.identity.user_id == USER_ID
    dumped = command.model_dump_json().lower()
    assert "correct-horse" not in dumped
    for secret in ("password", "token", "hash"):
        assert secret not in dumped


def test_the_command_captures_the_sessions_the_cleanup_will_need(client, users, deletions):
    """Read before the rows go, because file cleanup needs them after."""
    client.request("DELETE", PATH, json={"password": "pw", "confirm": "DELETE"})
    (command,) = deletions
    assert command.session_ids == (SESSION_ID,)


def test_account_cleanup_does_not_block_the_application_loop(monkeypatch):
    from visionagent.api.routes import user_rt

    entered = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def slow_delete(command):
        entered.set()
        order.append("cleanup-started")
        assert release.wait(timeout=1)
        order.append("cleanup-finished")
        return {"sessions": 0}

    monkeypatch.setattr(user_rt, "delete_account", slow_delete)

    async def exercise():
        task = asyncio.create_task(user_rt.delete_my_account(_command()))
        assert await asyncio.to_thread(entered.wait, 1)
        await asyncio.sleep(0)
        order.append("unrelated-coroutine-ran")
        release.set()
        result = await task
        assert result["removed"] == {"sessions": 0}

    asyncio.run(exercise())
    assert order == [
        "cleanup-started",
        "unrelated-coroutine-ran",
        "cleanup-finished",
    ]


def test_the_workflow_takes_a_command_not_a_credential():
    """A workflow that re-checks a password is one that can be called without
    an authorization; this one cannot."""
    import inspect

    params = inspect.signature(account.delete_account).parameters
    assert list(params) == ["command"]
    assert "password" not in params


def test_deletion_gate_closes_before_jobs_are_cancelled_or_store_lock_is_taken(
    monkeypatch,
):
    order: list[str] = []

    def gate(_uid: str, *, enabled: bool) -> bool:
        order.append(f"gate:{enabled}")
        return True

    @contextmanager
    def lock(_uid: str, *, timeout_s: float):
        order.append("tenant-lock")
        yield

    @contextmanager
    def staging_lock(_uid: str, *, timeout_s: float):
        order.append("staging-lock:enter")
        try:
            yield
        finally:
            order.append("staging-lock:exit")

    monkeypatch.setattr(account, "_set_account_deletion_gate", gate)
    monkeypatch.setattr(
        account,
        "_adopt_account_deletion_gate",
        lambda _uid: order.append("adopt") or True,
    )
    monkeypatch.setattr(
        account,
        "cancel_user_uploads",
        lambda _uid, **_kwargs: order.append("cancel"),
    )
    monkeypatch.setattr(
        account,
        "cleanup_upload_jobs",
        lambda _uid, **_kwargs: order.append("cleanup"),
    )
    monkeypatch.setattr(
        account,
        "discard_user_uploads_for_account_deletion",
        lambda _uid: order.append("discard"),
    )
    monkeypatch.setattr(
        account,
        "_delete_stores",
        lambda _uid, _sessions, **_kwargs: order.append("delete") or {},
    )
    monkeypatch.setattr(account, "upload_staging_lock", staging_lock)
    monkeypatch.setattr("visionagent.database.graph.tenant_lock", lock)

    account.delete_account(_command())

    assert order == [
        "staging-lock:enter",
        "gate:True",
        "cancel",
        "cleanup",
        "tenant-lock",
        "adopt",
        "cancel",
        "discard",
        "delete",
        "staging-lock:exit",
    ]


def test_a_waiting_delete_does_not_reopen_another_callers_gate_on_timeout(
    monkeypatch,
):
    transitions: list[bool] = []

    def gate(_uid: str, *, enabled: bool) -> bool:
        transitions.append(enabled)
        return False

    @contextmanager
    def timeout(_uid: str, *, timeout_s: float):
        raise TimeoutError("busy")
        yield  # pragma: no cover

    @contextmanager
    def staging_lock(_uid: str, *, timeout_s: float):
        yield

    monkeypatch.setattr(account, "_set_account_deletion_gate", gate)
    monkeypatch.setattr(account, "cancel_user_uploads", lambda _uid, **_kwargs: 0)
    monkeypatch.setattr(account, "cleanup_upload_jobs", lambda _uid, **_kwargs: 0)
    monkeypatch.setattr(account, "upload_staging_lock", staging_lock)
    monkeypatch.setattr("visionagent.database.graph.tenant_lock", timeout)

    with pytest.raises(TimeoutError, match="busy"):
        account.delete_account(_command())

    assert transitions == [True]


def test_staging_coordinator_timeout_happens_before_gate_mutation(monkeypatch):
    @contextmanager
    def timeout(_uid: str, *, timeout_s: float):
        raise TimeoutError("staging busy")
        yield  # pragma: no cover

    monkeypatch.setattr(account, "upload_staging_lock", timeout)
    monkeypatch.setattr(
        account,
        "_set_account_deletion_gate",
        lambda *_args, **_kwargs: pytest.fail("gate changed without coordinator"),
    )

    with pytest.raises(TimeoutError, match="staging busy"):
        account.delete_account(_command())


def test_an_in_lock_failure_reopens_before_releasing_the_tenant_lock(monkeypatch):
    order: list[str] = []

    def gate(_uid: str, *, enabled: bool) -> bool:
        order.append(f"gate:{enabled}")
        return True

    @contextmanager
    def lock(_uid: str, *, timeout_s: float):
        order.append("lock:enter")
        try:
            yield
        finally:
            order.append("lock:exit")

    @contextmanager
    def staging_lock(_uid: str, *, timeout_s: float):
        yield

    monkeypatch.setattr(account, "_set_account_deletion_gate", gate)
    monkeypatch.setattr(account, "_adopt_account_deletion_gate", lambda _uid: True)
    monkeypatch.setattr(account, "cancel_user_uploads", lambda _uid, **_kwargs: 0)
    monkeypatch.setattr(account, "cleanup_upload_jobs", lambda _uid, **_kwargs: 0)
    monkeypatch.setattr(account, "upload_staging_lock", staging_lock)
    monkeypatch.setattr(
        account,
        "discard_user_uploads_for_account_deletion",
        lambda _uid: 0,
    )
    monkeypatch.setattr(
        account,
        "_delete_stores",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("store unavailable")
        ),
    )
    monkeypatch.setattr("visionagent.database.graph.tenant_lock", lock)

    with pytest.raises(RuntimeError, match="store unavailable"):
        account.delete_account(_command())

    assert order == ["gate:True", "lock:enter", "gate:False", "lock:exit"]


def test_concurrent_deletions_cannot_reopen_the_gate_under_an_active_delete(
    monkeypatch,
):
    """The staging fence coordinates the complete deletion attempt.

    Caller A fails before the tenant lock while caller B is waiting. A must
    roll its gate transition back before B can close/adopt the gate; otherwise
    A could admit a new upload while B is erasing the account.
    """
    staging_mutex = threading.Lock()
    caller_a_inside = threading.Event()
    caller_b_waiting = threading.Event()
    events: list[str] = []
    events_lock = threading.Lock()
    gate_closed = False

    def record(value: str) -> None:
        with events_lock:
            events.append(value)

    @contextmanager
    def staging_lock(_uid: str, *, timeout_s: float):
        name = threading.current_thread().name
        if name == "delete-b":
            caller_b_waiting.set()
        assert staging_mutex.acquire(timeout=timeout_s)
        record(f"{name}:staging-enter")
        if name == "delete-a":
            caller_a_inside.set()
        try:
            yield
        finally:
            record(f"{name}:staging-exit")
            staging_mutex.release()

    def gate(_uid: str, *, enabled: bool) -> bool:
        nonlocal gate_closed
        changed = gate_closed != enabled
        gate_closed = enabled
        record(f"{threading.current_thread().name}:gate:{enabled}")
        return changed

    def cleanup(_uid: str, **_kwargs) -> int:
        if threading.current_thread().name == "delete-a":
            assert caller_b_waiting.wait(timeout=1.0)
            raise RuntimeError("pre-tenant cleanup failed")
        return 0

    @contextmanager
    def tenant_lock(_uid: str, *, timeout_s: float):
        record("delete-b:tenant-enter")
        yield

    monkeypatch.setattr(account, "upload_staging_lock", staging_lock)
    monkeypatch.setattr(account, "_set_account_deletion_gate", gate)
    monkeypatch.setattr(account, "cancel_user_uploads", lambda *_a, **_k: 0)
    monkeypatch.setattr(account, "cleanup_upload_jobs", cleanup)
    monkeypatch.setattr(
        account,
        "_adopt_account_deletion_gate",
        lambda _uid: record("delete-b:adopt") or True,
    )
    monkeypatch.setattr(
        account,
        "discard_user_uploads_for_account_deletion",
        lambda _uid: 0,
    )
    monkeypatch.setattr(
        account,
        "_delete_stores",
        lambda *_a, **_k: record("delete-b:delete") or {},
    )
    monkeypatch.setattr("visionagent.database.graph.tenant_lock", tenant_lock)

    failures: list[BaseException] = []

    def run_delete() -> None:
        try:
            account.delete_account(_command())
        except RuntimeError as error:
            failures.append(error)

    first = threading.Thread(target=run_delete, name="delete-a")
    second = threading.Thread(target=run_delete, name="delete-b")
    first.start()
    assert caller_a_inside.wait(timeout=1.0)
    second.start()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive() and not second.is_alive()
    assert [type(error) for error in failures] == [RuntimeError]
    assert events.index("delete-a:gate:False") < events.index("delete-b:gate:True")
    assert events.index("delete-b:gate:True") < events.index("delete-b:adopt")
    assert events.index("delete-b:adopt") < events.index("delete-b:delete")
    assert gate_closed, "successful caller owns the gate until its account delete"


# ------------------------------------------------------------- the workflow
def _command(user_id: str = "7", sessions=("s1", "s2")) -> DeleteAccountCommand:
    return DeleteAccountCommand(
        identity=JobIdentity(run_id="r1", user_id=user_id),
        authorization=AuthorizationScope(allowed_tools=frozenset(ToolName)),
        session_ids=tuple(sessions),
    )


class _Row:
    def __init__(self, rowcount: int, rows=(), row=None) -> None:
        self.rowcount = rowcount
        self._rows = rows
        self._row = row

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._row


class _Db:
    """Postgres as a log: which statements ran, and whether the user existed."""

    def __init__(self, user_exists: bool = True, session_ids=("s1",)) -> None:
        self.statements: list[str] = []
        self.user_exists = user_exists
        self.session_ids = tuple(session_ids)
        self.deletion_requested = False
        self.deleted: list[object] = []
        self.committed = False

    def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append(sql)
        if "SELECT session_id FROM sessions" in sql:
            return _Row(
                len(self.session_ids),
                [SimpleNamespace(session_id=sid) for sid in self.session_ids],
            )
        if "UPDATE users SET deletion_requested = :enabled" in sql:
            enabled = bool(params["enabled"])
            changed = self.user_exists and self.deletion_requested != enabled
            if changed:
                self.deletion_requested = enabled
            return _Row(int(changed))
        if "UPDATE users SET deletion_requested = TRUE" in sql:
            if self.user_exists:
                self.deletion_requested = True
                return _Row(1, row=SimpleNamespace(id=7))
            return _Row(0)
        return _Row(1 if self.user_exists else 0)

    def query(self, _model):
        return self

    def filter(self, *_a):
        return self

    def first(self):
        return object() if self.user_exists else None

    def delete(self, obj):
        self.deleted.append(obj)

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

    def close(self):
        pass


class _Chunkstore:
    def __init__(self, fail: bool = False) -> None:
        self.fail, self.dropped = fail, []

    def delete_index(self, *, index_name: str) -> int:
        if self.fail:
            raise RuntimeError("elasticsearch is down")
        self.dropped.append(index_name)
        return 3


@pytest.fixture
def stores(monkeypatch, tmp_path):
    """Every store the workflow touches, faked, with the order recorded."""
    import dataclasses
    import sys

    import visionagent.service.vectorstore as vs
    from visionagent.config.settings import settings
    from visionagent.database.postgres.upload_jobs import InMemoryUploadJobRepository
    from visionagent.pipeline import ingest

    order: list[str] = []
    db = _Db()
    chunks = _Chunkstore()

    def get_db():
        order.append("postgres")
        yield db

    def build_chunkstore():
        order.append("elasticsearch")
        return chunks

    monkeypatch.setattr(account, "get_db", get_db)
    monkeypatch.setattr(vs, "build_chunkstore", build_chunkstore)
    monkeypatch.setattr(ingest, "_JOB_REPOSITORY", InMemoryUploadJobRepository())
    # Settings is frozen; the workflow imports the module attribute at call
    # time, so a replaced copy on the module is what it sees.
    # (The package re-exports the instance under the module's own name, so the
    # module is reached through sys.modules rather than a dotted path.)
    isolated_settings = dataclasses.replace(
        settings,
        graph_dir=tmp_path / "graph",
        storage_dir=tmp_path / "uploads",
        state_dir=tmp_path / "state",
    )
    monkeypatch.setattr(
        sys.modules["visionagent.config.settings"], "settings", isolated_settings
    )
    monkeypatch.setattr(ingest, "settings", isolated_settings)
    (tmp_path / "graph").mkdir()
    (tmp_path / "uploads" / "session_context").mkdir(parents=True)
    (tmp_path / "graph" / "graph_7.graphml").write_text("<g/>")
    (tmp_path / "graph" / "vdb_nodes_7.json").write_text("{}")
    (tmp_path / "uploads" / "session_context" / "s1_context.json").write_text("{}")
    return {"order": order, "db": db, "chunks": chunks, "root": tmp_path}


def test_external_stores_go_first_and_postgres_last(stores):
    removed = account.delete_account(_command())

    assert stores["order"] == [
        "postgres",
        "postgres",
        "postgres",
        "elasticsearch",
        "postgres",
    ]
    assert stores["chunks"].dropped == ["7"]
    assert not (stores["root"] / "graph" / "graph_7.graphml").exists()
    assert not (stores["root"] / "graph" / "vdb_nodes_7.json").exists()
    assert not (stores["root"] / "uploads" / "session_context" / "s1_context.json").exists()
    assert stores["db"].committed and len(stores["db"].deleted) == 1
    assert removed["chunks"] == 3


def test_a_failing_index_drop_leaves_the_account_to_retry_from(stores):
    """The reverse order committed Postgres first, then logged the failure:
    the row was gone, the chunks were searchable by nobody, and there was no
    account left to run the deletion again."""
    stores["chunks"].fail = True

    with pytest.raises(RuntimeError, match="elasticsearch is down"):
        account.delete_account(_command())

    assert stores["order"] == [
        "postgres",
        "postgres",
        "postgres",
        "elasticsearch",
        "postgres",
    ]
    gate_updates = [sql for sql in stores["db"].statements if "deletion_requested" in sql]
    assert len(gate_updates) == 3, "adoption and failure must leave the admission gate open"
    assert "IS DISTINCT FROM" in gate_updates[0]
    assert stores["db"].deletion_requested is False
    assert (stores["root"] / "graph" / "graph_7.graphml").exists()
    assert stores["db"].deleted == [], "the account row must survive for a retry"


def test_a_second_deletion_is_a_no_op_not_an_error(stores):
    account.delete_account(_command())

    stores["db"].user_exists = False
    stores["db"].deleted.clear()
    again = account.delete_account(_command())

    assert again == {"sessions": 0, "messages": 0, "documents": 0, "chunks": 3}
    assert stores["db"].deleted == []


def test_a_retry_adopts_a_gate_left_closed_by_a_crashed_deletion(stores):
    stores["db"].deletion_requested = True

    removed = account.delete_account(_command())

    assert removed["sessions"] == 1
    assert len(stores["db"].deleted) == 1
    gate_updates = [
        sql for sql in stores["db"].statements if "deletion_requested" in sql
    ]
    assert "IS DISTINCT FROM" in gate_updates[0]
    assert "RETURNING id" in gate_updates[1]


def test_cleanup_uses_the_authoritative_post_gate_session_set(stores):
    root = stores["root"] / "uploads" / "session_context"
    (root / "late_context.json").write_text("{}")
    stores["db"].session_ids = ("late",)

    account.delete_account(_command(sessions=("s1",)))

    assert not (root / "s1_context.json").exists()
    assert not (root / "late_context.json").exists()
    assert any(
        "DELETE FROM messages AS message USING sessions AS session" in sql
        for sql in stores["db"].statements
    )


def test_account_deletion_erases_bytes_from_a_crashed_staging_ticket(stores):
    """A hard crash after fsync still leaves an owner-bearing row to erase."""
    from visionagent.pipeline import ingest

    process_id = "a" * 32
    job = IngestJob.model_validate(
        {
            "identity": {"run_id": "run-crashed-stage", "user_id": USER_ID},
            "authorization": {"can_upload": True},
            "file_names": ["private.txt"],
        }
    )
    files_data = [("private.txt", "private.txt", b"sensitive")]
    staged = ingest.plan_staged_files(files_data)
    repository = ingest._jobs()
    repository.create_staging(
        process_id,
        job,
        staged,
        ({"role": "upload_progress", "step": "upload"},),
        staging_node_id="crashed-node",
    )
    with ingest.upload_staging_lock(USER_ID):
        ingest.persist_staged_files(process_id, files_data, planned=staged)

    directory = stores["root"] / "state" / "upload_jobs" / process_id
    assert directory.is_dir()
    assert repository.get(process_id).status.value == "staging"

    account.delete_account(_command(user_id=USER_ID))

    assert not directory.exists()
    assert repository.get(process_id) is None
