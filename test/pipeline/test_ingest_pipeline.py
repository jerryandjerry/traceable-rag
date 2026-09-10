"""Document ingest, over HTTP: the ticket reaches the worker, the stages
arrive in order, and a failed stage is reported rather than swallowed.

The live suite covers the real parser and the real stores. What this tier
proves without them is the workflow's shape: the route mints a complete
IngestJob and hands it to the pipeline; the pipeline runs every file under
that job and reports per-file outcomes; the Postgres record carries the
error; and a graph-stage failure ends the batch as failed, not completed.

The background process is replaced by a thread so the fakes apply inside it.
"""
from __future__ import annotations

import asyncio
import dataclasses
import io
import os
import signal
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import pytest

from test.pipeline.conftest import USER_ID
from visionagent.models import IngestJob, IngestRun, ParsedChunk
from visionagent.pipeline import ingest


class _Thread:
    """multiprocessing.Process, in-process, so monkeypatches reach the worker."""

    def __init__(self, target, args) -> None:
        self._t = threading.Thread(target=target, args=args, daemon=True)
        self.pid = os.getpid()

    def start(self) -> None:
        self._t.start()

    def is_alive(self) -> bool:
        return self._t.is_alive()

    def kill(self) -> None:
        pass

    def join(self, timeout: float | None = None) -> None:
        self._t.join(timeout)


def _never_returning_item(*_args) -> None:
    while True:
        time.sleep(1)


def _item_exits_without_message(connection, *_args) -> None:
    connection.close()


class _ActiveLeaseStore:
    def owns_active_lease(self, process_id: str, owner: str) -> bool:
        return (process_id, owner) == ("process", "owner")


class _AcceptsUploads:
    def accepts_upload_work(self, _user_id: str) -> bool:
        return True


def test_item_execution_timeout_kills_spawned_native_work(monkeypatch, tmp_path):
    staged = tmp_path / "staged.txt"
    staged.write_bytes(b"body")
    monkeypatch.setattr(ingest, "_ITEM_EXECUTION_TARGET", _never_returning_item)
    monkeypatch.setattr(
        ingest,
        "settings",
        dataclasses.replace(ingest.settings, parse_timeout_s=0.2),
    )

    started = time.monotonic()
    with pytest.raises(ingest.ItemExecutionTimeout):
        ingest._execute_item_with_deadline(
            staged_path=staged,
            original_name="staged.txt",
            user_id=USER_ID,
            run_id="timeout-run",
            process_id="process",
            owner="owner",
            progress_callback=lambda *_args: None,
            cancelled=lambda: False,
        )
    assert time.monotonic() - started < 3


def test_item_child_exit_without_message_is_an_ambiguous_failure(
    monkeypatch, tmp_path
):
    staged = tmp_path / "staged.txt"
    staged.write_bytes(b"body")
    monkeypatch.setattr(ingest, "_ITEM_EXECUTION_TARGET", _item_exits_without_message)

    with pytest.raises(ingest.ItemExecutionFailure, match="without a result"):
        ingest._execute_item_with_deadline(
            staged_path=staged,
            original_name="staged.txt",
            user_id=USER_ID,
            run_id="exit-run",
            process_id="process",
            owner="owner",
            progress_callback=lambda *_args: None,
            cancelled=lambda: False,
        )


def test_item_isolation_spawn_failure_is_normalized(monkeypatch, tmp_path):
    real_context = ingest._ITEM_PROCESS_CONTEXT

    class _BrokenProcess:
        def start(self) -> None:
            raise OSError("process table full")

    class _BrokenContext:
        Pipe = staticmethod(real_context.Pipe)

        @staticmethod
        def Process(**_kwargs):
            return _BrokenProcess()

    monkeypatch.setattr(ingest, "_ITEM_PROCESS_CONTEXT", _BrokenContext())
    with pytest.raises(ingest.ItemExecutionFailure, match="could not start"):
        ingest._execute_item_with_deadline(
            staged_path=tmp_path / "staged.txt",
            original_name="staged.txt",
            user_id=USER_ID,
            run_id="spawn-run",
            process_id="process",
            owner="owner",
            progress_callback=lambda *_args: None,
            cancelled=lambda: False,
        )


def test_nonfinal_ambiguous_failure_compensates_before_checkpoint_reset(
    worker, monkeypatch
):
    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    job = IngestJob.model_validate({
        "identity": {"run_id": "ambiguous", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })
    process_id = ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])
    repository = ingest._jobs()
    active = repository.get(process_id)
    assert active is not None and active.lease_owner is not None
    order: list[str] = []
    original_retry = repository.retry_document
    monkeypatch.setattr(
        ingest,
        "_execute_item_with_deadline",
        lambda **_kwargs: (_ for _ in ()).throw(
            ingest.ItemExecutionFailure("provider exited")
        ),
    )
    monkeypatch.setattr(
        ingest,
        "_compensate_logical_document",
        lambda user_id, name, **_kwargs: order.append(
            f"compensate:{user_id}:{name}"
        ),
    )

    def retry_after_compensation(*args, **kwargs):
        order.append("reset")
        return original_retry(*args, **kwargs)

    monkeypatch.setattr(repository, "retry_document", retry_after_compensation)
    ingest.process_files_worker(process_id, active.lease_owner, ingest._UPLOAD_NODE_ID)

    assert order == [f"compensate:{USER_ID}:a.txt", "reset"]
    retry = repository.get(process_id)
    assert retry is not None and retry.status.value == "queued"
    assert retry.items[0].status.value == "pending"


def test_expired_cancel_at_attempt_budget_recovers_as_cancelled(
    worker, monkeypatch
):
    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    job = IngestJob.model_validate({
        "identity": {"run_id": "cancel-budget", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })
    process_id = ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])
    repository = ingest._jobs()
    active = repository.get(process_id)
    assert active is not None and active.lease_owner is not None
    repository._records[process_id] = active.model_copy(
        update={"max_attempts": 1}
    )
    assert repository.activate(process_id, active.lease_owner, os.getpid(), None, -1)
    assert repository.begin_item(process_id, active.lease_owner, 0)
    assert repository.request_cancel(process_id, USER_ID)[:2] == (True, True)
    assert repository.requeue_expired() == 1
    queued = repository.get(process_id)
    assert queued is not None
    repository._records[process_id] = queued.model_copy(
        update={"next_attempt_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    recovery_owner = "cancel-recovery"
    assert repository.reserve_next(
        recovery_owner, ingest._UPLOAD_NODE_ID, 45, process_id=process_id
    ) == process_id
    compensated: list[str] = []
    monkeypatch.setattr(
        ingest,
        "_compensate_logical_document",
        lambda _user_id, name, **_kwargs: compensated.append(name),
    )

    ingest.process_files_worker(process_id, recovery_owner, ingest._UPLOAD_NODE_ID)

    terminal = repository.get(process_id)
    assert terminal is not None
    assert terminal.status.value == "cancelled"
    assert terminal.attempt_count == 2
    assert all(item.status.value != "processing" for item in terminal.items)
    assert compensated == ["a.txt"]


def test_expired_inflight_item_at_budget_reconciles_then_fails(
    worker, monkeypatch
):
    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    job = IngestJob.model_validate({
        "identity": {"run_id": "failed-budget", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })
    process_id = ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])
    repository = ingest._jobs()
    active = repository.get(process_id)
    assert active is not None and active.lease_owner is not None
    repository._records[process_id] = active.model_copy(update={"max_attempts": 1})
    assert repository.activate(process_id, active.lease_owner, os.getpid(), None, -1)
    assert repository.begin_item(process_id, active.lease_owner, 0)
    assert repository.requeue_expired() == 1
    recovery_owner = "failure-recovery"
    assert repository.reserve_next(
        recovery_owner, ingest._UPLOAD_NODE_ID, 45, process_id=process_id
    ) == process_id
    compensated: list[str] = []
    monkeypatch.setattr(
        ingest,
        "_compensate_logical_document",
        lambda _user_id, name, **_kwargs: compensated.append(name),
    )

    ingest.process_files_worker(process_id, recovery_owner, ingest._UPLOAD_NODE_ID)

    terminal = repository.get(process_id)
    assert terminal is not None
    assert terminal.status.value == "failed"
    assert all(item.status.value != "processing" for item in terminal.items)
    assert compensated == ["a.txt"]


def test_compensation_refuses_to_delete_search_when_graph_fails_closed(
    monkeypatch,
):
    calls: list[str] = []

    class _Connection:
        def send(self, value) -> None:
            calls.append(f"result:{value}")

        def close(self) -> None:
            pass

    class _Graph:
        async def delete_file_data(self, _name):
            calls.append("graph")
            return {"success": False, "error": "ambiguous legacy aggregate"}

    class _Store:
        def delete_document(self, **_kwargs):
            calls.append("search")

    monkeypatch.setattr(ingest, "GraphRAGService", lambda _user_id: _Graph())
    monkeypatch.setattr(ingest, "UserRepository", _AcceptsUploads)
    monkeypatch.setattr(ingest, "_store", lambda: _Store())
    monkeypatch.setattr(ingest, "_jobs", lambda: _ActiveLeaseStore())
    monkeypatch.setattr(ingest, "tenant_lock", lambda _user_id: nullcontext())
    monkeypatch.setattr(ingest.signal, "setitimer", lambda *_args: None)

    ingest._isolated_compensation_main(
        _Connection(), USER_ID, "a.pdf", 1, "process", "owner"
    )

    assert calls[0] == "graph"
    assert "search" not in calls
    assert calls[-1] == "result:RuntimeError"


def test_compensation_retry_converges_after_search_failure(monkeypatch):
    graph_calls = 0
    search_calls = 0
    results: list[object] = []

    class _Connection:
        def send(self, value) -> None:
            results.append(value)

        def close(self) -> None:
            pass

    class _Graph:
        async def delete_file_data(self, _name):
            nonlocal graph_calls
            graph_calls += 1
            return {"success": True}

    class _Store:
        def delete_document(self, **_kwargs):
            nonlocal search_calls
            search_calls += 1
            if search_calls == 1:
                raise OSError("temporary search outage")

    monkeypatch.setattr(ingest, "GraphRAGService", lambda _user_id: _Graph())
    monkeypatch.setattr(ingest, "UserRepository", _AcceptsUploads)
    monkeypatch.setattr(ingest, "_store", lambda: _Store())
    monkeypatch.setattr(ingest, "_jobs", lambda: _ActiveLeaseStore())
    monkeypatch.setattr(ingest, "tenant_lock", lambda _user_id: nullcontext())
    monkeypatch.setattr(ingest.signal, "setitimer", lambda *_args: None)

    ingest._isolated_compensation_main(
        _Connection(), USER_ID, "a.pdf", 1, "process", "owner"
    )
    ingest._isolated_compensation_main(
        _Connection(), USER_ID, "a.pdf", 1, "process", "owner"
    )

    assert graph_calls == search_calls == 2
    assert results == ["OSError", None]


def test_compensation_does_not_recreate_stores_after_account_gate_closes(
    monkeypatch,
):
    results: list[object] = []

    class _Connection:
        def send(self, value) -> None:
            results.append(value)

        def close(self) -> None:
            pass

    class _DeletedAccount:
        def accepts_upload_work(self, _user_id):
            return False

    monkeypatch.setattr(ingest, "UserRepository", _DeletedAccount)
    monkeypatch.setattr(ingest, "_jobs", lambda: _ActiveLeaseStore())
    monkeypatch.setattr(ingest, "tenant_lock", lambda _user_id: nullcontext())
    monkeypatch.setattr(ingest.signal, "setitimer", lambda *_args: None)
    monkeypatch.setattr(
        ingest,
        "GraphRAGService",
        lambda _user_id: pytest.fail("graph store must not reopen"),
    )
    monkeypatch.setattr(
        ingest,
        "_store",
        lambda: pytest.fail("search store must not reopen"),
    )

    ingest._isolated_compensation_main(
        _Connection(), USER_ID, "a.pdf", 1, "process", "owner"
    )

    assert results == [None]


def test_stale_compensation_is_fenced_inside_tenant_lock(monkeypatch):
    """Owner A cannot delete data after owner B takes the reservation."""
    results: list[object] = []
    inside_lock = False

    class _Connection:
        def send(self, value) -> None:
            results.append(value)

        def close(self) -> None:
            pass

    @contextmanager
    def _lock(_user_id):
        nonlocal inside_lock
        inside_lock = True
        try:
            yield
        finally:
            inside_lock = False

    class _ReplacementOwnedJob:
        def owns_active_lease(self, process_id, owner):
            assert inside_lock
            assert (process_id, owner) == ("process", "owner-a")
            return False

    monkeypatch.setattr(ingest, "tenant_lock", _lock)
    monkeypatch.setattr(ingest, "_jobs", lambda: _ReplacementOwnedJob())
    monkeypatch.setattr(ingest.signal, "setitimer", lambda *_args: None)
    monkeypatch.setattr(
        ingest,
        "GraphRAGService",
        lambda _user_id: pytest.fail("stale compensation touched graph storage"),
    )
    monkeypatch.setattr(
        ingest,
        "_store",
        lambda: pytest.fail("stale compensation touched search storage"),
    )

    ingest._isolated_compensation_main(
        _Connection(), USER_ID, "a.pdf", 1, "process", "owner-a"
    )

    assert results == [ingest.UploadLeaseLost.__name__]


def test_stale_item_is_fenced_inside_tenant_lock(monkeypatch):
    results: list[object] = []
    inside_lock = False

    class _Connection:
        def send(self, value) -> None:
            results.append(value)

        def close(self) -> None:
            pass

    @contextmanager
    def _lock(_user_id):
        nonlocal inside_lock
        inside_lock = True
        try:
            yield
        finally:
            inside_lock = False

    class _ReplacementOwnedJob:
        def owns_active_lease(self, _process_id, _owner):
            assert inside_lock
            return False

    monkeypatch.setattr(ingest, "tenant_lock", _lock)
    monkeypatch.setattr(ingest, "_jobs", lambda: _ReplacementOwnedJob())
    monkeypatch.setattr(ingest.signal, "setitimer", lambda *_args: None)
    monkeypatch.setattr(
        ingest,
        "execute_insert_process_sync",
        lambda *_args, **_kwargs: pytest.fail("stale item executed"),
    )

    ingest._isolated_item_main(
        _Connection(),
        "/tmp/staged",
        "a.pdf",
        USER_ID,
        "run",
        1,
        "part",
        "process",
        "owner-a",
    )

    assert results == [("error", ingest.UploadLeaseLost.__name__)]


def test_stale_metadata_finalization_is_fenced_inside_tenant_lock(monkeypatch):
    results: list[object] = []

    class _Connection:
        def send(self, value) -> None:
            results.append(value)

        def close(self) -> None:
            pass

    class _ReplacementOwnedJob:
        def owns_active_lease(self, _process_id, _owner):
            return False

    monkeypatch.setattr(ingest, "tenant_lock", lambda _user_id: nullcontext())
    monkeypatch.setattr(ingest, "_jobs", lambda: _ReplacementOwnedJob())
    monkeypatch.setattr(ingest.signal, "setitimer", lambda *_args: None)
    monkeypatch.setattr(
        ingest,
        "insert_knowledgebase",
        lambda *_args: pytest.fail("stale metadata writer executed"),
    )

    ingest._isolated_metadata_main(
        _Connection(),
        (USER_ID, "a.pdf", 1, 0.1, None),
        1,
        "process",
        "owner-a",
    )

    assert results == [ingest.UploadLeaseLost.__name__]


@pytest.fixture
def worker(monkeypatch, tmp_path):
    """The stages, faked: what each file reports and what it returns."""
    seen: dict[str, Any] = {"runs": [], "records": [], "fail": None}

    def fake_execute(
        file_path,
        file_name,
        user_id,
        callback=None,
        *,
        run_id=None,
        tenant_lock_held=False,
        part_identity=None,
    ):
        assert tenant_lock_held
        seen["runs"].append({"file": file_name, "user": user_id, "run_id": run_id})
        for progress, message in (
            (0.1, "Starting document parsing"),
            (0.3, "Parsing complete, starting chunk processing..."),
            (0.5, "Processing chunks and generating embeddings"),
            (0.8, "Processing complete, starting database insertion"),
            (0.9, "Inserting into database"),
            (1.0, "File processing completed successfully"),
        ):
            callback(progress, message)
        run = IngestRun(run_id=run_id or "r", user_id=user_id, file_name=file_name,
                        indexed_count=4, entity_count=2, relation_count=1)
        if seen["fail"]:
            run.error = seen["fail"]
        return run

    def fake_record(user_id, file_name, total_chunks=None, process_time=None, error=None):
        seen["records"].append({"user": user_id, "file": file_name,
                                "chunks": total_chunks, "error": error})

    monkeypatch.setattr(ingest, "execute_insert_process_sync", fake_execute)
    monkeypatch.setattr(
        ingest,
        "_execute_item_with_deadline",
        lambda *, staged_path, original_name, user_id, run_id, process_id,
        owner, progress_callback, cancelled, part_identity=None: ingest.execute_insert_process_sync(
            staged_path,
            original_name,
            user_id,
            progress_callback,
            run_id=run_id,
            tenant_lock_held=True,
        ),
    )
    monkeypatch.setattr(ingest, "insert_knowledgebase", fake_record)
    monkeypatch.setattr(
        ingest,
        "_finalize_metadata_with_deadline",
        lambda arguments, cancelled=None, **_kwargs: ingest.insert_knowledgebase(
            *arguments
        ),
    )
    # This fixture replaces every external store with in-process fakes. A real
    # spawned compensation child cannot observe its in-memory lease record;
    # model the successful, fenced cleanup here. Tests of compensation and
    # lease transfer exercise the real child boundary separately above.
    monkeypatch.setattr(
        ingest,
        "_compensate_logical_document",
        lambda _user_id, _name, **_kwargs: None,
    )
    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _Thread)
    monkeypatch.setattr(ingest, "_LOCAL_UPLOAD_PROCESSES", {})
    monkeypatch.setattr(ingest, "_UPLOAD_SHUTDOWN_REQUESTED", threading.Event())
    from visionagent.database.postgres.upload_jobs import InMemoryUploadJobRepository

    monkeypatch.setattr(ingest, "_JOB_REPOSITORY", InMemoryUploadJobRepository())

    class _ExistingUser:
        def accepts_upload_work(self, user_id):
            return True

    monkeypatch.setattr(ingest, "UserRepository", _ExistingUser)
    monkeypatch.setattr(
        ingest,
        "settings",
        dataclasses.replace(
            ingest.settings,
            state_dir=tmp_path,
            graph_dir=tmp_path / "graph",
            upload_retry_backoff_s=0.0,
        ),
    )
    return seen


def _upload(client, name="notes.txt", body=b"some text"):
    r = client.post("/start-processing", files=[("files", (name, io.BytesIO(body), "text/plain"))])
    assert r.status_code == 200, r.text
    return r.json()["process_id"]


def _wait(client, process_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/process-status/{process_id}")
        assert r.status_code == 200, r.text
        status = r.json()
        if status["status"] in ("completed", "failed", "cancelled"):
            return status
        time.sleep(0.05)
    raise AssertionError("the worker never finished")


# ------------------------------------------------------------ the ticket
def test_the_complete_ticket_reaches_the_worker(client, worker):
    """Every file run retains the identity and run ID from its API-issued job."""
    process_id = _upload(client)
    assert uuid.UUID(process_id).hex == process_id, "the public process id must be an opaque UUID"
    status = _wait(client, process_id)

    assert status["status"] == "completed", status
    (run,) = worker["runs"]
    assert run["user"] == USER_ID
    assert run["run_id"] == status["run_id"], "the job's run id must reach every file"


def test_the_route_fills_in_the_file_names(client, worker, monkeypatch):
    seen: list[IngestJob] = []
    real = ingest.start_processing

    def spy(job, uploads):
        seen.append(job)
        return real(job, uploads)

    monkeypatch.setattr(ingest, "start_processing", spy)
    from visionagent.api.routes import file_upload_rt

    monkeypatch.setattr(file_upload_rt.ingest, "start_processing", spy)
    _wait(client, _upload(client, name="a.txt"))

    (job,) = seen
    assert isinstance(job, IngestJob)
    assert job.file_names == ("a.txt",)
    assert job.identity.user_id == USER_ID


# ------------------------------------------------------------- the stages
def test_the_stages_arrive_in_order_and_the_document_is_recorded(client, worker):
    process_id = _upload(client)
    status = _wait(client, process_id)

    steps = [m["step"] for m in status["progress"]]
    for earlier, later in (("upload", "parse_pending"), ("parse_pending", "parse_finish"),
                           ("parse_finish", "encode_pending"), ("encode_pending", "encode_finish"),
                           ("encode_finish", "database_pending"),
                           ("database_pending", "database_finish"),
                           ("database_finish", "complete")):
        assert steps.index(earlier) < steps.index(later), steps
    assert "error" not in steps

    (record,) = worker["records"]
    assert record == {"user": USER_ID, "file": "notes.txt", "chunks": 4, "error": None}


def test_the_progress_stream_carries_the_same_steps(client, worker):
    process_id = _upload(client)
    with client.stream("GET", f"/get-process-progress/{process_id}") as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert '"step": "complete"' in body
    assert '"step": "error"' not in body


# ------------------------------------------------------ a failed stage
def test_a_graph_failure_marks_the_document_and_fails_the_batch(client, worker):
    """The chunks are indexed and searchable, and the document is reported as
    partial: on the row, in the progress, and in the batch status. Before, the
    failure was printed and the batch said "completed successfully"."""
    worker["fail"] = "graph extraction failed: RuntimeError: model refused"
    process_id = _upload(client)
    status = _wait(client, process_id)

    assert status["status"] == "failed"
    errors = [m["message"] for m in status["progress"] if m["step"] == "error"]
    assert any("graph extraction failed" in e for e in errors), status["progress"]

    (record,) = worker["records"]
    assert record["chunks"] == 4, "the chunks were still indexed"
    assert record["error"] and "graph extraction failed" in record["error"]


def test_the_document_listing_carries_the_error(client, monkeypatch):
    from visionagent.api.routes import history_rt
    from visionagent.models import FilestResponse

    class _Repo:
        def list_for_user(self, user_id):
            return [FilestResponse(user_id=user_id, file_name="a.pdf", created_at="t",
                                   updated_at="t", total_chunks=4, process_time=1.0,
                                   error="graph extraction failed: x")]

    monkeypatch.setattr(history_rt, "KnowledgeBaseRepository", _Repo)
    r = client.get("/get_files/")
    assert r.status_code == 200
    assert r.json()[0]["error"] == "graph extraction failed: x"


# -------------------------------------------------------------- ownership
def test_another_user_cannot_read_the_progress(client, worker):
    process_id = _upload(client)
    r = client.get(f"/process-status/{process_id}", headers={"X-Test-User": "99"})
    assert r.status_code == 401


def test_unauthorized_upload_is_rejected_before_multipart_parsing(
    client, users, monkeypatch
):
    from starlette.requests import Request

    def must_not_parse(*_args, **_kwargs):
        raise AssertionError("an unauthorized body must not be parsed or spooled")

    users.exists = False
    monkeypatch.setattr(Request, "form", must_not_parse)
    response = client.post(
        "/start-processing",
        files=[("files", ("private.txt", io.BytesIO(b"private"), "text/plain"))],
    )

    assert response.status_code == 401


def test_upload_is_refused_cleanly_if_account_deletion_started(client, monkeypatch):
    """A ticket may be authorized just before the deletion gate is committed."""
    from visionagent.api.routes import file_upload_rt

    def refuse(*_args, **_kwargs):
        raise ingest.UploadAccountUnavailable("deletion gate is closed")

    monkeypatch.setattr(file_upload_rt.ingest, "start_processing", refuse)
    response = client.post(
        "/start-processing",
        files=[("files", ("notes.txt", io.BytesIO(b"text"), "text/plain"))],
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Account deletion is in progress; upload was not accepted"
    }


def test_owner_ticket_exists_and_is_not_dispatchable_before_bytes_are_written(
    worker, monkeypatch
):
    repository = ingest._jobs()
    observed: dict[str, Any] = {}
    real_persist = ingest.persist_staged_files

    class _DeferredProcess:
        pid = None

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    def inspect_persist(process_id, files_data, *, planned=None):
        ticket = repository.get(
            process_id, include_items=True, include_progress=False
        )
        assert ticket is not None
        observed["status"] = ticket.status.value
        observed["owner"] = ticket.user_id
        observed["directory_exists"] = ingest._job_directory(process_id).exists()
        observed["reserved"] = repository.reserve_next(
            "too-early", "node", 45, process_id=process_id
        )
        return real_persist(process_id, files_data, planned=planned)

    monkeypatch.setattr(ingest, "persist_staged_files", inspect_persist)
    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    job = IngestJob.model_validate(
        {
            "identity": {"run_id": "run-owner-ticket", "user_id": USER_ID},
            "authorization": {"can_upload": True},
            "file_names": ["private.txt"],
        }
    )

    process_id = ingest.start_processing(job, [("private.txt", io.BytesIO(b"secret"))])

    assert observed == {
        "status": "staging",
        "owner": USER_ID,
        "directory_exists": False,
        "reserved": None,
    }
    assert ingest.upload_job(process_id, progress=False).status.value == "processing"


def test_cancelled_staging_ticket_cannot_create_bytes(worker, monkeypatch):
    repository = ingest._jobs()
    lock_entries = 0

    @contextmanager
    def cancellation_wins(_user_id: str, *, timeout_s=None):
        nonlocal lock_entries
        lock_entries += 1
        if lock_entries == 1:
            (process_id,) = repository.active_ids_for_user(USER_ID)
            assert repository.request_cancel(process_id, USER_ID)[:2] == (True, True)
        yield

    def must_not_persist(*_args, **_kwargs):
        raise AssertionError("cancelled staging must not write bytes")

    monkeypatch.setattr(ingest, "upload_staging_lock", cancellation_wins)
    monkeypatch.setattr(ingest, "persist_staged_files", must_not_persist)
    job = IngestJob.model_validate(
        {
            "identity": {"run_id": "run-delete-wins", "user_id": USER_ID},
            "authorization": {"can_upload": True},
            "file_names": ["private.txt"],
        }
    )

    with pytest.raises(ingest.UploadAccountUnavailable):
        ingest.start_processing(job, [("private.txt", io.BytesIO(b"secret"))])

    (process_id,) = repository.terminal_ids_for_user(USER_ID)
    ticket = repository.get(process_id, include_items=False, include_progress=False)
    assert ticket is not None and ticket.status.value == "cancelled"
    assert ticket.staging_cleaned
    assert not ingest._job_directory(process_id).exists()


def test_gate_loss_after_fsync_removes_staged_bytes(worker, monkeypatch):
    repository = ingest._jobs()
    persisted: list[str] = []
    real_persist = ingest.persist_staged_files

    def persist_then_observe(process_id, files_data, *, planned=None):
        result = real_persist(process_id, files_data, planned=planned)
        assert ingest._job_directory(process_id).is_dir()
        persisted.append(process_id)
        return result

    monkeypatch.setattr(ingest, "persist_staged_files", persist_then_observe)
    monkeypatch.setattr(repository, "finalize_staging", lambda *_args: False)
    job = IngestJob.model_validate(
        {
            "identity": {"run_id": "run-gate-loss", "user_id": USER_ID},
            "authorization": {"can_upload": True},
            "file_names": ["private.txt"],
        }
    )

    with pytest.raises(ingest.UploadAccountUnavailable):
        ingest.start_processing(job, [("private.txt", io.BytesIO(b"secret"))])

    (process_id,) = persisted
    ticket = repository.get(process_id, include_items=False, include_progress=True)
    assert ticket is not None and ticket.status.value == "failed"
    assert ticket.staging_cleaned
    assert ticket.progress[-1].message == "Upload staging did not complete"
    assert not ingest._job_directory(process_id).exists()


def test_parent_reserves_the_job_before_spawning(worker, monkeypatch):
    repository = ingest._jobs()
    observed: dict[str, Any] = {}

    class _InspectProcess:
        pid = 987

        def __init__(self, target, args) -> None:
            process_id, owner, node_id = args
            record = repository.get(process_id)
            assert record is not None
            observed.update(
                status=record.status.value,
                owner=record.lease_owner,
                expected_owner=owner,
                node=record.worker_node_id,
                expected_node=node_id,
            )

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _InspectProcess)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-reserve", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })

    ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])

    assert observed["status"] == "processing"
    assert observed["owner"] == observed["expected_owner"]
    assert observed["node"] == observed["expected_node"]


def test_durable_upload_name_matches_postgres_character_limit() -> None:
    ingest.validate_durable_file_name("文" * 255)
    with pytest.raises(ingest.UploadNameInvalid):
        ingest.validate_durable_file_name("文" * 256)


@pytest.mark.parametrize("name", [" report.pdf", "report.pdf ", "a | b.pdf"])
def test_new_document_names_preserve_graph_identity(name: str) -> None:
    with pytest.raises(ingest.UploadNameInvalid):
        ingest.stage_files([(name, io.BytesIO(b"body"))])


def test_prepared_upload_cannot_bypass_graph_name_validation() -> None:
    upload = ingest.PreparedUpload(
        original_name="a | b.pdf",
        part_name="a | b.pdf",
        source=io.BytesIO(b"body"),
    )
    with pytest.raises(ingest.UploadNameInvalid):
        ingest.plan_staged_files([upload])


def test_planned_pdf_part_name_cannot_overflow_durable_schema() -> None:
    upload = ingest.PreparedUpload(
        original_name="a" * 251 + ".pdf",
        part_name="part_000_" + "a" * 251 + ".pdf",
        source=io.BytesIO(b"pdf"),
        start_page=0,
        end_page=1,
    )
    with pytest.raises(ingest.UploadNameInvalid):
        ingest.plan_staged_files([upload])


def test_dispatch_and_supervisor_leave_jobs_queued_at_node_capacity(
    worker, monkeypatch
):
    class _DeferredProcess:
        pid = 987

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    monkeypatch.setattr(
        ingest,
        "settings",
        dataclasses.replace(ingest.settings, max_upload_workers=1),
    )
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-capacity", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })

    first = ingest.start_processing(job, [("a.txt", io.BytesIO(b"first"))])
    second_job = job.model_copy(update={"file_names": ("b.txt",)})
    second = ingest.start_processing(second_job, [("b.txt", io.BytesIO(b"second"))])

    assert ingest.upload_job(first, progress=False).status.value == "processing"
    assert ingest.upload_job(second, progress=False).status.value == "queued"
    assert ingest.recover_upload_jobs() == 0
    assert ingest.upload_job(second, progress=False).status.value == "queued"


def test_active_document_name_is_an_immutable_upload_conflict(worker, monkeypatch):
    class _DeferredProcess:
        pid = 987

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-conflict", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })
    ingest.start_processing(job, [("a.txt", io.BytesIO(b"first"))])
    with pytest.raises(ingest.UploadDocumentConflict):
        ingest.start_processing(job, [("a.txt", io.BytesIO(b"changed"))])


def test_active_document_http_conflict_does_not_stage_second_copy(client, worker, monkeypatch):
    class _DeferredProcess:
        pid = 987

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    first = client.post(
        "/start-processing", files=[("files", ("same.txt", b"one", "text/plain"))]
    )
    assert first.status_code == 200
    root = ingest.settings.state_dir / "upload_jobs"
    before = set(root.iterdir())
    second = client.post(
        "/start-processing", files=[("files", ("same.txt", b"changed", "text/plain"))]
    )
    assert second.status_code == 409
    assert second.json()["detail"] == (
        "A document with this name already exists or is being processed"
    )
    assert set(root.iterdir()) == before


def test_duplicate_normalized_names_in_one_batch_are_rejected_before_bytes(client, worker) -> None:
    worker_settings = ingest.settings.state_dir / "upload_jobs"
    before = set(worker_settings.iterdir()) if worker_settings.exists() else set()
    response = client.post(
        "/start-processing",
        files=[
            ("files", ("caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt", b"one", "text/plain")),
            ("files", ("cafe\N{COMBINING ACUTE ACCENT}.txt", b"two", "text/plain")),
        ],
    )
    assert response.status_code == 409
    assert response.json()["detail"] == (
        "A document with this name already exists or is being processed"
    )
    after = set(worker_settings.iterdir()) if worker_settings.exists() else set()
    assert after == before
    with pytest.raises(ingest.UploadDocumentConflict):
        ingest.stage_files([
            ("caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt", io.BytesIO(b"one")),
            ("cafe\N{COMBINING ACUTE ACCENT}.txt", io.BytesIO(b"two")),
        ])


def test_spawn_failure_returns_the_reservation_to_the_queue(worker, monkeypatch):
    class _BrokenProcess:
        pid = None

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            raise OSError("process table full")

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _BrokenProcess)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-release", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })

    process_id = ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])
    record = ingest.upload_job(process_id, progress=False)

    assert record is not None
    assert record.status.value == "queued"
    assert record.lease_owner is None
    repository = ingest._jobs()
    repository._records[process_id] = record.model_copy(
        update={"next_attempt_at": datetime.now(UTC) - timedelta(seconds=1)}
    )

    class _DeferredProcess:
        pid = 456

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    assert ingest._dispatch_upload_worker(process_id=process_id)
    assert ingest.upload_job(process_id, progress=False).status.value == "processing"


def test_start_persists_one_nonduplicated_initial_progress_event(worker, monkeypatch):
    class _DeferredProcess:
        pid = None

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-initial-event", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })

    process_id = ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])
    record = ingest.upload_job(process_id)

    assert record is not None
    assert [event.step for event in record.progress] == ["upload"]
    assert record.progress[0].wire_dict() == {
        "id": record.progress[0].id,
        "role": "upload_progress",
        "step": "upload",
        "message": "Starting file processing...",
        "created_at": record.progress[0].wire_dict()["created_at"],
    }


def test_local_child_handles_are_reaped_and_closed(worker, monkeypatch):
    lifecycle: list[str] = []

    class _ExitedProcess:
        pid = 4567
        daemon = False

        def __init__(self, target, args) -> None:
            self.alive = False

        def start(self) -> None:
            lifecycle.append("start")

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout=None) -> None:
            lifecycle.append("join")

        def terminate(self) -> None:
            lifecycle.append("terminate")

        def kill(self) -> None:
            lifecycle.append("kill")

        def close(self) -> None:
            lifecycle.append("close")

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _ExitedProcess)

    assert ingest._launch_upload_worker("process", "owner", ingest._UPLOAD_NODE_ID) == 4567
    assert len(ingest._LOCAL_UPLOAD_PROCESSES) == 1
    assert ingest._reap_upload_workers() == 1
    assert ingest._LOCAL_UPLOAD_PROCESSES == {}
    assert lifecycle == ["start", "join", "close"]


def test_shutdown_terminates_then_force_kills_and_reaps_local_children(
    worker, monkeypatch
):
    lifecycle: list[str] = []

    class _StubbornProcess:
        pid = 7654
        daemon = False

        def __init__(self, target, args) -> None:
            self.alive = True

        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout=None) -> None:
            lifecycle.append("join")

        def terminate(self) -> None:
            lifecycle.append("terminate")

        def kill(self) -> None:
            lifecycle.append("kill")
            self.alive = False

        def close(self) -> None:
            lifecycle.append("close")

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _StubbornProcess)
    ingest._launch_upload_worker("process", "owner", ingest._UPLOAD_NODE_ID)

    assert ingest.shutdown_upload_workers(timeout_s=0) == 1
    assert lifecycle == ["terminate", "join", "kill", "join", "join", "close"]
    assert ingest._LOCAL_UPLOAD_PROCESSES == {}


def test_shutdown_gate_prevents_a_racing_dispatch_and_releases_its_reservation(
    worker, monkeypatch
):
    starts = 0

    class _NeverStarted:
        pid = None
        daemon = False

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            nonlocal starts
            starts += 1

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _NeverStarted)
    assert ingest.shutdown_upload_workers(timeout_s=0) == 0
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-during-shutdown", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })

    process_id = ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])

    record = ingest.upload_job(process_id, progress=False)
    assert record is not None
    assert record.status.value == "queued"
    assert record.lease_owner is None
    assert starts == 0


def test_sigterm_before_the_first_item_requeues_without_starting_it(worker, monkeypatch):
    from contextlib import contextmanager

    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    @contextmanager
    def signal_while_waiting(_user_id, **_kwargs):
        signal.raise_signal(signal.SIGTERM)
        yield

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    monkeypatch.setattr(ingest, "tenant_lock", signal_while_waiting)
    monkeypatch.setattr(
        ingest,
        "execute_insert_process_sync",
        lambda *_args, **_kwargs: pytest.fail("SIGTERM started a fresh item"),
    )
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-pre-item-stop", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })
    process_id = ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])
    repository = ingest._jobs()
    reservation = repository.get(process_id, include_items=True, include_progress=False)
    assert reservation is not None and reservation.lease_owner is not None

    ingest.process_files_worker(
        process_id, reservation.lease_owner, ingest._UPLOAD_NODE_ID
    )

    recovered = repository.get(process_id, include_items=True, include_progress=True)
    assert recovered is not None
    assert recovered.status.value == "queued"
    assert recovered.items[0].status.value == "pending"
    assert recovered.progress[-1].step == "recovery"
    assert not recovered.staging_cleaned


def test_worker_releases_tenant_lock_before_waiting_for_staging_cleanup(
    worker, monkeypatch
):
    """Preserve the global staging -> tenant lock order used by deletion."""

    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    tenant_held = False
    order: list[str] = []

    @contextmanager
    def tracked_tenant_lock(_user_id, **_kwargs):
        nonlocal tenant_held
        tenant_held = True
        order.append("tenant-enter")
        try:
            yield
        finally:
            tenant_held = False
            order.append("tenant-exit")

    def tracked_staging_cleanup(_process_id: str) -> bool:
        assert not tenant_held
        order.append("staging-cleanup")
        return True

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    monkeypatch.setattr(ingest, "tenant_lock", tracked_tenant_lock)
    monkeypatch.setattr(ingest, "_cleanup_one_terminal_staging", tracked_staging_cleanup)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-lock-order", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })
    process_id = ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])
    reservation = ingest.upload_job(process_id, progress=False)
    assert reservation is not None and reservation.lease_owner is not None

    ingest.process_files_worker(
        process_id,
        reservation.lease_owner,
        ingest._UPLOAD_NODE_ID,
    )

    assert order[-2:] == ["tenant-exit", "staging-cleanup"]
    assert order.count("tenant-enter") == order.count("tenant-exit")


def test_non_cancellation_sigterm_checkpoints_then_resumes_remaining_items(
    worker, monkeypatch
):
    import signal

    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    calls: list[str] = []

    def stop_during_first(file_path, file_name, user_id, callback=None, **kwargs):
        calls.append(file_name)
        if len(calls) == 1:
            signal.raise_signal(signal.SIGTERM)
        return IngestRun(
            run_id=kwargs["run_id"],
            user_id=user_id,
            file_name=file_name,
            indexed_count=1,
        )

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    monkeypatch.setattr(ingest, "execute_insert_process_sync", stop_during_first)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-host-stop", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt", "b.txt"],
    })
    process_id = ingest.start_processing(
        job, [("a.txt", io.BytesIO(b"first")), ("b.txt", io.BytesIO(b"second"))]
    )
    repository = ingest._jobs()
    first = repository.get(process_id, include_items=True, include_progress=False)
    assert first is not None and first.lease_owner is not None

    ingest.process_files_worker(process_id, first.lease_owner, ingest._UPLOAD_NODE_ID)

    queued = repository.get(process_id, include_items=True, include_progress=True)
    assert queued is not None
    assert queued.status.value == "queued"
    assert [item.status.value for item in queued.items] == ["completed", "pending"]
    assert queued.progress[-1].step == "recovery"
    assert calls == ["a.txt"]
    assert not queued.staging_cleaned

    second_owner = "host-recovery-owner"
    assert repository.reserve_next(second_owner, ingest._UPLOAD_NODE_ID, 45) == process_id
    ingest.process_files_worker(process_id, second_owner, ingest._UPLOAD_NODE_ID)

    completed = repository.get(process_id, include_items=True, include_progress=False)
    assert completed is not None
    assert completed.status.value == "completed"
    assert [item.status.value for item in completed.items] == ["completed", "completed"]
    assert calls == ["a.txt", "b.txt"]


def test_final_attempt_shutdown_compensates_and_refunds_attempt(
    worker, monkeypatch
):
    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-final-shutdown", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })
    process_id = ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])
    repository = ingest._jobs()
    reserved = repository.get(process_id, include_items=True, include_progress=False)
    assert reserved is not None and reserved.lease_owner is not None
    repository._records[process_id] = reserved.model_copy(
        update={"max_attempts": 1}
    )
    compensated: list[str] = []

    def stop_active_item(**kwargs):
        signal.raise_signal(signal.SIGTERM)
        assert kwargs["cancelled"]()
        raise ingest.UploadCancelled(process_id)

    monkeypatch.setattr(ingest, "_execute_item_with_deadline", stop_active_item)
    monkeypatch.setattr(
        ingest,
        "_compensate_logical_document",
        lambda _user_id, name, **_kwargs: compensated.append(name),
    )

    ingest.process_files_worker(
        process_id, reserved.lease_owner, ingest._UPLOAD_NODE_ID
    )

    queued = repository.get(process_id, include_items=True, include_progress=False)
    assert queued is not None
    assert queued.status.value == "queued"
    assert queued.attempt_count == 0
    assert queued.items[0].status.value == "pending"
    assert queued.items[0].indexed_count == 0
    assert compensated == ["a.txt"]
    assert not queued.staging_cleaned


def test_split_document_shutdown_then_cancel_compensates_all_siblings(
    worker, monkeypatch
):
    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    split_parts = [
        ingest.PreparedUpload(
            "split.pdf", "part_000_split.pdf", io.BytesIO(b"first part")
        ),
        ingest.PreparedUpload(
            "split.pdf", "part_001_split.pdf", io.BytesIO(b"second part")
        ),
    ]
    monkeypatch.setattr(ingest, "stage_files", lambda _uploads: split_parts)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-split-stop-cancel", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["split.pdf"],
    })
    process_id = ingest.start_processing(
        job, [("split.pdf", io.BytesIO(b"source PDF"))]
    )
    repository = ingest._jobs()
    first = repository.get(process_id, include_items=True, include_progress=False)
    assert first is not None and first.lease_owner is not None
    executions = 0

    def stop_after_first(**kwargs):
        nonlocal executions
        executions += 1
        signal.raise_signal(signal.SIGTERM)
        return IngestRun(
            run_id=kwargs["run_id"],
            user_id=kwargs["user_id"],
            file_name=kwargs["original_name"],
            indexed_count=3,
        )

    monkeypatch.setattr(ingest, "_execute_item_with_deadline", stop_after_first)
    compensated: list[str] = []
    monkeypatch.setattr(
        ingest,
        "_compensate_logical_document",
        lambda _user_id, name, **_kwargs: compensated.append(name),
    )

    ingest.process_files_worker(process_id, first.lease_owner, ingest._UPLOAD_NODE_ID)

    queued = repository.get(process_id, include_items=True, include_progress=False)
    assert queued is not None and queued.status.value == "queued"
    assert [item.status.value for item in queued.items] == ["completed", "pending"]
    assert worker["records"] == []
    assert repository.request_cancel(process_id, USER_ID)[:2] == (True, True)
    cancellation = repository.get(
        process_id, include_items=True, include_progress=False
    )
    assert cancellation is not None
    assert cancellation.status.value == "queued"
    assert cancellation.cancellation_requested
    recovery_owner = "split-cancel-recovery"
    assert repository.reserve_next(
        recovery_owner, ingest._UPLOAD_NODE_ID, 45, process_id=process_id
    ) == process_id

    ingest.process_files_worker(process_id, recovery_owner, ingest._UPLOAD_NODE_ID)

    terminal = repository.get(process_id, include_items=True, include_progress=False)
    assert terminal is not None and terminal.status.value == "cancelled"
    assert [item.status.value for item in terminal.items] == ["failed", "failed"]
    assert [item.indexed_count for item in terminal.items] == [0, 0]
    assert terminal.total_chunks_inserted == 0
    assert terminal.processed_files == terminal.total_files == 2
    assert executions == 1
    assert compensated == ["split.pdf"]
    assert worker["records"] == []


def test_queued_cancel_recovers_metadata_after_last_item_checkpoint(
    worker, monkeypatch
):
    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-metadata-gap", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })
    process_id = ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])
    repository = ingest._jobs()
    first = repository.get(process_id, include_items=True, include_progress=False)
    assert first is not None and first.lease_owner is not None
    assert repository.begin_item(process_id, first.lease_owner, 0)
    assert repository.finish_item(
        process_id,
        first.lease_owner,
        0,
        indexed_count=4,
        process_time=1.0,
        error=None,
    )
    assert repository.requeue_owned(
        process_id, first.lease_owner, {"step": "recovery"}
    )

    assert repository.request_cancel(process_id, USER_ID)[:2] == (True, True)
    pending = repository.get(process_id, include_items=True, include_progress=False)
    assert pending is not None
    assert pending.status.value == "queued"
    assert pending.cancellation_requested
    recovery_owner = "metadata-cancel-recovery"
    assert repository.reserve_next(
        recovery_owner, ingest._UPLOAD_NODE_ID, 45, process_id=process_id
    ) == process_id

    ingest.process_files_worker(process_id, recovery_owner, ingest._UPLOAD_NODE_ID)

    terminal = repository.get(process_id, include_items=True, include_progress=False)
    assert terminal is not None and terminal.status.value == "cancelled"
    assert terminal.items[0].status.value == "completed"
    assert terminal.items[0].indexed_count == 4
    assert terminal.total_chunks_inserted == 4
    assert [record["file"] for record in worker["records"]] == ["a.txt"]


def test_expired_processing_checkpoint_is_compensated_before_replay(
    worker, monkeypatch
):
    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-expired-replay", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })
    process_id = ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])
    repository = ingest._jobs()
    first = repository.get(process_id, include_items=True, include_progress=False)
    assert first is not None and first.lease_owner is not None
    assert repository.activate(process_id, first.lease_owner, os.getpid(), None, -1)
    assert repository.begin_item(process_id, first.lease_owner, 0)
    assert repository.requeue_expired() == 1
    expired = repository.get(process_id, include_items=True, include_progress=False)
    assert expired is not None
    assert expired.items[0].status.value == "processing"
    repository._records[process_id] = expired.model_copy(
        update={"next_attempt_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    recovery_owner = "crash-recovery"
    assert repository.reserve_next(
        recovery_owner, ingest._UPLOAD_NODE_ID, 45, process_id=process_id
    ) == process_id
    order: list[str] = []
    monkeypatch.setattr(
        ingest,
        "_compensate_logical_document",
        lambda _user_id, name, **_kwargs: order.append(f"compensate:{name}"),
    )
    monkeypatch.setattr(
        ingest,
        "_execute_item_with_deadline",
        lambda **kwargs: (
            order.append(f"execute:{kwargs['original_name']}")
            or IngestRun(
                run_id=kwargs["run_id"],
                user_id=kwargs["user_id"],
                file_name=kwargs["original_name"],
                indexed_count=4,
            )
        ),
    )

    ingest.process_files_worker(process_id, recovery_owner, ingest._UPLOAD_NODE_ID)

    completed = repository.get(process_id, include_items=True, include_progress=False)
    assert completed is not None and completed.status.value == "completed"
    assert order == ["compensate:a.txt", "execute:a.txt"]


def test_checkpoint_exception_never_terminalizes_ambiguous_item(
    worker, monkeypatch
):
    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-checkpoint-failure", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })
    process_id = ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])
    repository = ingest._jobs()
    first = repository.get(process_id, include_items=True, include_progress=False)
    assert first is not None and first.lease_owner is not None
    original_finish_item = repository.finish_item
    finish_calls = 0
    order: list[str] = []

    def fail_first_checkpoint(*args, **kwargs):
        nonlocal finish_calls
        finish_calls += 1
        if finish_calls == 1:
            raise OSError("checkpoint connection lost")
        return original_finish_item(*args, **kwargs)

    monkeypatch.setattr(repository, "finish_item", fail_first_checkpoint)
    monkeypatch.setattr(
        ingest,
        "_compensate_logical_document",
        lambda _user_id, name, **_kwargs: order.append(f"compensate:{name}"),
    )

    ingest.process_files_worker(process_id, first.lease_owner, ingest._UPLOAD_NODE_ID)

    retry = repository.get(process_id, include_items=True, include_progress=False)
    assert retry is not None
    assert retry.status.value == "queued"
    assert retry.items[0].status.value == "pending"
    assert retry.items[0].indexed_count == 0
    assert not retry.staging_cleaned
    assert finish_calls == 1
    assert order == ["compensate:a.txt"]
    repository._records[process_id] = retry.model_copy(
        update={"next_attempt_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    recovery_owner = "checkpoint-recovery"
    assert repository.reserve_next(
        recovery_owner, ingest._UPLOAD_NODE_ID, 45, process_id=process_id
    ) == process_id
    ingest.process_files_worker(process_id, recovery_owner, ingest._UPLOAD_NODE_ID)

    completed = repository.get(process_id, include_items=True, include_progress=False)
    assert completed is not None and completed.status.value == "completed"
    assert order == ["compensate:a.txt"]
    assert finish_calls == 2
    assert len(worker["runs"]) == 2


def test_cancellation_after_item_commit_finalizes_metadata_before_terminal(
    worker, monkeypatch
):
    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-cancel-boundary", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })
    process_id = ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])
    repository = ingest._jobs()
    reserved = repository.get(process_id, include_items=True, include_progress=False)
    assert reserved is not None and reserved.lease_owner is not None
    owner = reserved.lease_owner
    original_finish_item = repository.finish_item

    def finish_then_cancel(*args, **kwargs):
        finished = original_finish_item(*args, **kwargs)
        assert repository.request_cancel(process_id, USER_ID)[:2] == (True, True)
        return finished

    monkeypatch.setattr(repository, "finish_item", finish_then_cancel)
    ingest.process_files_worker(process_id, owner, ingest._UPLOAD_NODE_ID)

    terminal = repository.get(process_id, include_items=True, include_progress=True)
    assert terminal is not None
    assert terminal.status.value == "cancelled"
    assert terminal.items[0].status.value == "completed"
    assert terminal.staging_cleaned
    assert [record["file"] for record in worker["records"]] == ["a.txt"]
    assert [event.step for event in terminal.progress][-2:] == [
        "cancellation_pending",
        "cancelled",
    ]


def test_cancellation_winning_the_atomic_begin_starts_no_new_item(
    worker, monkeypatch
):
    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-cancel-before-begin", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })
    process_id = ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])
    repository = ingest._jobs()
    reserved = repository.get(process_id, include_items=True, include_progress=False)
    assert reserved is not None and reserved.lease_owner is not None
    original_begin = repository.begin_item

    def cancel_then_begin(pid: str, owner: str, sequence: int) -> bool:
        assert repository.request_cancel(process_id, USER_ID)[:2] == (True, True)
        return original_begin(pid, owner, sequence)

    monkeypatch.setattr(repository, "begin_item", cancel_then_begin)
    ingest.process_files_worker(
        process_id,
        reserved.lease_owner,
        ingest._UPLOAD_NODE_ID,
    )

    terminal = repository.get(process_id, include_items=True, include_progress=False)
    assert terminal is not None
    assert terminal.status.value == "cancelled"
    assert terminal.items[0].status.value == "failed"
    assert terminal.processed_files == terminal.total_files == 1
    assert worker["runs"] == []


def test_two_document_cancellation_preserves_only_finalized_document(
    worker, monkeypatch
):
    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-two-doc-cancel", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt", "b.txt"],
    })
    process_id = ingest.start_processing(
        job, [("a.txt", io.BytesIO(b"first")), ("b.txt", io.BytesIO(b"second"))]
    )
    repository = ingest._jobs()
    reserved = repository.get(process_id, include_items=True, include_progress=False)
    assert reserved is not None and reserved.lease_owner is not None
    calls: list[str] = []

    def execute_or_cancel(**kwargs):
        original_name = kwargs["original_name"]
        calls.append(f"execute:{original_name}")
        if original_name == "b.txt":
            assert repository.request_cancel(process_id, USER_ID)[:2] == (True, True)
            raise ingest.UploadCancelled(process_id)
        return IngestRun(
            run_id=kwargs["run_id"],
            user_id=kwargs["user_id"],
            file_name=original_name,
            indexed_count=4,
        )

    monkeypatch.setattr(ingest, "_execute_item_with_deadline", execute_or_cancel)
    monkeypatch.setattr(
        ingest,
        "_compensate_logical_document",
        lambda _user_id, name, **_kwargs: calls.append(f"compensate:{name}"),
    )

    ingest.process_files_worker(
        process_id, reserved.lease_owner, ingest._UPLOAD_NODE_ID
    )

    terminal = repository.get(process_id, include_items=True, include_progress=False)
    assert terminal is not None
    assert terminal.status.value == "cancelled"
    assert [item.status.value for item in terminal.items] == ["completed", "failed"]
    assert [item.indexed_count for item in terminal.items] == [4, 0]
    assert terminal.total_chunks_inserted == 4
    assert terminal.processed_files == terminal.total_files == 2
    assert [record["file"] for record in worker["records"]] == ["a.txt"]
    assert calls == ["execute:a.txt", "execute:b.txt", "compensate:b.txt"]


def test_two_document_final_attempt_failure_preserves_finalized_document(
    worker, monkeypatch
):
    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-two-doc-failure", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt", "b.txt"],
    })
    process_id = ingest.start_processing(
        job, [("a.txt", io.BytesIO(b"first")), ("b.txt", io.BytesIO(b"second"))]
    )
    repository = ingest._jobs()
    reserved = repository.get(process_id, include_items=True, include_progress=False)
    assert reserved is not None and reserved.lease_owner is not None
    repository._records[process_id] = reserved.model_copy(
        update={"max_attempts": 1}
    )
    calls: list[str] = []

    def execute_or_fail(**kwargs):
        original_name = kwargs["original_name"]
        calls.append(f"execute:{original_name}")
        if original_name == "b.txt":
            raise ingest.ItemExecutionFailure("child exited after commit")
        return IngestRun(
            run_id=kwargs["run_id"],
            user_id=kwargs["user_id"],
            file_name=original_name,
            indexed_count=4,
        )

    monkeypatch.setattr(ingest, "_execute_item_with_deadline", execute_or_fail)
    monkeypatch.setattr(
        ingest,
        "_compensate_logical_document",
        lambda _user_id, name, **_kwargs: calls.append(f"compensate:{name}"),
    )

    ingest.process_files_worker(
        process_id, reserved.lease_owner, ingest._UPLOAD_NODE_ID
    )

    terminal = repository.get(process_id, include_items=True, include_progress=False)
    assert terminal is not None
    assert terminal.status.value == "failed"
    assert [item.status.value for item in terminal.items] == ["completed", "failed"]
    assert [item.indexed_count for item in terminal.items] == [4, 0]
    assert terminal.total_chunks_inserted == 4
    assert terminal.processed_files == terminal.total_files == 2
    assert [record["file"] for record in worker["records"]] == ["a.txt"]
    assert calls == ["execute:a.txt", "execute:b.txt", "compensate:b.txt"]


def test_kill_route_distinguishes_pending_from_terminal_cancellation(
    client, worker, monkeypatch
):
    class _DeferredProcess:
        pid = None

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    process_id = _upload(client)
    repository = ingest._jobs()
    reservation = repository.get(process_id, include_items=False, include_progress=False)
    assert reservation is not None and reservation.lease_owner is not None
    assert repository.activate(process_id, reservation.lease_owner, 987654, "start", 45)
    monkeypatch.setattr(ingest, "_process_start_token", lambda _pid: "different")

    response = client.post(f"/kill-processing/{process_id}")

    assert response.status_code == 200
    assert response.json()["process_status"] == "cancellation_pending"
    assert "requested" in response.json()["message"].lower()
    pending = repository.get(process_id, include_items=False, include_progress=True)
    assert pending is not None
    assert pending.status.value == "processing"
    assert pending.progress[-1].step == "cancellation_pending"

    assert repository.settle_cancellations_for_user(USER_ID) == 1
    response = client.post(f"/kill-processing/{process_id}")
    assert response.status_code == 200
    assert response.json()["process_status"] == "cancelled"


def test_kill_route_does_not_claim_an_already_completed_job_was_cancelled(
    client, worker
):
    process_id = _upload(client)
    assert _wait(client, process_id)["status"] == "completed"

    response = client.post(f"/kill-processing/{process_id}")

    assert response.status_code == 200
    assert response.json()["process_status"] == "completed"
    assert "not applied" in response.json()["message"]


def test_cancelled_metadata_failure_retries_without_replaying_completed_item(
    worker, monkeypatch
):
    class _DeferredProcess:
        pid = os.getpid()

        def __init__(self, target, args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ingest, "_UPLOAD_PROCESS_FACTORY", _DeferredProcess)
    monkeypatch.setattr(ingest, "UPLOAD_LEASE_SECONDS", -1.0)
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-cancel-retry", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })
    process_id = ingest.start_processing(job, [("a.txt", io.BytesIO(b"body"))])
    repository = ingest._jobs()
    first = repository.get(process_id, include_items=True, include_progress=False)
    assert first is not None and first.lease_owner is not None
    first_owner = first.lease_owner
    original_finish_item = repository.finish_item
    original_metadata = ingest.insert_knowledgebase
    metadata_attempts = 0

    def finish_then_cancel(*args, **kwargs):
        finished = original_finish_item(*args, **kwargs)
        assert repository.request_cancel(process_id, USER_ID)[:2] == (True, True)
        return finished

    def flaky_metadata(*args, **kwargs):
        nonlocal metadata_attempts
        metadata_attempts += 1
        if metadata_attempts == 1:
            raise OSError("temporary metadata outage")
        return original_metadata(*args, **kwargs)

    monkeypatch.setattr(repository, "finish_item", finish_then_cancel)
    monkeypatch.setattr(ingest, "insert_knowledgebase", flaky_metadata)
    ingest.process_files_worker(process_id, first_owner, ingest._UPLOAD_NODE_ID)

    retryable = repository.get(process_id, include_items=True, include_progress=False)
    assert retryable is not None
    assert retryable.status.value == "processing"
    assert retryable.cancellation_requested
    assert retryable.items[0].status.value == "completed"
    assert not retryable.staging_cleaned
    assert len(worker["runs"]) == 1

    assert repository.requeue_expired() == 1
    retry = repository.get(process_id)
    assert retry is not None
    repository._records[process_id] = retry.model_copy(
        update={"next_attempt_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    second_owner = "retry-owner"
    assert repository.reserve_next(second_owner, ingest._UPLOAD_NODE_ID, -1.0) == process_id
    ingest.process_files_worker(process_id, second_owner, ingest._UPLOAD_NODE_ID)

    terminal = repository.get(process_id, include_items=True, include_progress=False)
    assert terminal is not None
    assert terminal.status.value == "cancelled"
    assert terminal.staging_cleaned
    assert metadata_attempts == 2
    assert len(worker["runs"]) == 1
    assert [record["file"] for record in worker["records"]] == ["a.txt"]


def test_cancellation_never_signals_a_pid_owned_by_another_node(worker, monkeypatch):
    repository = ingest._jobs()
    process_id = uuid.uuid4().hex
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-remote", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })
    from visionagent.models import StagedUpload

    repository.create(
        process_id,
        job,
        (StagedUpload(
            sequence=0,
            original_name="a.txt",
            part_name="a.txt",
            storage_key="0.txt",
        ),),
        ({"step": "upload"},),
        staging_node_id="remote-node",
    )
    assert repository.reserve_next("remote-owner", "remote-node", 45) == process_id
    assert repository.activate(process_id, "remote-owner", 4321, "same-token", 45)
    signals: list[tuple[int, int]] = []
    removed: list[str] = []
    monkeypatch.setattr(ingest, "_process_start_token", lambda _pid: "same-token")
    monkeypatch.setattr(ingest.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(
        ingest, "_remove_staged_files", lambda staged_id: removed.append(staged_id)
    )

    assert ingest.cancel_processing(process_id, USER_ID)
    assert signals == []
    assert removed == []
    assert repository.settle_cancellations_for_user(USER_ID) == 1
    assert ingest.cleanup_upload_jobs(USER_ID, require_all=True) == 1
    assert removed == [process_id]
    assert ingest.upload_job(process_id, progress=False) is None


def test_failed_staging_cleanup_keeps_the_durable_retry_handle(
    worker, monkeypatch
):
    process_id = uuid.uuid4().hex
    ingest.persist_staged_files(process_id, [("a.txt", "a.txt", b"sensitive")])
    repository = ingest._jobs()
    job = IngestJob.model_validate({
        "identity": {"run_id": "run-cleanup-failure", "user_id": USER_ID},
        "authorization": {"can_upload": True},
        "file_names": ["a.txt"],
    })
    from visionagent.models import StagedUpload

    repository.create(
        process_id,
        job,
        (StagedUpload(
            sequence=0,
            original_name="a.txt",
            part_name="a.txt",
            storage_key="000000-a.txt",
        ),),
        ({"step": "upload"},),
        staging_node_id=ingest._UPLOAD_NODE_ID,
    )
    assert repository.request_cancel(process_id, USER_ID)[:2] == (True, True)

    def cannot_remove(_process_id: str) -> None:
        raise OSError("read-only shared volume")

    monkeypatch.setattr(ingest, "_remove_staged_files", cannot_remove)
    assert ingest.cleanup_upload_jobs(USER_ID) == 0
    assert ingest.upload_job(process_id, progress=False) is not None
    with pytest.raises(RuntimeError, match="erase all durable upload staging"):
        ingest.cleanup_upload_jobs(USER_ID, require_all=True)


def test_supervisor_reclaims_old_untracked_staging(worker, tmp_path):
    process_id = uuid.uuid4().hex
    directory = tmp_path / "upload_jobs" / process_id
    directory.mkdir(parents=True)
    (directory / "000000-deadbeef.txt").write_bytes(b"sensitive")

    assert ingest.cleanup_orphan_staging(min_age_seconds=0) == 1
    assert not directory.exists()


def test_supervisor_expires_and_erases_a_crashed_owner_ticket(
    worker, monkeypatch
):
    process_id = uuid.uuid4().hex
    files_data = [("private.txt", "private.txt", b"sensitive")]
    staged = ingest.plan_staged_files(files_data)
    repository = ingest._jobs()
    job = IngestJob.model_validate(
        {
            "identity": {"run_id": "run-expired-stage", "user_id": USER_ID},
            "authorization": {"can_upload": True},
            "file_names": ["private.txt"],
        }
    )
    repository.create_staging(
        process_id,
        job,
        staged,
        ({"role": "upload_progress", "step": "upload"},),
        staging_node_id="crashed-node",
    )
    with ingest.upload_staging_lock(USER_ID):
        ingest.persist_staged_files(process_id, files_data, planned=staged)
    monkeypatch.setattr(ingest, "UPLOAD_ORPHAN_GRACE_SECONDS", 0.0)

    assert ingest.recover_upload_jobs() == 0

    record = repository.get(process_id, include_items=False, include_progress=True)
    assert record is not None and record.status.value == "failed"
    assert record.staging_cleaned
    assert record.progress[-1].message == "Upload staging expired before queueing"
    assert not ingest._job_directory(process_id).exists()


def test_orphan_janitor_preserves_fresh_untracked_staging(worker, tmp_path):
    process_id = uuid.uuid4().hex
    staged = ingest.persist_staged_files(
        process_id,
        [("fresh.txt", "fresh.txt", b"request still creating its durable row")],
    )

    assert staged
    assert ingest.cleanup_orphan_staging() == 0
    assert (tmp_path / "upload_jobs" / process_id).is_dir()


def test_worker_node_identity_is_independent_of_shared_state(monkeypatch, tmp_path):
    monkeypatch.delenv("VISIONAGENT_NODE_ID", raising=False)
    monkeypatch.setattr(ingest.socket, "gethostname", lambda: "runtime-node-a")
    monkeypatch.setattr(
        ingest,
        "settings",
        dataclasses.replace(ingest.settings, state_dir=tmp_path),
    )

    first = ingest._worker_node_id()
    monkeypatch.setattr(
        ingest,
        "settings",
        dataclasses.replace(ingest.settings, state_dir=tmp_path / "another-mount"),
    )
    second = ingest._worker_node_id()

    assert first == second
    assert not list(tmp_path.glob(".upload-node-id*"))


# -------------------------------------------------------------- the seams
def test_the_upload_route_reaches_the_pipeline_not_the_stores():
    from pathlib import Path

    src = Path(ingest.__file__).resolve().parents[1]
    route = (src / "api" / "routes" / "file_upload_rt.py").read_text(encoding="utf-8")
    assert "visionagent.pipeline" in route
    for store in ("service.vectorstore", "service.parsers", "database.", "multiprocessing",
                  "pypdf", "insert_knowledgebase"):
        assert store not in route, f"the route still reaches {store} directly"


def test_the_pipeline_writes_through_the_slot_not_the_vendor_client():
    from pathlib import Path

    src = Path(ingest.__file__).read_text(encoding="utf-8")
    for name in ("visionagent.vendor", "ESConnection"):
        assert name not in src, "ingest must write through ChunkStore, never ESConnection"
    assert "to_document" not in src, "the document shape is the store's, not the pipeline's"
    assert "execute_insert_process_fallback" not in src


def test_a_missing_document_is_reported_not_guessed(monkeypatch):
    """404 is the route's translation of DocumentNotFound, so the pipeline has
    to distinguish "no such document" from "nothing to delete"."""
    import asyncio

    class _Documents:
        def names_matching_canonical(self, _user_id, _name):
            return ()

    class _Users:
        def accepts_upload_work(self, _user_id):
            return True

    monkeypatch.setattr(ingest, "KnowledgeBaseRepository", _Documents)
    monkeypatch.setattr(ingest, "UserRepository", _Users)
    with pytest.raises(ingest.DocumentNotFound):
        asyncio.run(ingest.delete_document("42", "absent.pdf"))


def test_a_document_delete_drops_the_chunks_before_the_row(monkeypatch):
    """While the row stands the document is listed and the delete can be
    retried; the reverse order left an unlisted document's chunks behind."""
    import asyncio

    order: list[str] = []

    class _Documents:
        def names_matching_canonical(self, _user_id, name):
            return (name,)

        def delete_exact(self, _user_id, _name):
            order.append("postgres")
            return True

    class _Users:
        def accepts_upload_work(self, _user_id):
            return True

    class _Store:
        def delete_document(self, *, doc_name, index_name):
            order.append("elasticsearch")
            return 5

    class _Graph:
        def __init__(self, user_id):
            pass

        async def delete_file_data(self, file_name):
            order.append("graph")
            return {"success": True, "entities_deleted": 1, "relationships_deleted": 0}

    monkeypatch.setattr(ingest, "KnowledgeBaseRepository", _Documents)
    monkeypatch.setattr(ingest, "UserRepository", _Users)
    monkeypatch.setattr(ingest, "_store", lambda: _Store())
    monkeypatch.setattr(ingest, "GraphRAGService", _Graph)

    removed = asyncio.run(ingest.delete_document("42", "a.pdf"))

    assert order == ["graph", "elasticsearch", "postgres"]
    assert removed == {"chunks": 5, "entities": 1, "relations": 0}


def test_delete_resolves_legacy_nfd_name_and_uses_exact_storage_identity(monkeypatch):
    """A name returned by an old row remains deletable after NFC admission."""
    import asyncio

    legacy_name = "cafe\N{COMBINING ACUTE ACCENT}.pdf"
    used: list[tuple[str, str]] = []

    class _Documents:
        def names_matching_canonical(self, user_id, canonical_name):
            assert (user_id, canonical_name) == ("42", "caf\N{LATIN SMALL LETTER E WITH ACUTE}.pdf")
            return (legacy_name,)

        def delete_exact(self, user_id, stored_name):
            used.append(("postgres", stored_name))
            return user_id == "42"

    class _Users:
        def accepts_upload_work(self, _user_id):
            return True

    class _Store:
        def delete_document(self, *, doc_name, index_name):
            used.append(("elasticsearch", doc_name))
            return int(index_name == "42")

    class _Graph:
        def __init__(self, _user_id):
            pass

        async def delete_file_data(self, file_name):
            used.append(("graph", file_name))
            return {"success": True}

    monkeypatch.setattr(ingest, "KnowledgeBaseRepository", _Documents)
    monkeypatch.setattr(ingest, "UserRepository", _Users)
    monkeypatch.setattr(ingest, "_store", lambda: _Store())
    monkeypatch.setattr(ingest, "GraphRAGService", _Graph)

    asyncio.run(ingest.delete_document("42", legacy_name))

    assert used == [
        ("graph", legacy_name),
        ("elasticsearch", legacy_name),
        ("postgres", legacy_name),
    ]


def test_delete_fails_closed_on_legacy_canonical_name_collision(monkeypatch):
    import asyncio

    class _Documents:
        def names_matching_canonical(self, _user_id, _canonical_name):
            return ("cafe\N{COMBINING ACUTE ACCENT}.pdf", "caf\N{LATIN SMALL LETTER E WITH ACUTE}.pdf")

    class _Users:
        def accepts_upload_work(self, _user_id):
            return True

    class _Graph:
        def __init__(self, _user_id):
            raise AssertionError("ambiguous identity must not touch graph storage")

    monkeypatch.setattr(ingest, "KnowledgeBaseRepository", _Documents)
    monkeypatch.setattr(ingest, "UserRepository", _Users)
    monkeypatch.setattr(ingest, "GraphRAGService", _Graph)

    with pytest.raises(ingest.DocumentNameAmbiguous):
        asyncio.run(ingest.delete_document("42", "caf\N{LATIN SMALL LETTER E WITH ACUTE}.pdf"))


def test_delete_rechecks_account_gate_after_tenant_lock(monkeypatch):
    """A delete waiting behind account erasure cannot recreate graph files."""
    import asyncio

    inside_lock = False

    @contextmanager
    def _lock(*_args, **_kwargs):
        nonlocal inside_lock
        inside_lock = True
        try:
            yield
        finally:
            inside_lock = False

    class _Users:
        def accepts_upload_work(self, _user_id):
            assert inside_lock
            return False

    class _Documents:
        def names_matching_canonical(self, *_args):
            raise AssertionError("an erased account must not resolve documents")

    class _Graph:
        def __init__(self, _user_id):
            raise AssertionError("an erased account must not recreate graph files")

    monkeypatch.setattr(ingest, "tenant_lock", _lock)
    monkeypatch.setattr(ingest, "UserRepository", _Users)
    monkeypatch.setattr(ingest, "KnowledgeBaseRepository", _Documents)
    monkeypatch.setattr(ingest, "GraphRAGService", _Graph)

    with pytest.raises(ingest.DocumentNotFound):
        asyncio.run(ingest.delete_document("42", "a.pdf"))


# ------------------------------------------------------- truthful outcomes
class _GraphOk:
    def __init__(self, user_id):
        pass

    async def process_chunks_for_graphrag(self, chunks, file_name, callback=None):
        return {"success": True, "entities": 2, "relationships": 1}


class _GraphDown(_GraphOk):
    async def process_chunks_for_graphrag(self, chunks, file_name, callback=None):
        raise RuntimeError("DashScope at https://secret-host refused: 401")


class _Store:
    def __init__(self, accept: int | None = None) -> None:
        self.accept = accept

    def index(self, *, chunks, index_name, doc_name, progress=None):
        return len(chunks) if self.accept is None else self.accept

    def delete_document(self, *, doc_name, index_name):
        return 0


@pytest.fixture
def stages(monkeypatch):
    """parse and the stores faked; the pipeline's own decisions are real."""
    monkeypatch.setattr(ingest, "parse", lambda name, path, callback=None: [
        ParsedChunk(id="parser-1", content="first chunk of text"),
        ParsedChunk(id="parser-2", content="second chunk of text"),
    ])
    monkeypatch.setattr(ingest, "GraphRAGService", _GraphOk)
    monkeypatch.setattr(ingest, "_store", lambda: _Store())
    return monkeypatch


def test_a_graph_provider_failure_is_recorded_without_its_text(stages):
    """Provider failure is visible without persisting sensitive exception text."""
    stages.setattr(ingest, "GraphRAGService", _GraphDown)
    run = ingest.execute_insert_process_sync("/tmp/x.pdf", "x.pdf", "42")
    assert run.error == "graph extraction failed"
    assert "secret-host" not in run.error
    assert run.indexed_count == 2, "the chunks are still indexed"


def test_a_graph_persistence_failure_aborts_before_search_indexing(stages):
    from visionagent.database.graph import GraphPersistenceError

    class _GraphDiskFailure(_GraphOk):
        async def process_chunks_for_graphrag(self, chunks, file_name, callback=None):
            raise GraphPersistenceError("partly published graph snapshot")

    class MustNotIndex(_Store):
        def index(self, *, chunks, index_name, doc_name, progress=None):
            raise AssertionError("search indexing must wait for graph reconciliation")

    stages.setattr(ingest, "GraphRAGService", _GraphDiskFailure)
    stages.setattr(ingest, "_store", lambda: MustNotIndex())
    with pytest.raises(GraphPersistenceError, match="partly published"):
        ingest.execute_insert_process_sync("/tmp/x.pdf", "x.pdf", "42")


def test_fewer_accepted_chunks_than_prepared_is_not_a_success(stages):
    stages.setattr(ingest, "_store", lambda: _Store(accept=1))
    run = ingest.execute_insert_process_sync("/tmp/x.pdf", "x.pdf", "42")
    assert run.error == "indexed 1 of 2 chunks"


def test_chunk_identity_separates_documents_occurrences_and_retries(stages):
    captured: list[list[str]] = []

    class CaptureStore(_Store):
        def index(self, *, chunks, index_name, doc_name, progress=None):
            captured.append([chunk.id for chunk in chunks])
            return len(chunks)

    stages.setattr(ingest, "parse", lambda name, path, callback=None: [
        ParsedChunk(id="parser-a", content="identical boilerplate"),
        ParsedChunk(id="parser-b", content="identical boilerplate"),
    ])
    stages.setattr(ingest, "_store", lambda: CaptureStore())
    ingest.execute_insert_process_sync(
        "/tmp/staged-a", "first.pdf", "42", part_identity="part-0"
    )
    ingest.execute_insert_process_sync(
        "/tmp/staged-b", "second.pdf", "42", part_identity="part-0"
    )
    ingest.execute_insert_process_sync(
        "/tmp/retry-path", "first.pdf", "42", part_identity="part-0"
    )

    assert captured[0][0] != captured[0][1]
    assert set(captured[0]).isdisjoint(captured[1])
    assert captured[2] == captured[0]


def test_the_graph_service_propagates_the_provider_failure(monkeypatch):
    import asyncio

    import visionagent.service.vectorstore.graphstore.service as svc

    class _Repo:
        working_dir = "/tmp"
        node_vdb = edge_vdb = knowledge_graph = None

    class _Llm:
        async def complete(self, **kw):
            raise RuntimeError("provider down")

    monkeypatch.setattr(svc, "_llm", lambda: _Llm())
    service = svc.GraphRAGService("42", repository=_Repo())
    with pytest.raises(RuntimeError, match="provider down"):
        asyncio.run(service._call_llm("extract"))


def test_the_graph_delete_reports_what_failed(monkeypatch):
    """A graph mutation failure propagates; it is never reported as success."""
    import asyncio

    import visionagent.service.vectorstore.graphstore.service as svc

    class _Broken:
        metadatas: ClassVar[list] = []

        async def delete_by_doc_id(self, doc_id):
            raise OSError("read-only file system")

    class _Graph:
        class graph:
            @staticmethod
            def nodes(data=True):
                return []

            @staticmethod
            def edges(data=True):
                return []

        async def remove_document(self, doc_id):
            raise OSError("read-only file system")

    class _Repo:
        working_dir = "/tmp"
        node_vdb = _Broken()
        edge_vdb = _Broken()
        knowledge_graph = _Graph()

        def save(self):
            raise OSError("read-only file system")

    with pytest.raises(OSError, match="read-only file system"):
        asyncio.run(svc.GraphRAGService("42", repository=_Repo()).delete_file_data("a.pdf"))


def test_a_failed_graph_delete_keeps_the_row(monkeypatch):
    """The row is the retry handle. Removing it after a failed cleanup left
    graph data with nothing pointing at it."""
    import asyncio

    deleted: list[str] = []

    class _Documents:
        def names_matching_canonical(self, _user_id, name):
            return (name,)

        def delete_exact(self, _user_id, name):
            deleted.append(name)
            return True

    class _Users:
        def accepts_upload_work(self, _user_id):
            return True

    class _Graph:
        def __init__(self, user_id):
            pass

        async def delete_file_data(self, file_name):
            return {"success": False, "error": "save: read-only file system"}

    monkeypatch.setattr(ingest, "KnowledgeBaseRepository", _Documents)
    monkeypatch.setattr(ingest, "UserRepository", _Users)
    monkeypatch.setattr(ingest, "_store", lambda: _Store())
    monkeypatch.setattr(ingest, "GraphRAGService", _Graph)
    with pytest.raises(RuntimeError, match="graph cleanup failed"):
        asyncio.run(ingest.delete_document("42", "a.pdf"))
    assert deleted == []


# ------------------------------------------------------- page coverage
def _blank_pdf(pages: int) -> bytes:
    import io

    from pypdf import PdfWriter

    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


@pytest.mark.parametrize("pages,chars_per_page", [(10, 100), (7, 300), (13, 1000), (1, 5000), (25, 60)])
def test_every_page_appears_exactly_once_across_the_parts(
    worker, monkeypatch, pages, chars_per_page
):
    """Floor division left the remainder unassigned: a 10-page file split
    four ways emitted pages 1-8 only."""
    from pypdf import PdfReader

    import visionagent.pipeline.ingest as mod

    class _Page:
        def extract_text(self):
            return "x" * chars_per_page

    class _Pdf:
        def __init__(self) -> None:
            self.pages = [_Page() for _ in range(pages)]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import pdfplumber

    monkeypatch.setattr(pdfplumber, "open", lambda path: _Pdf())
    monkeypatch.setattr(mod, "CHAR_LIMIT", 250)

    parts = mod.stage_files([("doc.pdf", _blank_pdf(pages))])
    assert all(part.original_name == "doc.pdf" for part in parts)
    assert all(part.source is parts[0].source for part in parts)
    process_id = uuid.uuid4().hex
    staged = mod.persist_staged_files(process_id, parts)
    counted = sum(
        len(PdfReader(mod._staged_path(process_id, item)).pages) for item in staged
    )
    assert counted == pages, f"{pages} pages in, {counted} out across {len(parts)} part(s)"


# ------------------------------------------------------- the account is gone
def test_a_worker_stops_when_the_account_no_longer_exists(client, worker, monkeypatch):
    """An upload authorized just before the account was deleted must not
    recreate the index, the graph or the row."""
    class _DeletedAfterQueue:
        checks = 0

        def accepts_upload_work(self, user_id):
            type(self).checks += 1
            return type(self).checks == 1

    monkeypatch.setattr(ingest, "UserRepository", _DeletedAfterQueue)
    status = _wait(client, _upload(client))
    assert status["status"] == "failed"
    assert any("no longer exists" in m["message"] for m in status["progress"])
    assert worker["runs"] == [], "no file was processed"
    assert worker["records"] == [], "no row was written"


def test_upload_errors_reach_the_client_as_a_reference_not_a_traceback(client, worker, monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("psycopg2 could not connect to db-host:5432 (password=hunter2)")

    from visionagent.database.postgres import upload_jobs

    monkeypatch.setattr(
        upload_jobs,
        "settings",
        dataclasses.replace(upload_jobs.settings, max_upload_attempts=1),
    )
    monkeypatch.setattr(ingest, "execute_insert_process_sync", explode)
    status = _wait(client, _upload(client))
    assert status["status"] == "failed"
    text = " ".join(m["message"] for m in status["progress"])
    assert "db-host" not in text and "hunter2" not in text
    assert "Reference:" in text
    assert worker["records"] == []


# ------------------------------------------------------- server-side limits
@pytest.fixture
def small_limits(monkeypatch):
    import dataclasses

    from visionagent.api.routes import file_upload_rt
    from visionagent.config.settings import settings

    monkeypatch.setattr(
        file_upload_rt,
        "settings",
        dataclasses.replace(
            settings,
            max_upload_files=2,
            max_upload_bytes=64,
            max_upload_total_bytes=96,
        ),
    )


def test_too_many_files_are_refused_before_anything_is_read(client, worker, small_limits):
    files = [("files", (f"f{i}.txt", io.BytesIO(b"x"), "text/plain")) for i in range(3)]
    r = client.post("/start-processing", files=files)
    assert r.status_code == 413
    assert worker["runs"] == []


def test_an_oversized_file_is_refused(client, worker, small_limits):
    r = client.post("/start-processing",
                    files=[("files", ("big.txt", io.BytesIO(b"x" * 65), "text/plain"))])
    assert r.status_code == 413
    assert worker["runs"] == []


def test_an_oversized_aggregate_is_refused(client, worker, small_limits):
    files = [
        ("files", ("a.txt", io.BytesIO(b"a" * 60), "text/plain")),
        ("files", ("b.txt", io.BytesIO(b"b" * 60), "text/plain")),
    ]
    response = client.post("/start-processing", files=files)

    assert response.status_code == 413
    assert response.json() == {"detail": "upload exceeds 96 aggregate bytes"}
    assert worker["runs"] == []


def test_upload_rejects_any_multipart_key_other_than_files(
    client, worker, small_limits
):
    response = client.post(
        "/start-processing",
        files=[("attachment", ("a.txt", io.BytesIO(b"x"), "text/plain"))],
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "only the files multipart field is accepted"
    }
    assert worker["runs"] == []


def test_route_hands_the_pipeline_rewound_spooled_streams(
    client, worker, small_limits, monkeypatch
):
    from visionagent.api.routes import file_upload_rt

    observed: list[tuple[str, int, bytes, bool]] = []
    observed_sources: list[Any] = []

    def inspect(job, uploads):
        for name, source in uploads:
            observed_sources.append(source)
            position = source.tell()
            content = source.read(1 << 20)
            observed.append((name, position, content, isinstance(source, bytes)))
        return "a" * 32

    monkeypatch.setattr(file_upload_rt.ingest, "start_processing", inspect)
    response = client.post(
        "/start-processing",
        files=[("files", ("a.txt", io.BytesIO(b"streamed"), "text/plain"))],
    )

    assert response.status_code == 200
    assert observed == [("a.txt", 0, b"streamed", False)]
    assert observed_sources[0].closed, "FastAPI retains ownership of the request spool"


def test_route_cancellation_waits_until_the_staging_thread_releases_the_spool(
    monkeypatch,
):
    from starlette.datastructures import UploadFile

    from visionagent.api.routes import file_upload_rt

    entered = threading.Event()
    release = threading.Event()
    upload = UploadFile(file=io.BytesIO(b"streamed"), filename="a.txt", size=8)
    job = IngestJob.model_validate(
        {
            "identity": {"run_id": "run-cancelled-request", "user_id": USER_ID},
            "authorization": {"can_upload": True},
        }
    )

    def slow_start(_job, uploads):
        source = uploads[0][1]
        assert source.tell() == 0
        entered.set()
        assert release.wait(2)
        assert not source.closed
        return "a" * 32

    monkeypatch.setattr(file_upload_rt.ingest, "start_processing", slow_start)

    async def cancel_during_handoff() -> None:
        task = asyncio.create_task(file_upload_rt._start_file_processing(job, [upload]))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not upload.file.closed
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await upload.close()

    asyncio.run(cancel_during_handoff())


# ------------------------------------------------------- one writer per user
def test_the_tenant_lock_serialises_writers_and_bounds_the_wait(monkeypatch, tmp_path):
    import dataclasses
    import sys
    import threading
    import time

    from visionagent.config.settings import settings
    from visionagent.database.graph import tenant_lock

    monkeypatch.setattr(sys.modules["visionagent.database.graph.repository"], "settings",
                        dataclasses.replace(settings, graph_dir=tmp_path))

    order: list[str] = []
    held = threading.Event()

    def first():
        with tenant_lock("7"):
            held.set()
            order.append("first-in")
            time.sleep(0.5)
            order.append("first-out")

    t = threading.Thread(target=first)
    t.start()
    held.wait(2)
    with pytest.raises(TimeoutError), tenant_lock("7", timeout_s=0.1):
        pass
    with tenant_lock("7"):
        order.append("second-in")
    t.join()
    assert order == ["first-in", "first-out", "second-in"]
