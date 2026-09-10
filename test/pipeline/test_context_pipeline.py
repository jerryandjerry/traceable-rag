"""Attached session context over HTTP: add, read, remove, and clear.

These tests drive the real route, pipeline, and store against a temporary
directory; only the VLM extractor is replaced.
"""
from __future__ import annotations

import asyncio
import io
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from test.pipeline.conftest import SESSION_ID, USER_ID
from visionagent.pipeline import context


@pytest.fixture
def store(monkeypatch, tmp_path):
    """The one context store, rooted in a temporary directory."""
    import visionagent.database.graph.repository as graph_repository
    from visionagent.database.session_context import session_context_manager

    storage_dir = tmp_path / "uploads"
    context_dir = storage_dir / "session_context"
    monkeypatch.setattr(session_context_manager, "context_dir", str(context_dir))
    context_dir.mkdir(parents=True)
    monkeypatch.setattr(
        graph_repository,
        "settings",
        type("Settings", (), {"graph_dir": tmp_path / "graph"})(),
    )

    # The route stages the upload under settings.storage_dir; point it at the
    # temporary directory too (Settings is frozen, so a replaced copy).
    import dataclasses

    from visionagent.api.routes import add_context_rt
    from visionagent.config.settings import settings

    test_settings = dataclasses.replace(settings, storage_dir=storage_dir)
    monkeypatch.setattr(add_context_rt, "settings", test_settings)
    return session_context_manager


@pytest.fixture
def vlm(monkeypatch):
    async def extract(path):
        return {"success": True, "content": f"text of {path.rsplit('/', 1)[-1]}",
                "pages_processed": 1}

    monkeypatch.setattr(
        context, "vlm_processor", type("V", (), {"process_file": staticmethod(extract)})()
    )


def _add(client, name="a.pdf"):
    return client.post(
        f"/add_context/?session_id={SESSION_ID}",
        files=[("files", (name, io.BytesIO(b"%PDF-1.4"), "application/pdf"))],
    )


# ---------------------------------------------------------------- the journey
def test_add_then_read_then_remove_then_clear(client, store, vlm):
    r = _add(client, "a.pdf")
    assert r.status_code == 200, r.text
    assert r.json()["timing"]["individual_files"][0]["status"] == "success"
    _add(client, "b.pdf")

    r = client.get(f"/get_context/{SESSION_ID}")
    assert r.status_code == 200
    assert [f["file_name"] for f in r.json()["files"]] == ["a.pdf", "b.pdf"]
    assert r.json()["context_length"] > 0
    assert "text of a.pdf" in store.get_session_context(SESSION_ID)

    r = client.delete(f"/remove_file/{SESSION_ID}?file_name=a.pdf")
    assert r.status_code == 200
    raw_root = Path(store.context_dir).parent / "context_files" / SESSION_ID
    assert not (raw_root / "a.pdf").exists()
    assert [f["file_name"] for f in client.get(f"/get_context/{SESSION_ID}").json()["files"]] == ["b.pdf"]
    assert "text of a.pdf" not in store.get_session_context(SESSION_ID)

    r = client.delete(f"/remove_file/{SESSION_ID}?file_name=a.pdf")
    assert r.status_code == 404, "removing what is not attached is a 404, not a silent 200"

    r = client.delete(f"/clear_context/{SESSION_ID}")
    assert r.status_code == 200
    assert not (raw_root / "b.pdf").exists()
    assert client.get(f"/get_context/{SESSION_ID}").json()["files"] == []


def test_corrupt_context_storage_is_not_treated_as_empty_or_absent(client, store):
    Path(store._get_session_file(SESSION_ID)).write_text("{broken", encoding="utf-8")

    assert client.get(f"/get_context/{SESSION_ID}").status_code == 500
    assert client.delete(
        f"/remove_file/{SESSION_ID}?file_name=a.pdf"
    ).status_code == 500


def test_context_filename_policy_is_bounded_and_path_safe(client, vlm):
    too_long = "a" * 252 + ".pdf"
    traversal = "../secret.pdf"

    assert _add(client, too_long).status_code == 422
    assert _add(client, traversal).status_code == 422


def test_a_foreign_session_is_refused_on_every_operation(client, store, vlm):
    for method, path in (
        ("POST", "/add_context/?session_id=not-mine"),
        ("GET", "/get_context/not-mine"),
        ("DELETE", "/remove_file/not-mine?file_name=a.pdf"),
        ("DELETE", "/clear_context/not-mine"),
    ):
        kwargs = {"files": [("files", ("a.pdf", io.BytesIO(b"x"), "application/pdf"))]} \
            if method == "POST" else {}
        r = client.request(method, path, **kwargs)
        assert r.status_code == 404, (method, path, r.status_code)


def test_foreign_session_is_rejected_before_multipart_form_parsing(
    client, store, monkeypatch
):
    from starlette.requests import Request

    def must_not_parse(*_args, **_kwargs):
        raise AssertionError("foreign-session content must not be parsed or spooled")

    monkeypatch.setattr(Request, "form", must_not_parse)
    response = client.post(
        "/add_context/?session_id=not-mine",
        files=[("files", ("private.pdf", io.BytesIO(b"private"), "application/pdf"))],
    )

    assert response.status_code == 404


@pytest.fixture
def context_small_limits(store, monkeypatch):
    import dataclasses

    from visionagent.api.routes import add_context_rt

    monkeypatch.setattr(
        add_context_rt,
        "settings",
        dataclasses.replace(
            add_context_rt.settings,
            max_upload_files=2,
            max_upload_bytes=4,
            max_upload_total_bytes=6,
        ),
    )


def test_context_upload_rejects_too_many_files(
    client, vlm, context_small_limits
):
    files = [
        ("files", (f"{index}.txt", io.BytesIO(b"x"), "text/plain"))
        for index in range(3)
    ]
    response = client.post(f"/add_context/?session_id={SESSION_ID}", files=files)
    assert response.status_code == 413


def test_context_upload_rejects_one_oversized_file(
    client, vlm, context_small_limits
):
    response = client.post(
        f"/add_context/?session_id={SESSION_ID}",
        files=[("files", ("large.txt", io.BytesIO(b"12345"), "text/plain"))],
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "large.txt exceeds 4 bytes"}


def test_context_upload_rejects_oversized_aggregate(
    client, vlm, context_small_limits
):
    files = [
        ("files", ("a.txt", io.BytesIO(b"1234"), "text/plain")),
        ("files", ("b.txt", io.BytesIO(b"5678"), "text/plain")),
    ]
    response = client.post(f"/add_context/?session_id={SESSION_ID}", files=files)
    assert response.status_code == 413
    assert response.json() == {"detail": "upload exceeds 6 aggregate bytes"}


def test_context_upload_accepts_exact_aggregate_limit(
    client, vlm, context_small_limits
):
    files = [
        ("files", ("a.txt", io.BytesIO(b"123"), "text/plain")),
        ("files", ("b.txt", io.BytesIO(b"456"), "text/plain")),
    ]
    response = client.post(f"/add_context/?session_id={SESSION_ID}", files=files)
    assert response.status_code == 200, response.text
    assert response.json()["timing"]["files_processed"] == 2


def test_context_upload_rejects_any_multipart_key_other_than_files(
    client, vlm, context_small_limits
):
    response = client.post(
        f"/add_context/?session_id={SESSION_ID}",
        files=[("attachment", ("a.txt", io.BytesIO(b"x"), "text/plain"))],
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": "only the files multipart field is accepted"
    }


def test_malformed_context_multipart_remains_a_400(client, context_small_limits):
    response = client.post(
        f"/add_context/?session_id={SESSION_ID}",
        content=b"not multipart",
        headers={"content-type": "multipart/form-data"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]


def test_a_failed_extraction_is_reported_per_file_and_stores_nothing(client, store, monkeypatch):
    async def failed(_path):
        return {"success": False, "error": "unreadable"}

    monkeypatch.setattr(
        context, "vlm_processor",
        type("V", (), {"process_file": staticmethod(failed)})(),
    )
    r = _add(client, "bad.pdf")
    assert r.status_code == 200
    (entry,) = r.json()["timing"]["individual_files"]
    assert entry["status"] == "failed"
    assert entry["error"] == "File processing failed"
    assert "unreadable" not in r.text, "parser/provider details must stay server-side"
    assert client.get(f"/get_context/{SESSION_ID}").json()["files"] == []
    from visionagent.api.routes import add_context_rt

    raw = add_context_rt.settings.storage_dir / "context_files" / SESSION_ID / "bad.pdf"
    assert not raw.exists()


def test_context_raw_file_copy_is_streamed_atomic_and_durable(tmp_path):
    from visionagent.api.routes import add_context_rt

    target = tmp_path / "context" / "large.bin"
    target.parent.mkdir()

    class BoundedReader(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 1 << 20
            return super().read(size)

    add_context_rt._write_file(BoundedReader(b"payload"), str(target))

    assert target.read_bytes() == b"payload"
    assert list(target.parent.glob(".context-upload-*.tmp")) == []


def test_failed_context_stream_copy_publishes_no_partial_file(tmp_path):
    from visionagent.api.routes import add_context_rt

    target = tmp_path / "context" / "broken.bin"
    target.parent.mkdir()

    class BrokenReader(io.BytesIO):
        reads = 0

        def read(self, size=-1):
            self.reads += 1
            if self.reads > 1:
                raise OSError("source disappeared")
            return super().read(3)

    with pytest.raises(OSError, match="source disappeared"):
        add_context_rt._write_file(BrokenReader(b"payload"), str(target))

    assert not target.exists()
    assert list(target.parent.glob(".context-upload-*.tmp")) == []


def test_context_cancellation_waits_for_copy_then_removes_uncommitted_raw_file(
    store, monkeypatch
):
    from starlette.datastructures import UploadFile

    from visionagent.api.routes import add_context_rt

    entered = threading.Event()
    release = threading.Event()

    def delayed_publish(source, file_path: str) -> None:
        assert source.read(1 << 20) == b"private"
        with open(file_path, "wb") as target:
            target.write(b"private")
        entered.set()
        assert release.wait(2)

    async def must_not_extract(*_args, **_kwargs):
        raise AssertionError("cancelled raw staging must not reach extraction")

    @asynccontextmanager
    async def write_scope(_user_id: str, _session_id: str):
        yield

    monkeypatch.setattr(add_context_rt, "_write_file", delayed_publish)
    monkeypatch.setattr(
        add_context_rt.context_workflow,
        "context_write_scope",
        write_scope,
    )
    monkeypatch.setattr(
        add_context_rt.context_workflow,
        "add_file_to_context",
        must_not_extract,
    )
    upload = UploadFile(
        file=io.BytesIO(b"private"),
        filename="private.txt",
        size=7,
    )

    async def scenario() -> None:
        operation = asyncio.create_task(
            add_context_rt._add_context_files(
                session_id=SESSION_ID,
                files=[upload],
                user_id=USER_ID,
            )
        )
        assert await asyncio.to_thread(entered.wait, 2)
        operation.cancel()
        await asyncio.sleep(0)
        assert not operation.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await operation

    asyncio.run(scenario())

    raw = (
        add_context_rt.settings.storage_dir
        / "context_files"
        / SESSION_ID
        / "private.txt"
    )
    assert not raw.exists()


# ------------------------------------------------------------------ the seams
def test_a_raising_extractor_is_reported_not_propagated(monkeypatch, store):
    async def boom(_path):
        raise RuntimeError("model down")

    monkeypatch.setattr(
        context, "vlm_processor", type("V", (), {"process_file": staticmethod(boom)})()
    )

    out = asyncio.run(context.add_file_to_context("s1", "/tmp/a.pdf", "a.pdf"))

    assert out["success"] is False
    assert "model down" in out["error"]
    assert store.get_session_files("s1") == []


def test_the_store_cannot_reach_a_parser():
    """The database layer cannot import a parser service slot."""
    from pathlib import Path

    import visionagent.database.session_context as sc

    src = Path(sc.__file__).read_text(encoding="utf-8")
    assert "vlm_processor" not in src
    assert "service.parsers" not in src


def test_the_route_holds_no_store():
    """Every operation goes through the pipeline; the route never imports the
    store, so there is one place the context format lives."""
    from pathlib import Path

    src = Path(context.__file__).resolve().parents[1]
    route = (src / "api" / "routes" / "add_context_rt.py").read_text(encoding="utf-8")
    assert "session_context_manager" not in route
    assert "database." not in route


def test_there_is_one_context_store():
    """One repository owns the session-context file format."""
    import visionagent.database.postgres.repositories as repos

    assert not hasattr(repos, "SessionContextRepository")


def test_a_failed_write_is_reported_not_a_success(client, store, vlm, monkeypatch):
    """A storage failure cannot be reported as a successful attachment."""
    import visionagent.database.session_context as sc

    def full(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(sc.os, "replace", full)
    r = _add(client, "a.pdf")
    assert r.status_code == 200
    (entry,) = r.json()["timing"]["individual_files"]
    assert entry["status"] == "failed"
    assert entry["error"] == "File processing failed"
    assert "disk full" not in r.text, "filesystem details must stay server-side"
    assert client.get(f"/get_context/{SESSION_ID}").json()["files"] == []


def test_nothing_is_created_on_disk_for_a_session_the_caller_does_not_own(client, store, vlm):
    from visionagent.api.routes import add_context_rt

    r = client.post("/add_context/?session_id=not-mine",
                    files=[("files", ("a.pdf", io.BytesIO(b"x"), "application/pdf"))])
    assert r.status_code == 404
    assert not (add_context_rt.settings.storage_dir / "context_files" / "not-mine").exists()


def test_a_gate_closed_after_authorization_prevents_all_context_mutations(
    client, store, vlm, monkeypatch
):
    from visionagent.api.routes import add_context_rt

    assert _add(client).status_code == 200

    class _DeletingAccount:
        def accepts_upload_work(self, _user_id: str) -> bool:
            return False

    monkeypatch.setattr(context, "UserRepository", _DeletingAccount)

    added = _add(client, "late.pdf")
    removed = client.delete(f"/remove_file/{SESSION_ID}?file_name=a.pdf")
    cleared = client.delete(f"/clear_context/{SESSION_ID}")

    assert [added.status_code, removed.status_code, cleared.status_code] == [409, 409, 409]
    assert [item["file_name"] for item in store.get_session_files(SESSION_ID)] == ["a.pdf"]
    assert not (
        add_context_rt.settings.storage_dir
        / "context_files"
        / SESSION_ID
        / "late.pdf"
    ).exists()


def test_clear_and_remove_share_the_serialized_context_write_scope(
    monkeypatch,
):
    from visionagent.api.routes import add_context_rt

    active = 0
    maximum_active = 0

    async def scenario() -> None:
        nonlocal active, maximum_active
        lock = asyncio.Lock()

        @asynccontextmanager
        async def scope(_user_id: str, _session_id: str):
            nonlocal active, maximum_active
            async with lock:
                active += 1
                maximum_active = max(maximum_active, active)
                try:
                    yield
                finally:
                    active -= 1

        def slow_clear(_session_id: str) -> None:
            time.sleep(0.05)

        def slow_remove(_session_id: str, _file_name: str) -> bool:
            time.sleep(0.05)
            return True

        monkeypatch.setattr(add_context_rt, "get_current_user_id", lambda _credentials: "42")
        monkeypatch.setattr(add_context_rt, "verify_session_owner", lambda *_args: None)
        monkeypatch.setattr(context, "context_write_scope", scope)
        monkeypatch.setattr(context, "clear_context", slow_clear)
        monkeypatch.setattr(context, "remove_file_from_context", slow_remove)

        await asyncio.gather(
            add_context_rt.clear_context("session", credentials=object()),
            add_context_rt.remove_file_from_context(
                "session", "file.pdf", credentials=object()
            ),
        )

    asyncio.run(scenario())
    assert maximum_active == 1


def test_cancelling_context_lock_wait_does_not_leak_a_future_tenant_lock(
    store,
    users,
    sessions,
):
    from visionagent.database.graph import tenant_lock

    async def scenario() -> None:
        with tenant_lock("42"):
            waiter = asyncio.create_task(context.context_write_scope("42", SESSION_ID).__aenter__())
            await asyncio.sleep(0.08)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter

        async with context.context_write_scope("42", SESSION_ID):
            pass

    asyncio.run(scenario())


def test_session_deletion_erases_attached_context_before_the_database_row(
    client,
    store,
    vlm,
    sessions,
):
    from visionagent.api.routes import add_context_rt

    assert _add(client).status_code == 200
    context_json = (
        add_context_rt.settings.storage_dir
        / "session_context"
        / f"{SESSION_ID}_context.json"
    )
    raw_directory = (
        add_context_rt.settings.storage_dir / "context_files" / SESSION_ID
    )
    assert context_json.exists()
    assert raw_directory.exists()

    response = client.delete(f"/delete_session/{SESSION_ID}")

    assert response.status_code == 200, response.text
    assert not context_json.exists()
    assert not raw_directory.exists()
    assert sessions.owner_of(SESSION_ID) is None


def test_session_deletion_keeps_the_database_retry_handle_if_file_erasure_fails(
    client,
    store,
    vlm,
    sessions,
    monkeypatch,
):
    from visionagent.api.routes import add_context_rt
    from visionagent.pipeline import session as session_workflow

    assert _add(client).status_code == 200
    context_json = (
        add_context_rt.settings.storage_dir
        / "session_context"
        / f"{SESSION_ID}_context.json"
    )
    raw_directory = (
        add_context_rt.settings.storage_dir / "context_files" / SESSION_ID
    )
    assert context_json.exists()
    assert raw_directory.exists()

    def unavailable(_session_id):
        raise OSError("volume unavailable")

    monkeypatch.setattr(
        session_workflow.session_context_manager,
        "_remove_raw_session_directory",
        unavailable,
    )

    response = client.delete(f"/delete_session/{SESSION_ID}")

    assert response.status_code == 500
    assert raw_directory.exists()
    assert context_json.exists(), "raw erasure failure must preserve the retry manifest"
    assert sessions.owner_of(SESSION_ID) == USER_ID
