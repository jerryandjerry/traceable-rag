"""API error-boundary and request-correlation contracts."""
from __future__ import annotations

import ast
import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from visionagent.api.observability import RequestContextMiddleware, request_id


def test_application_lifespan_closes_the_shared_llm_once(monkeypatch):
    from visionagent.api import main

    closed = 0
    workers_stopped = 0

    class Pipeline:
        async def aclose(self) -> None:
            nonlocal closed
            closed += 1

    async def supervise() -> None:
        await asyncio.Event().wait()

    def stop_workers() -> None:
        nonlocal workers_stopped
        workers_stopped += 1

    monkeypatch.setattr(main, "configure_logging", lambda: None)
    monkeypatch.setattr(main, "prepare_storage", lambda: None)
    monkeypatch.setattr(main, "QueryPipeline", Pipeline)
    monkeypatch.setattr(main, "supervise_upload_jobs", supervise)
    monkeypatch.setattr(main, "shutdown_upload_workers", stop_workers)

    async def exercise() -> None:
        application = SimpleNamespace(state=SimpleNamespace())
        async with main.lifespan(application):
            assert isinstance(application.state.pipeline, Pipeline)

    asyncio.run(exercise())
    assert workers_stopped == 1
    assert closed == 1


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    return app


def test_request_id_is_preserved_across_an_await_and_returned_as_a_header():
    app = _app()

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def body():
            before = request_id()
            await asyncio.sleep(0)
            yield f"{before}|{request_id()}"

        return StreamingResponse(body())

    response = TestClient(app).get(
        "/stream", headers={"X-Request-ID": "edge-01:abc"}
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "edge-01:abc"
    assert response.text == "edge-01:abc|edge-01:abc"


def test_concurrent_requests_do_not_overwrite_correlation_context():
    app = _app()

    @app.get("/echo")
    async def echo() -> dict[str, str]:
        before = request_id()
        await asyncio.sleep(0.01)
        return {"before": before, "after": request_id()}

    async def exercise() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return list(
                await asyncio.gather(
                    client.get("/echo", headers={"X-Request-ID": "request-A"}),
                    client.get("/echo", headers={"X-Request-ID": "request-B"}),
                )
            )

    first, second = asyncio.run(exercise())
    assert first.json() == {"before": "request-A", "after": "request-A"}
    assert second.json() == {"before": "request-B", "after": "request-B"}
    assert first.headers["X-Request-ID"] == "request-A"
    assert second.headers["X-Request-ID"] == "request-B"


def test_concurrent_nested_logs_keep_their_request_correlation(caplog):
    app = _app()
    service_logger = logging.getLogger("visionagent.service.correlation_test")

    @app.get("/nested-log")
    async def nested_log() -> dict[str, str]:
        before = request_id()
        await asyncio.sleep(0.01)

        def blocking_service_call() -> None:
            service_logger.info("nested service completed", extra={"before": before})

        await asyncio.to_thread(blocking_service_call)
        return {"request_id": request_id()}

    async def exercise() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return list(
                await asyncio.gather(
                    client.get("/nested-log", headers={"X-Request-ID": "nested-A"}),
                    client.get("/nested-log", headers={"X-Request-ID": "nested-B"}),
                )
            )

    with caplog.at_level(logging.INFO, logger="visionagent.service.correlation_test"):
        responses = asyncio.run(exercise())

    assert [response.json() for response in responses] == [
        {"request_id": "nested-A"},
        {"request_id": "nested-B"},
    ]
    records = [
        record
        for record in caplog.records
        if record.name == "visionagent.service.correlation_test"
    ]
    assert {(record.before, record.request_id) for record in records} == {
        ("nested-A", "nested-A"),
        ("nested-B", "nested-B"),
    }


def test_invalid_caller_request_id_is_replaced_not_logged_verbatim():
    app = _app()

    @app.get("/ok")
    async def ok() -> dict[str, str]:
        return {"request_id": request_id()}

    response = TestClient(app).get("/ok", headers={"X-Request-ID": "spaces are unsafe"})
    generated = response.headers["X-Request-ID"]
    assert len(generated) == 32
    assert generated.isalnum()
    assert response.json() == {"request_id": generated}


def test_unhandled_exception_is_logged_and_answered_without_its_text(caplog):
    app = _app()
    secret = "postgresql://internal-host/private?password=do-not-leak"

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError(secret)

    with caplog.at_level(logging.ERROR, logger="visionagent.api.observability"):
        response = TestClient(app).get(
            "/boom", headers={"X-Request-ID": "support-ticket-42"}
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert response.headers["X-Request-ID"] == "support-ticket-42"
    assert secret not in response.text

    event = next(
        json.loads(record.getMessage())
        for record in caplog.records
        if json.loads(record.getMessage()).get("event") == "unhandled_request_exception"
    )
    assert event == {
        "event": "unhandled_request_exception",
        "request_id": "support-ticket-42",
        "method": "GET",
        "path": "/boom",
        "response_started": False,
        "exception_type": "RuntimeError",
    }


def test_cancelled_stream_is_logged_as_cancelled_not_completed(caplog):
    entered = asyncio.Event()

    async def never_finishes(scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        entered.set()
        await asyncio.Event().wait()

    middleware = RequestContextMiddleware(never_finishes)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/cancelled-stream",
        "headers": [],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message) -> None:
        return None

    async def exercise() -> None:
        task = asyncio.create_task(middleware(scope, receive, send))
        await entered.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    with caplog.at_level(logging.INFO, logger="visionagent.api.observability"):
        asyncio.run(exercise())

    event = next(
        json.loads(record.getMessage())
        for record in caplog.records
        if json.loads(record.getMessage()).get("event") == "http_request_finished"
        and json.loads(record.getMessage()).get("path") == "/cancelled-stream"
    )
    assert event["status_code"] == 200
    assert event["outcome"] == "cancelled"


def test_expected_http_status_and_detail_are_unchanged():
    app = _app()

    @app.get("/busy")
    async def busy() -> None:
        raise HTTPException(status_code=409, detail="An upload is still running; try again")

    response = TestClient(app).get("/busy")
    assert response.status_code == 409
    assert response.json() == {"detail": "An upload is still running; try again"}
    assert response.headers["X-Request-ID"]


def _uses_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def test_registered_routes_never_return_broad_exception_text(app_dir: Path):
    """Unexpected database/provider text belongs in logs, never ``detail``."""
    route_dir = app_dir / "api" / "routes"
    registered = {
        "add_context_rt.py",
        "ai_search_rt.py",
        "file_upload_rt.py",
        "graphml_rt.py",
        "history_rt.py",
        "user_rt.py",
    }
    offenders: list[str] = []
    for path in sorted(route_dir.glob("*.py")):
        if path.name not in registered:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for handler in (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)):
            if getattr(handler.type, "id", None) != "Exception" or not handler.name:
                continue
            for call in (node for node in ast.walk(handler) if isinstance(node, ast.Call)):
                if getattr(call.func, "id", None) != "HTTPException":
                    continue
                detail = next(
                    (keyword.value for keyword in call.keywords if keyword.arg == "detail"),
                    None,
                )
                if detail is not None and _uses_name(detail, handler.name):
                    offenders.append(f"{path.name}:{call.lineno}")
    assert not offenders, "broad exception text returned by: " + ", ".join(offenders)


def test_registered_routes_contain_no_print_debugging(app_dir: Path):
    route_dir = app_dir / "api" / "routes"
    registered = {
        "add_context_rt.py",
        "ai_search_rt.py",
        "file_upload_rt.py",
        "graphml_rt.py",
        "history_rt.py",
        "user_rt.py",
    }
    offenders: list[str] = []
    for name in sorted(registered):
        tree = ast.parse((route_dir / name).read_text(encoding="utf-8"))
        offenders.extend(
            f"{name}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"
        )
    assert not offenders, "route print debugging remains at: " + ", ".join(offenders)
