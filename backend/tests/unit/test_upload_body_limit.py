"""The upload transport fence runs before multipart temp-file spooling."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from typing import Any

import pytest
from fastapi import FastAPI, File, UploadFile

from visionagent.api.body_limits import UploadBodyLimitMiddleware


def _scope(
    *, path: str = "/start-processing", content_length: int | None = None
) -> dict[str, Any]:
    headers = [] if content_length is None else [(b"content-length", str(content_length).encode())]
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("test", 1),
        "server": ("test", 80),
        "root_path": "",
    }


def _run(
    *,
    chunks: Iterable[bytes],
    content_length: int | None = None,
    path: str = "/start-processing",
) -> tuple[list[dict[str, Any]], list[bytes], int]:
    chunks_iter = iter(chunks)
    sent: list[dict[str, Any]] = []
    delivered: list[bytes] = []
    calls = 0

    async def receive() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        try:
            body = next(chunks_iter)
        except StopIteration:
            body = b""
        return {"type": "http.request", "body": body, "more_body": bool(body)}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def consume(scope, receive_from_middleware, send_to_client) -> None:
        while True:
            message = await receive_from_middleware()
            delivered.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await send_to_client({"type": "http.response.start", "status": 204, "headers": []})
        await send_to_client({"type": "http.response.body", "body": b""})

    middleware = UploadBodyLimitMiddleware(consume, max_body_bytes=8)
    asyncio.run(middleware(_scope(path=path, content_length=content_length), receive, send))
    return sent, delivered, calls


def _response(sent: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    return status, json.loads(body)


def test_content_length_rejects_before_any_body_is_read() -> None:
    sent, delivered, calls = _run(chunks=[b"not read"], content_length=9)

    assert _response(sent) == (413, {"detail": "upload request exceeds 8 bytes"})
    assert calls == 0
    assert delivered == []


def test_chunked_body_is_stopped_when_received_bytes_cross_the_limit() -> None:
    sent, delivered, calls = _run(chunks=[b"12345", b"6789"])

    assert _response(sent) == (413, {"detail": "upload request exceeds 8 bytes"})
    assert calls == 2
    assert delivered == [b"12345"]


def test_context_content_length_rejects_before_body_spooling() -> None:
    sent, delivered, calls = _run(
        chunks=[b"not read"],
        content_length=9,
        path="/add_context/",
    )

    assert _response(sent) == (413, {"detail": "upload request exceeds 8 bytes"})
    assert calls == 0
    assert delivered == []


def test_context_chunked_body_without_length_is_stopped_at_the_raw_limit() -> None:
    sent, delivered, calls = _run(
        chunks=[b"12345", b"6789"],
        path="/add_context/",
    )

    assert _response(sent) == (413, {"detail": "upload request exceeds 8 bytes"})
    assert calls == 2
    assert delivered == [b"12345"]


def test_context_dishonest_content_length_cannot_bypass_stream_counting() -> None:
    sent, delivered, calls = _run(
        chunks=[b"12345", b"6789"],
        content_length=1,
        path="/add_context/",
    )

    assert _response(sent) == (413, {"detail": "upload request exceeds 8 bytes"})
    assert calls == 2
    assert delivered == [b"12345"]


@pytest.mark.parametrize("path", ["/start-processing", "/add_context/"])
def test_fastapi_multipart_parser_preserves_streaming_overflow_as_413(
    path: str,
) -> None:
    boundary = b"visionagent-boundary"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="files"; filename="a.txt"\r\n'
        b"Content-Type: text/plain\r\n\r\n"
        b"sensitive upload bytes\r\n--" + boundary + b"--\r\n"
    )
    app = FastAPI()
    app.add_middleware(
        UploadBodyLimitMiddleware,
        max_body_bytes=64,
        max_concurrent_uploads=1,
    )
    handler_called = False

    @app.post(path)
    async def upload_endpoint(files: list[UploadFile] = File(...)) -> dict[str, int]:
        nonlocal handler_called
        handler_called = True
        return {"files": len(files)}

    scope = _scope(path=path)
    scope["headers"] = [
        (b"content-type", b"multipart/form-data; boundary=" + boundary),
    ]
    pieces = iter((body[:48], body[48:]))
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        try:
            chunk = next(pieces)
        except StopIteration:
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.request", "body": chunk, "more_body": True}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))

    assert _response(sent) == (
        413,
        {"detail": "upload request exceeds 64 bytes"},
    )
    assert not handler_called


def test_limit_is_scoped_to_the_upload_endpoint() -> None:
    sent, delivered, calls = _run(
        chunks=[b"123456789", b""], content_length=9, path="/login"
    )

    assert next(message["status"] for message in sent if message["type"] == "http.response.start") == 204
    assert delivered == [b"123456789", b""]
    assert calls == 2


def test_concurrent_upload_overflow_is_rejected_before_reading_its_body() -> None:
    first_sent: list[dict[str, Any]] = []
    second_sent: list[dict[str, Any]] = []
    second_receive_calls = 0

    async def scenario() -> None:
        nonlocal second_receive_calls
        entered = asyncio.Event()
        release = asyncio.Event()

        async def held_request(scope, receive, send) -> None:
            entered.set()
            await release.wait()
            await send(
                {"type": "http.response.start", "status": 204, "headers": []}
            )
            await send({"type": "http.response.body", "body": b""})

        async def first_receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"x", "more_body": False}

        async def second_receive() -> dict[str, Any]:
            nonlocal second_receive_calls
            second_receive_calls += 1
            return {"type": "http.request", "body": b"x", "more_body": False}

        async def first_send(message: dict[str, Any]) -> None:
            first_sent.append(message)

        async def second_send(message: dict[str, Any]) -> None:
            second_sent.append(message)

        middleware = UploadBodyLimitMiddleware(
            held_request,
            max_body_bytes=8,
            max_concurrent_uploads=1,
        )
        first = asyncio.create_task(
            middleware(
                _scope(path="/add_context/", content_length=1),
                first_receive,
                first_send,
            )
        )
        await entered.wait()
        await middleware(
            _scope(path="/add_context/", content_length=1),
            second_receive,
            second_send,
        )
        release.set()
        await first

    asyncio.run(scenario())

    assert _response(second_sent) == (
        429,
        {"detail": "too many uploads are already being received"},
    )
    headers = next(
        message["headers"]
        for message in second_sent
        if message["type"] == "http.response.start"
    )
    assert (b"retry-after", b"1") in headers
    assert second_receive_calls == 0
    assert next(
        message["status"]
        for message in first_sent
        if message["type"] == "http.response.start"
    ) == 204
