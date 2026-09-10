"""Request correlation, structured API logs, and the last-resort error boundary.

Request-id selection and binding live at the HTTP boundary, rather than in an
agent job or service module.  The lower-layer logging carrier is a
:class:`ContextVar`: unlike user/session identity it authorizes nothing, and
Python copies it into each request task (and into worker-thread calls made by
that task) without requests overwriting one another.

This is a pure ASGI middleware rather than ``BaseHTTPMiddleware``.  The latter
returns from ``call_next`` before a streaming response has finished, which can
clear request context while an SSE generator is still emitting or logging.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid

from fastapi import HTTPException, status
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from visionagent.utils.observability import (
    EventLogger,
    get_event_logger,
    reset_request_id,
    set_request_id,
)
from visionagent.utils.observability import request_id as _request_id_value

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_HEADER_BYTES = REQUEST_ID_HEADER.lower().encode("ascii")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def request_id() -> str:
    """Return the current request's correlation id, or ``-`` off-request."""
    return _request_id_value()


def _new_request_id(scope: Scope) -> str:
    for name, value in scope.get("headers", []):
        if name.lower() != _REQUEST_ID_HEADER_BYTES:
            continue
        try:
            candidate = bytes(value).decode("ascii")
        except UnicodeDecodeError:
            break
        # Never allow control characters or unbounded caller input into logs.
        if _REQUEST_ID_PATTERN.fullmatch(candidate):
            return candidate
        break
    return uuid.uuid4().hex


def get_api_logger(name: str) -> EventLogger:
    return get_event_logger(name)


logger = get_api_logger(__name__)


def translated_http_error(
    exc: Exception,
    *,
    operation: str,
    detail: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    **fields: object,
) -> HTTPException:
    """Log an unexpected cause and return a stable, non-sensitive HTTP error.

    Expected domain failures should be translated directly by their route.
    This helper is for infrastructure/provider/database exceptions whose text
    may contain SQL, hosts, paths, credentials, or other internal details.
    """
    logger.exception(
        "api_operation_failed",
        exc,
        operation=operation,
        status_code=status_code,
        exception_type=type(exc).__name__,
        **fields,
    )
    return HTTPException(status_code=status_code, detail=detail)


class RequestContextMiddleware:
    """Correlate one HTTP request and sanitize otherwise unhandled failures."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        current_id = _new_request_id(scope)
        token = set_request_id(current_id)
        started_at = time.perf_counter()
        response_started = False
        status_code: int | None = None
        outcome = "completed"

        async def send_with_request_id(message: Message) -> None:
            nonlocal response_started, status_code
            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != _REQUEST_ID_HEADER_BYTES
                ]
                headers.append((_REQUEST_ID_HEADER_BYTES, current_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 - this is the process error boundary
            outcome = "failed"
            logger.exception(
                "unhandled_request_exception",
                exc,
                method=scope.get("method", ""),
                path=scope.get("path", ""),
                response_started=response_started,
                exception_type=type(exc).__name__,
            )
            if response_started:
                # HTTP cannot replace headers or status after streaming began.
                # The query and upload SSE routes therefore translate their
                # own generator failures into their established error frames.
                raise
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"},
            )
            await response(scope, receive, send_with_request_id)
        finally:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            logger.info(
                "http_request_finished",
                method=scope.get("method", ""),
                path=scope.get("path", ""),
                status_code=status_code,
                outcome=outcome,
                duration_ms=duration_ms,
            )
            reset_request_id(token)
