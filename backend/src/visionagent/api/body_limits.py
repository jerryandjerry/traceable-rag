"""Raw HTTP request limits that must run before multipart parsing."""
from __future__ import annotations

import asyncio

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from visionagent.config.settings import settings

# File-content bytes are checked exactly by the route. Multipart boundaries
# and headers also occupy the raw body, so the earlier transport fence permits
# a small, fixed amount of framing rather than silently reducing the file cap.
MULTIPART_OVERHEAD_ALLOWANCE = 1 << 20


class _UploadBodyTooLarge(HTTPException):
    """A 413 that FastAPI's body parser must re-raise rather than mask as 400."""

    def __init__(self, limit: int) -> None:
        super().__init__(
            status_code=413,
            detail=f"upload request exceeds {limit} bytes",
        )


class UploadBodyLimitMiddleware:
    """Admit a bounded number of bounded bodies before multipart parsing.

    ``Content-Length`` permits rejection without reading a byte. A missing or
    dishonest header is handled by counting ``http.request`` messages, so
    chunked uploads are bounded too. A per-process semaphore rejects excess
    uploads without reading their bodies. The route subsequently enforces
    exact per-file and aggregate *content* limits after framing is removed.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int | None = None,
        max_concurrent_uploads: int | None = None,
    ) -> None:
        self.app = app
        self.max_body_bytes = (
            max_body_bytes
            if max_body_bytes is not None
            else settings.max_upload_total_bytes + MULTIPART_OVERHEAD_ALLOWANCE
        )
        if self.max_body_bytes <= 0:
            raise ValueError("upload body limit must be positive")
        concurrency = (
            max_concurrent_uploads
            if max_concurrent_uploads is not None
            else settings.max_concurrent_uploads
        )
        if concurrency <= 0:
            raise ValueError("upload concurrency limit must be positive")
        self._admission = asyncio.Semaphore(concurrency)

    @staticmethod
    def _guards(scope: Scope) -> bool:
        return (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path")
            in {"/start-processing", "/start-processing/", "/add_context", "/add_context/"}
        )

    @staticmethod
    def _content_length(scope: Scope) -> int | None:
        for name, value in scope.get("headers", []):
            if name.lower() != b"content-length":
                continue
            try:
                parsed = int(bytes(value))
            except ValueError:
                return None
            return parsed if parsed >= 0 else None
        return None

    async def _reject(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        status_code: int,
        detail: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        response = JSONResponse(
            status_code=status_code,
            content={"detail": detail},
            headers=headers,
        )
        await response(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._guards(scope):
            await self.app(scope, receive, send)
            return

        content_length = self._content_length(scope)
        if content_length is not None and content_length > self.max_body_bytes:
            await self._reject(
                scope,
                receive,
                send,
                status_code=413,
                detail=f"upload request exceeds {self.max_body_bytes} bytes",
            )
            return
        # There is no await between this check and acquire, so one event-loop
        # task cannot steal the observed permit. Reject instead of queueing:
        # queued clients may continue transmitting bodies into server buffers.
        if self._admission.locked():
            await self._reject(
                scope,
                receive,
                send,
                status_code=429,
                detail="too many uploads are already being received",
                headers={"Retry-After": "1"},
            )
            return
        await self._admission.acquire()

        received = 0

        async def receive_limited() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise _UploadBodyTooLarge(self.max_body_bytes)
            return message

        try:
            try:
                await self.app(scope, receive_limited, send)
            except _UploadBodyTooLarge as error:
                # Multipart parsing consumes the body before the upload work
                # begins, so no response can have started when this trips.
                await self._reject(
                    scope,
                    receive,
                    send,
                    status_code=413,
                    detail=str(error.detail),
                )
        finally:
            self._admission.release()
