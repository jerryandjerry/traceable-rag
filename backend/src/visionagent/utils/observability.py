"""Process-wide logging configuration and request correlation metadata.

Only observability metadata is ambient here.  A request id identifies a log
trace; it never authenticates a user or authorizes access.  The HTTP boundary
validates and binds the value, while this lower-layer module makes it visible
to logs emitted by pipeline, service, database, and provider code.

The configuration composes with stdlib logging instead of replacing it:

* a ``LogRecord`` factory stamps the request id when the record is created, so
  queue handlers and worker-thread handoffs cannot read a later request's id;
* one marked root handler renders first-party records as one JSON object;
* records still propagate, preserving uvicorn/gunicorn integration and
  pytest's ``caplog`` handler;
* repeated configuration never installs another factory wrapper or handler.

Raw exception messages are deliberately not rendered.  Provider and database
exceptions frequently contain URLs, SQL, paths, credentials, or user data.
The record retains the exception type and source frames needed to diagnose the
failure without copying the exception text into operational logs.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import traceback
from collections.abc import Callable
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

_APPLICATION_LOGGER = "visionagent"
_DEFAULT_REQUEST_ID = "-"
_request_id: ContextVar[str] = ContextVar(
    "visionagent_request_id", default=_DEFAULT_REQUEST_ID
)
_configuration_lock = threading.Lock()

_STANDARD_RECORD_FIELDS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"asctime", "message", "request_id"}
_INTERNAL_EVENT = "_visionagent_event"
_INTERNAL_FIELDS = "_visionagent_fields"
_INTERNAL_EXCEPTION = "_visionagent_exception"
_FACTORY_MARKER = "_visionagent_structured_record_factory"
_HANDLER_MARKER = "_visionagent_structured_handler"
_LogRecordFactory = Callable[..., logging.LogRecord]
_ExcInfo = (
    tuple[type[BaseException], BaseException, TracebackType | None]
    | tuple[None, None, None]
)


def request_id() -> str:
    """Return this execution context's correlation id, or ``-`` off-request."""
    return _request_id.get()


def set_request_id(value: str) -> Token[str]:
    """Bind a validated correlation id and return the token needed to reset it."""
    return _request_id.set(value)


def reset_request_id(token: Token[str]) -> None:
    """Restore the correlation context that preceded ``set_request_id``."""
    _request_id.reset(token)


def _safe_json_default(value: object) -> str:
    """Serialize useful known metadata without falling back to object reprs."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, BaseException):
        return type(value).__name__
    return f"<{type(value).__name__}>"


def _safe_exception(exc_info: _ExcInfo | None) -> dict[str, object] | None:
    """Reduce ``exc_info`` to its type and frames, never its message or values."""
    if not exc_info or exc_info[0] is None:
        return None
    exc_type, _exc, tb = exc_info
    frames = [
        {
            "file": Path(frame.filename).name,
            "line": frame.lineno,
            "function": frame.name,
        }
        for frame in traceback.extract_tb(tb)[-32:]
    ]
    return {"type": exc_type.__name__, "frames": frames}


def _record_factory_wrapper(previous: _LogRecordFactory) -> _LogRecordFactory:
    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        if record.name == _APPLICATION_LOGGER or record.name.startswith(
            _APPLICATION_LOGGER + "."
        ):
            # Capture correlation now. Formatting may happen later in another
            # thread when a QueueHandler is used.
            record.request_id = request_id()
            sanitized = _safe_exception(record.exc_info)
            if sanitized is not None:
                setattr(record, _INTERNAL_EXCEPTION, sanitized)
                # Every downstream handler sees the sanitized representation;
                # none can append the raw exception message after the JSON.
                record.exc_info = None
                record.exc_text = None
        return record

    setattr(factory, _FACTORY_MARKER, True)
    return factory


def _log_level(value: str | int | None) -> int:
    if isinstance(value, int):
        return value
    name = (value or os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    resolved = logging.getLevelName(name)
    return resolved if isinstance(resolved, int) else logging.INFO


class StructuredJSONFormatter(logging.Formatter):
    """Render one stable JSON object for an application ``LogRecord``."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, UTC).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")
        event = getattr(record, _INTERNAL_EVENT, None)
        fields = getattr(record, _INTERNAL_FIELDS, None)
        payload: dict[str, object] = {}
        if isinstance(fields, dict):
            payload.update(fields)

        # ``extra={...}`` on ordinary stdlib calls is also structured data.
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_FIELDS or key.startswith("_visionagent_"):
                continue
            payload.setdefault(key, value)

        payload.update(
            {
                "timestamp": timestamp,
                "level": record.levelname,
                "logger": record.name,
                "event": str(event or "application_log"),
                "request_id": str(
                    getattr(record, "request_id", _DEFAULT_REQUEST_ID)
                ),
                "source": {
                    "file": Path(record.pathname).name,
                    "line": record.lineno,
                    "function": record.funcName,
                },
                "process": {"pid": record.process, "name": record.processName},
                "thread": {"id": record.thread, "name": record.threadName},
            }
        )
        if event is None:
            payload["message"] = record.getMessage()
        exception = getattr(record, _INTERNAL_EXCEPTION, None)
        if exception is not None:
            payload["exception"] = exception

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=_safe_json_default,
            separators=(",", ":"),
        )


class _ApplicationRecordFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == _APPLICATION_LOGGER or record.name.startswith(
            _APPLICATION_LOGGER + "."
        )


class _ExcludeApplicationRecords(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.name == _APPLICATION_LOGGER
            or record.name.startswith(_APPLICATION_LOGGER + ".")
        )


class _ApplicationRoutingFormatter(logging.Formatter):
    """Use JSON for this application and preserve a host's formatter otherwise."""

    def __init__(self, host_formatter: logging.Formatter | None) -> None:
        super().__init__()
        self._host_formatter = host_formatter or logging.Formatter()
        self._application_formatter = StructuredJSONFormatter()

    def format(self, record: logging.LogRecord) -> str:
        if record.name == _APPLICATION_LOGGER or record.name.startswith(
            _APPLICATION_LOGGER + "."
        ):
            return self._application_formatter.format(record)
        return self._host_formatter.format(record)


def _is_test_capture_handler(handler: logging.Handler) -> bool:
    """Keep pytest's record collector additive without treating it as a sink."""
    handler_type = type(handler)
    return (
        handler_type.__name__ == "LogCaptureHandler"
        and handler_type.__module__ == "_pytest.logging"
    )


def _exclude_application_records(handler: logging.Handler) -> None:
    if not any(isinstance(item, _ExcludeApplicationRecords) for item in handler.filters):
        handler.addFilter(_ExcludeApplicationRecords())


def _include_application_records(handler: logging.Handler) -> None:
    for item in list(handler.filters):
        if isinstance(item, _ExcludeApplicationRecords):
            handler.removeFilter(item)


def configure_logging(level: str | int | None = None) -> None:
    """Install the correlation factory and the single application JSON sink."""
    with _configuration_lock:
        current_factory = logging.getLogRecordFactory()
        if not getattr(current_factory, _FACTORY_MARKER, False):
            logging.setLogRecordFactory(_record_factory_wrapper(current_factory))

        root = logging.getLogger()
        selected = [
            handler
            for handler in root.handlers
            if getattr(handler, _HANDLER_MARKER, False)
        ]
        candidates = [
            handler
            for handler in root.handlers
            if not getattr(handler, _HANDLER_MARKER, False)
            and not isinstance(handler, logging.NullHandler)
            and not _is_test_capture_handler(handler)
        ]
        if selected:
            application_sink = selected[0]
            # Repair an impossible-but-safe state produced by external handler
            # copying or an older configuration implementation.
            for duplicate in selected[1:]:
                setattr(duplicate, _HANDLER_MARKER, False)
                _exclude_application_records(duplicate)
        elif candidates:
            # gunicorn and embedding hosts may configure the root before the
            # application is imported. Reuse their destinations instead of
            # installing a second console sink.
            application_sink = candidates[0]
            setattr(application_sink, _HANDLER_MARKER, True)
        else:
            application_sink = logging.StreamHandler(sys.stderr)
            application_sink.setLevel(logging.NOTSET)
            application_sink.setFormatter(StructuredJSONFormatter())
            application_sink.addFilter(_ApplicationRecordFilter())
            setattr(application_sink, _HANDLER_MARKER, True)
            root.addHandler(application_sink)

        _include_application_records(application_sink)
        if not isinstance(
            application_sink.formatter,
            (StructuredJSONFormatter, _ApplicationRoutingFormatter),
        ):
            application_sink.setFormatter(
                _ApplicationRoutingFormatter(application_sink.formatter)
            )

        # Exactly one production handler receives an application record.
        # pytest's LogCaptureHandler is deliberately additive and stores the
        # same record for assertions without becoming an operational sink.
        for handler in root.handlers:
            if (
                handler is application_sink
                or isinstance(handler, logging.NullHandler)
                or _is_test_capture_handler(handler)
            ):
                continue
            _exclude_application_records(handler)

        # Set the application namespace, not the root logger: uvicorn, pytest,
        # and a hosting process retain control of their own logging levels.
        logging.getLogger(_APPLICATION_LOGGER).setLevel(_log_level(level))


def get_logger(name: str = _APPLICATION_LOGGER) -> logging.Logger:
    """Return a stdlib logger after ensuring central configuration exists."""
    configure_logging()
    return logging.getLogger(name)


class EventLogger:
    """Event-oriented facade used where named dimensions are known."""

    def __init__(self, name: str) -> None:
        configure_logging()
        self._logger = logging.getLogger(name)

    def _emit(
        self,
        level: int,
        event: str,
        fields: dict[str, object],
        *,
        exc: BaseException | None = None,
    ) -> None:
        if not self._logger.isEnabledFor(level):
            return
        # Keep getMessage() a self-contained JSON event for collectors or test
        # handlers that intentionally use their own formatter. The central
        # formatter reads the private attributes and adds its stable envelope.
        event_payload = {**fields, "event": event, "request_id": request_id()}
        message = json.dumps(
            event_payload,
            ensure_ascii=False,
            default=_safe_json_default,
            separators=(",", ":"),
        )
        exc_info = None if exc is None else (type(exc), exc, exc.__traceback__)
        self._logger.log(
            level,
            message,
            exc_info=exc_info,
            extra={_INTERNAL_EVENT: event, _INTERNAL_FIELDS: dict(fields)},
            stacklevel=3,
        )

    def debug(self, event: str, **fields: object) -> None:
        self._emit(logging.DEBUG, event, fields)

    def info(self, event: str, **fields: object) -> None:
        self._emit(logging.INFO, event, fields)

    def warning(
        self, event: str, *, exc: BaseException | None = None, **fields: object
    ) -> None:
        self._emit(logging.WARNING, event, fields, exc=exc)

    def error(
        self, event: str, *, exc: BaseException | None = None, **fields: object
    ) -> None:
        self._emit(logging.ERROR, event, fields, exc=exc)

    def exception(self, event: str, exc: BaseException, **fields: object) -> None:
        self._emit(logging.ERROR, event, fields, exc=exc)


def get_event_logger(name: str) -> EventLogger:
    return EventLogger(name)
