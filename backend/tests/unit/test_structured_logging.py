"""Central structured-logging contracts."""
from __future__ import annotations

import io
import json
import logging

from visionagent.utils.observability import (
    StructuredJSONFormatter,
    configure_logging,
    get_event_logger,
    reset_request_id,
    set_request_id,
)


def _isolated_json_logger(name: str) -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredJSONFormatter())
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, stream


def test_plain_stdlib_log_is_a_correlated_json_event() -> None:
    logger, stream = _isolated_json_logger("visionagent.service.test")
    token = set_request_id("request-structured-1")
    try:
        logger.info(
            "indexed %d chunks",
            7,
            extra={"run_id": "run-7", "chunk_count": 7},
        )
    finally:
        reset_request_id(token)

    event = json.loads(stream.getvalue())
    assert event["level"] == "INFO"
    assert event["logger"] == "visionagent.service.test"
    assert event["event"] == "application_log"
    assert event["request_id"] == "request-structured-1"
    assert event["message"] == "indexed 7 chunks"
    assert event["run_id"] == "run-7"
    assert event["chunk_count"] == 7
    assert event["timestamp"].endswith("Z")
    assert event["source"]["file"] == "test_structured_logging.py"


def test_exception_output_keeps_diagnostics_without_the_exception_message() -> None:
    logger, stream = _isolated_json_logger("visionagent.database.test")
    secret = "postgresql://private-host/db?password=do-not-log"

    try:
        raise RuntimeError(secret)
    except RuntimeError:
        logger.exception("database operation failed")

    rendered = stream.getvalue()
    event = json.loads(rendered)
    assert secret not in rendered
    assert event["message"] == "database operation failed"
    assert event["exception"]["type"] == "RuntimeError"
    assert event["exception"]["frames"][-1]["file"] == "test_structured_logging.py"


def test_event_logger_fields_cannot_forge_correlation_and_exceptions_are_safe() -> None:
    name = "visionagent.api.event_test"
    _logger, stream = _isolated_json_logger(name)
    event_logger = get_event_logger(name)
    token = set_request_id("real-request-id")
    try:
        try:
            raise ValueError("secret provider response")
        except ValueError as exc:
            event_logger.exception(
                "provider_call_failed",
                exc,
                operation="answer",
                request_id="forged-request-id",
            )
    finally:
        reset_request_id(token)

    rendered = stream.getvalue()
    event = json.loads(rendered)
    assert "secret provider response" not in rendered
    assert event["event"] == "provider_call_failed"
    assert event["request_id"] == "real-request-id"
    assert event["operation"] == "answer"
    assert event["exception"]["type"] == "ValueError"
    assert event["source"]["file"] == "test_structured_logging.py"


def test_configuration_is_idempotent() -> None:
    configure_logging()
    factory = logging.getLogRecordFactory()
    handlers = list(logging.getLogger().handlers)
    configure_logging()
    configure_logging()

    assert logging.getLogger().handlers == handlers
    assert logging.getLogRecordFactory() is factory


def test_preconfigured_host_handler_is_reused_without_duplicate_output() -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    stream = io.StringIO()
    host_handler = logging.StreamHandler(stream)
    host_handler.setFormatter(logging.Formatter("HOST %(message)s"))
    try:
        for handler in original_handlers:
            root.removeHandler(handler)
        root.addHandler(host_handler)

        configure_logging()
        logging.getLogger("visionagent.pipeline.hosted").info("one record")
        logging.getLogger("hosting.runtime").warning("host record")

        assert root.handlers == [host_handler]
        lines = stream.getvalue().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["message"] == "one record"
        assert lines[1] == "HOST host record"
    finally:
        root.removeHandler(host_handler)
        for handler in original_handlers:
            root.addHandler(handler)


def test_only_one_of_multiple_or_late_host_handlers_receives_application_logs() -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    streams = [io.StringIO(), io.StringIO(), io.StringIO()]
    handlers = [logging.StreamHandler(stream) for stream in streams]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("HOST %(message)s"))
    try:
        for handler in original_handlers:
            root.removeHandler(handler)
        root.addHandler(handlers[0])
        root.addHandler(handlers[1])
        configure_logging()

        logging.getLogger("visionagent.service.multi_host").info("first")
        assert len(streams[0].getvalue().splitlines()) == 1
        assert streams[1].getvalue() == ""

        # A host handler installed later is reconciled at API lifespan start.
        root.addHandler(handlers[2])
        configure_logging()
        logging.getLogger("visionagent.service.multi_host").info("second")

        assert len(streams[0].getvalue().splitlines()) == 2
        assert streams[1].getvalue() == ""
        assert streams[2].getvalue() == ""
    finally:
        for handler in handlers:
            root.removeHandler(handler)
        for handler in original_handlers:
            root.addHandler(handler)


def test_reconciliation_preserves_pytest_record_capture(caplog) -> None:
    name = "visionagent.service.caplog_contract"
    with caplog.at_level(logging.INFO, logger=name):
        configure_logging()
        logging.getLogger(name).info("captured after reconciliation")

    assert any(
        record.name == name and record.getMessage() == "captured after reconciliation"
        for record in caplog.records
    )


def test_invalid_log_level_falls_back_to_info() -> None:
    configure_logging("not-a-level")
    assert logging.getLogger("visionagent").level == logging.INFO
