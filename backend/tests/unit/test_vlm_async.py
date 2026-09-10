"""Async and cancellation contracts for attached-context VLM processing."""
from __future__ import annotations

import asyncio
import inspect
import io
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI
from starlette.datastructures import UploadFile

from visionagent.api.routes import add_context_rt
from visionagent.pipeline import context as context_pipeline
from visionagent.service.parsers.vlm import VLMParser
from visionagent.service.parsers.vlm.processor import (
    PageTranscription,
    VLMProcessor,
    VLMResourceLimitError,
)


def run(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


def test_vlm_request_contract_is_native_async() -> None:
    assert inspect.iscoroutinefunction(VLMProcessor.process_file)
    assert inspect.iscoroutinefunction(VLMProcessor.extract_text_from_images)
    assert inspect.iscoroutinefunction(VLMParser.parse_async)
    assert inspect.iscoroutinefunction(context_pipeline.add_file_to_context)


def test_rendering_and_base64_work_do_not_block_the_event_loop(monkeypatch) -> None:
    """The loop must make progress while synchronous conversion is running."""
    processor = VLMProcessor()
    conversion_started = threading.Event()
    loop_progressed = threading.Event()

    def blocking_conversion(file_path: str) -> list[str]:
        conversion_started.set()
        if not loop_progressed.wait(timeout=1):
            raise AssertionError("conversion ran on the event-loop thread")
        return ["encoded-page"]

    monkeypatch.setattr(processor, "_convert_file_to_images", blocking_conversion)

    async def exercise() -> None:
        conversion = asyncio.create_task(processor.convert_file_to_images("drawing.pdf"))
        while not conversion_started.is_set():
            await asyncio.sleep(0)
        loop_progressed.set()
        assert await conversion == ["encoded-page"]

    run(exercise())


def test_cancelling_vlm_processing_cancels_request_and_closes_transport() -> None:
    request_started = asyncio.Event()
    request_cancelled = asyncio.Event()
    client_closed = asyncio.Event()

    async def create(**kwargs: Any) -> Any:
        request_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            request_cancelled.set()
            raise

    class FakeClient:
        chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        async def close(self) -> None:
            # Cleanup itself may need to yield while cancellation unwinds.
            await asyncio.sleep(0)
            client_closed.set()

    fake_client = cast(AsyncOpenAI, FakeClient())
    processor = VLMProcessor(client_factory=lambda: fake_client)

    async def exercise() -> None:
        task = asyncio.create_task(processor.extract_text_from_images(["encoded-page"]))
        await request_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert request_cancelled.is_set()
        assert client_closed.is_set()

    run(exercise())


def test_page_extraction_preserves_boundaries_blanks_and_errors() -> None:
    responses: list[str | Exception] = [
        "first paragraph\n\nsecond paragraph",
        "   ",
        RuntimeError("provider failed"),
    ]
    client_closed = asyncio.Event()

    async def create(**kwargs: Any) -> Any:
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
        )

    class FakeClient:
        chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        async def close(self) -> None:
            client_closed.set()

    processor = VLMProcessor(client_factory=lambda: cast(AsyncOpenAI, FakeClient()))

    async def exercise() -> None:
        pages = await processor.extract_text_from_images(["one", "two", "three"])
        assert pages == [
            PageTranscription(
                page_number=1,
                content="Page 1: first paragraph\n\nsecond paragraph",
            ),
            PageTranscription(page_number=2, content=""),
            PageTranscription(
                page_number=3,
                content="Page 3: [Error extracting text]",
            ),
        ]
        assert client_closed.is_set()

    run(exercise())


def test_file_result_keeps_structured_pages_and_compatible_aggregate(
    monkeypatch,
) -> None:
    pages = [
        PageTranscription(
            page_number=1,
            content="Page 1: first paragraph\n\nsecond paragraph",
        ),
        PageTranscription(page_number=2, content=""),
        PageTranscription(page_number=3, content="Page 3: final"),
    ]
    processor = VLMProcessor()

    async def convert(file_path: str) -> list[str]:
        return ["one", "two", "three"]

    async def extract(images: list[str]) -> list[PageTranscription]:
        assert images == ["one", "two", "three"]
        return pages

    monkeypatch.setattr(processor, "convert_file_to_images", convert)
    monkeypatch.setattr(processor, "extract_text_from_images", extract)

    async def exercise() -> None:
        result = await processor.process_file("drawing.png")
        assert result["success"] is True
        assert result["pages"] == pages
        assert result["pages_processed"] == 3
        assert result["content"] == (
            "Page 1: first paragraph\n\nsecond paragraph\n\nPage 3: final"
        )

    run(exercise())


def test_vlm_document_has_one_wall_clock_timeout_and_closes_client(
    monkeypatch,
) -> None:
    request_started = asyncio.Event()
    client_closed = asyncio.Event()
    calls = 0

    async def create(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        request_started.set()
        if calls > 1:
            await asyncio.Event().wait()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="text"))]
        )

    class FakeClient:
        chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        async def close(self) -> None:
            client_closed.set()

    processor = VLMProcessor(
        client_factory=lambda: cast(AsyncOpenAI, FakeClient()),
        document_timeout_s=0.03,
    )

    rendered: list[int] = []

    monkeypatch.setattr(processor, "_pdf_page_count", lambda path: 3)

    def render(file_path: str, page_index: int) -> str:
        rendered.append(page_index)
        return f"page-{page_index + 1}"

    monkeypatch.setattr(processor, "_render_pdf_page", render)

    async def exercise() -> None:
        result = await asyncio.wait_for(processor.process_file("drawing.pdf"), timeout=1)
        assert request_started.is_set()
        assert result["success"] is False
        assert calls == 2, "the deadline was incorrectly reset for every page"
        assert rendered == [0, 1], "timeout scheduled a render for a later page"
        assert client_closed.is_set()

    run(exercise())


def test_pdf_processing_has_one_live_render_and_keeps_order(monkeypatch) -> None:
    live_pages: set[int] = set()
    max_live_pages = 0
    violations: list[str] = []
    rendered: list[int] = []
    transcribed: list[int] = []
    client_closed = asyncio.Event()

    def render(file_path: str, page_index: int) -> str:
        nonlocal max_live_pages
        if live_pages:
            violations.append("next page rendered before current page was consumed")
        live_pages.add(page_index)
        max_live_pages = max(max_live_pages, len(live_pages))
        rendered.append(page_index)
        return f"encoded-{page_index}"

    async def create(**kwargs: Any) -> Any:
        image_url = kwargs["messages"][0]["content"][1]["image_url"]["url"]
        page_index = int(image_url.rsplit("encoded-", 1)[1])
        if live_pages != {page_index}:
            violations.append("provider did not receive exactly the live page")
        transcribed.append(page_index)
        live_pages.discard(page_index)
        await asyncio.sleep(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"text-{page_index}"))]
        )

    class FakeClient:
        chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        async def close(self) -> None:
            client_closed.set()

    processor = VLMProcessor(client_factory=lambda: cast(AsyncOpenAI, FakeClient()))
    monkeypatch.setattr(processor, "_pdf_page_count", lambda path: 25)
    monkeypatch.setattr(processor, "_render_pdf_page", render)

    async def exercise() -> None:
        result = await processor.process_file("large.pdf")
        assert result["success"] is True
        assert [page.page_number for page in result["pages"]] == list(range(1, 26))
        assert rendered == list(range(25))
        assert transcribed == list(range(25))
        assert max_live_pages == 1
        assert not live_pages
        assert not violations
        assert client_closed.is_set()

    run(exercise())


def test_cancellation_stops_after_current_render_and_releases_resources(
    monkeypatch,
) -> None:
    render_started = threading.Event()
    release_render = threading.Event()
    render_finished = threading.Event()
    client_closed = asyncio.Event()
    render_calls: list[int] = []
    open_resources = 0

    def render(file_path: str, page_index: int) -> str:
        nonlocal open_resources
        render_calls.append(page_index)
        open_resources += 1
        render_started.set()
        try:
            if not release_render.wait(timeout=2):
                raise AssertionError("test did not release the current render")
            return "encoded"
        finally:
            open_resources -= 1
            render_finished.set()

    class FakeClient:
        chat = SimpleNamespace(completions=SimpleNamespace(create=None))

        async def close(self) -> None:
            client_closed.set()

    processor = VLMProcessor(client_factory=lambda: cast(AsyncOpenAI, FakeClient()))
    monkeypatch.setattr(processor, "_pdf_page_count", lambda path: 500)
    monkeypatch.setattr(processor, "_render_pdf_page", render)

    async def exercise() -> None:
        operation = asyncio.create_task(processor.process_file("large.pdf"))
        await asyncio.to_thread(render_started.wait, 1)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert render_calls == [0]
        assert client_closed.is_set()

        release_render.set()
        assert await asyncio.to_thread(render_finished.wait, 1)
        assert open_resources == 0
        assert render_calls == [0]

    run(exercise())


def test_pdf_limits_fail_before_unsafe_render_and_close_documents(
    monkeypatch,
) -> None:
    import visionagent.service.parsers.vlm.processor as processor_module

    closed_documents = 0
    render_attempted = False

    class FakePixmap:
        width = 100
        height = 100

        def tobytes(self, output: str) -> bytes:
            return b"png"

    class FakePage:
        rect = SimpleNamespace(width=100.0, height=100.0)

        def get_images(self, *, full: bool) -> list[tuple[int, ...]]:
            return [(1, 0, 20_000, 20_000)]

        def get_pixmap(self, *, matrix: Any) -> FakePixmap:
            nonlocal render_attempted
            render_attempted = True
            return FakePixmap()

    class FakeDocument:
        def __enter__(self) -> FakeDocument:
            return self

        def __exit__(self, *args: Any) -> None:
            nonlocal closed_documents
            closed_documents += 1

        def __len__(self) -> int:
            return 3

        def load_page(self, page_index: int) -> FakePage:
            return FakePage()

    fake_fitz = SimpleNamespace(
        open=lambda path: FakeDocument(),
        Matrix=lambda x, y: (x, y),
    )
    monkeypatch.setattr(processor_module, "fitz", fake_fitz)
    monkeypatch.setattr(processor_module, "PYMUPDF_AVAILABLE", True)

    page_limited = VLMProcessor(max_pdf_pages=2)
    with pytest.raises(VLMResourceLimitError, match="page count"):
        page_limited._pdf_page_count("large.pdf")

    image_limited = VLMProcessor(max_source_image_pixels=10_000)
    with pytest.raises(VLMResourceLimitError, match="contains an image"):
        image_limited._render_pdf_page("large.pdf", 0)

    render_limited = VLMProcessor(max_render_pixels=10_000)
    with pytest.raises(VLMResourceLimitError, match="render size"):
        render_limited._render_pdf_page("large.pdf", 0)

    assert not render_attempted
    assert closed_documents == 3


def test_sync_vlm_ingest_adapter_works_inside_an_active_event_loop(
    monkeypatch,
    tmp_path,
) -> None:
    """Ingest owns a loop but still calls the parser's sync contract."""
    import visionagent.service.parsers.vlm.processor as processor_module

    async def process_file(path: str) -> dict[str, Any]:
        await asyncio.sleep(0)
        return {
            "success": True,
            "content": "Page 1: text",
            "pages": [PageTranscription(page_number=1, content="Page 1: text")],
            "pages_processed": 1,
            "file_path": path,
        }

    monkeypatch.setattr(processor_module.vlm_processor, "process_file", process_file)

    async def exercise() -> None:
        chunks = VLMParser().parse(
            file_path=tmp_path / "drawing.pdf",
            file_name="drawing.pdf",
        )
        assert [chunk.content for chunk in chunks] == ["Page 1: text"]

    run(exercise())


def test_context_storage_file_io_runs_off_the_event_loop(monkeypatch) -> None:
    conversion_awaited = asyncio.Event()
    store_started = threading.Event()
    loop_progressed = threading.Event()

    async def process_file(file_path: str) -> dict[str, Any]:
        conversion_awaited.set()
        await asyncio.sleep(0)
        return {
            "success": True,
            "content": "Page 1: text",
            "pages_processed": 1,
        }

    def store(*args: Any, **kwargs: Any) -> dict[str, Any]:
        store_started.set()
        if not loop_progressed.wait(timeout=1):
            raise AssertionError("session-context file I/O ran on the event-loop thread")
        return {"success": True, "content_length": 12, "pages_processed": 1}

    monkeypatch.setattr(context_pipeline.vlm_processor, "process_file", process_file)
    monkeypatch.setattr(
        context_pipeline.session_context_manager,
        "add_content_to_context",
        store,
    )

    async def exercise() -> None:
        operation = asyncio.create_task(
            context_pipeline.add_file_to_context("session", "/tmp/file.pdf", "file.pdf")
        )
        await conversion_awaited.wait()
        while not store_started.is_set():
            await asyncio.sleep(0)
        loop_progressed.set()
        assert (await operation)["success"] is True

    run(exercise())


def test_context_pipeline_does_not_turn_cancellation_into_a_failed_file(monkeypatch) -> None:
    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()

    async def process_file(file_path: str) -> dict[str, Any]:
        provider_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            provider_cancelled.set()
            raise

    monkeypatch.setattr(context_pipeline.vlm_processor, "process_file", process_file)

    async def exercise() -> None:
        task = asyncio.create_task(
            context_pipeline.add_file_to_context("session", "/tmp/file.pdf", "file.pdf")
        )
        await provider_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider_cancelled.is_set()

    run(exercise())


def test_add_context_awaits_pipeline_and_writes_upload_off_loop(
    monkeypatch,
    tmp_path,
) -> None:
    write_started = threading.Event()
    loop_progressed = threading.Event()
    pipeline_awaited = asyncio.Event()

    upload = UploadFile(file=io.BytesIO(b"pdf"), filename="drawing.pdf", size=3)

    def write_file(source, file_path: str) -> None:
        assert source.read(1 << 20) == b"pdf"
        write_started.set()
        if not loop_progressed.wait(timeout=1):
            raise AssertionError("upload filesystem write ran on the event-loop thread")

    async def add_file_to_context(
        session_id: str,
        file_path: str,
        file_name: str,
    ) -> dict[str, Any]:
        pipeline_awaited.set()
        await asyncio.sleep(0)
        return {"success": True, "content_length": 7, "pages_processed": 1}

    monkeypatch.setattr(add_context_rt, "get_current_user_id", lambda credentials: "42")
    monkeypatch.setattr(add_context_rt, "verify_session_owner", lambda *args: None)

    @asynccontextmanager
    async def context_write_scope(user_id: str, session_id: str):
        yield

    monkeypatch.setattr(
        add_context_rt.context_workflow,
        "context_write_scope",
        context_write_scope,
    )
    monkeypatch.setattr(
        add_context_rt,
        "settings",
        SimpleNamespace(
            storage_dir=tmp_path,
            max_upload_files=10,
            max_upload_bytes=1024,
            max_upload_total_bytes=1024,
        ),
    )
    monkeypatch.setattr(add_context_rt, "_write_file", write_file)
    monkeypatch.setattr(
        add_context_rt.context_workflow,
        "add_file_to_context",
        add_file_to_context,
    )

    async def exercise() -> None:
        operation = asyncio.create_task(
            add_context_rt._add_context_files(
                session_id="session",
                files=[upload],
                user_id="42",
            )
        )
        while not write_started.is_set():
            await asyncio.sleep(0)
        loop_progressed.set()
        response = await operation
        assert pipeline_awaited.is_set()
        item = response["timing"]["individual_files"][0]
        assert {key: value for key, value in item.items() if key != "processing_time"} == {
            "filename": "drawing.pdf",
            "content_length": 7,
            "pages_processed": 1,
            "status": "success",
        }
        assert isinstance(item["processing_time"], float)

    run(exercise())
