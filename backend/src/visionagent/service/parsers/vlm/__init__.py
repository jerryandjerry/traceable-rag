"""Vision-language parsing behind the DocumentParser Protocol.

The second ingest path. `VLMProcessor` renders each page to an image and asks a
vision model to transcribe it, which reaches content that layout-based OCR
misses -- dense figures, handwriting, scanned tables. Source pages stay
structured until this adapter emits the same `ParsedChunk` contract as deepdoc;
transcription formatting is never interpreted as a page boundary.

**On why VLMProcessor keeps its own client.** The vision call sends image parts
in the message body, which is not the shape of `visionagent.providers.llm.LLM` -- that
Protocol is text in, text out. Forcing multimodal input through it would mean
widening the interface for one caller. The client stays where the multimodal
concern is.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import xxhash

from visionagent.models import ParsedChunk
from visionagent.service.parsers.base import ParserError, ProgressCallback


class VLMParser:
    """Implements `visionagent.service.parsers.base.DocumentParser`."""

    name = "vlm"
    version = "qwen-vl-plus"
    supported_extensions = frozenset({".pdf", ".png", ".jpg", ".jpeg"})

    def parse(
        self,
        *,
        file_path: Path,
        file_name: str,
        progress: ProgressCallback | None = None,
    ) -> list[ParsedChunk]:
        """Synchronous adapter for the isolated ingestion worker.

        The provider implementation is async even here. The ingestion worker's
        orchestration already owns an event loop, despite exposing this legacy
        synchronous parser contract. In that case, run the private loop in a
        dedicated thread; calling ``asyncio.run`` on the worker's loop would
        fail before parsing began. Async request paths use :meth:`parse_async`
        directly and remain cancellable.
        """
        def run_parse() -> list[ParsedChunk]:
            return asyncio.run(
                self.parse_async(
                    file_path=file_path,
                    file_name=file_name,
                    progress=progress,
                )
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return run_parse()

        # This branch is used only by the isolated synchronous ingestion slot.
        # Its child process has one file operation in flight, so blocking that
        # orchestration loop while the dedicated provider loop runs cannot
        # starve unrelated requests.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-parser") as pool:
            return pool.submit(run_parse).result()

    async def parse_async(
        self,
        *,
        file_path: Path,
        file_name: str,
        progress: ProgressCallback | None = None,
    ) -> list[ParsedChunk]:
        """Transcribe a document into one chunk per non-blank source page.

        Provider-error placeholders are non-blank and therefore remain a chunk
        for their source page. Blank transcriptions remain absent because
        ``ParsedChunk.content`` cannot be empty.
        """
        from visionagent.service.parsers.vlm.processor import vlm_processor

        result = await vlm_processor.process_file(str(file_path))
        if result["success"] is False:
            raise ParserError(
                f"{self.name} failed on {file_name}: {result['error']}"
            )

        chunks: list[ParsedChunk] = []
        for page in result["pages"]:
            content = page.content.strip()
            # ParsedChunk has a non-empty-content invariant. A blank VLM
            # response therefore remains omitted, as before, while the
            # structured page number prevents later pages from being shifted.
            if not content:
                continue
            chunks.append(
                ParsedChunk(
                    id=xxhash.xxh64(content.encode("utf-8")).hexdigest(),
                    content=content,
                    page_nums=[page.page_number],
                )
            )
        return chunks
