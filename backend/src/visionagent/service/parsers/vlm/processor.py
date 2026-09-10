import asyncio
import base64
import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypedDict

from openai import AsyncOpenAI

from visionagent.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PageTranscription:
    """The transcription produced for one rendered source page.

    ``content`` is the exact text exposed by the VLM processor, including the
    existing ``Page N:`` prefix. A blank provider response is represented by an
    empty string so the caller can preserve the source page number without
    inventing a non-empty :class:`ParsedChunk`.
    """

    page_number: int
    content: str


class VLMProcessSuccess(TypedDict):
    """Successful file processing with lossless source-page boundaries."""

    success: Literal[True]
    content: str
    pages: list[PageTranscription]
    pages_processed: int
    file_path: str


class VLMProcessFailure(TypedDict):
    """Failed file processing; no page transcription is safe to consume."""

    success: Literal[False]
    error: str
    content: str


VLMProcessResult: TypeAlias = VLMProcessSuccess | VLMProcessFailure


class VLMResourceLimitError(ValueError):
    """A document exceeds a bound required for safe in-process rendering."""


_DEFAULT_MAX_PDF_PAGES = 2_000
_DEFAULT_MAX_RENDER_PIXELS = 40_000_000
_DEFAULT_MAX_SOURCE_IMAGE_PIXELS = 100_000_000
_DEFAULT_MAX_ENCODED_PAGE_BYTES = 25 * 1024 * 1024

# Try to import optional dependencies
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. PDF processing will not work.")

try:
    from PIL import Image  # noqa: F401  (import is the availability probe)
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Image processing may be limited.")

class VLMProcessor:
    """Render documents off-loop and transcribe their pages asynchronously.

    A client is scoped to one file so its HTTP pool is deterministically closed
    after success, provider failure, or task cancellation.  The factory is
    injectable to keep the cancellation boundary testable without a network.
    """

    def __init__(
        self,
        *,
        client_factory: Callable[[], AsyncOpenAI] | None = None,
        document_timeout_s: float | None = None,
        max_pdf_pages: int = _DEFAULT_MAX_PDF_PAGES,
        max_render_pixels: int = _DEFAULT_MAX_RENDER_PIXELS,
        max_source_image_pixels: int = _DEFAULT_MAX_SOURCE_IMAGE_PIXELS,
        max_encoded_page_bytes: int = _DEFAULT_MAX_ENCODED_PAGE_BYTES,
        render_scale: float = 2.0,
    ) -> None:
        self._document_timeout_s = (
            settings.parse_timeout_s
            if document_timeout_s is None
            else document_timeout_s
        )
        if self._document_timeout_s <= 0:
            raise ValueError("VLM document timeout must be positive")
        limits = {
            "max_pdf_pages": max_pdf_pages,
            "max_render_pixels": max_render_pixels,
            "max_source_image_pixels": max_source_image_pixels,
            "max_encoded_page_bytes": max_encoded_page_bytes,
        }
        for name, value in limits.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if render_scale <= 0:
            raise ValueError("render_scale must be positive")

        self._client_factory = client_factory or self._new_client
        self._max_pdf_pages = max_pdf_pages
        self._max_render_pixels = max_render_pixels
        self._max_source_image_pixels = max_source_image_pixels
        self._max_encoded_page_bytes = max_encoded_page_bytes
        self._render_scale = render_scale

    def _new_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            # The SDK also enforces the configured bound at the transport
            # layer. The asyncio.timeout around the page loop below covers
            # injected clients and every SDK retry under one document budget.
            timeout=self._document_timeout_s,
        )

    async def convert_file_to_images(self, file_path: str) -> list[str]:
        """Convert one non-PDF source file to bounded base64 image data.

        PDFs deliberately bypass this list-returning compatibility method and
        use the page-at-a-time path in :meth:`process_file`. Filesystem access,
        validation, and base64 conversion remain off the event loop.
        """
        return await asyncio.to_thread(self._convert_file_to_images, file_path)

    def _convert_file_to_images(self, file_path: str) -> list[str]:
        file_ext = os.path.splitext(file_path)[1].lower()
        images: list[str] = []
        
        try:
            if file_ext == '.pdf':
                raise ValueError("PDF conversion must use bounded page processing")
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                images = self._image_to_base64(file_path)
            else:
                logger.warning(
                    "unsupported file type",
                    extra={"file_extension": file_ext},
                )
                return []
                
        except Exception:
            logger.exception(
                "file-to-image conversion failed",
                extra={"file_extension": file_ext},
            )
            return []
            
        return images
    
    def _pdf_page_count(self, pdf_path: str) -> int:
        """Read and validate page count without retaining the native document."""
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF not available. Cannot process PDF files.")
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
        if page_count <= 0:
            raise ValueError("PDF contains no pages")
        if page_count > self._max_pdf_pages:
            raise VLMResourceLimitError(
                f"PDF page count {page_count} exceeds limit {self._max_pdf_pages}"
            )
        return page_count

    def _render_pdf_page(self, pdf_path: str, page_index: int) -> str:
        """Render one validated PDF page and release all native objects.

        The document is intentionally scoped to this one call. If cancellation
        stops awaiting its worker thread, only this render can finish in the
        background; no document-wide producer continues allocating images.
        """
        with fitz.open(pdf_path) as doc:
            if page_index < 0 or page_index >= len(doc):
                raise IndexError(f"PDF page index {page_index} is out of range")
            page = doc.load_page(page_index)

            width = math.ceil(float(page.rect.width) * self._render_scale)
            height = math.ceil(float(page.rect.height) * self._render_scale)
            render_pixels = width * height
            if render_pixels > self._max_render_pixels:
                raise VLMResourceLimitError(
                    f"PDF page {page_index + 1} render size {render_pixels} pixels "
                    f"exceeds limit {self._max_render_pixels}"
                )

            # Reject pathological embedded rasters before MuPDF expands them.
            for image_info in page.get_images(full=True):
                source_pixels = int(image_info[2]) * int(image_info[3])
                if source_pixels > self._max_source_image_pixels:
                    raise VLMResourceLimitError(
                        f"PDF page {page_index + 1} contains an image of "
                        f"{source_pixels} pixels; limit is "
                        f"{self._max_source_image_pixels}"
                    )

            pix = page.get_pixmap(
                matrix=fitz.Matrix(self._render_scale, self._render_scale)
            )
            actual_pixels = int(pix.width) * int(pix.height)
            if actual_pixels > self._max_render_pixels:
                raise VLMResourceLimitError(
                    f"PDF page {page_index + 1} rendered to {actual_pixels} pixels; "
                    f"limit is {self._max_render_pixels}"
                )
            image_data = pix.tobytes("png")
            if len(image_data) > self._max_encoded_page_bytes:
                raise VLMResourceLimitError(
                    f"PDF page {page_index + 1} encoded size {len(image_data)} bytes "
                    f"exceeds limit {self._max_encoded_page_bytes}"
                )
            return base64.b64encode(image_data).decode("ascii")
    
    def _image_to_base64(self, image_path: str) -> list[str]:
        """Validate and encode one raster without unbounded reads/decompression."""
        try:
            file_size = os.path.getsize(image_path)
            if file_size > self._max_encoded_page_bytes:
                raise VLMResourceLimitError(
                    f"image size {file_size} bytes exceeds limit "
                    f"{self._max_encoded_page_bytes}"
                )
            if not PIL_AVAILABLE:
                raise RuntimeError("PIL not available. Cannot validate image dimensions.")
            with Image.open(image_path) as image:
                source_pixels = int(image.width) * int(image.height)
                if source_pixels > self._max_source_image_pixels:
                    raise VLMResourceLimitError(
                        f"image has {source_pixels} pixels; limit is "
                        f"{self._max_source_image_pixels}"
                    )
                image.verify()
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('ascii')
                return [img_base64]
        except Exception:
            logger.exception("image encoding failed")
            return []
    
    async def _transcribe_page(
        self,
        client: AsyncOpenAI,
        image_base64: str,
        page_number: int,
    ) -> PageTranscription:
        """Transcribe one encoded page, preserving blank/error semantics."""
        try:
            # Keep this as a directly-awaited SDK call. Cancellation must reach
            # the HTTP request rather than abandon a blocking worker.
            create: Any = client.chat.completions.create
            response = await create(
                model="qwen-vl-plus",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please extract all text content from this image. Return only the text content, no explanations or formatting.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=4000,
                temperature=0.1,
            )
            text_content = (response.choices[0].message.content or "").strip()
            content = f"Page {page_number}: {text_content}" if text_content else ""
            return PageTranscription(page_number=page_number, content=content)
        except Exception:
            logger.exception(
                "image text extraction failed",
                extra={"image_number": page_number},
            )
            return PageTranscription(
                page_number=page_number,
                content=f"Page {page_number}: [Error extracting text]",
            )

    async def _close_client(self, client: AsyncOpenAI) -> None:
        try:
            await client.close()
        except Exception:
            logger.exception("VLM provider client cleanup failed")

    async def extract_text_from_images(
        self,
        images: list[str],
    ) -> list[PageTranscription]:
        """Use Qwen-VL to transcribe images without flattening page boundaries.

        The returned list has one entry per input image, in input order. Blank
        responses remain explicit empty entries and provider errors use an
        explicit placeholder entry.
        """
        if not images:
            return []

        pages: list[PageTranscription] = []
        client = self._client_factory()
        try:
            # PARSE_TIMEOUT_S is a per-document wall-clock budget, not a
            # fresh allowance for every page. One context bounds the complete
            # page sequence, including SDK retries.
            async with asyncio.timeout(self._document_timeout_s):
                for page_number, image_base64 in enumerate(images, start=1):
                    pages.append(
                        await self._transcribe_page(
                            client,
                            image_base64,
                            page_number,
                        )
                    )
        finally:
            await self._close_client(client)

        return pages

    async def process_file(self, file_path: str) -> VLMProcessResult:
        """Render and transcribe a file, retaining one result per source page.

        ``content`` is the blank-line-separated aggregate used by
        attached-context storage. Parser consumers use ``pages`` and never
        reconstruct structural boundaries from that display string.
        """
        try:
            # The same configured budget covers rendering and every page call.
            # A to_thread render cannot be force-stopped, but the request/worker
            # stops waiting for it when the document deadline expires. The PDF
            # path has no producer loop in that thread, so it cannot schedule or
            # retain any later page after cancellation.
            async with asyncio.timeout(self._document_timeout_s):
                if os.path.splitext(file_path)[1].lower() == ".pdf":
                    page_count = await asyncio.to_thread(
                        self._pdf_page_count,
                        file_path,
                    )
                    pages = []
                    client = self._client_factory()
                    try:
                        for page_index in range(page_count):
                            image_base64 = await asyncio.to_thread(
                                self._render_pdf_page,
                                file_path,
                                page_index,
                            )
                            pages.append(
                                await self._transcribe_page(
                                    client,
                                    image_base64,
                                    page_index + 1,
                                )
                            )
                            # Do not retain the potentially large base64 string
                            # while the next source page is rendered.
                            del image_base64
                    finally:
                        await self._close_client(client)
                    pages_processed = page_count
                else:
                    images = await self.convert_file_to_images(file_path)
                    if not images:
                        return {
                            "success": False,
                            "error": "Failed to convert file to images",
                            "content": ""
                        }
                    pages = await self.extract_text_from_images(images)
                    pages_processed = len(images)

                return {
                    "success": True,
                    "content": "\n\n".join(
                        page.content for page in pages if page.content
                    ),
                    "pages": pages,
                    "pages_processed": pages_processed,
                    "file_path": file_path
                }
            
        except Exception as e:
            logger.exception(
                "VLM file processing failed",
                extra={"file_extension": os.path.splitext(file_path)[1].lower()},
            )
            return {
                "success": False,
                "error": str(e),
                "content": ""
            }

# Global instance
vlm_processor = VLMProcessor()
