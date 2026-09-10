"""RAGFlow deepdoc behind the DocumentParser Protocol.

Wraps `rag.app.manual.chunk` and normalises its output into `ParsedChunk`.
No parsing logic is reimplemented here -- this is an adapter.
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import xxhash

from visionagent.models import ParsedChunk
from visionagent.service.parsers.base import ParserError, ProgressCallback


def _as_base64(image: Any) -> str | None:
    """Normalize a PIL image into the indexed base64 PNG representation."""
    if image is None or not hasattr(image, "save"):
        return None
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _int_list(value: Any) -> list[int]:
    """Normalize DeepDoc list-or-scalar integer fields without discarding values."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


class DeepDocParser:
    """Implements `visionagent.service.parsers.base.DocumentParser`.

    DeepDoc's language choice is deterministic: the vendored adapter inspects
    the available text instead of sampling the process-global random generator.
    The golden suite runs each fixture in a fresh process and pins the chunks.
    """

    name = "deepdoc"
    version = "ragflow-vendored"
    supported_extensions = frozenset({".pdf", ".docx"})

    def parse(
        self,
        *,
        file_path: Path,
        file_name: str,
        progress: ProgressCallback | None = None,
    ) -> list[ParsedChunk]:
        from visionagent.vendor.ragflow.rag.app.manual import chunk

        def _noop(prog: float | None = None, msg: str = "") -> None:
            return None

        try:
            raw = chunk(file_name, str(file_path), callback=progress or _noop)
        except Exception as exc:  # noqa: BLE001 -- deepdoc raises many types
            raise ParserError(f"{self.name} failed on {file_name}: {exc}") from exc

        out: list[ParsedChunk] = []
        for item in raw:
            content = item.get("content_with_weight") or ""
            if not content:
                # ParsedChunk requires content. Dropping would shift every
                # later chunk's position, so this is surfaced rather than
                # silently skipped.
                raise ParserError(
                    f"{self.name} produced an empty chunk in {file_name}; "
                    "chunk positions would shift if it were dropped"
                )
            out.append(
                ParsedChunk(
                    # Parser-local only. Ingest replaces this with the stable
                    # tenant/document/part/ordinal/content identity before the
                    # chunk reaches either persistent store.
                    id=xxhash.xxh64(content.encode("utf-8")).hexdigest(),
                    content=content,
                    content_tokens=item.get("content_ltks", ""),
                    content_tokens_fine=item.get("content_sm_ltks", ""),
                    doc_name_tokens=item.get("docnm_tks", ""),
                    page_nums=_int_list(item.get("page_num_int")),
                    top_offsets=_int_list(item.get("top_int")),
                    ref_images=[
                        b64 for b64 in (_as_base64(i) for i in item.get("ref_images") or [])
                        if b64
                    ],
                    image=_as_base64(item.get("image")),
                )
            )
        return out
