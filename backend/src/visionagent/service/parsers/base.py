"""The document parser slot (components #1, #8).

Parser implementations emit one validated model, so `pipeline/ingest.py` does
not depend on a parser-specific dictionary.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from visionagent.models import ParsedChunk

ProgressCallback = Callable[..., None]


class ParserError(RuntimeError):
    """The document could not be parsed."""


@runtime_checkable
class DocumentParser(Protocol):
    """File in, validated chunks out.

    Implementations must be safe to call repeatedly on one instance. Note that
    the deepdoc implementation is **not** currently safe across documents in one
    process -- see `DeepDocParser` -- which is why ingestion isolates each
    document.
    """

    name: str
    version: str
    supported_extensions: frozenset[str]

    def parse(
        self,
        *,
        file_path: Path,
        file_name: str,
        progress: ProgressCallback | None = None,
    ) -> list[ParsedChunk]: ...
