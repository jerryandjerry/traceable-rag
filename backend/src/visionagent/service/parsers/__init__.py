"""Slot 1: a document in, chunks out.

`vlm_processor` remains available as a compatibility export, but it is loaded
only when requested.  Eagerly constructing the VLM client made importing the
offline DeepDoc parser validate hosted-provider settings and create runtime
directories.  A parser import must not acquire an unrelated provider runtime.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from visionagent.service.parsers.base import (
    DocumentParser,
    ParserError,
    ProgressCallback,
)
from visionagent.service.parsers.factory import build_parser

if TYPE_CHECKING:
    from visionagent.service.parsers.vlm.processor import vlm_processor as vlm_processor


def __getattr__(name: str) -> Any:
    if name == "vlm_processor":
        from visionagent.service.parsers.vlm.processor import vlm_processor

        globals()[name] = vlm_processor
        return vlm_processor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DocumentParser", "ParserError", "ProgressCallback", "build_parser",
    "vlm_processor",
]
