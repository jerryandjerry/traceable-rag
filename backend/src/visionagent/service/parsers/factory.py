"""Construct the configured document parser."""
from __future__ import annotations

import os
from typing import Any

from visionagent.service.parsers.base import DocumentParser

_PARSERS = {"deepdoc", "vlm"}


def build_parser(name: str | None = None, **kwargs: Any) -> DocumentParser:
    chosen = (name or os.getenv("PARSER") or "deepdoc").lower()
    if chosen == "deepdoc":
        from visionagent.service.parsers.deepdoc import DeepDocParser

        return DeepDocParser(**kwargs)
    if chosen == "vlm":
        from visionagent.service.parsers.vlm import VLMParser

        return VLMParser(**kwargs)
    raise ValueError(f"unknown parser {chosen!r}; known: {sorted(_PARSERS)}")
