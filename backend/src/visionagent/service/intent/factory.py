"""Construct the configured intent parser.

Returns the module that implements the Protocol, not an adapter around it.
"""
from __future__ import annotations

import os
from typing import Any

from visionagent.service.intent.base import IntentParser

_PARSERS = {"llm", "keywords"}


def build_intent_parser(name: str | None = None, **kwargs: Any) -> IntentParser:
    chosen = (name or os.getenv("INTENT_PARSER") or "llm").lower()
    if chosen == "llm":
        from visionagent.service.intent import llm

        return llm
    if chosen == "keywords":
        from visionagent.service.intent import keywords

        return keywords
    raise ValueError(f"unknown intent parser {chosen!r}; known: {sorted(_PARSERS)}")
