"""Construct the configured answer generator.

Returns the module that implements the Protocol, not an adapter around it.
"""
from __future__ import annotations

import os
from typing import Any

from visionagent.service.answer.base import AnswerGenerator

_GENERATORS = {"cited"}


def build_answer_generator(name: str | None = None, **kwargs: Any) -> AnswerGenerator:
    chosen = (name or os.getenv("ANSWER") or "cited").lower()
    if chosen == "cited":
        from visionagent.service.answer import chat

        return chat
    raise ValueError(f"unknown answer generator {chosen!r}; known: {sorted(_GENERATORS)}")
