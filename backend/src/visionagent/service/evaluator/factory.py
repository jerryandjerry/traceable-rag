"""Construct the configured evaluator.

Returns the module that implements the Protocol, not an adapter around it.
"""
from __future__ import annotations

import os
from typing import Any

from visionagent.service.evaluator.base import Evaluator

_EVALUATORS = {"llm"}


def build_evaluator(name: str | None = None, **kwargs: Any) -> Evaluator:
    chosen = (name or os.getenv("EVALUATOR") or "llm").lower()
    if chosen == "llm":
        from visionagent.service.evaluator import llm

        return llm
    raise ValueError(f"unknown evaluator {chosen!r}; known: {sorted(_EVALUATORS)}")
