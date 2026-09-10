"""Construct the configured analyzer."""
from __future__ import annotations

import os
from typing import Any

from visionagent.service.vectorstore.elasticsearch.analyzer.base import Analyzer

_ANALYZERS = {"ragflow"}


def build_analyzer(name: str | None = None, **kwargs: Any) -> Analyzer:
    chosen = (name or os.getenv("ANALYZER") or "ragflow").lower()
    if chosen == "ragflow":
        from visionagent.service.vectorstore.elasticsearch.analyzer.ragflow import RagflowAnalyzer

        return RagflowAnalyzer(**kwargs)
    raise ValueError(f"unknown analyzer {chosen!r}; known: {sorted(_ANALYZERS)}")
