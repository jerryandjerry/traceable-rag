"""Construct the configured executer."""
from __future__ import annotations

import os
from typing import Any

from visionagent.service.executer.base import Executer

_EXECUTERS = {"concurrent"}


def build_executer(name: str | None = None, **kwargs: Any) -> Executer:
    chosen = (name or os.getenv("EXECUTER") or "concurrent").lower()
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if chosen == "concurrent":
        from visionagent.service.executer.concurrent import ConcurrentExecuter

        return ConcurrentExecuter(**kwargs)
    raise ValueError(f"unknown executer {chosen!r}; known: {sorted(_EXECUTERS)}")
