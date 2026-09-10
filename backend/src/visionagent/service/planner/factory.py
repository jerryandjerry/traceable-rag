"""Construct the configured planner.

Returns the module that implements the Protocol, not an adapter around it.
"""
from __future__ import annotations

import os
from typing import Any

from visionagent.service.planner.base import Planner

_PLANNERS = {"rule_based"}


def build_planner(name: str | None = None, **kwargs: Any) -> Planner:
    chosen = (name or os.getenv("PLANNER") or "rule_based").lower()
    if chosen == "rule_based":
        from visionagent.service.planner import rule_based

        return rule_based
    raise ValueError(f"unknown planner {chosen!r}; known: {sorted(_PLANNERS)}")
