"""Lifecycle boundary for provider resources shared by service slots.

The API owns a query pipeline, not provider clients. The pipeline closes this
service-level resource boundary during application shutdown, preserving the
one-way ``api -> pipeline -> service -> providers`` dependency.
"""
from __future__ import annotations

from visionagent.providers.llm import close_llm


async def close_runtime() -> None:
    """Release application-loop transports shared by query-time slots."""
    await close_llm()
