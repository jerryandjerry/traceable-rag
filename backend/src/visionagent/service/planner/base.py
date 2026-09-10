"""The planner slot: intents in, a Plan of retrievals out."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from visionagent.models import Intent, Plan, ToolName, WebSearchMode


@runtime_checkable
class Planner(Protocol):
    async def agent_plan(
        self,
        queries: list[str],
        intent: Intent,
        session_id: str | None = None,
        chat_id: str | None = None,
        available: list[ToolName] | None = None,
        web_search: WebSearchMode = WebSearchMode.AUTO,
    ) -> Plan: ...
