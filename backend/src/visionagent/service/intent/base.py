"""The intent slot: what the user is asking for, before any retrieval happens.

The Protocol is written around the existing function name --
`analyze_query_intent` in `visionagent.service.agent.agent` -- so that module
satisfies it as it stands.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from visionagent.models import Intent


@runtime_checkable
class IntentParser(Protocol):
    async def analyze_query_intent(self, queries: list[str]) -> Intent: ...

    async def analyze_chat_scenario(self, question: str) -> str:
        """"casual", "casual_web" or "professional" -- which path the turn takes.

        "casual_web" is casual with a web search first: the question is simple
        but its answer is current, and the model alone would guess.

        On the intent Protocol because deciding what kind of request this is
        happens before any retrieval, and the same module implements both.
        """
        ...
