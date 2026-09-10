"""The evaluator slot: is the gathered context enough, and if not, what next?

The Protocol is written around the existing function names --
`evaluate_context_sufficiency` and `reflection` in
`visionagent.service.agent.agent` -- so that module satisfies it as it stands.
Protocols are structural: no base class, no registration, and no adapter class
whose whole body forwards to the real function.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from visionagent.models import Evaluation, RetrievedChunk, ToolName, WebSearchMode


@runtime_checkable
class Evaluator(Protocol):
    async def evaluate_context_sufficiency(
        self, context_list: list[RetrievedChunk], query: str
    ) -> Evaluation: ...

    def is_sufficient(self, evaluation: Evaluation) -> bool:
        """Apply this evaluator's configured decision policy to a score."""
        ...

    async def reflection(
        self,
        user_query: str,
        context_list: list[RetrievedChunk],
        evaluation: Evaluation,
        web_search: WebSearchMode = WebSearchMode.AUTO,
        available: list[ToolName] | None = None,
    ) -> list[str] | None:
        """Refined queries for another round, or None to stop."""
        ...
