"""The executer slot: a Plan in, ToolResults out.

The observer Protocol names only what execution reports: steps and evidence.
``pipeline.Tracer`` satisfies it structurally, so this lower layer never imports
its caller.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from visionagent.models import (
    Evidence,
    Plan,
    ToolContext,
    ToolName,
    ToolResult,
    TraceStep,
)


class ExecuterError(RuntimeError):
    """The step could not run at all, as distinct from a tool failing."""


@runtime_checkable
class ExecutionObserver(Protocol):
    """What an executer reports while it works. Implemented by pipeline.Tracer."""

    def step(self, label: str, *, note: str | None = None
             ) -> AbstractContextManager[TraceStep]: ...

    def emit(self, kind: str, label: str, *, chunk_id: str | None = None,
             thumbnail: str | None = None) -> Evidence: ...

    def emitter(self) -> Callable[..., Evidence]: ...


@runtime_checkable
class Executer(Protocol):
    """Runs a Plan and returns one result per call."""

    name: str

    def names(self) -> Iterator[ToolName] | list[ToolName]:
        """The tools this executer can actually run."""
        ...

    async def run(
        self,
        plan: Plan,
        *,
        context: ToolContext,
        allowed_tools: frozenset[ToolName],
        observer: ExecutionObserver | None = None,
    ) -> list[ToolResult]:
        """One ToolResult per call in the plan, in the plan's order.

        A tool that fails or times out yields a result carrying the error
        rather than raising: one provider outage must not end the round, and
        the caller decides what partial results are worth.

        `allowed_tools` is the turn's authorization, re-checked here even
        though the planner was only offered authorized tools. A plan built
        anywhere else must not be able to reach a tool policy refused. It is
        required, not optional: an executer that ran unrestricted when the
        argument was omitted was fail-open for any caller that forgot it.
        """
        ...
