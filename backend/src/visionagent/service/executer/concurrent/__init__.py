"""Running a Plan: dispatch every call at once, then de-duplicate.

The slot owns dispatch and first-seen result deduplication, returning data ready
for the next pipeline step.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager

from visionagent.config.settings import settings
from visionagent.models import (
    Plan,
    RetrievedChunk,
    SourceType,
    StepStatus,
    ToolCall,
    ToolContext,
    ToolName,
    ToolResult,
    TraceStep,
)
from visionagent.service.executer.base import ExecutionObserver
from visionagent.service.executer.tools import ToolRegistry, build_registry

logger = logging.getLogger(__name__)

# What each tool is called in the progress trace.
_LABELS = {
    ToolName.RAG: "searching user knowledge base",
    ToolName.GRAPHRAG: "searching user graph base",
    ToolName.WEB_SEARCH: "searching online",
    ToolName.LLM: "searching added context",
}

_SOURCE_TYPE_BY_TOOL = {
    ToolName.RAG.value: SourceType.KNOWLEDGE_BASE.value,
    # Graph hits resolve to chunks of the user's own documents, so they cite
    # as knowledge base rather than as a separate source.
    ToolName.GRAPHRAG.value: SourceType.KNOWLEDGE_BASE.value,
    ToolName.WEB_SEARCH.value: SourceType.WEB_SEARCH.value,
    ToolName.LLM.value: SourceType.CURRENT_CONTEXT.value,
}
def gather_results(results: list[ToolResult]) -> list[RetrievedChunk]:
    """Gather tool chunks without reranking.

    Flattens every tool's chunks and de-duplicates by id, keeping first-seen
    order. Each chunk carries the source type assigned by its retrieval tool;
    dispatch must not overwrite that provenance.
    """
    seen: set[str] = set()
    merged: list[RetrievedChunk] = []
    for r in results:
        for c in r.chunks:
            if c.id in seen:
                continue
            seen.add(c.id)
            merged.append(c)

    total = sum(len(r.chunks) for r in results)
    logger.debug(
        "gathered tool results candidates=%d unique=%d",
        total,
        len(merged),
    )
    return merged


class ConcurrentExecuter:
    """Every call in the plan at once; one result each, in the plan's order."""

    name = "concurrent"

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        tool_timeout_s: float | None = None,
    ) -> None:
        self.registry = registry or build_registry()
        self.tool_timeout_s = tool_timeout_s or settings.tool_timeout_s

    def names(self) -> list[ToolName]:
        return list(self.registry.names())

    async def run(
        self,
        plan: Plan,
        *,
        context: ToolContext,
        allowed_tools: frozenset[ToolName],
        observer: ExecutionObserver | None = None,
    ) -> list[ToolResult]:
        """Run every call in the plan concurrently.

        One tool failing marks its step FAILED and the round continues with
        partial results -- `return_exceptions=True` rather than letting one
        provider outage end the request.
        """
        if not plan.calls:
            return []

        async def one(call: ToolCall) -> ToolResult:
            label = _LABELS.get(call.tool_name, call.tool_name.value)
            with _observe(observer, label) as step:
                try:
                    result = await asyncio.wait_for(
                        self.registry.run(
                            call.tool_name,
                            queries=call.query,
                            context=context,
                            emit=observer.emitter() if observer else None,
                            allowed_tools=allowed_tools,
                        ),
                        timeout=self.tool_timeout_s,
                    )
                except TimeoutError:
                    if step is not None:
                        step.status = StepStatus.TIMEOUT
                    return ToolResult(
                        tool_name=call.tool_name,
                        error=f"timed out after {self.tool_timeout_s:.0f}s",
                        timed_out=True,
                    )
                if result.error and step is not None:
                    step.status = StepStatus.FAILED
                if observer is not None:
                    for c in result.chunks:
                        observer.emit(
                            "chunk", c.doc_name or c.url or c.id, chunk_id=c.id
                        )
                return result

        gathered = await asyncio.gather(
            *(one(c) for c in plan.calls), return_exceptions=True
        )
        out: list[ToolResult] = []
        for call, r in zip(plan.calls, gathered, strict=True):
            if isinstance(r, asyncio.CancelledError):
                raise r
            if isinstance(r, ToolResult):
                out.append(r)
                continue
            logger.error(
                "tool execution raised unexpectedly tool=%s",
                call.tool_name.value,
                exc_info=(type(r), r, r.__traceback__),
            )
            out.append(
                ToolResult(tool_name=call.tool_name, error="tool execution failed")
            )
        return out


@contextmanager
def _observe(observer: ExecutionObserver | None, label: str) -> Iterator[TraceStep | None]:
    """Open a trace step if anyone is watching, otherwise do nothing.

    The observer is optional so the slot can be driven from a test or a script
    with no tracer at all.
    """
    if observer is None:
        yield None
        return
    with observer.step(label) as step:
        yield step
