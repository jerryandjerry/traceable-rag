"""The retrieval tool slot.

Every tool answers the same question -- "what do you have about this?" -- and
returns the same shape, so the orchestrator can dispatch them uniformly.

A tool is a **function**, not an object: `rag`, `graphrag`,
`web_search_answer`, and `direct_llm_answer` live under `visionagent.service.executer.tools`.
Protocols are structural, so those functions satisfy this one as they stand --
no base class and no adapter whose whole body forwards the call.

`retrieve` is async because the work is I/O. LLM, hosted embedding, rerank, and
public web-search providers use native async transports. Synchronous
Elasticsearch and graph/local file operations cross an explicit
`asyncio.to_thread` boundary; local sentence-transformers inference uses the
same bridge for CPU/GPU work. The registry gathers independent tools so they
overlap without blocking the event loop.
"""
from __future__ import annotations

from typing import Protocol

from visionagent.models import ToolContext, ToolResult

# A tool emits leaf detail ("reading <doc>", "found one relevant piece"); the
# orchestrator owns the step tree and the timing. Optional and defaulting to
# nothing, so a caller using a tool standalone never constructs one.
Emitter = object


class RetrievalTool(Protocol):
    """One source of evidence."""

    async def __call__(
        self, query: str, *, context: ToolContext, emit: Emitter | None = None
    ) -> ToolResult: ...
