"""Map each declared tool name to one retrieval function."""
from __future__ import annotations

import asyncio

from visionagent.models import ToolContext, ToolName, ToolResult
from visionagent.service.executer.tools.base import Emitter, RetrievalTool


class ToolRegistry:
    def __init__(self, tools: dict[ToolName, RetrievalTool] | None = None) -> None:
        self._tools: dict[ToolName, RetrievalTool] = dict(tools or {})

    def freeze(self) -> ToolRegistry:
        """Refuse further registration. Called once at construction.

        The application pipeline shares its registry across concurrent turns,
        so a register() call after start-up would change what tools exist
        underneath requests already running.
        """
        self._frozen = True
        return self

    def register(self, name: ToolName, tool: RetrievalTool) -> None:
        if getattr(self, "_frozen", False):
            raise RuntimeError(
                "the registry is frozen: tools are registered once at start-up, "
                "not while turns are running"
            )
        self._tools[name] = tool

    def get(self, name: ToolName | str) -> RetrievalTool:
        key = ToolName(name)
        if key not in self._tools:
            raise KeyError(f"no tool registered for {key!r}; have {sorted(self._tools)}")
        return self._tools[key]

    def has(self, name: ToolName | str) -> bool:
        try:
            return ToolName(name) in self._tools
        except ValueError:
            return False

    def names(self) -> list[ToolName]:
        return list(self._tools)

    # What each tool is for, in the planner prompt. Held here rather than on
    # the functions: a description is a property of the catalogue, not of the
    # retrieval itself.
    DESCRIPTIONS = {
        ToolName.RAG: (
            "Search the user's own indexed documents. Best for anything the "
            "uploaded corpus would cover: standards, specifications, drawings."
        ),
        ToolName.GRAPHRAG: (
            "Search the entity/relationship graph built from the user's "
            "documents. Best for questions about how things relate."
        ),
        ToolName.WEB_SEARCH: (
            "Search the public web. Best for current events, vendor products, "
            "or anything outside the user's corpus."
        ),
        ToolName.LLM: (
            "Answer from the model and the session's attached context, with no "
            "retrieval."
        ),
    }

    def describe(self) -> str:
        """The tool list for the planner prompt, generated not hardcoded."""
        return "\n".join(
            f"- {n.value}: {self.DESCRIPTIONS.get(n, '')}" for n in self._tools
        )

    async def run(self, name: ToolName | str, *, queries: list[str], context: ToolContext,
                  allowed_tools: frozenset[ToolName],
                  emit: Emitter | None = None) -> ToolResult:
        """Run one tool over every query in a call and merge the results.

        Merging here is what the four duplicated blocks each did by hand, each
        looping the queries and extending a list.

        `allowed_tools` is the turn's authorization and is required. It used
        to default to None meaning "not enforced", which made the one gateway
        every tool call passes through fail-open for any caller that forgot
        it. A test that wants everything passes frozenset(ToolName).
        """
        if ToolName(name) not in allowed_tools:
            # Defence in depth. The planner is already given only authorized
            # tools, so reaching here means a planner defect or a plan built
            # somewhere else. Refusing at the gateway means policy does not
            # depend on every upstream step being correct.
            return ToolResult(
                tool_name=ToolName(name),
                error=f"POLICY_DENIED: {ToolName(name).value} is not authorized for this turn",
            )

        tool = self.get(name)
        results = await asyncio.gather(
            *(tool(q, context=context, emit=emit) for q in queries),
            return_exceptions=True,
        )
        merged = ToolResult(tool_name=ToolName(name))
        errors: list[str] = []
        for r in results:
            if isinstance(r, BaseException):
                errors.append(str(r))
                continue
            merged.chunks.extend(r.chunks)
            for q in r.related_questions:
                if q not in merged.related_questions:
                    merged.related_questions.append(q)
            for field, key in (("images", "imageUrl"), ("videos", "link")):
                have = {m.get(key) for m in getattr(merged, field)}
                for m in getattr(r, field):
                    if m.get(key) not in have:
                        getattr(merged, field).append(m)
                        have.add(m.get(key))
            if r.error:
                errors.append(r.error)
        merged.error = "; ".join(errors) or None
        return merged


def build_registry() -> ToolRegistry:
    """The four tools this system ships -- the functions themselves."""
    from visionagent.service.executer.tools.graphrag import graphrag
    from visionagent.service.executer.tools.llm_direct import direct_llm_answer
    from visionagent.service.executer.tools.rag import rag
    from visionagent.service.executer.tools.web import web_search_answer

    return ToolRegistry({
        ToolName.RAG: rag,
        ToolName.GRAPHRAG: graphrag,
        ToolName.WEB_SEARCH: web_search_answer,
        ToolName.LLM: direct_llm_answer,
    }).freeze()
