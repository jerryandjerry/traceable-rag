"""The planner: intents in, a Plan of retrievals out."""
from __future__ import annotations

import asyncio

from visionagent.database.session_context import session_context_manager
from visionagent.models import Intent, Plan, ToolCall, ToolName, WebSearchMode

# Exact intent names and the tools they schedule. ``kb(...)`` is handled below.
_INTENT_TO_TOOLS = {
    "graphrag": (ToolName.GRAPHRAG,),
    "web_search": (ToolName.WEB_SEARCH,),
    "session_context": (ToolName.LLM,),
}
def _tools_for(intent_name: str) -> tuple[ToolName, ...]:
    """Map ``kb(...)`` variants to both KB retrievers; map other names exactly."""
    key = intent_name.strip().lower()
    if key.startswith("kb"):
        return (ToolName.RAG, ToolName.GRAPHRAG)
    return _INTENT_TO_TOOLS.get(key, ())
async def agent_plan(
    queries: list[str],
    intent: Intent,
    session_id: str | None = None,
    chat_id: str | None = None,
    available: list[ToolName] | None = None,
    web_search: WebSearchMode = WebSearchMode.AUTO,
) -> Plan:
    """
    Create action plans based on recognized intents

    Takes the whole Intent rather than its `intents` list: the pipeline is a
    chain of models, and unpacking one field to pass it on leaves the scenario
    and the graph keywords behind and drops back to an untyped argument.

    Args:
        queries: what to search for this round -- the question on the first
            round, the refined searches on a reflection round
        intent: the Intent from analyze_query_intent()
        session_id: when set, attached session context adds the LLM tool
        available: tool names the registry can actually run; anything outside
            this list is dropped rather than scheduled and then failing
        web_search: already resolved against policy. FORCE schedules a search
            this turn, DISABLED strips it from the plan whatever the intent
            asked for, AUTO leaves the decision to the intents.

    Returns:
        A Plan of retrievals to run this round
    """
    runnable = available if available is not None else [t.value for t in ToolName]
    calls: list[ToolCall] = []
    query_list = [q for q in queries if q] or [""]

    def add(tool: ToolName) -> None:
        if tool.value in runnable and not any(c.tool_name is tool for c in calls):
            calls.append(ToolCall(tool_name=tool, query=query_list))

    for name in intent.intents:
        for tool in _tools_for(name):
            add(tool)

    # Add session context if available
    if session_id:
        # The file-backed context store is deliberately synchronous. Isolate
        # that compatibility boundary here instead of making the async query
        # pipeline run the whole planner in a worker thread.
        session_context = await asyncio.to_thread(
            session_context_manager.get_session_context, session_id
        )
        if session_context:
            add(ToolName.LLM)

    if web_search is WebSearchMode.FORCE:
        add(ToolName.WEB_SEARCH)
    elif web_search is WebSearchMode.DISABLED:
        # Server policy overrides web calls proposed by intent analysis.
        calls = [c for c in calls if c.tool_name is not ToolName.WEB_SEARCH]

    # No intent matched anything runnable: answer from the model rather than
    # returning an empty plan, which reads downstream as "no results".
    if not calls:
        add(ToolName.LLM)

    return Plan(calls=calls)
