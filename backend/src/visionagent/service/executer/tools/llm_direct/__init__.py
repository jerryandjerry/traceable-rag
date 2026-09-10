"""The direct-answer tool over model knowledge and session context."""
from __future__ import annotations

import asyncio
import logging

from visionagent.models import RetrievedChunk, SourceType, ToolContext, ToolName, ToolResult
from visionagent.providers.llm import _llm
from visionagent.service.executer.tools.base import Emitter

logger = logging.getLogger(__name__)


async def direct_llm_answer(
    query: str,
    *,
    context: ToolContext,
    emit: Emitter | None = None,
    session_context: str | None = None,
) -> ToolResult:
    """
    Direct LLM generation using query and session context

    Args:
        query: User query
        session_context: Long context from chatting session. Left unset it is
            read from the current session, so every tool takes the same
            explicit execution context; no identity is read from ambient state.

    Returns:
        A ToolResult whose chunk is the model's own answer
    """
    if session_context is None:
        from visionagent.database.session_context import session_context_manager

        session_context = (
            await asyncio.to_thread(
                session_context_manager.get_session_context, context.session_id
            )
            or ""
        )
    llm_prompt = f'''
Based on the session context and your knowledge, provide a comprehensive answer to the user's query.

Session Context:
{session_context}

User Query: {query}

Provide a detailed, helpful response.
'''
    
    try:
        result = await _llm().complete(
            prompt=llm_prompt,
            system='You are a world-renown architect and educator.',
        )

        return ToolResult(
            tool_name=ToolName.LLM,
            chunks=[
                RetrievedChunk(
                    id=f"llm_{abs(hash(query)) % 10000}",
                    content=result,
                    source_type=SourceType.CURRENT_CONTEXT,
                    score=1.0,
                    doc_name='Direct LLM Response',
                )
            ] if (result or "").strip() else [],
        )

    except Exception:
        logger.exception("direct LLM generation failed")
        return ToolResult(
            tool_name=ToolName.LLM, error="direct llm failed"
        )
