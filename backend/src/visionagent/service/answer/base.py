"""The answer slot: the gathered context in, SSE frames out.

The Protocol is written around the existing function names --
`get_chat_completion` and `casual_chat_completion` in
`visionagent.service.answer.chat` -- so that module satisfies it as it stands.

Both are async generators. Cancellation therefore reaches the provider stream
and closes its network response or subprocess instead of abandoning a blocking
generator in a worker thread.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol, runtime_checkable

from visionagent.models import RetrievedChunk


@runtime_checkable
class AnswerGenerator(Protocol):
    def get_chat_completion(
        self,
        session_id: str,
        question: str,
        context_list: list[RetrievedChunk],
        user_id: str,
        final_prompt: str,
        related_questions: list[str],
        snippets: list[RetrievedChunk],
        *,
        run_id: str,
        citation_ids: Mapping[str, str],
        media: dict[str, list[dict[str, Any]]] | None = None,
    ) -> AsyncIterator[str]:
        """The knowledge answer: tokens, citations, media, related questions,
        and the conversation write.

        `media` is whatever the web tool found this turn, or None. The slot
        emits it; it never reaches a provider itself, so nothing here can run
        outside the turn's authorization."""
        ...

    def casual_chat_completion(
        self,
        session_id: str,
        question: str,
        user_id: str,
        final_prompt: str,
        web_context: list[RetrievedChunk] | None = None,
        *,
        run_id: str,
        related_questions: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """The casual answer: tokens only, no retrieval payload."""
        ...
