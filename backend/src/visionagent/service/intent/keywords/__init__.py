"""Keyword-only intent: no LLM call at all.

Useful as a fallback when the provider is down, and as the cheap end of the
scale the Protocol exists to make swappable. Always reports KNOWLEDGE, because
without a classifier the safe assumption is that retrieval is wanted.
"""
from __future__ import annotations

from visionagent.models import Intent, Scenario


async def analyze_query_intent(queries: list[str]) -> Intent:
    from visionagent.utils.keyword_extraction import (
        extract_keywords_advanced,
    )

    high, low = extract_keywords_advanced(" ".join(queries))
    return Intent(
        scenario=Scenario.KNOWLEDGE,
        intents=["kb(filter)"],
        keywords_high=list(high),
        keywords_low=list(low),
    )


async def analyze_chat_scenario(question: str) -> str:
    # No classifier, same reasoning as always reporting KNOWLEDGE above: the
    # safe assumption is that retrieval is wanted, so every turn takes the
    # professional path rather than skipping search on a guess.
    return "professional"
