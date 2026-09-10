"""Intent by LLM: the scenario, context types, and graph keywords.

Both classifiers fall back to the safe broad answer rather than raising: a
classifier failure should degrade retrieval, not fail the request.
"""
from __future__ import annotations

import json
import logging

from visionagent.models import Intent, Scenario
from visionagent.providers.llm import extract_json_content, middle_json_model

logger = logging.getLogger(__name__)


def _scenario_word(response: str) -> str:
    """The scenario word, whether it arrives bare, quoted, or wrapped in JSON."""
    text = (response or "").strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, str):
        text = parsed
    elif isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, str):
                text = value
                break
    return text.strip().strip('"\'').lower()
async def analyze_chat_scenario(query: str) -> str:
    """Classify a query as ``casual``, ``casual_web``, or ``professional``.

    "casual_web" is a casual question whose answer is not in the model --
    today's weather, a price, the news. The casual path runs a web search for
    it when the turn's mode is AUTO.
    """
    scenario_prompt = f'''
You are a chat scenario analyzer. Analyze the user's query to determine the appropriate chat scenario.

User Query: "{query}"

Determine the chat scenario type:

1. "casual": Pure conversation, greetings, personal chat, simple questions the model can answer on its own
2. "casual_web": Simple questions that need current information from the web: today's weather, news, prices, scores, recent events
3. "professional": Technical questions requiring knowledge base search, document analysis, or complex research

Guidelines:
- Greetings, personal questions, simple chat → "casual"
- Questions about current events, weather, news, stock prices → "casual_web"
- Technical questions, document analysis, research questions → "professional"
- Questions requiring knowledge base search → "professional"

Return only the scenario string: "casual", "casual_web" or "professional"
'''
    
    try:
        response = await middle_json_model(scenario_prompt)
        # Accept the bare, quoted, and object-wrapped forms providers return.
        scenario = _scenario_word(response)

        if scenario in ["casual", "casual_web", "professional"]:
            return scenario
        else:
            logger.warning(
                "invalid chat scenario value=%r; defaulting to professional",
                scenario,
            )
            return "professional"
            
    except Exception:
        logger.exception("chat scenario analysis failed; defaulting to professional")
        return "professional"


async def analyze_query_intent(queries: list[str]) -> Intent:
    """
    Analyze user queries to determine required context types

    Takes a list because a round may search several ways at once -- the first
    round has one query, a reflection round has the refined ones -- and it is
    still one task, so it returns one Intent.

    Returns an Intent: the scenario, the intent strings
    (['web_search'], ['kb(filter)'], ['session_context']) and the keywords the
    graph is entered with.
    """
    from visionagent.utils.keyword_extraction import (
        extract_keywords_advanced,
    )

    query = " ".join(q for q in queries if q).strip()

    # Scenario first, and the classification below only for a knowledge
    # question: a greeting must not pay for an intent classification whose
    # answer is discarded.
    scenario = (
        Scenario.CASUAL
        if str(await analyze_chat_scenario(query)).lower().startswith("casual")
        else Scenario.KNOWLEDGE
    )
    high, low = extract_keywords_advanced(query)
    if scenario is Scenario.CASUAL:
        return Intent(
            scenario=scenario,
            intents=[],
            keywords_high=list(high),
            keywords_low=list(low),
        )

    def _intent(intents: list[str]) -> Intent:
        return Intent(
            scenario=scenario,
            intents=list(intents),
            keywords_high=list(high),
            keywords_low=list(low),
        )

    intent_prompt = f'''
You are an intelligent query analyzer. Analyze the user's query to determine what context is needed to provide the best answer.

User Query: "{query}"

Based on the query, determine which context sources are needed. Return ONLY a JSON array of strings from these options:
- "kb(filter)": Search documents in knowledge base with filter (filter can be "all" or specific document IDs)
- "web_search": Search internet for current information, news, weather, stock prices, etc.
- "session_context": Use uploaded file context

Guidelines:
1. For current events, weather, stock prices, news: return ["web_search"]
2. For professional/technical questions: return ["kb(filter)"]
3. For questions about specific documents: return ["kb(filter)"]
4. For complex queries needing both: return ["kb(filter)", "web_search"]

Examples:
- "Tell me about machine learning algorithms" → ["kb(filter)"]
- "What's in the Tesla earnings report?" → ["kb(filter)"]
- "Compare our product with market trends" → ["kb(filter)", "web_search"]

Return only the JSON array:
'''
    
    try:
        result = await middle_json_model(intent_prompt)
        logger.debug("intent recognition response received characters=%d", len(result))
        
        try:
            intents = json.loads(result)
            if isinstance(intents, list):
                return _intent(intents)
        except Exception:
            pass

        json_list = extract_json_content(result)
        if json_list:
            intents = json.loads(json_list)
            if isinstance(intents, list):
                return _intent(intents)

        logger.warning("intent response was invalid; using broad retrieval fallback")
        return _intent(["kb(filter)", "web_search"])

    except Exception:
        logger.exception("intent analysis failed; using broad retrieval fallback")
        return _intent(["kb(filter)", "web_search"])
