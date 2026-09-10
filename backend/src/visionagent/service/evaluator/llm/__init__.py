"""Judging the gathered context, and asking what is missing.

`evaluate_context_sufficiency`, `is_sufficient`, and `reflection` are the names
the Evaluator Protocol declares, so this module owns both model evaluation and
the threshold policy that decides whether another round is needed.
"""
from __future__ import annotations

import json
import logging

from visionagent.config.settings import settings
from visionagent.models import Evaluation, RetrievedChunk, ToolName, WebSearchMode
from visionagent.providers.llm import extract_json_content, middle_json_model

logger = logging.getLogger(__name__)


def is_sufficient(evaluation: Evaluation) -> bool:
    """Apply the configured loop threshold to an evaluator score.

    `Evaluation` is intentionally only the typed model output.  The evaluator
    owns the policy that turns that output into a routing decision, and the
    pipeline owns the decision to start another retrieval round.
    """
    return evaluation.sufficient_score > settings.sufficient_threshold


async def reflection(
    user_query: str,
    context_list: list[RetrievedChunk],
    evaluation: Evaluation,
    web_search: WebSearchMode = WebSearchMode.AUTO,
    available: list[ToolName] | None = None,
) -> list[str] | None:
    """
    Perform reflection to generate additional queries based on insufficient context

    Args:
        user_query: Original user query
        context_list: Current gathered context
        evaluation: Evaluation from evaluate_context_sufficiency()
        web_search: FORCE adds a web query this round even when the model
            proposed none; DISABLED drops any it did propose;
            False lets the model decide (not a prohibition)
        available: tool names the registry can actually run; anything the model
            asks for outside this list is dropped

    Returns:
        Refined queries for another round, or None to stop. The loop re-enters
        at analyze_query_intent, so what comes back is what to search for --
        not which tools to run, which is the planner's job.
    """
    if is_sufficient(evaluation):
        return None
    context_summary = '\n'.join([
        f"{i+1}. {chunk.content[:500]}..."
        for i, chunk in enumerate(context_list)
    ])
    
    reflection_prompt = f'''
Analyze what additional information is needed to better answer the user's query.

Original Query: "{user_query}"
Evaluation Score: {evaluation.sufficient_score}
Evaluation Reason: {evaluation.reasons or 'Unknown'}
What is missing: {evaluation.comments or 'Unknown'}

Current Context:
{context_summary}

Generate up to 3 additional queries to fill information gaps.

Available Tools:
- "kb(filter)": Search specific documents 
- "web_search": Search internet for current information

Return JSON array of additional queries:
[
  {{
    "intent": "tool_type",
    "query": "specific_query"
  }}
]

If no additional queries would help, return: []
'''
    
    try:
        result = await middle_json_model(reflection_prompt)
        
        try:
            additional_queries = json.loads(result)
            if isinstance(additional_queries, list):
                if not additional_queries:
                    return None
                    
                kb_queries = []
                web_queries = []
                
                for item in additional_queries:
                    intent = item.get('intent', 'kb(filter)')
                    query = item.get('query', user_query)
                    
                    # Prefix, not equality: the prompt calls this "kb(filter)"
                    # and says the filter may be "all" or a document id, so the
                    # model emits kb(all) and the knowledge base was dropped
                    # from the follow-up round whenever it did.
                    if str(intent).strip().lower().startswith('kb'):
                        kb_queries.append(query)
                    elif intent == 'web_search':
                        web_queries.append(query)
                    elif intent == 'session_context':
                        pass

                refined = kb_queries + web_queries

                # The mode was resolved against policy before the turn started.
                # A forced toggle must survive the extra round, and a denial must
                # not be reintroduced by reflection.
                if web_search is WebSearchMode.FORCE and not web_queries:
                    refined.append(user_query)
                elif web_search is WebSearchMode.DISABLED:
                    refined = kb_queries

                return [q for q in refined if q] or None
        except Exception:
            pass
            
        json_list = extract_json_content(result)
        if json_list:
            additional_queries = json.loads(json_list)
            if isinstance(additional_queries, list) and additional_queries:
                return [user_query]

        return None
        
    except Exception:
        logger.exception("context reflection failed")
        return None
async def evaluate_context_sufficiency(
    context_list: list[RetrievedChunk], query: str
) -> Evaluation:
    """
    Evaluate if the gathered context is sufficient to answer the query.
    Returns an Evaluation:
    - sufficient_score: float between 0.0 and 1.0
    - reasons: string explanation of the score
    - comments: string comments for improvement
    """
    if not context_list:
        return Evaluation(
            sufficient_score=0.0,
            reasons="No context provided.",
            comments="Search again for more context.",
        )
    
    # Evaluate KB content quality
    evaluation_prompt = f"""
Evaluate if the following knowledge is sufficient to answer the user's query.

User Query: "{query}"

Knowledge:{chr(10).join([chunk.content for chunk in context_list])}

Rate the context quality from 0.0 to 1.0 and determine if it's sufficient.
You should consider 5 aspects below for scoring:
- coverage (weight 0.30): Does the context cover all key parts of the query?
- specificity (weight 0.30): Does it provide concrete, query-specific facts (not generic fluff)?
- correctness_likelihood (weight 0.20): Does the context appear internally plausible and well-sourced (no obvious errors)?
- consistency (weight 0.1): Are there contradictions across snippets?
- concision (weight 0.1): Is there enough signal without overwhelming noise?
Your final score is a combination of above 5 aspects, range from 0.0 to 1.0

Return only a JSON object with:
- "sufficient_score": float between 0.0 and 1.0
- "reasons": string explanation of your score
- "comments": string comments for improvement

JSON response:
"""
    
    try:
        response = await middle_json_model(evaluation_prompt)
        logger.debug("context evaluation response received characters=%d", len(response))
        evaluation = json.loads(response)
        
        if not all(key in evaluation for key in ['sufficient_score', 'comments', 'reasons']):
            raise ValueError("Invalid evaluation format")
        
        return Evaluation(
            sufficient_score=max(0.0, min(1.0, float(evaluation["sufficient_score"]))),
            reasons=str(evaluation.get("reasons") or ""),
            comments=str(evaluation.get("comments") or ""),
        )

    except Exception:
        logger.exception("context evaluation failed")
        # A failed evaluator must not trigger another round of retrieval, so the
        # fallback is "sufficient".
        return Evaluation(sufficient_score=1.0, reasons="", comments="")
