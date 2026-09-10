"""The model call, and the JSON handling every caller needs around it.

Not a slot: turning a prompt into text is not a step in either pipeline; intent,
the evaluator, the direct-answer tool, and the answer generator all call it.
The provider is swappable through LLM_PROVIDER, and its application-loop client
is closed by the API lifespan.
"""
from __future__ import annotations

import re
from typing import Any

from visionagent.providers.llm.base import LLM
from visionagent.providers.llm.factory import build_llm

_LLM: LLM | None = None


def _llm() -> LLM:
    """One client for the module. Ten separate OpenAI() constructions across
    eight files is what the llm/ slot exists to remove."""
    global _LLM
    if _LLM is None:
        _LLM = build_llm()
    return _LLM


async def close_llm() -> None:
    """Close and clear the application-loop singleton, if it was created."""
    global _LLM
    client, _LLM = _LLM, None
    if client is not None:
        await client.aclose()


def extract_json_content(input_str: Any) -> Any:
    """Return the outermost JSON-array substring, or ``None``."""
    pattern = r'(\[[\s\S]*\])'
    match = re.search(pattern, input_str)
    return match.group(1) if match else None


async def middle_json_model(prompt: str) -> str:
    """Run the shared JSON-constrained model without blocking the event loop."""
    return await _llm().complete(
        prompt=prompt,
        system='You are a helpful assistant. Please respond with JSON format.',
        json_object=True,
    )
