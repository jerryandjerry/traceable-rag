"""The model call, and the JSON handling every caller needs around it.

Not a slot: turning a prompt into text is not a step in either pipeline, it is
something intent, the evaluator, the direct-answer tool and the answer generator
all call. The provider is swappable through LLM_PROVIDER.
"""
from visionagent.providers.llm.base import LLM, LLMError
from visionagent.providers.llm.factory import build_llm
from visionagent.providers.llm.helpers import (
    _llm,
    close_llm,
    extract_json_content,
    middle_json_model,
)

__all__ = [
    "LLM",
    "LLMError",
    "_llm",
    "build_llm",
    "close_llm",
    "extract_json_content",
    "middle_json_model",
]
