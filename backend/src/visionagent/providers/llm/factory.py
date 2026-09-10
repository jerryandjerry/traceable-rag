"""Construct the configured LLM.

Adding a provider is one file plus one line here. Nothing upstream changes.
"""
from __future__ import annotations

import os
from typing import Any

from visionagent.providers.llm.base import LLM

_PROVIDERS = {"dashscope", "claude-cli"}


def build_llm(provider: str | None = None, **kwargs: Any) -> LLM:
    """Return the LLM named by `provider`, or `LLM_PROVIDER`, or the default."""
    name = (provider or os.getenv("LLM_PROVIDER") or "dashscope").lower()
    if name == "dashscope":
        from visionagent.providers.llm.dashscope import DashScopeLLM

        return DashScopeLLM(**kwargs)
    if name in ("claude-cli", "claude"):
        from visionagent.providers.llm.claude_cli import ClaudeCLI

        return ClaudeCLI(**kwargs)
    raise ValueError(f"unknown LLM provider {name!r}; known: {sorted(_PROVIDERS)}")
