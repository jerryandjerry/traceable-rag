"""Reranker clients.

The scoring call to an external service. What to do with the scores -- sort,
cut to top_n, drop anything under the floor -- is the rerank slot's, and is the
same whichever provider answers.
"""
from visionagent.providers.rerank.dashscope.provider import DashScopeReranker

__all__ = ["DashScopeReranker"]
