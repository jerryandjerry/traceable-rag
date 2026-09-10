"""Errors from the rerank step."""
from __future__ import annotations


class RerankError(RuntimeError):
    """The provider failed.

    Here rather than on the slot so a provider can raise it without importing
    the slot it implements: a Protocol is structural, and an implementation
    that has to import its own contract is not independent of it.
    """
