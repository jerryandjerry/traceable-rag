"""The text analyzer slot.

Lucene's term for the chain that turns text into search terms: split,
normalise, lemmatise, stem, expand. "Tokenizer" names only the first stage and
collides with the ML sense of the word used elsewhere in this codebase.

RAGFlow reimplements this chain in Python and hands Elasticsearch a
pre-analyzed string -- `conf/mapping.json` maps `*_ltks` with
`"analyzer": "whitespace"`, so ES does nothing but re-split on spaces.

Index-time and query-time analysis must be identical or BM25 silently stops
matching: no error, just fewer results. The store holds one Analyzer and calls
it from both `index()` and `search()`, so that holds by construction.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Analyzer(Protocol):
    """Text to search terms."""

    name: str

    def analyze(self, text: str) -> str:
        """Space-joined coarse terms. Stored as `content_ltks`."""
        ...

    def analyze_fine(self, text: str) -> str:
        """Space-joined fine-grained terms. Stored as `content_sm_ltks`."""
        ...

    def synonyms(self, term: str) -> list[str]:
        """Expansions for one term; empty when there are none."""
        ...
