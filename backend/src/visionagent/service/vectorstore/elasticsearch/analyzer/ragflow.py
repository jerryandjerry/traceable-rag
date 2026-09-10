"""RAGFlow's analyzer: NLTK + WordNet + Porter for English, trie-based
maximum-matching segmentation for Chinese, plus a synonym dictionary."""
from __future__ import annotations


class RagflowAnalyzer:
    """Implements `visionagent.service.vectorstore.elasticsearch.analyzer.base.Analyzer`."""

    name = "ragflow"

    def __init__(self) -> None:
        from visionagent.vendor.ragflow.rag.nlp import rag_tokenizer
        from visionagent.vendor.ragflow.rag.nlp.synonym import Dealer

        self._tok = rag_tokenizer
        self._syn = Dealer()

    def analyze(self, text: str) -> str:
        return str(self._tok.tokenize(text))

    def analyze_fine(self, text: str) -> str:
        return str(self._tok.fine_grained_tokenize(self.analyze(text)))

    def synonyms(self, term: str) -> list[str]:
        return list(self._syn.lookup(term) or [])
