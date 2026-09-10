"""Elasticsearch, as this system reaches it.

The connection, the vendored hybrid-search engine and the chunk-id lookup. What
gets stored and how a document is shaped is the vectorstore slot's decision;
this package owns the client and answers queries.

One asymmetry worth naming: the vendored Dealer embeds the query itself inside
search, so this is the single place the storage layer calls an embedding
provider. Rewriting that engine to accept a vector is out of scope, and the
layering guard names the exception rather than letting it pass unnoticed.
"""
from visionagent.database.elasticsearch.chunks import (
    chunks_of_document,
    retrieve_chunks_from_es,
)
from visionagent.database.elasticsearch.retrieval import retrieve_content
from visionagent.database.elasticsearch.store import ElasticsearchStore

__all__ = [
    "ElasticsearchStore", "chunks_of_document", "retrieve_chunks_from_es",
    "retrieve_content",
]
