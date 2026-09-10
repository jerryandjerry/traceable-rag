"""Elasticsearch: what a chunk looks like once it is stored.

The document shape, the BM25 tokens produced by analyzer/, and the kNN vector.
The client, the search engine and the delete primitives live in
database/elasticsearch; this decides what goes in.
"""
from visionagent.service.vectorstore.base import ChunkStore, ChunkStoreError
from visionagent.service.vectorstore.elasticsearch.store import ElasticsearchChunkStore
from visionagent.service.vectorstore.factory import build_chunkstore

__all__ = [
    "ChunkStore",
    "ChunkStoreError",
    "ElasticsearchChunkStore",
    "build_chunkstore",
]
