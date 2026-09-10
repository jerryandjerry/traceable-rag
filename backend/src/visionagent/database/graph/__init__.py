"""The per-user knowledge graph on disk.

Three files per user under the graph directory: a GraphML graph of entities and
relations, and two JSON vector indexes over their embeddings. This package owns
the formats, the file handles and the backend choice. What goes into them --
which entities exist, how they merge -- is the vectorstore slot's decision.

`GraphRepository` is the front door for the live path. It is what a retrieval
tool reads and what the indexing slot writes through, so neither has to know
that a graph is three files or where they live.
"""
from visionagent.database.graph.identity import (
    GraphVectorIdentityError,
    canonical_edge_vector_ids,
    edge_vector_id,
)
from visionagent.database.graph.lightrag_storage import (
    GRAPH_FIELD_SEP,
    NetworkXGraphStorage,
    nanoVectorDB,
)
from visionagent.database.graph.repository import (
    GraphPersistenceError,
    GraphRepository,
    async_tenant_lock,
    atomic_write,
    tenant_lock,
)

__all__ = [
    "GRAPH_FIELD_SEP", "GraphPersistenceError", "GraphRepository",
    "GraphVectorIdentityError", "NetworkXGraphStorage",
    "async_tenant_lock", "atomic_write", "tenant_lock",
    "canonical_edge_vector_ids", "edge_vector_id",
    "nanoVectorDB",
]
