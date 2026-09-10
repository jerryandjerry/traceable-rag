import asyncio
import json
import logging
import os
from typing import Any

import networkx as nx
import numpy as np

from visionagent.config.settings import settings
from visionagent.database.graph.provenance import (
    placeholder_node_from_edges,
    remove_document_contributions,
)
from visionagent.providers.embedding import build_embedder

logger = logging.getLogger(__name__)

GRAPH_FIELD_SEP = " | "
DEFAULT_TOP_K = 5

class BaseVectorStorage:
    """JSON-backed vector storage used by the graph repository."""
    
    def __init__(self, embedding_dim: int, storage_file: str, namespace: str = "default") -> None:
        self.embedding_dim = embedding_dim
        # Persisted vector width is part of an index's format. Override
        # GRAPH_EMBEDDING_DIM to read another width, or migrate the index.
        self._embedder = build_embedder(dimensions=embedding_dim)
        self.storage_file = storage_file
        self.namespace = namespace
        self.ids: list[str] = []
        self.embeddings: list[list[float]] = []
        self.metadatas: list[dict[str, Any]] = []

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Upsert data to storage (LightRAG compatible)."""
        for item_id, item_data in data.items():
            if item_id in self.ids:
                # Aggregate text may change when one document contribution is
                # removed, so replacement must also replace its embedding.
                idx = self.ids.index(item_id)
                if self.metadatas[idx] == item_data:
                    continue
                embedding = await self._generate_embedding_for_upsert(
                    item_data.get("content", "")
                )
                self.metadatas[idx] = item_data
                self.embeddings[idx] = embedding
            else:
                content = item_data.get("content", "")
                embedding = await self._generate_embedding_for_upsert(content)
                # Mutate the three parallel arrays only after the await. A
                # cancelled provider call must not leave their lengths out of
                # sync.
                self.ids.append(item_id)
                self.embeddings.append(embedding)
                self.metadatas.append(item_data)

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding; query failures must reach the tool result."""
        return await self._embedder.aembed(text)

    async def _generate_embedding_for_upsert(self, text: str) -> list[float]:
        """Generate a persisted vector; provider failure aborts graph ingest."""
        return await self._generate_embedding(text)

    async def aclose(self) -> None:
        """Release the embedder's async transport, if one was opened."""
        await self._embedder.aclose()

    def save(self, file_path: str) -> None:
        """Write the vector arrays and metadata to JSON."""
        data = {
            "ids": self.ids,
            "embeddings": self.embeddings,
            "metadatas": self.metadatas
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, file_path: str) -> None:
        """Load from JSON file."""
        if os.path.exists(file_path):
            with open(file_path, encoding='utf-8') as f:
                data = json.load(f)
                self.ids = data.get("ids", [])
                self.embeddings = data.get("embeddings", [])
                self.metadatas = data.get("metadatas", [])

    async def query(self, query: str, top_k: int, ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Return the highest-scoring graph vectors."""
        if not self.embeddings or not self.metadatas:
            return []
        
        query_embedding = await self._generate_embedding(query)
        return await asyncio.to_thread(
            self._rank_embedding,
            query_embedding,
            top_k,
            ids,
        )

    def _rank_embedding(
        self,
        query_embedding: list[float],
        top_k: int,
        ids: list[str] | None,
    ) -> list[dict[str, Any]]:
        """CPU-only cosine ranking over the in-memory graph vectors."""
        if not self.embeddings:
            return []

        matrix: Any = np.asarray(self.embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0          # a zero row would poison every score
        q: Any = np.asarray(query_embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn == 0:
            return []
        sims = matrix / norms @ (q / qn)

        # The persisted query contract ranks globally before applying an ID filter.
        results = []
        for idx in np.argsort(-sims)[:top_k]:
            if ids is None or self.ids[idx] in ids:
                results.append({**self.metadatas[idx], "score": float(sims[idx])})
        return results
    

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all items related to a specific document ID."""
        indices_to_delete = []
        for i, metadata in enumerate(self.metadatas):
            if metadata.get('docnm') == doc_id:
                indices_to_delete.append(i)
        
        for i in reversed(indices_to_delete):
            del self.ids[i]
            del self.embeddings[i]
            del self.metadatas[i]
        
        logger.info("graph node vectors deleted count=%d", len(indices_to_delete))

    def delete_ids(self, item_ids: list[str]) -> int:
        indices = [index for index, item_id in enumerate(self.ids) if item_id in item_ids]
        for index in reversed(indices):
            del self.ids[index]
            del self.embeddings[index]
            del self.metadatas[index]
        return len(indices)


class nanoVectorDB(BaseVectorStorage):
    """Graph vector store loaded from one JSON file."""
    
    def __init__(self, embedding_dim: int, storage_file: str, namespace: str = "default"):
        super().__init__(embedding_dim, storage_file, namespace)
        self.load(storage_file)
    
    async def query(self, query: str, top_k: int, ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Return the highest-scoring graph vectors."""
        return await super().query(query, top_k, ids)
    

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all items related to a specific document ID."""
        indices_to_delete = []
        for i, metadata in enumerate(self.metadatas):
            if metadata.get('docnm') == doc_id:
                indices_to_delete.append(i)
        
        for i in reversed(indices_to_delete):
            del self.ids[i]
            del self.embeddings[i]
            del self.metadatas[i]
        
        logger.info("graph edge vectors deleted count=%d", len(indices_to_delete))


class BaseGraphStorage:
    """NetworkX graph operations used by the graph repository."""
    
    def __init__(self, namespace: str, working_dir: str | None = None):
        working_dir = working_dir or str(settings.graph_dir)
        self.namespace = namespace
        self.working_dir = working_dir
        self.graph = nx.Graph()
        os.makedirs(working_dir, exist_ok=True)

    async def upsert_node(self, node_id: str, node_data: dict[str, Any]) -> None:
        """Insert or replace a node."""
        self.graph.add_node(node_id, **node_data)

    async def upsert_edge(self, source: str, target: str, edge_data: dict[str, Any]) -> None:
        """Insert or replace an edge."""
        self.graph.add_edge(source, target, **edge_data)

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Return a copy of one node's attributes."""
        if self.graph.has_node(node_id):
            return dict(self.graph.nodes[node_id])
        return None

    async def has_node(self, node_id: str) -> bool:
        """Return whether a node exists."""
        return bool(self.graph.has_node(node_id))

    async def has_edge(self, source: str, target: str) -> bool:
        """Return whether an edge exists."""
        return bool(self.graph.has_edge(source, target))

    async def get_edge(self, source: str, target: str) -> dict[str, Any] | None:
        """Return a copy of one edge's attributes."""
        if self.graph.has_edge(source, target):
            return dict(self.graph.edges[source, target])
        return None

    def save(self, file_path: str) -> None:
        """Write the graph to GraphML."""
        nx.write_graphml(self.graph, file_path)

    def load(self, file_path: str) -> None:
        """Load from GraphML file."""
        if os.path.exists(file_path):
            self.graph = nx.read_graphml(file_path)

    async def remove_document(self, doc_id: str) -> dict[str, Any]:
        """Remove one document's ledger entries and rebuild shared records."""
        updated_edges: dict[tuple[str, str], dict[str, Any]] = {}
        deleted_edges: dict[tuple[str, str], dict[str, Any]] = {}
        for source, target, attrs in list(self.graph.edges(data=True)):
            rebuilt = remove_document_contributions(dict(attrs), doc_id, edge=True)
            if rebuilt is None:
                self.graph.remove_edge(source, target)
                deleted_edges[(source, target)] = dict(attrs)
            elif rebuilt != dict(attrs):
                self.graph.edges[source, target].clear()
                self.graph.edges[source, target].update(rebuilt)
                updated_edges[(source, target)] = rebuilt

        updated_nodes: dict[str, dict[str, Any]] = {}
        deleted_nodes: list[str] = []
        for node, attrs in list(self.graph.nodes(data=True)):
            rebuilt = remove_document_contributions(dict(attrs), doc_id, edge=False)
            if rebuilt is None:
                incident = [dict(data) for *_ends, data in self.graph.edges(node, data=True)]
                if incident:
                    rebuilt = placeholder_node_from_edges(
                        str(node), incident, created_at=attrs.get("created_at", 0)
                    )
                    self.graph.nodes[node].clear()
                    self.graph.nodes[node].update(rebuilt)
                    updated_nodes[str(node)] = rebuilt
                else:
                    self.graph.remove_node(node)
                    deleted_nodes.append(str(node))
            elif rebuilt != dict(attrs):
                self.graph.nodes[node].clear()
                self.graph.nodes[node].update(rebuilt)
                updated_nodes[str(node)] = rebuilt

        # Degree is derived state and must reflect edge removal.
        for node in self.graph.nodes:
            self.graph.nodes[node]["degree"] = self.graph.degree(node)
        return {
            "updated_nodes": updated_nodes,
            "deleted_nodes": deleted_nodes,
            "updated_edges": updated_edges,
            "deleted_edges": deleted_edges,
        }

    async def delete_by_doc_id(self, doc_id: str) -> None:
        await self.remove_document(doc_id)


class NetworkXGraphStorage(BaseGraphStorage):
    """GraphML-backed NetworkX storage."""
    
    def __init__(self, namespace: str, working_dir: str | None = None):
        working_dir = working_dir or str(settings.graph_dir)
        super().__init__(namespace, working_dir)
        self.graph_file = os.path.join(working_dir, f"graph_{namespace}.graphml")
        self.load(self.graph_file)

    def save(self, file_path: str | None = None) -> None:
        """Write the graph to its configured GraphML file."""
        save_path = file_path or self.graph_file
        super().save(save_path)
    
    async def get_nodes_batch(self, node_names: list[str]) -> dict[str, dict[str, Any]]:
        """Return attributes for the requested nodes that exist."""
        result = {}
        for node_name in node_names:
            if self.graph.has_node(node_name):
                result[node_name] = dict(self.graph.nodes[node_name])
        return result
    
    async def get_edges_batch(self, edge_pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
        """Return attributes for the requested edges that exist."""
        result = {}
        for edge_pair in edge_pairs:
            src = edge_pair.get("src")
            tgt = edge_pair.get("tgt")
            if src and tgt and self.graph.has_edge(src, tgt):
                result[(src, tgt)] = dict(self.graph.edges[src, tgt])
        return result
