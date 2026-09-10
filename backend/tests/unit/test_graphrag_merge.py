"""Replay safety for LightRAG-compatible graph merges."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import pytest

from visionagent.database.graph import GraphPersistenceError, GraphRepository, edge_vector_id
from visionagent.database.graph.identity import canonical_edge_vector_ids
from visionagent.database.graph.lightrag_storage import BaseGraphStorage, BaseVectorStorage
from visionagent.database.graph.provenance import (
    DOCUMENT_NAMES_FIELD,
    LegacyGraphProvenanceError,
    dump_document_names,
    remove_document_contributions,
)
from visionagent.service.vectorstore.graphstore.lightrag_utils import (
    GRAPH_FIELD_SEP,
    _merge_edges_then_upsert,
    _merge_nodes_then_upsert,
    compute_mdhash_id,
)
from visionagent.service.vectorstore.graphstore.service import GraphRAGService

P = ParamSpec("P")
T = TypeVar("T")


def run_async(function: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, T]:
    """Run an async unit-test body without adding a pytest plugin dependency."""

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return asyncio.run(function(*args, **kwargs))

    return wrapper


class RecordingNetwork:
    """Small undirected graph needed by the merge helpers' degree updates."""

    def __init__(self) -> None:
        self.node_data: dict[str, dict[str, Any]] = {}
        self.edge_data: dict[tuple[str, str], dict[str, Any]] = {}

    def __bool__(self) -> bool:
        return bool(self.node_data)

    def __contains__(self, node_id: object) -> bool:
        return isinstance(node_id, str) and self.has_node(node_id)

    @staticmethod
    def edge_key(source: str, target: str) -> tuple[str, str]:
        return (source, target) if source <= target else (target, source)

    def has_node(self, node_id: str) -> bool:
        return node_id in self.node_data

    def add_node(self, node_id: str, node_data: dict[str, Any]) -> None:
        self.node_data.setdefault(node_id, {}).update(node_data)

    def has_edge(self, source: str, target: str) -> bool:
        return self.edge_key(source, target) in self.edge_data

    def add_edge(self, source: str, target: str, edge_data: dict[str, Any]) -> None:
        self.edge_data[self.edge_key(source, target)] = dict(edge_data)

    def degree(self, node_id: str) -> int:
        return sum(node_id in edge for edge in self.edge_data)


class RecordingGraph:
    """The async storage surface used by the merge helpers, with write counts."""

    def __init__(self) -> None:
        self.graph = RecordingNetwork()
        self.node_upserts = 0
        self.edge_upserts = 0

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        if not self.graph.has_node(node_id):
            return None
        return dict(self.graph.node_data[node_id])

    async def has_node(self, node_id: str) -> bool:
        return bool(self.graph.has_node(node_id))

    async def upsert_node(self, node_id: str, node_data: dict[str, Any]) -> None:
        self.node_upserts += 1
        self.graph.add_node(node_id, node_data)

    async def has_edge(self, source: str, target: str) -> bool:
        return bool(self.graph.has_edge(source, target))

    async def get_edge(self, source: str, target: str) -> dict[str, Any] | None:
        if not self.graph.has_edge(source, target):
            return None
        return dict(self.graph.edge_data[self.graph.edge_key(source, target)])

    async def upsert_edge(self, source: str, target: str, edge_data: dict[str, Any]) -> None:
        self.edge_upserts += 1
        self.graph.add_edge(source, target, edge_data)


class RecordingVDB:
    def __init__(self) -> None:
        self.writes: list[dict[str, dict[str, Any]]] = []

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        self.writes.append(copy.deepcopy(data))


def node_record(
    source_id: str,
    *,
    description: str = "first description",
    docnm: str = "first.pdf",
) -> dict[str, Any]:
    return {
        "entity_type": "category",
        "description": description,
        "source_id": source_id,
        "docnm": docnm,
    }


def edge_record(
    source_id: str,
    *,
    weight: float,
    description: str,
    keywords: str,
    docnm: str,
) -> dict[str, Any]:
    return {
        "weight": weight,
        "description": description,
        "keywords": keywords,
        "source_id": source_id,
        "docnm": docnm,
    }


@run_async
async def test_edge_vector_identity_keeps_ambiguous_concatenations_distinct() -> None:
    old_collision = hashlib.md5(b"ABC").hexdigest()
    canonical, changed = canonical_edge_vector_ids(
        [f"rel-{old_collision}", f"rel-{old_collision}"],
        [
            {"src_id": "AB", "tgt_id": "C"},
            {"src_id": "A", "tgt_id": "BC"},
        ],
    )
    assert changed
    assert canonical == [edge_vector_id("AB", "C"), edge_vector_id("A", "BC")]
    assert len(set(canonical)) == 2

    graph = RecordingGraph()
    vectors = RecordingVDB()
    service: Any = object.__new__(GraphRAGService)
    service.knowledge_graph = graph
    service.global_config = {}
    service.edge_vdb = vectors
    await service._process_edges(
        ("AB", "C"),
        [edge_record("one", weight=1, description="one", keywords="one", docnm="a")],
    )
    await service._process_edges(
        ("A", "BC"),
        [edge_record("two", weight=1, description="two", keywords="two", docnm="b")],
    )
    assert {next(iter(write)) for write in vectors.writes} == set(canonical)


def test_graph_repository_mid_save_failure_is_typed_and_retry_converges(
    tmp_path,
) -> None:
    class Writer:
        def __init__(self, payload: str, *, fail_once: bool = False) -> None:
            self.payload = payload
            self.fail_once = fail_once

        def save(self, path: str) -> None:
            if self.fail_once:
                self.fail_once = False
                raise OSError("simulated edge publication failure")
            from pathlib import Path

            Path(path).write_text(self.payload, encoding="utf-8")

    repository: Any = object.__new__(GraphRepository)
    repository.user_id = "42"
    repository.working_dir = str(tmp_path)
    repository.node_vdb = Writer("nodes-v1")
    repository.edge_vdb = Writer("edges-v1", fail_once=True)
    repository.knowledge_graph = Writer("graph-v1")

    with pytest.raises(GraphPersistenceError, match="edge vectors") as failure:
        repository.save()
    assert isinstance(failure.value.__cause__, OSError)
    assert (tmp_path / "vdb_nodes_42.json").read_text(encoding="utf-8") == "nodes-v1"
    assert not (tmp_path / "vdb_edges_42.json").exists()
    assert not (tmp_path / "graph_42.graphml").exists()

    repository.save()
    assert (tmp_path / "vdb_edges_42.json").read_text(encoding="utf-8") == "edges-v1"
    assert (tmp_path / "graph_42.graphml").read_text(encoding="utf-8") == "graph-v1"
    assert not list(tmp_path.glob("*.tmp"))


@run_async
async def test_node_replay_is_a_noop_even_if_reextraction_differs() -> None:
    graph = RecordingGraph()
    config = {"force_llm_summary_on_merge": 5}

    await _merge_nodes_then_upsert("Valve", [node_record("chunk-1")], graph, config)
    stored_before = await graph.get_node("Valve")
    writes_before = graph.node_upserts

    replay_result = await _merge_nodes_then_upsert(
        "Valve",
        [
            node_record(
                "chunk-1",
                description="different nondeterministic retry output",
                docnm="wrong.pdf",
            )
        ],
        graph,
        config,
    )

    assert await graph.get_node("Valve") == stored_before
    assert graph.node_upserts == writes_before
    assert {key: value for key, value in replay_result.items() if key != "entity_name"} == stored_before


@run_async
async def test_node_merge_keeps_a_genuinely_new_source_once() -> None:
    graph = RecordingGraph()
    config = {"force_llm_summary_on_merge": 5}
    await _merge_nodes_then_upsert("Valve", [node_record("chunk-1")], graph, config)

    merged = await _merge_nodes_then_upsert(
        "Valve",
        [
            node_record("chunk-1", description="must be ignored"),
            node_record("chunk-2", description="second description", docnm="second.pdf"),
        ],
        graph,
        config,
    )
    stored_after_new_source = await graph.get_node("Valve")

    assert set(merged["source_id"].split(GRAPH_FIELD_SEP)) == {
        "chunk-1",
        "chunk-2",
    }
    assert merged["description"] == "first description | second description"
    assert merged["docnm"] == "first.pdf | second.pdf"

    await _merge_nodes_then_upsert(
        "Valve",
        [node_record("chunk-2", description="must also be ignored")],
        graph,
        config,
    )
    assert await graph.get_node("Valve") == stored_after_new_source


@run_async
async def test_relation_replay_does_not_double_weight_or_rewrite_edge() -> None:
    graph = RecordingGraph()
    config: dict[str, Any] = {}
    first = edge_record(
        "chunk-1",
        weight=2.5,
        description="first relation",
        keywords="first",
        docnm="first.pdf",
    )
    await _merge_edges_then_upsert("A", "B", [first], graph, config)
    stored_before = await graph.get_edge("A", "B")
    edge_writes_before = graph.edge_upserts
    node_writes_before = graph.node_upserts

    replay_result = await _merge_edges_then_upsert(
        "A",
        "B",
        [
            edge_record(
                "chunk-1",
                weight=99.0,
                description="different nondeterministic retry output",
                keywords="different",
                docnm="wrong.pdf",
            )
        ],
        graph,
        config,
    )

    assert replay_result == stored_before
    assert await graph.get_edge("A", "B") == stored_before
    assert graph.edge_upserts == edge_writes_before
    assert graph.node_upserts == node_writes_before


@run_async
async def test_relation_merge_counts_only_genuinely_new_sources() -> None:
    graph = RecordingGraph()
    config: dict[str, Any] = {}
    await _merge_edges_then_upsert(
        "A",
        "B",
        [
            edge_record(
                "chunk-1",
                weight=2.5,
                description="first relation",
                keywords="first",
                docnm="first.pdf",
            )
        ],
        graph,
        config,
    )

    merged = await _merge_edges_then_upsert(
        "A",
        "B",
        [
            edge_record(
                "chunk-1",
                weight=100.0,
                description="replayed relation",
                keywords="replayed",
                docnm="wrong.pdf",
            ),
            edge_record(
                "chunk-2",
                weight=3.0,
                description="second relation",
                keywords="second",
                docnm="second.pdf",
            ),
        ],
        graph,
        config,
    )
    assert merged is not None
    stored_after_new_source = await graph.get_edge("A", "B")

    assert merged["weight"] == 5.5
    assert set(merged["source_id"].split(GRAPH_FIELD_SEP)) == {
        "chunk-1",
        "chunk-2",
    }
    assert merged["description"] == "first relation | second relation"
    assert merged["keywords"] == "first, second"
    assert merged["docnm"] == "first.pdf | second.pdf"

    await _merge_edges_then_upsert(
        "A",
        "B",
        [
            edge_record(
                "chunk-2",
                weight=200.0,
                description="retry must be ignored",
                keywords="retry",
                docnm="wrong-again.pdf",
            )
        ],
        graph,
        config,
    )
    assert await graph.get_edge("A", "B") == stored_after_new_source


@run_async
async def test_reversed_mixed_merge_preserves_the_stored_edge_orientation() -> None:
    graph = RecordingGraph()
    await _merge_edges_then_upsert(
        "A",
        "B",
        [edge_record(
            "chunk-1",
            weight=1.0,
            description="first",
            keywords="first",
            docnm="first.pdf",
        )],
        graph,
        {},
    )

    merged = await _merge_edges_then_upsert(
        "B",
        "A",
        [edge_record(
            "chunk-2",
            weight=2.0,
            description="second",
            keywords="second",
            docnm="second.pdf",
        )],
        graph,
        {},
    )

    assert merged is not None
    assert (merged["src_id"], merged["tgt_id"]) == ("A", "B")
    stored = await graph.get_edge("A", "B")
    assert stored is not None
    assert (stored["src_id"], stored["tgt_id"]) == ("A", "B")


@run_async
async def test_pure_replay_keeps_node_and_edge_vdb_metadata_byte_stable() -> None:
    graph = RecordingGraph()
    node_vdb = RecordingVDB()
    edge_vdb = RecordingVDB()
    service: Any = object.__new__(GraphRAGService)
    service.knowledge_graph = graph
    service.global_config = {}
    service.node_vdb = node_vdb
    service.edge_vdb = edge_vdb

    node = node_record("chunk-1")
    edge = edge_record(
        "chunk-1",
        weight=2.0,
        description="first",
        keywords="first",
        docnm="first.pdf",
    )
    await service._process_entity_name("A", [node])
    await service._process_edges(("A", "B"), [edge])
    first_node_metadata = copy.deepcopy(node_vdb.writes[-1])
    first_edge_metadata = copy.deepcopy(edge_vdb.writes[-1])

    await service._process_entity_name(
        "A", [node_record("chunk-1", description="retry drift", docnm="wrong.pdf")]
    )
    await service._process_edges(
        ("A", "B"),
        [edge_record(
            "chunk-1",
            weight=99.0,
            description="retry drift",
            keywords="wrong",
            docnm="wrong.pdf",
        )],
    )

    assert node_vdb.writes[-1] == first_node_metadata
    assert edge_vdb.writes[-1] == first_edge_metadata


@pytest.mark.parametrize("merge_kind", ["node", "edge"])
@run_async
async def test_merge_rejects_unkeyed_contributions(merge_kind: str) -> None:
    graph = RecordingGraph()
    with pytest.raises(ValueError, match="non-empty source_id"):
        if merge_kind == "node":
            await _merge_nodes_then_upsert("A", [{}], graph, {})
        else:
            await _merge_edges_then_upsert("A", "B", [{}], graph, {})


@run_async
async def test_shared_node_and_edge_remove_only_one_document_contribution() -> None:
    graph = RecordingGraph()
    await _merge_nodes_then_upsert(
        "Valve", [node_record("doc-a:0", description="only A", docnm="a.pdf")], graph, {}
    )
    await _merge_nodes_then_upsert(
        "Valve", [node_record("doc-b:0", description="only B", docnm="b.pdf")], graph, {}
    )
    await _merge_edges_then_upsert(
        "Valve", "Pipe",
        [edge_record("doc-a:0", weight=2, description="A edge", keywords="alpha", docnm="a.pdf")],
        graph, {},
    )
    await _merge_edges_then_upsert(
        "Pipe", "Valve",
        [edge_record("doc-b:0", weight=3, description="B edge", keywords="beta", docnm="b.pdf")],
        graph, {},
    )

    node = remove_document_contributions(
        (await graph.get_node("Valve")) or {}, "a.pdf", edge=False
    )
    edge = remove_document_contributions(
        (await graph.get_edge("Valve", "Pipe")) or {}, "a.pdf", edge=True
    )
    assert node is not None and node["docnm"] == "b.pdf"
    assert node["source_id"] == "doc-b:0" and node["description"] == "only B"
    assert edge is not None and edge["docnm"] == "b.pdf"
    assert edge["source_id"] == "doc-b:0"
    assert edge["description"] == "B edge"
    assert edge["keywords"] == "beta" and edge["weight"] == 3
    assert "only A" not in str(node) and "A edge" not in str(edge)


def test_legacy_target_aggregate_fails_closed_instead_of_leaking_or_overdeleting() -> None:
    legacy = {
        "description": "cannot attribute fragments",
        "source_id": "old-a | old-b",
        "docnm": "a.pdf | b.pdf",
    }
    with pytest.raises(LegacyGraphProvenanceError, match="requires reindexing"):
        remove_document_contributions(legacy, "a.pdf", edge=False)
    with pytest.raises(LegacyGraphProvenanceError, match="requires reindexing"):
        remove_document_contributions(legacy, "a.pdf | b.pdf", edge=False)
    assert remove_document_contributions(legacy, "other.pdf", edge=False) == legacy


@run_async
async def test_service_deletion_rewrites_graph_and_vector_metadata_together(tmp_path) -> None:
    graph = BaseGraphStorage("tenant", str(tmp_path))
    await _merge_nodes_then_upsert(
        "Valve", [node_record("a:0", description="A", docnm="a.pdf")], graph, {}
    )
    await _merge_nodes_then_upsert(
        "Valve", [node_record("b:0", description="B", docnm="b.pdf")], graph, {}
    )
    await _merge_edges_then_upsert(
        "Valve", "Pipe",
        [edge_record("a:0", weight=2, description="A edge", keywords="alpha", docnm="a.pdf")],
        graph, {},
    )
    await _merge_edges_then_upsert(
        "Valve", "Pipe",
        [edge_record("b:0", weight=3, description="B edge", keywords="beta", docnm="b.pdf")],
        graph, {},
    )

    class VDB:
        def __init__(self) -> None:
            self.rows: dict[str, dict[str, Any]] = {}
            self.ids: list[str] = []
            self.metadatas: list[dict[str, Any]] = []

        async def upsert(self, rows):
            self.rows.update(copy.deepcopy(rows))

        def delete_ids(self, ids):
            return sum(self.rows.pop(item, None) is not None for item in ids)

    class Repo:
        saved = False

        def save(self):
            self.saved = True

    service: Any = object.__new__(GraphRAGService)
    service.user_id = "42"
    service.knowledge_graph = graph
    service.node_vdb = VDB()
    service.edge_vdb = VDB()
    service.repository = Repo()
    result = await service._delete_file_data("a.pdf")

    assert result["success"]
    assert result["graph_nodes_updated"] == 2
    assert result["graph_edges_updated"] == 1
    node = await graph.get_node("Valve")
    edge = await graph.get_edge("Valve", "Pipe")
    assert node is not None and node["docnm"] == "b.pdf"
    assert edge is not None and edge["docnm"] == "b.pdf"
    assert next(iter(service.node_vdb.rows.values()))["source_id"] == "b:0"
    assert next(iter(service.edge_vdb.rows.values()))["source_id"] == "b:0"
    assert json.loads(
        next(iter(service.node_vdb.rows.values()))[DOCUMENT_NAMES_FIELD]
    ) == ["b.pdf"]
    assert json.loads(
        next(iter(service.edge_vdb.rows.values()))[DOCUMENT_NAMES_FIELD]
    ) == ["b.pdf"]
    assert service.repository.saved


@run_async
async def test_compensation_repairs_shared_vectors_published_before_old_graph(
    tmp_path,
) -> None:
    """A failed vector-first save may expose new derived rows with old GraphML."""
    graph = BaseGraphStorage("tenant", str(tmp_path))
    await _merge_nodes_then_upsert(
        "Valve",
        [node_record("b:0", description="only B", docnm="b.pdf")],
        graph,
        {},
    )
    await _merge_edges_then_upsert(
        "Valve",
        "Pipe",
        [
            edge_record(
                "b:0",
                weight=2,
                description="B edge",
                keywords="beta",
                docnm="b.pdf",
            )
        ],
        graph,
        {},
    )

    class VDB:
        def __init__(self, item_id: str, metadata: dict[str, Any]) -> None:
            self.ids = [item_id]
            self.metadatas = [metadata]

        async def upsert(self, rows: dict[str, dict[str, Any]]) -> None:
            for item_id, metadata in copy.deepcopy(rows).items():
                if item_id in self.ids:
                    self.metadatas[self.ids.index(item_id)] = metadata
                else:
                    self.ids.append(item_id)
                    self.metadatas.append(metadata)

        def delete_ids(self, ids: list[str]) -> int:
            removed = 0
            for item_id in ids:
                if item_id in self.ids:
                    index = self.ids.index(item_id)
                    self.ids.pop(index)
                    self.metadatas.pop(index)
                    removed += 1
            return removed

    class Repo:
        def save(self) -> None:
            return None

    # These rows are the new derived files from document A+B. The GraphML
    # above is deliberately still the old B-only ledger.
    node_vectors = VDB(
        compute_mdhash_id("Valve", prefix="ent-"),
        {
            "entity_name": "Valve",
            "entity_type": "category",
            "content": "Valve\nonly A | only B",
            "source_id": "a:0 | b:0",
            "docnm": "a.pdf | b.pdf",
            DOCUMENT_NAMES_FIELD: dump_document_names({"a.pdf", "b.pdf"}),
        },
    )
    edge_vectors = VDB(
        edge_vector_id("Valve", "Pipe"),
        {
            "src_id": "Valve",
            "tgt_id": "Pipe",
            "keywords": "alpha, beta",
            "content": "Valve Pipe alpha, beta A edge | B edge",
            "source_id": "a:0 | b:0",
            "docnm": "a.pdf | b.pdf",
            DOCUMENT_NAMES_FIELD: dump_document_names({"a.pdf", "b.pdf"}),
            "weight": 3,
        },
    )
    service: Any = object.__new__(GraphRAGService)
    service.user_id = "42"
    service.knowledge_graph = graph
    service.node_vdb = node_vectors
    service.edge_vdb = edge_vectors
    service.repository = Repo()

    await service._delete_file_data("a.pdf")

    assert node_vectors.metadatas[0]["docnm"] == "b.pdf"
    assert node_vectors.metadatas[0]["source_id"] == "b:0"
    assert "only A" not in node_vectors.metadatas[0]["content"]
    assert edge_vectors.metadatas[0]["docnm"] == "b.pdf"
    assert edge_vectors.metadatas[0]["source_id"] == "b:0"
    assert "A edge" not in edge_vectors.metadatas[0]["content"]


@run_async
async def test_lossless_vector_provenance_removes_delimiter_named_orphans(
    tmp_path,
) -> None:
    """A vector-first partial publish must not parse a filename as two docs."""
    graph = BaseGraphStorage("tenant", str(tmp_path))
    file_name = "a | b.pdf"

    class VDB:
        def __init__(self, item_id: str, metadata: dict[str, Any]) -> None:
            self.ids = [item_id]
            self.metadatas = [metadata]

        async def upsert(self, _rows):
            raise AssertionError("an orphan has no authoritative row to rebuild")

        def delete_ids(self, ids):
            for item_id in ids:
                if item_id in self.ids:
                    index = self.ids.index(item_id)
                    self.ids.pop(index)
                    self.metadatas.pop(index)
            return len(ids)

    class EmptyVDB(VDB):
        def __init__(self) -> None:
            self.ids = []
            self.metadatas = []

    class Repo:
        def save(self):
            return None

    node_vectors = VDB(
        compute_mdhash_id("Missing", prefix="ent-"),
        {
            "entity_name": "Missing",
            "docnm": file_name,
            DOCUMENT_NAMES_FIELD: dump_document_names({file_name}),
        },
    )
    service: Any = object.__new__(GraphRAGService)
    service.user_id = "42"
    service.knowledge_graph = graph
    service.node_vdb = node_vectors
    service.edge_vdb = EmptyVDB()
    service.repository = Repo()

    await service._delete_file_data(file_name)

    assert node_vectors.ids == []


@run_async
async def test_malformed_vector_document_provenance_fails_before_mutation(
    tmp_path,
) -> None:
    graph = BaseGraphStorage("tenant", str(tmp_path))

    class VDB:
        ids = ["orphan"]
        metadatas = [
            {
                "entity_name": "Missing",
                "docnm": "a.pdf",
                DOCUMENT_NAMES_FIELD: "not-json",
            }
        ]

        def delete_ids(self, _ids):
            raise AssertionError("malformed provenance must not mutate vectors")

        async def upsert(self, _rows):
            raise AssertionError("malformed provenance must not mutate vectors")

    class EmptyVDB:
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        def delete_ids(self, _ids):
            return 0

        async def upsert(self, _rows):
            return None

    service: Any = object.__new__(GraphRAGService)
    service.user_id = "42"
    service.knowledge_graph = graph
    service.node_vdb = VDB()
    service.edge_vdb = EmptyVDB()
    service.repository = type("Repo", (), {"save": lambda self: None})()

    with pytest.raises(LegacyGraphProvenanceError, match="malformed"):
        await service._delete_file_data("a.pdf")


@run_async
async def test_identical_vector_upsert_is_a_provider_free_byte_stable_noop() -> None:
    store: Any = object.__new__(BaseVectorStorage)
    store.ids = ["entity"]
    store.metadatas = [{"content": "stable", "docnm": "a.pdf"}]
    store.embeddings = [[1.0, 2.0]]
    calls = 0

    async def embed(_content: str) -> list[float]:
        nonlocal calls
        calls += 1
        return [9.0, 9.0]

    store._generate_embedding_for_upsert = embed
    before_metadata = copy.deepcopy(store.metadatas)
    before_embeddings = copy.deepcopy(store.embeddings)
    await store.upsert({"entity": {"content": "stable", "docnm": "a.pdf"}})
    assert calls == 0
    assert store.metadatas == before_metadata
    assert store.embeddings == before_embeddings


@run_async
async def test_delete_removes_exact_orphan_vector_but_fails_closed_for_shared_legacy_orphan(
    tmp_path,
) -> None:
    graph = BaseGraphStorage("tenant", str(tmp_path))

    class VDB:
        def __init__(self, docnm: str) -> None:
            self.ids = ["orphan"]
            self.metadatas = [{"entity_name": "Missing", "docnm": docnm}]

        def delete_ids(self, ids):
            for item_id in ids:
                if item_id in self.ids:
                    index = self.ids.index(item_id)
                    self.ids.pop(index)
                    self.metadatas.pop(index)
            return len(ids)

        async def upsert(self, _rows):
            return None

    class EmptyVDB(VDB):
        def __init__(self) -> None:
            self.ids = []
            self.metadatas = []

    class Repo:
        def save(self):
            return None

    service: Any = object.__new__(GraphRAGService)
    service.user_id = "42"
    service.knowledge_graph = graph
    service.repository = Repo()
    service.node_vdb = VDB("a.pdf")
    service.edge_vdb = EmptyVDB()
    await service._delete_file_data("a.pdf")
    assert service.node_vdb.ids == []

    service.node_vdb = VDB("a.pdf | b.pdf")
    with pytest.raises(LegacyGraphProvenanceError, match="requires reindexing"):
        await service._delete_file_data("a.pdf")
    assert service.node_vdb.ids == ["orphan"]
