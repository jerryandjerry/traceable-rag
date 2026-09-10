"""Graph-vector migration must handle same-width provider changes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _migration_module() -> ModuleType:
    script = Path(__file__).parents[2] / "scripts" / "migrate_graph_embeddings.py"
    spec = importlib.util.spec_from_file_location("graph_embedding_migration", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _index(path: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "ids": ["node-1"],
        "embeddings": [[0.1, 0.2]],
        "metadatas": [
            {
                "content": "stored source text",
                "kind": "entity",
                "src_id": "A",
                "tgt_id": "B",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_same_width_is_unchanged_without_force(tmp_path: Path, monkeypatch) -> None:
    migration = _migration_module()
    path = tmp_path / "vdb_nodes_1.json"
    before = _index(path)

    def should_not_build(*args, **kwargs):
        raise AssertionError("same-width migration should be skipped without --force")

    import visionagent.providers.embedding as embedding

    monkeypatch.setattr(embedding, "build_embedder", should_not_build)
    assert migration.migrate(path, target_dim=2, dry_run=False) == (1, 2, 2)
    assert json.loads(path.read_text(encoding="utf-8")) == before
    assert not path.with_suffix(".json.bak").exists()


def test_force_reembeds_same_width_and_preserves_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    migration = _migration_module()
    path = tmp_path / "vdb_edges_1.json"
    before = _index(path)

    class FakeEmbedder:
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            assert texts == ["stored source text"]
            return [[0.9, 0.8]]

    import visionagent.providers.embedding as embedding

    monkeypatch.setattr(
        embedding,
        "build_embedder",
        lambda *, dimensions: FakeEmbedder(),
    )
    assert migration.migrate(
        path,
        target_dim=2,
        dry_run=False,
        force=True,
    ) == (1, 2, 2)

    after = json.loads(path.read_text(encoding="utf-8"))
    from visionagent.database.graph import edge_vector_id

    assert after["ids"] == [edge_vector_id("A", "B")]
    assert after["metadatas"] == before["metadatas"]
    assert after["embeddings"] == [[0.9, 0.8]]
    assert path.with_suffix(".json.bak").exists()


def test_legacy_edge_ids_migrate_without_reembedding(
    tmp_path: Path, monkeypatch
) -> None:
    migration = _migration_module()
    path = tmp_path / "vdb_edges_1.json"
    payload = {
        "ids": ["rel-legacy"],
        "embeddings": [[0.1, 0.2]],
        "metadatas": [
            {
                "src_id": "AB",
                "tgt_id": "C",
                "content": "AB C relation",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    def should_not_build(*args, **kwargs):
        raise AssertionError("an id-only migration must preserve the embedding")

    import visionagent.providers.embedding as embedding

    monkeypatch.setattr(embedding, "build_embedder", should_not_build)
    assert migration.edge_ids_need_migration(path)
    assert migration.migrate(path, target_dim=2, dry_run=False) == (1, 2, 2)

    from visionagent.database.graph import edge_vector_id

    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["ids"] == [edge_vector_id("AB", "C")]
    assert after["embeddings"] == payload["embeddings"]
    assert after["metadatas"] == payload["metadatas"]
    assert path.with_suffix(".json.bak").exists()
