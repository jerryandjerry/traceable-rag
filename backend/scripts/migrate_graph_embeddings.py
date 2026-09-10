#!/usr/bin/env python
# ruff: noqa: T201 -- command-line migration reports progress to its operator
"""Re-embed the GraphRAG vector indices at the configured width.

    python backend/scripts/migrate_graph_embeddings.py --dry-run
    python backend/scripts/migrate_graph_embeddings.py
    python backend/scripts/migrate_graph_embeddings.py --force

The graph vectors were written at 1536 because LightRAG's example code uses
OpenAI's `text-embedding-ada-002` width and the constant was copied along with
it -- the comment at the call site read "OpenAI embedding dimension". Nothing
about this system chose it: chunk vectors use the configured `EMBEDDING_DIM`
(1024) against the same DashScope model, which simply honours whatever width it
is asked for.

The two indices are never compared, so the mismatch was not a correctness bug.
It was 50% more storage and compute for the graph half than for the chunk half,
for no reason.

**This does not re-extract anything.** Each vector's metadata carries the exact
`content` it was built from, so the migration is an embedding pass over stored
text -- no LLM calls, no re-parsing of documents. Legacy edge ids are also
rewritten from ambiguous endpoint concatenation to the canonical pair identity.

Every changed file is backed up to `<name>.bak` before being written, and the
vector count and metadata are asserted unchanged afterwards.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# The embedder splits at the provider's limit; this only bounds memory.
BATCH = 100


def _canonical_ids(path: Path, data: dict[str, object]) -> tuple[list[str], bool]:
    ids = data.get("ids") or []
    metadatas = data.get("metadatas") or []
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise SystemExit(f"{path.name}: vector ids are malformed; reindex required")
    if not isinstance(metadatas, list) or not all(
        isinstance(item, dict) for item in metadatas
    ):
        raise SystemExit(f"{path.name}: vector metadata is malformed; reindex required")
    if not path.name.startswith("vdb_edges_"):
        return ids, False

    from visionagent.database.graph.identity import (
        GraphVectorIdentityError,
        canonical_edge_vector_ids,
    )

    try:
        return canonical_edge_vector_ids(ids, metadatas)
    except GraphVectorIdentityError as error:
        raise SystemExit(f"{path.name}: {error}") from error


def edge_ids_need_migration(path: Path) -> bool:
    """Return whether an edge index still carries pre-v1 pair identities."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path.name}: vector index is not an object")
    return _canonical_ids(path, data)[1]


def migrate(
    path: Path,
    *,
    target_dim: int,
    dry_run: bool,
    force: bool = False,
) -> tuple[int, int, int]:
    """Re-embed when needed and migrate legacy edge-pair ids in the same write."""
    from visionagent.providers.embedding import build_embedder

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path.name}: vector index is not an object")
    vectors = data.get("embeddings") or []
    metadatas = data.get("metadatas") or []
    if not vectors:
        return 0, 0, 0

    canonical_ids, ids_changed = _canonical_ids(path, data)

    old_dim = len(vectors[0])
    if old_dim == target_dim and not force and not ids_changed:
        return len(vectors), old_dim, old_dim

    if dry_run:
        return len(vectors), old_dim, target_dim

    if old_dim != target_dim or force:
        texts = [m.get("content", "") for m in metadatas]
        if len(texts) != len(vectors) or not all(texts):
            raise SystemExit(
                f"{path.name}: {sum(1 for text in texts if not text)} vectors "
                "have no stored content, so they cannot be re-embedded without "
                "re-extraction"
            )
        embedder = build_embedder(dimensions=target_dim)
        fresh: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            fresh.extend(embedder.embed_batch(texts[i:i + BATCH]))

        assert len(fresh) == len(vectors), "vector count changed"
        assert all(len(v) == target_dim for v in fresh), "provider ignored the width"
    else:
        fresh = vectors

    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    data["ids"] = canonical_ids
    data["embeddings"] = fresh
    from visionagent.database.graph import atomic_write

    encoded = json.dumps(data, ensure_ascii=False)
    atomic_write(path, lambda target: Path(target).write_text(encoded, encoding="utf-8"))

    # Re-read and check the intended vector/id migration retained metadata.
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["ids"] == data["ids"], "ids changed"
    assert after["metadatas"] == metadatas, "metadata changed"
    return len(vectors), old_dim, target_dim


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dim", type=int, default=None,
                    help="target width; defaults to EMBEDDING_DIM")
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-embed even when the vector width is unchanged (provider/model migration)",
    )
    args = ap.parse_args()

    from visionagent.config.settings import settings

    target = args.dim or settings.embedding_dimensions
    files = sorted(settings.graph_dir.glob("vdb_*.json"))
    if not files:
        print(f"no vdb_*.json under {settings.graph_dir}", file=sys.stderr)
        return 1

    print(f"target width: {target}  ({len(files)} index files)\n")
    total = skipped = 0
    for path in files:
        ids_changed = edge_ids_need_migration(path)
        n, old, new = migrate(
            path,
            target_dim=target,
            dry_run=args.dry_run,
            force=args.force,
        )
        if n == 0:
            continue
        if old == new and not args.force and not ids_changed:
            skipped += 1
            continue
        total += n
        if old == new and ids_changed and not args.force:
            verb = "would migrate ids for" if args.dry_run else "migrated ids for"
        else:
            verb = "would re-embed" if args.dry_run else "re-embedded"
        if old != new:
            width = f"{old} -> {new}"
        elif args.force:
            width = f"{old} -> {new} (forced)"
        else:
            width = f"{old} (edge ids -> v1)"
        if ids_changed and (old != new or args.force):
            width += ", edge ids -> v1"
        print(f"  {verb} {n:4} vectors  {width}  {path.name}")

    print(f"\n{total} vectors, {skipped} files already at {target}")
    if args.dry_run:
        print("dry run: nothing written")
    else:
        print("backups written alongside each file as *.bak")
        print(f"\nSet GRAPH_EMBEDDING_DIM={target} (or leave it unset once the "
              "default changes) so new extractions match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
