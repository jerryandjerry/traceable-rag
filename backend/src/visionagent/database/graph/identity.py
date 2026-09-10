"""Stable identities for graph vector records.

Graph edges are pairs, not concatenated strings. Encoding each endpoint with
its byte length makes the pair boundary explicit before hashing, so pairs such
as (``"AB"``, ``"C"``) and (``"A"``, ``"BC"``) no longer have identical hash
input.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any


class GraphVectorIdentityError(ValueError):
    """An existing vector index cannot be assigned identities without loss."""


def edge_vector_id(source: str, target: str) -> str:
    """Return the versioned, orientation-preserving identity for one edge."""
    digest = hashlib.sha256(b"visionagent.graph.edge.v1\0")
    for endpoint in (source, target):
        encoded = endpoint.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"rel-{digest.hexdigest()}"


def canonical_edge_vector_ids(
    ids: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
) -> tuple[list[str], bool]:
    """Canonicalize edge ids from authoritative endpoint metadata.

    Vector order and metadata are preserved. Multiple rows for the same
    endpoint pair are ambiguous aggregates and are rejected.
    """
    if len(ids) != len(metadatas):
        raise GraphVectorIdentityError(
            "edge vector ids and metadata have different lengths; reindex required"
        )

    canonical: list[str] = []
    seen: dict[str, int] = {}
    changed = False
    for index, (stored_id, metadata) in enumerate(zip(ids, metadatas, strict=True)):
        source = metadata.get("src_id")
        target = metadata.get("tgt_id")
        if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
            raise GraphVectorIdentityError(
                f"edge vector row {index} has no usable endpoints; reindex required"
            )
        item_id = edge_vector_id(source, target)
        if item_id in seen:
            raise GraphVectorIdentityError(
                "multiple edge vector rows describe the same endpoint pair; "
                "reindex required"
            )
        seen[item_id] = index
        canonical.append(item_id)
        changed = changed or item_id != stored_id
    return canonical, changed
