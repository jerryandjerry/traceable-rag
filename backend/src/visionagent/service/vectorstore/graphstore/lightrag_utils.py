import hashlib
import logging
import time
from collections import Counter
from typing import Any

from visionagent.database.graph.provenance import (
    CONTRIBUTIONS_FIELD,
)
from visionagent.database.graph.provenance import (
    dump_contributions as _dump_contributions,
)
from visionagent.database.graph.provenance import (
    extend_contributions as _extend_contributions,
)
from visionagent.database.graph.provenance import (
    load_contributions as _load_contributions,
)

GRAPH_FIELD_SEP = " | "

logger = logging.getLogger(__name__)


def _source_ids(value: Any) -> set[str]:
    """Return the atomic source ids carried by one graph contribution."""
    if not isinstance(value, str):
        return set()
    return set(split_string_by_multi_markers(value, [GRAPH_FIELD_SEP]))


def _new_source_contributions(
    records: list[dict[str, Any]], existing_source_ids: set[str]
) -> list[dict[str, Any]]:
    """Keep only contributions whose source has not already been applied.

    A durable upload may restart after the graph files were saved but before
    the item checkpoint was committed.  The retry extracts the same chunk
    again.  ``source_id`` is the chunk id, so adding a record whose ids are
    already present would count the same contribution twice.

    We intentionally compare every record with the sources that existed
    *before this merge*.  One extraction may legitimately emit several
    records for the same entity or relation from a new chunk; all of those
    records belong to that chunk's first contribution and must be aggregated.
    """
    new_records: list[dict[str, Any]] = []
    for record in records:
        record_source_ids = _source_ids(record.get("source_id"))
        if not record_source_ids:
            raise ValueError("graph contributions require a non-empty source_id")
        if record_source_ids.issubset(existing_source_ids):
            continue
        new_records.append(record)
    return new_records


def _merged_source_id(existing_source_ids: set[str], records: list[dict[str, Any]]) -> str:
    source_ids = set(existing_source_ids)
    for record in records:
        source_ids.update(_source_ids(record.get("source_id")))
    return GRAPH_FIELD_SEP.join(sorted(source_ids))


def compute_mdhash_id(text: str, prefix: str = "") -> str:
    """Compute the stable ID format used by the graph vector files."""
    md5_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{prefix}{md5_hash}"


def split_string_by_multi_markers(text: str, markers: list[str]) -> list[str]:
    """Split a persisted aggregate field on its primary separator."""
    if not text:
        return []

    primary_marker = markers[0] if markers else GRAPH_FIELD_SEP
    parts = text.split(primary_marker)
    return [part.strip() for part in parts if part.strip()]


def build_docnm_kwd(already_docnm_kwds: list[str], new_data: list[dict[str, Any]], entity_name: str) -> str:
    """Merge and serialize the document names that contributed a fact."""

    all_docnm_kwds = set(already_docnm_kwds)

    for item in new_data:
        if isinstance(item, dict) and "docnm" in item:
            all_docnm_kwds.add(item["docnm"])

    all_docnm_kwds = {docnm_kwd for docnm_kwd in all_docnm_kwds if docnm_kwd and docnm_kwd != "unknown"}

    if not all_docnm_kwds:
        return "unknown"

    return GRAPH_FIELD_SEP.join(sorted(all_docnm_kwds))


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict[str, Any]],
    knowledge_graph_inst: Any,
    global_config: dict[str, Any],
    pipeline_status: dict[str, Any] | None = None,
    pipeline_status_lock: Any = None,
    llm_response_cache: Any = None,
) -> dict[str, Any]:
    """Merge new source contributions into one persisted graph node."""
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_docnm_kwds = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    existing_contributions = (
        _load_contributions(already_node) if already_node is not None else {}
    )
    if already_node:
        already_entity_types.append(already_node.get("entity_type", ""))
        already_source_ids.extend(
            split_string_by_multi_markers(already_node.get("source_id", ""), [GRAPH_FIELD_SEP])
        )
        already_docnm_kwds.extend(
            split_string_by_multi_markers(already_node.get("docnm", ""), [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node.get("description", ""))

    existing_source_ids = set(already_source_ids)
    new_nodes_data = _new_source_contributions(nodes_data, existing_source_ids)

    # A durable retry with no new source IDs must not mutate the aggregate.
    if already_node and not new_nodes_data:
        node_data = dict(already_node)
        node_data.setdefault("entity_id", entity_name)
        node_data["entity_name"] = entity_name
        return node_data

    entity_type = (
        sorted(
            Counter([dp.get("entity_type", "") for dp in new_nodes_data] + already_entity_types).items(),
            key=lambda x: x[1],
            reverse=True,
        )[0][0]
        if new_nodes_data or already_entity_types
        else "unknown"
    )

    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp.get("description", "") for dp in new_nodes_data] + already_description))
    )

    source_id = _merged_source_id(existing_source_ids, new_nodes_data)
    docnm_kwd = build_docnm_kwd(already_docnm_kwds, new_nodes_data, entity_name)

    node_data = dict(
        entity_id=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        docnm=docnm_kwd,
        created_at=(already_node.get("created_at", int(time.time())) if already_node else int(time.time())),
    )
    # Fresh records carry a lossless ledger. Legacy aggregates remain marked
    # as legacy rather than fabricating provenance that deletion could trust.
    if existing_contributions is not None:
        node_data[CONTRIBUTIONS_FIELD] = _dump_contributions(
            _extend_contributions(existing_contributions, new_nodes_data)
        )

    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )

    try:
        graph = knowledge_graph_inst.graph
        if graph and entity_name in graph:
            degree = graph.degree(entity_name)
            await knowledge_graph_inst.upsert_node(
                entity_name,
                node_data={**node_data, "degree": degree},
            )
    except Exception:
        logger.warning("could not calculate graph node degree", exc_info=True)

    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict[str, Any]],
    knowledge_graph_inst: Any,
    global_config: dict[str, Any],
    pipeline_status: dict[str, Any] | None = None,
    pipeline_status_lock: Any = None,
    llm_response_cache: Any = None,
    added_entities: list[Any] | None = None,
) -> dict[str, Any] | None:
    """Merge new source contributions into one persisted graph edge."""
    if src_id == tgt_id:
        return None

    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_docnm_kwds = []

    already_edge = None
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        if already_edge:
            # NetworkX is undirected here. A later extraction may present the
            # same relationship as B->A, but its persisted/VDB identity must
            # retain the orientation chosen by the first contribution.
            stored_src = already_edge.get("src_id")
            stored_tgt = already_edge.get("tgt_id")
            if isinstance(stored_src, str) and isinstance(stored_tgt, str):
                src_id, tgt_id = stored_src, stored_tgt
            already_weights.append(already_edge.get("weight", 1.0))
            if already_edge.get("source_id"):
                already_source_ids.extend(
                    split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
                )
            if already_edge.get("docnm"):
                already_docnm_kwds.extend(
                    split_string_by_multi_markers(already_edge["docnm"], [GRAPH_FIELD_SEP])
                )
            if already_edge.get("description"):
                already_description.append(already_edge["description"])
            if already_edge.get("keywords"):
                already_keywords.extend(
                    split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
                )

    existing_contributions = (
        _load_contributions(already_edge) if already_edge is not None else {}
    )

    existing_source_ids = set(already_source_ids)
    new_edges_data = _new_source_contributions(edges_data, existing_source_ids)

    # A durable retry with no new source IDs must not mutate the aggregate.
    if already_edge and not new_edges_data:
        edge_data = dict(already_edge)
        edge_data.setdefault("src_id", src_id)
        edge_data.setdefault("tgt_id", tgt_id)
        return edge_data

    weight = sum([dp.get("weight", 1.0) for dp in new_edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(
            set(
                [dp.get("description", "") for dp in new_edges_data if dp.get("description")]
                + already_description
            )
        )
    )

    all_keywords: set[Any] = set()
    for keyword_str in already_keywords:
        if keyword_str:
            all_keywords.update(k.strip() for k in keyword_str.split(",") if k.strip())
    for edge in new_edges_data:
        if edge.get("keywords"):
            all_keywords.update(k.strip() for k in edge["keywords"].split(",") if k.strip())
    keywords = ", ".join(sorted(all_keywords))

    source_id = _merged_source_id(existing_source_ids, new_edges_data)
    docnm_kwd = build_docnm_kwd(already_docnm_kwds, new_edges_data, f"{src_id}_{tgt_id}")
    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        weight=weight,
        source_id=source_id,
        docnm=docnm_kwd,
        created_at=(already_edge.get("created_at", int(time.time())) if already_edge else int(time.time())),
    )
    if existing_contributions is not None:
        edge_data[CONTRIBUTIONS_FIELD] = _dump_contributions(
            _extend_contributions(existing_contributions, new_edges_data)
        )

    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            placeholder_records = [
                {
                    "entity_type": "unknown",
                    "description": "Description not available in text.",
                    "source_id": edge["source_id"],
                    "docnm": edge.get("docnm", "unknown"),
                }
                for edge in edges_data
            ]
            missing_node_data = {
                "entity_id": need_insert_id,
                "entity_type": "unknown",
                "description": "Description not available in text.",
                "source_id": source_id,
                "docnm": docnm_kwd,
                "created_at": int(time.time()),
                CONTRIBUTIONS_FIELD: _dump_contributions(
                    _extend_contributions({}, placeholder_records)
                ),
            }
            await knowledge_graph_inst.upsert_node(need_insert_id, node_data=missing_node_data)

    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=edge_data,
    )

    try:
        graph = knowledge_graph_inst.graph
        if graph:
            if src_id in graph:
                src_degree = graph.degree(src_id)
                src_node = await knowledge_graph_inst.get_node(src_id)
                if src_node:
                    await knowledge_graph_inst.upsert_node(
                        src_id,
                        node_data={**src_node, "degree": src_degree},
                    )

            if tgt_id in graph:
                tgt_degree = graph.degree(tgt_id)
                tgt_node = await knowledge_graph_inst.get_node(tgt_id)
                if tgt_node:
                    await knowledge_graph_inst.upsert_node(
                        tgt_id,
                        node_data={**tgt_node, "degree": tgt_degree},
                    )
    except Exception:
        logger.warning("could not update graph edge degrees", exc_info=True)

    return edge_data
