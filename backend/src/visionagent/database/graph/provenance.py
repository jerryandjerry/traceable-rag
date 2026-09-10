"""Lossless contribution ledger for aggregate graph records."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

GRAPH_FIELD_SEP = " | "
CONTRIBUTIONS_FIELD = "contributions_json"
DOCUMENT_NAMES_FIELD = "document_names_json"


def source_ids(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {part.strip() for part in value.split(GRAPH_FIELD_SEP) if part.strip()}


def load_contributions(record: dict[str, Any]) -> dict[str, list[dict[str, Any]]] | None:
    raw = record.get(CONTRIBUTIONS_FIELD)
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    result: dict[str, list[dict[str, Any]]] = {}
    for source_id, entries in value.items():
        if not isinstance(source_id, str) or not isinstance(entries, list):
            return None
        if not all(isinstance(entry, dict) for entry in entries):
            return None
        result[source_id] = [dict(entry) for entry in entries]
    return result


def contribution_document_names(record: dict[str, Any]) -> set[str] | None:
    """Return exact document identities from a contribution ledger.

    ``docnm`` is a display aggregate separated by :data:`GRAPH_FIELD_SEP`, so
    it cannot represent a filename containing that text.  The contribution
    ledger is keyed independently and retains each original ``docnm`` value.
    """
    ledger = load_contributions(record)
    if ledger is None:
        return None
    return {
        doc_name
        for entries in ledger.values()
        for entry in entries
        if isinstance((doc_name := entry.get("docnm")), str) and doc_name
    }


def dump_document_names(names: set[str]) -> str:
    """Encode exact vector provenance without the full contribution ledger."""
    return json.dumps(sorted(names), ensure_ascii=False, separators=(",", ":"))


def load_document_names(record: dict[str, Any]) -> set[str] | None:
    """Read lossless vector provenance; malformed present data fails closed."""
    if DOCUMENT_NAMES_FIELD not in record:
        return None
    raw = record[DOCUMENT_NAMES_FIELD]
    if not isinstance(raw, str):
        raise LegacyGraphProvenanceError("malformed graph vector document provenance")
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise LegacyGraphProvenanceError(
            "malformed graph vector document provenance"
        ) from error
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise LegacyGraphProvenanceError("malformed graph vector document provenance")
    return set(value)


def extend_contributions(
    existing: dict[str, list[dict[str, Any]]], records: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    result = {key: [dict(item) for item in values] for key, values in existing.items()}
    existing_sources = set(existing)
    for record in records:
        ids = source_ids(record.get("source_id"))
        if len(ids) != 1:
            raise ValueError("new graph contributions require one atomic source_id")
        source_id = next(iter(ids))
        if source_id not in existing_sources:
            result.setdefault(source_id, []).append(
                {key: value for key, value in record.items() if key != CONTRIBUTIONS_FIELD}
            )
    return result


def dump_contributions(value: dict[str, list[dict[str, Any]]]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def placeholder_node_from_edges(
    node_name: str, edge_records: list[dict[str, Any]], *, created_at: Any
) -> dict[str, Any]:
    ledger: dict[str, list[dict[str, Any]]] = {}
    for edge in edge_records:
        edge_ledger = load_contributions(edge)
        if edge_ledger is None:
            raise LegacyGraphProvenanceError(
                f"legacy incident edge for {node_name!r} requires reindexing"
            )
        for source, entries in edge_ledger.items():
            ledger.setdefault(source, []).extend(
                {
                    "entity_type": "unknown",
                    "description": "Description not available in text.",
                    "source_id": source,
                    "docnm": entry.get("docnm", "unknown"),
                }
                for entry in entries
            )
    docs = sorted(
        {str(entry["docnm"]) for entries in ledger.values() for entry in entries if entry.get("docnm")}
    )
    return {
        "entity_id": node_name,
        "entity_type": "unknown",
        "description": "Description not available in text.",
        "source_id": GRAPH_FIELD_SEP.join(sorted(ledger)),
        "docnm": GRAPH_FIELD_SEP.join(docs) or "unknown",
        "created_at": created_at,
        CONTRIBUTIONS_FIELD: dump_contributions(ledger),
    }


class LegacyGraphProvenanceError(RuntimeError):
    """A legacy aggregate references the target but has no removable ledger."""


def remove_document_contributions(
    record: dict[str, Any], doc_name: str, *, edge: bool
) -> dict[str, Any] | None:
    ledger = load_contributions(record)
    if ledger is None:
        raw_doc_names = str(record.get("docnm", ""))
        if raw_doc_names == doc_name or doc_name in source_ids(raw_doc_names):
            raise LegacyGraphProvenanceError(
                f"legacy graph aggregate for {doc_name!r} requires reindexing"
            )
        return dict(record)
    kept = {
        source: entries
        for source, entries in ledger.items()
        if not any(entry.get("docnm") == doc_name for entry in entries)
    }
    if kept == ledger:
        return dict(record)
    records = [record for source in sorted(kept) for record in kept[source]]
    if not records:
        return None
    rebuilt = {
        key: value
        for key, value in record.items()
        if key not in {"entity_type", "description", "source_id", "docnm", "keywords", "weight"}
    }
    rebuilt.update(
        description=GRAPH_FIELD_SEP.join(
            sorted({str(item["description"]) for item in records if item.get("description")})
        ),
        source_id=GRAPH_FIELD_SEP.join(sorted(kept)),
        docnm=GRAPH_FIELD_SEP.join(
            sorted({str(item["docnm"]) for item in records if item.get("docnm")})
        ) or "unknown",
        **{CONTRIBUTIONS_FIELD: dump_contributions(kept)},
    )
    if edge:
        keywords = {
            word.strip()
            for item in records
            for word in str(item.get("keywords", "")).split(",")
            if word.strip()
        }
        rebuilt.update(
            keywords=", ".join(sorted(keywords)),
            weight=sum(float(item.get("weight", 1.0)) for item in records),
        )
    else:
        types = Counter(str(item.get("entity_type", "")) for item in records)
        rebuilt["entity_type"] = sorted(types.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
    return rebuilt
