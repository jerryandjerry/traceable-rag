"""Elasticsearch reads used by GraphRAG and document workflows."""

import logging
from typing import Any

from visionagent.vendor.ragflow.rag.utils.es_conn import ESConnection

logger = logging.getLogger(__name__)


def retrieve_chunks_from_es(chunk_ids: list[str], user_id: str) -> list[dict[str, Any]]:
    """Batch-read chunk IDs from one tenant's Elasticsearch index."""
    if not chunk_ids:
        return []

    # De-duplicate without losing the caller's order.
    unique_chunk_ids = list(dict.fromkeys(chunk_ids))

    es_connection = ESConnection()
    index_name = str(user_id)
    docs = [
        {"_index": index_name, "_id": chunk_id}
        for chunk_id in unique_chunk_ids
    ]
    response = es_connection.es.mget(body={"docs": docs})

    filtered_chunks = []
    for chunk_id, doc_response in zip(
        unique_chunk_ids, response["docs"], strict=True
    ):
        if not doc_response.get("found", False):
            logger.debug("graph chunk not found chunk_id=%s", chunk_id)
            continue

        # Keep the raw client response intact while removing storage-only fields.
        chunk = dict(doc_response["_source"])
        # RetrievedChunk images are base64 strings; discard malformed entries.
        stored_images = chunk.get("ref_images", [])
        chunk["ref_images"] = (
            [value for value in stored_images if isinstance(value, str)]
            if isinstance(stored_images, list)
            else []
        )
        for key in ("content_ltks", "content_sm_ltks", "image"):
            chunk.pop(key, None)
        filtered_chunks.append(chunk)

    return filtered_chunks


def chunks_of_document(user_id: str, file_name: str) -> list[dict[str, Any]]:
    """Every stored chunk of one document, in reading order.

    The storage layer owns the query, projected fields, and reading-order sort.
    """
    from visionagent.vendor.ragflow.rag.utils.es_conn import ESConnection

    index_name = str(user_id)
    es = ESConnection()
    if not es.es.indices.exists(index=index_name):
        return []

    response = es.es.search(
        index=index_name,
        body={
            "query": {"match": {"docnm": file_name}},
            "sort": [
                {"page_num": {"order": "asc"}},
                {"top_int": {"order": "asc"}},
            ],
            "size": 1000,
            "_source": ["id", "docnm", "page_num", "top_int",
                        "content_with_weight", "image"],
        },
    )
    return [
        {
            "id": hit["_id"],
            "docnm": hit["_source"].get("docnm", ""),
            "page_num": hit["_source"].get("page_num", 0),
            "top_int": hit["_source"].get("top_int", 0),
            "content_with_weight": hit["_source"].get("content_with_weight", ""),
            "image": hit["_source"].get("image", ""),
        }
        for hit in response["hits"]["hits"]
    ]
