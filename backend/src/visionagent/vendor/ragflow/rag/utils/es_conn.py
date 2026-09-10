#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import copy
import json
import logging
import os
import re
import time
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# Elasticsearch DSL (Domain Specific Language) Python library
from elasticsearch_dsl import Q, Search

from visionagent.vendor.ragflow.api.utils.file_utils import get_project_base_directory
from visionagent.vendor.ragflow.rag.nlp import is_english
from visionagent.vendor.ragflow.rag.utils import singleton
from visionagent.vendor.ragflow.rag.utils.doc_store_conn import (
    FusionExpr,
    MatchDenseExpr,
    MatchExpr,
    MatchTextExpr,
    OrderByExpr,
)

load_dotenv()

ATTEMPT_TIME = 2
PAGERANK_FLD = "pagerank_fea"
TAG_FLD = "tag_feas"

logger = logging.getLogger(__name__)
ES_URL = os.getenv("ES_URL")


def _bulk_failures(response: Any, expected_ids: list[str]) -> list[str]:
    """Validate a bulk response and return its exact rejected document ids.

    A missing/malformed item is not evidence of success.  Callers may turn a
    structurally complete response into an accepted count, but transport errors
    and ambiguous responses must raise so ingest cannot checkpoint phantom
    chunks.
    """
    if not isinstance(response, Mapping) or not isinstance(response.get("errors"), bool):
        raise RuntimeError("Elasticsearch returned a malformed bulk response")
    items = response.get("items")
    if not isinstance(items, list) or len(items) != len(expected_ids):
        raise RuntimeError("Elasticsearch bulk response did not account for every document")

    failures: list[str] = []
    for expected_id, item in zip(expected_ids, items, strict=True):
        if not isinstance(item, Mapping) or len(item) != 1:
            raise RuntimeError("Elasticsearch returned a malformed bulk item")
        action, result = next(iter(item.items()))
        if action not in {"create", "delete", "index", "update"} or not isinstance(result, Mapping):
            raise RuntimeError("Elasticsearch returned a malformed bulk item")
        actual_id = result.get("_id")
        status = result.get("status")
        if actual_id != expected_id or not isinstance(status, int):
            raise RuntimeError("Elasticsearch bulk item identity/status is ambiguous")
        error = result.get("error")
        succeeded = 200 <= status < 300 and error is None
        if not succeeded:
            failures.append(f"{expected_id}:{error or f'HTTP {status}'}")

    if bool(failures) != response["errors"]:
        raise RuntimeError("Elasticsearch bulk error summary disagrees with its items")
    return failures


def _deleted_count(response: Any) -> int:
    """Return a delete count only when Elasticsearch proves full completion."""
    if not isinstance(response, Mapping):
        raise RuntimeError("Elasticsearch returned a malformed delete response")
    deleted = response.get("deleted")
    total = response.get("total")
    failures = response.get("failures")
    version_conflicts = response.get("version_conflicts")
    timed_out = response.get("timed_out")
    shards = response.get("_shards")
    if (
        not isinstance(deleted, int)
        or isinstance(deleted, bool)
        or deleted < 0
        or not isinstance(total, int)
        or isinstance(total, bool)
        or total < 0
        or not isinstance(failures, list)
        or not isinstance(version_conflicts, int)
        or isinstance(version_conflicts, bool)
        or not isinstance(timed_out, bool)
        or not isinstance(shards, Mapping)
        or not isinstance(shards.get("failed"), int)
    ):
        raise RuntimeError("Elasticsearch returned a malformed delete response")
    if timed_out or failures or version_conflicts or shards["failed"]:
        raise RuntimeError("Elasticsearch did not fully delete the requested documents")
    if deleted != total:
        raise RuntimeError("Elasticsearch delete response did not account for every document")
    return deleted

@singleton
class ESConnection():
    def __init__(self):
        self.info = {}
        logger.info("Use Elasticsearch localhost as the document engine")
        self.es = Elasticsearch(
            ES_URL,  # Elasticsearch 主机地址
            basic_auth=("elastic", "infini_rag_flow"),  # 用户名和密码
            verify_certs=False,  # 禁用 SSL 证书验证
            timeout=600
        )

        fp_mapping = os.path.join(get_project_base_directory(), "conf", "mapping.json")
        self.mapping = json.load(open(fp_mapping, "r"))


    """
    Helper functions for search result
    """

    def getTotal(self, res):
        if isinstance(res["hits"]["total"], type({})):
            return res["hits"]["total"]["value"]
        return res["hits"]["total"]

    def getChunkIds(self, res):
        return [d["_id"] for d in res["hits"]["hits"]]
    

    def getHighlight(self, res, keywords: list[str], fieldnm: str):
        ans = {}
        for d in res["hits"]["hits"]:
            hlts = d.get("highlight")
            if not hlts:
                continue
            txt = "...".join([a for a in list(hlts.items())[0][1]])
            if not is_english(txt.split()):
                ans[d["_id"]] = txt
                continue

            txt = d["_source"][fieldnm]
            txt = re.sub(r"[\r\n]", " ", txt, flags=re.IGNORECASE | re.MULTILINE)
            txts = []
            for t in re.split(r"[.?!;\n]", txt):
                for w in keywords:
                    t = re.sub(r"(^|[ .?/'\"\(\)!,:;-])(%s)([ .?/'\"\(\)!,:;-])" % re.escape(w), r"\1<em>\2</em>\3", t,
                               flags=re.IGNORECASE | re.MULTILINE)
                if not re.search(r"<em>[^<>]+</em>", t, flags=re.IGNORECASE | re.MULTILINE):
                    continue
                txts.append(t)
            ans[d["_id"]] = "...".join(txts) if txts else "...".join([a for a in list(hlts.items())[0][1]])

        return ans
    

    def getAggregation(self, res, fieldnm: str):
        agg_field = "aggs_" + fieldnm
        if "aggregations" not in res or agg_field not in res["aggregations"]:
            return list()
        bkts = res["aggregations"][agg_field]["buckets"]
        return [(b["key"], b["doc_count"]) for b in bkts]

    def getFields(self, res, fields: list[str]) -> dict[str, dict]:
        res_fields = {}
        if not fields:
            return {}
        for d in self.__getSource(res):
            m = {n: d.get(n) for n in fields if d.get(n) is not None}
            for n, v in m.items():
                if isinstance(v, list):
                    m[n] = v
                    continue
                if not isinstance(v, str):
                    m[n] = str(m[n])
                # if n.find("tks") > 0:
                #     m[n] = rmSpace(m[n])

            if m:
                res_fields[d["id"]] = m
        return res_fields


    def __getSource(self, res):
        rr = []
        for d in res["hits"]["hits"]:
            d["_source"]["id"] = d["_id"]
            d["_source"]["_score"] = d["_score"]
            rr.append(d["_source"])
        return rr

    """
    Database operations
    """
    def insert(self, documents: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:
        # Refers to https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html
        logger.debug(
            "Elasticsearch bulk insert starting",
            extra={"document_count": len(documents), "index_name": indexName},
        )
        
        operations = []
        expected_ids: list[str] = []
        for d in documents:
            assert "_id" not in d
            assert "id" in d
            d_copy = copy.deepcopy(d)
            meta_id = d_copy.pop("id", "")
            if not isinstance(meta_id, str) or not meta_id:
                raise ValueError("Elasticsearch documents require a non-empty string id")
            expected_ids.append(meta_id)
            operations.append(
                {"index": {"_index": indexName, "_id": meta_id}})
            operations.append(d_copy)

        last_error: Exception | None = None
        for attempt in range(ATTEMPT_TIME):
            try:
                # wait_for blocks until the new chunks are searchable. With
                # refresh=False the pipeline reported "completed" while the
                # document was still invisible, so opening it straight after
                # ingestion showed zero chunks.
                r = self.es.bulk(index=(indexName), operations=operations,
                                 refresh="wait_for", timeout="60s")
                logger.debug(
                    "Elasticsearch bulk insert completed",
                    extra={"document_count": len(operations) // 2, "errors": bool(r.get("errors"))},
                )
                return _bulk_failures(r, expected_ids)
            except Exception as error:
                last_error = error
                logger.warning("Elasticsearch bulk insert failed", exc_info=True)
                if attempt + 1 < ATTEMPT_TIME and re.search(
                    r"(Timeout|time out)", str(error), re.IGNORECASE
                ):
                    time.sleep(3)
        raise RuntimeError("Elasticsearch bulk insert failed after retries") from last_error
    
    # ====================  This is the matchExprs serach method =========================================
    def search(
            self, selectFields: list[str],
            highlightFields: list[str],
            condition: dict,
            matchExprs: list[MatchExpr],
            orderBy: OrderByExpr,
            offset: int,
            limit: int,
            indexNames: str | list[str],
            knowledgebaseIds: list[str],
            aggFields: list[str] = [],
            rank_feature: dict | None = None
    ):
        """
        Refers to https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html
        """
        # =============== 1. building boolean query ===================
        # elasticsearch_dsl.Q is a class to build search query for es search, an alternative of passing an json obejct
        # different type of query(countless types, check es doc): 
        # q = Q('term', content = 'kwd')
        # q = Q('match', content = 'description')
        # q = Q('range', age={'gte': 18, 'lte': 65})
        # Compound query q = Q('bool', must = [], should= [], must_not=[], filter=[])
        # used to combine/nest multiple queries for searching
        # must = AND, should = OR, must_not = Neither, filter= AND without scoring
        # score used to evaluate the relevancy, high relevancy is consider a match in must[] -> used in content search
        # filter doesn't use score, which require a exact match(yes or no) -> used in keyword or condition filter

        # elasticsearch_dsl.Search is a class to execute search queries. 
        # Search.query(Q()) takes built query and execute text serach, 

        # ======= 1.1 filter ========
        if isinstance(indexNames, str):
            indexNames = indexNames.split(",")
        assert isinstance(indexNames, list) and len(indexNames) > 0
        assert "_id" not in condition
        bqry = Q("bool", must=[]) # initialize a bool query
        condition["kb_id"] = knowledgebaseIds
        for k, v in condition.items():
            if k == "available_int":
                if v == 0:
                    # filter used for exact match, no scoring(unlike)
                    bqry.filter.append(Q("range", available_int={"lt": 1}))
                else:
                    bqry.filter.append(
                        Q("bool", must_not=Q("range", available_int={"lt": 1})))
                continue
            if not v:
                continue
            if isinstance(v, list):
                bqry.filter.append(Q("terms", **{k: v}))
            elif isinstance(v, str) or isinstance(v, int):
                bqry.filter.append(Q("term", **{k: v}))
            else:
                raise Exception(
                    f"Condition `{str(k)}={str(v)}` value type is {str(type(v))}, expected to be int, str or list.")

        # ======= 1.2 must [] ========
        s = Search()
        vector_similarity_weight = 0.5
        # if FusionExpr is used, update the weight
        for m in matchExprs:
            if isinstance(m, FusionExpr) and m.method == "weighted_sum" and "weights" in m.fusion_params:
                assert len(matchExprs) == 3 and isinstance(matchExprs[0], MatchTextExpr) and isinstance(matchExprs[1],
                                                                                                        MatchDenseExpr) and isinstance(
                    matchExprs[2], FusionExpr)
                weights = m.fusion_params["weights"]
                vector_similarity_weight = float(weights.split(",")[1])
        
        for m in matchExprs:
            # for text search, use Q('query string')
            if isinstance(m, MatchTextExpr):
                # print(f'[DEBUG ES_conn.search()] text search params: {m.matching_text=},{m.fields=}')
                minimum_should_match = m.extra_options.get("minimum_should_match", 0.0)
                if isinstance(minimum_should_match, float):
                    minimum_should_match = str(int(minimum_should_match * 100)) + "%" # turn float to percentage
                # bqry is an object of {must:[], }, append the result form Q()
                bqry.must.append(Q("query_string", fields=m.fields,
                                   type="best_fields", query=m.matching_text,
                                   minimum_should_match=minimum_should_match,
                                   boost=1))
                bqry.boost = 1.0 - vector_similarity_weight

            # for vector search, use Search.knn(), adds knn query to the Search object
            # defualt HNSW algorithm for knn in es
            elif isinstance(m, MatchDenseExpr):
                assert (bqry is not None)
                similarity = 0.0
                if "similarity" in m.extra_options:
                    similarity = m.extra_options["similarity"]
                # print(f'[DEBUG ES_conn.search()] vector search params: {m.topn=}, {similarity=}')
                s = s.knn(m.vector_column_name, # field name that contain vector    
                          m.topn, # num of result to return, top k
                          m.topn * 2, # num of candidate to examine (two times)
                          query_vector=list(m.embedding_data),
                          filter=bqry.to_dict(),
                          similarity=similarity, # threshold, use 0.2 here
                          )

        # rank feature: relevance tuning - making popular, highly-rated, or authoritative documents rank higher in search results, even if their text match isn't perfect
        if bqry and rank_feature:
            for fld, sc in rank_feature.items():
                if fld != PAGERANK_FLD:
                    fld = f"{TAG_FLD}.{fld}"
                bqry.should.append(Q("rank_feature", field=fld, linear={}, boost=sc))

        # add bool query to Search object
        if bqry:
            s = s.query(bqry)

        # ======= 1.3 additional features ========
        # Highlighting shows which parts of the text matched the query
        # will return both original and highlighted version of selected fields, here "content_ltks", "title_tks"
        for field in highlightFields:
            s = s.highlight(field)

        if orderBy:
            orders = list()
            for field, order in orderBy.fields:
                order = "asc" if order == 0 else "desc"
                if field in ["page_num_int", "top_int"]:
                    order_info = {"order": order, "unmapped_type": "float",
                                  "mode": "avg", "numeric_type": "double"}
                elif field.endswith("_int") or field.endswith("_flt"):
                    order_info = {"order": order, "unmapped_type": "float"}
                else:
                    order_info = {"order": order, "unmapped_type": "text"}
                orders.append({field: order_info})
            s = s.sort(*orders)

        for fld in aggFields:
            s.aggs.bucket(f'aggs_{fld}', 'terms', field=fld, size=1000000)

        if limit > 0:
            s = s[offset:offset + limit]
        q = s.to_dict()
        logger.debug(
            "Elasticsearch search starting",
            extra={
                "index_count": 1 if isinstance(indexNames, str) else len(indexNames),
                "offset": offset,
                "limit": limit,
            },
        )

        for i in range(ATTEMPT_TIME):
            try:
                #print(json.dumps(q, ensure_ascii=False))
                res = self.es.search(index=indexNames,
                                     body=q,
                                     timeout="600s",
                                     # search_type="dfs_query_then_fetch",
                                     track_total_hits=True,
                                     _source=True)
                if str(res.get("timed_out", "")).lower() == "true":
                    raise Exception("Es Timeout.")
                logger.debug(
                    "Elasticsearch search completed",
                    extra={"timed_out": bool(res.get("timed_out", False))},
                )
                return res
            except Exception as e:
                logger.exception("Elasticsearch search failed")
                if str(e).find("Timeout") > 0:
                    continue
                raise
        logger.error("ESConnection.search timeout for 3 times!")
        raise Exception("ESConnection.search timeout.")

    def delete(self, condition: dict, indexName: str, knowledgebaseId: str = None) -> int:
        """
        Delete documents based on conditions
        :param condition: Dictionary of field-value pairs to match for deletion
        :param indexName: Name of the ES index
        :param knowledgebaseId: Knowledge base ID (optional, for compatibility)
        :return: Number of deleted documents
        """
        try:
            # Build the delete query
            query = {"bool": {"must": []}}
            
            for field, value in condition.items():
                if isinstance(value, list):
                    query["bool"]["must"].append({"terms": {field: value}})
                else:
                    query["bool"]["must"].append({"term": {field: value}})
            
            # Execute delete by query
            response = self.es.delete_by_query(
                index=indexName,
                body={"query": query},
                conflicts="proceed"  # Continue even if there are conflicts
            )
            
            deleted_count = _deleted_count(response)
            logger.info(
                "Elasticsearch delete completed",
                extra={"deleted_count": deleted_count, "index_name": indexName},
            )
            
            return deleted_count
            
        except Exception:
            logger.exception("Elasticsearch delete failed")
            raise
