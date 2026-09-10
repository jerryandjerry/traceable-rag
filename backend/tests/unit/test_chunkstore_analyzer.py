"""Tests for the analyzer/ and chunkstore/ slots.

Intended function of the analyzer component:
    analyze(text)      -> exactly the space-joined coarse terms ES stores as
                          content_ltks: lowercased, punctuation stripped,
                          lemmatised then Porter-stemmed
    analyze_fine(text) -> exactly the fine-grained split of that output
    synonyms(term)     -> exactly the expansions, [] when there are none

Intended function of the chunk store component:
    to_document(chunk) -> exactly the Elasticsearch document, field for field.
                          The Python-side rename (content vs
                          content_with_weight) must NOT reach the index.
    index(chunks)      -> exactly the number accepted
    search(query)      -> exactly the hits as RetrievedChunk, analyzed fields
                          and the raw vector dropped
"""
from __future__ import annotations

import pytest

from visionagent.models import ParsedChunk, RetrievedChunk, SourceType
from visionagent.service.vectorstore.elasticsearch import (
    ChunkStore,
    ChunkStoreError,
    ElasticsearchChunkStore,
    build_chunkstore,
)
from visionagent.service.vectorstore.elasticsearch.analyzer import Analyzer, build_analyzer


class FakeAnalyzer:
    name = "fake"

    def analyze(self, text: str) -> str:
        return " ".join(w.lower() for w in text.split())

    def analyze_fine(self, text: str) -> str:
        return self.analyze(text)

    def synonyms(self, term: str) -> list[str]:
        return {"curb": ["kerb"]}.get(term, [])


class FakeEmbedder:
    name = "fake"
    dimensions = 3

    def embed(self, text: str) -> list[float]:
        return [1.0, 2.0, 3.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class FakeConnection:
    """Stands in for ESConnection."""

    def __init__(self, failures=None, hits=None, deleted=0) -> None:
        self.failures = failures or []
        self.hits = hits or []
        self.deleted = deleted
        self.inserted: list[dict] = []
        self.delete_calls: list[tuple] = []

    def insert(self, documents, indexName, knowledgebaseId=None):  # noqa: N803
        self.inserted = documents
        return self.failures

    def delete(self, condition, indexName, knowledgebaseId=None):  # noqa: N803
        self.delete_calls.append((condition, indexName))
        return self.deleted

    class _Indices:
        @staticmethod
        def exists(index):
            return True

    class _Es:
        indices = None

    # The writer checks the index exists before a delete, so that a tenant
    # with no index yet deletes as zero rather than raising.
    es = type("Es", (), {"indices": _Indices()})()


def store(**kw) -> ElasticsearchChunkStore:
    """The indexing slot: what a document looks like, and writing it."""
    kw.setdefault("analyzer", FakeAnalyzer())
    kw.setdefault("embedder", FakeEmbedder())
    kw.setdefault("connection", FakeConnection())
    return ElasticsearchChunkStore(**kw)


def reader(connection=None):
    """The storage layer: the connection, the search, and reading a hit back.

    A separate object from `store()` above, which is the point of the split:
    the slot shapes documents and can only write, and this can only read.
    """
    from visionagent.database.elasticsearch import ElasticsearchStore

    return ElasticsearchStore(connection or FakeConnection())


# ============================================================ analyzer: exact
def test_analyzer_lowercases_strips_punctuation_lemmatises_and_stems():
    """Splitting alone would leave 'sidewalks' unmatched by a query for
    'sidewalk'. The other two stages are what make BM25 work."""
    assert build_analyzer().analyze("The sidewalks widths shall be 1.8m minimum") == (
        "the sidewalk width shall be 1 8m minimum"
    )


def test_analyzer_stems_inflected_forms_to_a_shared_root():
    assert build_analyzer().analyze("Curb extensions reduce crossing distances") == (
        "curb extens reduc cross distanc"
    )


def test_analyzer_measurements_and_codes_lose_their_punctuation():
    """A real limitation worth knowing: '1.8m' and 'R-4' do not survive intact,
    which is a concrete reason someone might swap this component."""
    a = build_analyzer()
    assert a.analyze("1.8m") == "1 8m"
    assert a.analyze("R-4 zoning") == "r 4 zone"


def test_analyzer_returns_empty_for_empty_input():
    assert build_analyzer().analyze("") == ""


def test_analyzer_satisfies_the_protocol_and_a_fake_can_replace_it():
    assert isinstance(build_analyzer(), Analyzer)
    assert isinstance(FakeAnalyzer(), Analyzer)


def test_analyzer_factory_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown analyzer"):
        build_analyzer("lucene")


# ========================================================== chunkstore: exact
def test_to_document_produces_exactly_the_elasticsearch_field_set():
    doc = store().to_document(
        ParsedChunk(id="parser-local", content="Sidewalk widths.",
                    content_tokens="sidewalk width", content_tokens_fine="sidewalk width",
                    doc_name_tokens="curb guide", page_nums=[4, 5], top_offsets=[120, 8],
                    ref_images=["REF64"], image="IMG64"),
        index_name="user7", doc_name="curb.pdf",
    )
    assert set(doc) == {
        "id", "chunk_id", "content_with_weight", "content_ltks", "content_sm_ltks",
        "important_kwd", "important_tks", "question_kwd", "question_tks",
        "create_time", "create_timestamp_flt", "page_num", "top_int", "kb_id",
        "docnm_tks", "doc_id", "docnm", "ref_images", "image", "q_3_vec",
    }


def test_to_document_keeps_the_on_disk_names_not_the_python_names():
    """The typed `content` field persists as `content_with_weight`."""
    doc = store().to_document(
        ParsedChunk(id="x", content="Sidewalk widths."), index_name="u", doc_name="d.pdf"
    )
    assert doc["content_with_weight"] == "Sidewalk widths."
    assert "content" not in doc


def test_to_document_maps_every_value_exactly():
    doc = store().to_document(
        ParsedChunk(id="x", content="Sidewalk widths.", content_tokens="sidewalk width",
                    content_tokens_fine="sidewalk wid", doc_name_tokens="curb guide",
                    page_nums=[4, 5], top_offsets=[120, 8], ref_images=["REF64"],
                    image="IMG64"),
        index_name="user7", doc_name="curb.pdf",
    )
    assert doc["content_ltks"] == "sidewalk width"
    assert doc["content_sm_ltks"] == "sidewalk wid"
    assert doc["docnm_tks"] == "curb guide"
    assert doc["page_num"] == 4, "the first page, not the whole list"
    assert doc["top_int"] == 120
    assert doc["kb_id"] == "user7"
    assert doc["docnm"] == "curb.pdf"
    assert doc["ref_images"] == ["REF64"]
    assert doc["image"] == "IMG64"
    assert doc["q_3_vec"] == [1.0, 2.0, 3.0]
    assert doc["important_kwd"] == doc["important_tks"] == []


def test_to_document_analyses_when_the_parser_supplied_nothing():
    """A replacement parser need not analyse -- that is the store's job."""
    doc = store().to_document(
        ParsedChunk(id="x", content="Sidewalk Widths"), index_name="u", doc_name="Curb Guide.pdf"
    )
    assert doc["content_ltks"] == "sidewalk widths"
    assert doc["docnm_tks"] == "curb guide.pdf"


def test_chunk_store_honors_pipeline_assigned_persisted_identity():
    """The adapter must not collapse distinct pipeline occurrences by text."""
    s = store()
    a = s.to_document(ParsedChunk(id="assigned-a", content="same"), index_name="userA", doc_name="d.pdf")
    b = s.to_document(ParsedChunk(id="assigned-b", content="same"), index_name="userA", doc_name="other.pdf")
    assert a["chunk_id"] == "assigned-a"
    assert b["chunk_id"] == "assigned-b"
    assert a["id"] == a["chunk_id"], "ES _id and the retrievable chunk_id are the same value"


def test_doc_id_is_stable_for_the_same_document_and_user():
    s = store()
    first = s.document_id(doc_name="curb.pdf", index_name="user7")
    assert first == s.document_id(doc_name="curb.pdf", index_name="user7")
    assert first != s.document_id(doc_name="curb.pdf", index_name="user8")


def test_vector_field_name_follows_the_embedder_width():
    """The mapping's q_<dim>_vec field follows the configured embedding width."""
    class Wide(FakeEmbedder):
        dimensions = 5

        def embed(self, text: str) -> list[float]:
            return [0.0] * 5

    doc = store(embedder=Wide()).to_document(
        ParsedChunk(id="x", content="c"), index_name="u", doc_name="d.pdf"
    )
    assert "q_5_vec" in doc and "q_3_vec" not in doc


def test_index_returns_exactly_the_accepted_count():
    conn = FakeConnection(failures=[])
    n = store(connection=conn).index(
        chunks=[ParsedChunk(id="a", content="one"), ParsedChunk(id="b", content="two")],
        index_name="u", doc_name="d.pdf",
    )
    assert n == 2
    assert [d["content_with_weight"] for d in conn.inserted] == ["one", "two"]


def test_index_subtracts_rejected_documents():
    conn = FakeConnection(failures=["doc-b failed"])
    n = store(connection=conn).index(
        chunks=[ParsedChunk(id="a", content="one"), ParsedChunk(id="b", content="two")],
        index_name="u", doc_name="d.pdf",
    )
    assert n == 1


def test_index_of_nothing_is_zero_and_touches_no_connection():
    conn = FakeConnection()
    assert store(connection=conn).index(chunks=[], index_name="u", doc_name="d.pdf") == 0
    assert conn.inserted == []


def test_index_failure_becomes_a_chunkstoreerror():
    class Boom(FakeConnection):
        def insert(self, documents, indexName, knowledgebaseId=None):  # noqa: N803
            raise ConnectionError("es unreachable")

    with pytest.raises(ChunkStoreError, match="insert failed"):
        store(connection=Boom()).index(
            chunks=[ParsedChunk(id="a", content="one")], index_name="u", doc_name="d.pdf"
        )


def test_to_chunk_maps_a_hit_exactly_and_drops_matching_only_fields():
    s = reader()
    got = s.to_chunk({
        "chunk_id": "c1", "content_with_weight": "Sidewalk widths.",
        "docnm": "curb.pdf", "page_num": 4, "similarity": 0.83,
        "ref_images": ["REF64"],
        # matching-only: must not reach the client
        "content_ltks": "sidewalk width", "content_sm_ltks": "sidewalk wid",
        "q_1024_vec": [0.1] * 1024, "image": "IMG64",
    })
    assert got == RetrievedChunk(
        id="c1", content="Sidewalk widths.", source_type=SourceType.KNOWLEDGE_BASE,
        score=0.83, doc_name="curb.pdf", page_num=4, images=["REF64"],
    )


def test_to_chunk_clamps_similarity_into_the_contract_bounds():
    assert reader().to_chunk({"chunk_id": "c", "content_with_weight": "x",
                             "similarity": 1.4}).score == 1.0
    assert reader().to_chunk({"chunk_id": "c", "content_with_weight": "x",
                             "similarity": -0.2}).score == 0.0


def test_to_chunk_tolerates_a_hit_with_no_score_or_page():
    got = reader().to_chunk({"chunk_id": "c", "content_with_weight": "x"})
    assert got.score == 0.0 and got.page_num is None and got.images == []


def test_search_returns_exactly_the_hits_in_order():
    class Dealer:
        @staticmethod
        def retrieval(**kw):
            return {"chunks": [
                {"chunk_id": "c1", "content_with_weight": "first", "similarity": 0.9},
                {"chunk_id": "c2", "content_with_weight": "second", "similarity": 0.4},
            ]}

    s = reader()
    s._dealer = Dealer()
    assert [(c.id, c.score) for c in s.search(index_name="u", query="q")] == [
        ("c1", 0.9), ("c2", 0.4)
    ]


def test_search_returns_empty_when_the_store_has_nothing():
    s = reader()
    s._dealer = type("D", (), {"retrieval": staticmethod(lambda **kw: None)})()
    assert s.search(index_name="u", query="q") == []


def test_delete_document_targets_the_scoped_doc_id():
    conn = FakeConnection(deleted=7)
    s = store(connection=conn)
    assert s.delete_document(doc_name="curb.pdf", index_name="user7") == 7
    condition, index = conn.delete_calls[0]
    assert condition == {"doc_id": s.document_id(doc_name="curb.pdf", index_name="user7")}
    assert index == "user7"


def test_store_satisfies_the_protocol_and_the_factory_rejects_unknowns():
    assert isinstance(build_chunkstore(), ChunkStore)
    with pytest.raises(ValueError, match="unknown chunk store"):
        build_chunkstore("pinecone")


def test_one_analyzer_serves_both_write_and_read():
    """Index-time and query-time analysis diverging is silent: BM25 just stops
    matching. Holding a single Analyzer is what makes it impossible."""
    a = FakeAnalyzer()
    s = store(analyzer=a)
    assert s._analyzer is a
