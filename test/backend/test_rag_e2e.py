"""End-to-end retrieval tests: ingest a real document, then query it."""
import json

import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def ingested_document(api, sample_pdf):
    """Push a real PDF all the way into Elasticsearch."""
    with sample_pdf.open("rb") as fh:
        r = api.req(
            "POST",
            "/start-processing",
            files={"files": (sample_pdf.name, fh, "application/pdf")},
        )
    r.raise_for_status()
    process_id = r.json()["process_id"]

    stream = api.req(
        "GET", f"/get-process-progress/{process_id}", stream=True, timeout=900
    )
    errors = []
    for raw in stream.iter_lines():
        if not raw or not raw.startswith(b"data: "):
            continue
        try:
            msg = json.loads(raw[6:].decode("utf-8", "replace"))
        except json.JSONDecodeError:
            continue
        if msg.get("step") == "error":
            errors.append(msg.get("message", ""))
        if msg.get("step") == "complete":
            break

    if errors:
        pytest.fail("ingestion failed:\n" + "\n".join(errors))
    return sample_pdf.name


def test_ingested_document_is_listed(api, ingested_document):
    files = api.req("GET", "/get_files/").json()
    names = {f.get("file_name") or f.get("filename") for f in files}
    assert any(ingested_document.split(".")[0][:30] in str(n) for n in names), (
        f"{ingested_document} not in {names}"
    )


def test_document_chunks_are_retrievable(needs_provider, api, ingested_document):
    r = api.req("GET", f"/document-chunks/{ingested_document}")
    assert r.status_code == 200
    chunks = r.json().get("chunks", [])
    assert chunks, "document produced no chunks"
    first = chunks[0]
    for field in ("id", "content_with_weight"):
        assert field in first, f"chunk missing {field}: {first.keys()}"


def test_knowledge_base_search_returns_grounded_citations(needs_provider, api, session_id, ingested_document):
    """Require the answer's cited passage to contain the fixture fact."""
    r = api.req(
        "POST",
        f"/ai_search/?session_id={session_id}",
        json={
            "message": "According to the uploaded document, what is RAGFlow "
                       "designed to turn raw documents into?",
            "web_search": False,
            "deep_research": False,
        },
        stream=True,
        timeout=600,
    )
    assert r.status_code == 200

    answer, citations, progress = "", [], ""
    for raw in r.iter_lines():
        if not raw or not raw.startswith(b"data: "):
            continue
        body = raw[6:].decode("utf-8", "replace")
        if body == "[DONE]":
            break
        try:
            msg = json.loads(body)
        except json.JSONDecodeError:
            continue
        assert msg.get("role") != "error", f"stream error: {msg}"
        if msg.get("role") == "workflow_progress":
            progress = msg.get("content", "")
        if msg.get("role") == "assistant" and not msg.get("thinking"):
            answer += msg.get("content", "")
        if msg.get("citations"):
            citations = msg["citations"]

    assert answer.strip(), "no answer produced"
    assert citations, "no citations returned -- nothing was retrieved"

    normalized_answer = " ".join(answer.lower().split())
    assert "reliable" in normalized_answer and "context" in normalized_answer, answer

    cited_text = " ".join(
        str(c.get("content_with_weight", "")) for c in citations
    ).lower()
    assert "reliable context" in " ".join(cited_text.split()), (
        "the citations do not contain the fact claimed by the answer"
    )

    # Which tool the orchestrator picks is LLM-driven and varies, so do not
    # assert a specific source_type. What must hold is that every citation is
    # grounded in retrieved text.
    for c in citations:
        assert c.get("source_type"), f"citation without a source: {c}"
        assert c.get("content_with_weight"), f"empty citation: {c.get('citation_id')}"

    # A completed knowledge-base search reports measurable work.
    assert "searching user knowledge base (0.00s)" not in progress, (
        "knowledge base search returned instantly -- it is failing silently.\n"
        f"progress:\n{progress}"
    )


def test_citation_ids_are_unique_and_typed(api, session_id, ingested_document):
    r = api.req(
        "POST",
        f"/ai_search/?session_id={session_id}",
        json={"message": "Summarise the uploaded document in two sentences.",
              "web_search": False, "deep_research": False},
        stream=True,
        timeout=600,
    )
    citations = []
    for raw in r.iter_lines():
        if raw and raw.startswith(b"data: "):
            body = raw[6:].decode("utf-8", "replace")
            if body == "[DONE]":
                break
            try:
                msg = json.loads(body)
            except json.JSONDecodeError:
                continue
            if msg.get("citations"):
                citations = msg["citations"]

    if not citations:
        pytest.skip("no citations returned for this query")

    ids = [c["citation_id"] for c in citations]
    assert len(ids) == len(set(ids)), f"duplicate citation ids: {ids}"
    for c in citations:
        assert c.get("source_type"), f"citation without source_type: {c}"
        assert c.get("content_with_weight"), f"empty citation body: {c['citation_id']}"


def test_web_search_produces_web_citations(api, session_id):
    """web_search=True must actually reach the web tool."""
    r = api.req(
        "POST",
        f"/ai_search/?session_id={session_id}",
        json={"message": "What is today's weather in Toronto?",
              "web_search": True, "deep_research": False},
        stream=True,
        timeout=600,
    )
    assert r.status_code == 200
    progress = ""
    for raw in r.iter_lines():
        if raw and raw.startswith(b"data: "):
            body = raw[6:].decode("utf-8", "replace")
            if body == "[DONE]":
                break
            try:
                msg = json.loads(body)
            except json.JSONDecodeError:
                continue
            if msg.get("role") == "workflow_progress":
                progress = msg.get("content", "")
    assert "searching online" in progress, (
        f"web tool never ran with web_search=True; progress was:\n{progress}"
    )
