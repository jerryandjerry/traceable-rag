"""One complete journey: clean account -> document -> parsed -> three answers.

    pytest test/journey                    # from the repo root
    make test-journey                      # backend and browser halves

The suite verifies that an ingested document produces searchable chunks and a
graph, then follows every answer citation back to the fixture passage that
supports the claim.

The stages are ordered and share state through session-scoped fixtures, so a
failure names the stage that broke rather than collapsing the whole journey.
"""
from __future__ import annotations

import json
import re

import pytest
from conftest import sse

pytestmark = pytest.mark.e2e

# `☒ searching user knowledge base (1.23s)` -- the timing the UI renders.
DURATION = re.compile(r"\(\d+\.\d{2}s\)")
# The citation markers the answer prompt is told to emit.
CITATION_MARKER = re.compile(r"\[(?:doc|web)\]\[([^\]]+)\]")


# ============================================================ stage 1: clean
def test_the_account_starts_with_nothing(clean_user):
    """A journey that begins with leftovers proves nothing about ingestion."""
    assert clean_user.req("GET", "/get_files/").json() == []
    assert clean_user.req("GET", "/get_sessions/").json()["sessions"] == []


# =========================================================== stage 2: ingest
def test_parsing_runs_to_completion(parsed):
    steps = [f.get("step") for f in parsed["frames"]]
    assert "complete" in steps, f"parse never completed; steps were {steps}"


def test_the_parsed_document_is_listed(clean_user, parsed, sample_doc):
    names = {
        f.get("file_name") or f.get("filename")
        for f in clean_user.req("GET", "/get_files/").json()
    }
    assert sample_doc["path"].name in names, f"not listed: {names}"


def test_the_document_produced_chunks(clean_user, parsed, sample_doc, needs_provider):
    """Chunks that lost the document's facts cannot ground an answer, however
    many of them there are."""
    chunks = clean_user.req(
        "GET", f"/document-chunks/{parsed['filename']}").json().get("chunks", [])
    assert chunks, "the document produced no chunks"
    body = " ".join(str(c.get("content_with_weight", "")) for c in chunks).lower()
    for spec in sample_doc["questions"]:
        for token in spec["must_contain"]:
            assert token.lower() in body, (
                f"the chunk text lost {token!r} ({spec['fact']}), so no answer "
                "can be grounded in it"
            )


def test_ingestion_built_a_graph(clean_user, parsed, repo_root):
    """Successful ingestion writes a non-empty graph index for the user."""
    # The live-stack suite runs the API locally and inspects its state directory.
    graph_dir = repo_root / "backend" / "var" / "graph"
    uid = clean_user.user_id
    nodes = graph_dir / f"vdb_nodes_{uid}.json"
    if not nodes.exists():
        pytest.fail(
            f"no graph written for user {uid}: GraphRAG failed silently during ingest"
        )
    data = json.loads(nodes.read_text(encoding="utf-8"))
    assert data.get("embeddings"), "graph index written but empty"


# ========================================================== stage 3: answers
@pytest.fixture(scope="session")
def answers(clean_user, sample_doc, parsed, provider_available):
    """Ask every question once; the assertions below read the results."""
    out = []
    for spec in sample_doc["questions"]:
        session_id = clean_user.req("POST", "/create_session/").json()["session_id"]
        frames = sse(clean_user.req(
            "POST", f"/ai_search/?session_id={session_id}",
            json={"message": spec["question"], "web_search": False,
                  "deep_research": False},
            stream=True, timeout=900,
        ))
        answer = "".join(
            str(f.get("content", "")) for f in frames
            if f.get("role") == "assistant" and f.get("content")
        )
        citations = next(
            (f["citations"] for f in reversed(frames) if f.get("citations")), [])
        progress = next(
            (f.get("content", "") for f in reversed(frames)
             if f.get("role") == "workflow_progress"), "")
        out.append({"spec": spec, "answer": answer, "citations": citations,
                    "progress": progress, "frames": frames})
    return out


def _by_question(answers):
    return [(a["spec"]["question"], a) for a in answers]


def test_every_question_was_answered(answers):
    for spec_q, a in _by_question(answers):
        assert a["answer"].strip(), f"no answer produced for {spec_q!r}"


def test_no_question_reported_an_error(answers):
    for spec_q, a in _by_question(answers):
        errors = [f.get("content") for f in a["frames"] if f.get("role") == "error"]
        assert not errors, f"{spec_q!r} errored: {errors}"


def test_every_answer_states_the_documents_fact(answers):
    """The answer must state the fixture fact; citation tests prove its source."""
    for spec_q, a in _by_question(answers):
        for token in a["spec"]["answer_should_contain"]:
            assert token.lower() in a["answer"].lower(), (
                f"{spec_q!r} was answered without the document's fact "
                f"{a['spec']['fact']!r}; retrieval did not reach the answer.\n"
                f"answer was: {a['answer'][:400]}"
            )


def test_every_answer_cites_the_knowledge_base(answers):
    for spec_q, a in _by_question(answers):
        assert a["citations"], f"{spec_q!r} produced no citations"
        kinds = {c.get("source_type") for c in a["citations"]}
        assert kinds == {"knowledge_base"}, (
            f"{spec_q!r} cited {kinds}; with web search off and a document "
            "ingested, every citation must come from the knowledge base"
        )


def test_citations_carry_the_documents_content(answers, sample_doc):
    """A citation that does not carry its passage cannot be checked by a reader.

    Provenance is asserted on the stored document name rather than text that
    might appear in an answer or passage.
    """
    stem = sample_doc["path"].stem
    for spec_q, a in _by_question(answers):
        bodies = [str(c.get("content_with_weight", "")) for c in a["citations"]]
        assert all(bodies), f"{spec_q!r} returned an empty citation body"
        names = [
            str(c.get("docnm") or c.get("docnm_kwd") or c.get("title") or "")
            for c in a["citations"]
        ]
        assert any(stem in n for n in names), (
            f"{spec_q!r} cited chunks that do not come from the ingested "
            f"document: {names}"
        )


def _cited_in(answer: str, citations: list[dict]) -> list[dict]:
    """The citations the answer actually points at, in marker order.

    Resolved the way components/markdown does: exact, prefix-stripped, or by
    trailing ordinal, because the model spells the marker three different ways.
    """
    by_id = {str(c.get("citation_id")): c for c in citations}
    out = []
    for marker in CITATION_MARKER.findall(answer):
        bare = marker.removeprefix("cite_")
        hit = by_id.get(marker) or by_id.get(bare)
        if hit is None:
            tail = re.search(r"(\d+)$", bare)
            if tail:
                hit = next((c for k, c in by_id.items()
                            if k.endswith(f"_{tail.group(1)}")), None)
        if hit is not None and hit not in out:
            out.append(hit)
    return out


def test_the_marker_in_the_answer_points_at_the_chunk_holding_the_fact(answers):
    """The load-bearing assertion.

    It does not matter whether the model already knows this document. What
    matters is that its citation marker resolves to a real chunk, that the chunk
    came from the document just ingested, and that the fact is actually in it.
    An answer from memory cites nothing; an answer citing the wrong chunk fails
    here even though its prose reads correctly.
    """
    for spec_q, a in _by_question(answers):
        cited = _cited_in(a["answer"], a["citations"])
        assert cited, (
            f"{spec_q!r} was answered with no marker resolving to a citation, so "
            "nothing ties the claim to a retrieved chunk"
        )
        for token in a["spec"]["must_contain"]:
            holding = [
                c for c in cited
                if token.lower() in str(c.get("content_with_weight", "")).lower()
            ]
            assert holding, (
                f"{spec_q!r} cites {[c.get('citation_id') for c in cited]}, and "
                f"none of those chunks contains {token!r} ({a['spec']['fact']}). "
                "The answer may be right for the wrong reason."
            )


def test_the_cited_chunk_came_from_the_ingested_document(answers, sample_doc):
    """A marker pointing at a web result or the model's own context is not a
    citation of this document."""
    stem = sample_doc["path"].stem
    for spec_q, a in _by_question(answers):
        for c in _cited_in(a["answer"], a["citations"]):
            assert c.get("source_type") == "knowledge_base", (
                f"{spec_q!r} cites {c.get('citation_id')} which is "
                f"{c.get('source_type')}, not the knowledge base"
            )
            name = str(c.get("docnm") or c.get("docnm_kwd") or c.get("title") or "")
            assert stem in name, (
                f"{spec_q!r} cites {c.get('citation_id')} from {name!r}, "
                f"not from {stem}"
            )


def _resolves(marker: str, known: set[str]) -> bool:
    """Whether a citation marker names a citation that was actually sent.

    Resolved the way components/markdown does: exact, prefix-stripped, or by
    trailing ordinal. The prompt asks for [doc][cite_XXX] *and* says XXX must
    equal the supplied id, and the ids look like knowledge_base_940127_003 --
    so the model compromises differently almost every answer: cite_003,
    cite_940127_003, cite_knowledge_base_940127_003. Ordinals are unique within
    one answer, so the tail identifies the citation whichever spelling arrives.
    """
    bare = marker.removeprefix("cite_")
    if marker in known or bare in known:
        return True
    tail = re.search(r"(\d+)$", bare)
    return bool(tail) and any(k.endswith(f"_{tail.group(1)}") for k in known)


def test_answers_reference_citations_that_exist(answers):
    """A marker naming an id that was never sent renders as literal text in the
    answer, so the reader sees `[doc][cite_...]` instead of a source.

    The marker is `cite_` + the citation_id: the prompt asks for the form
    [doc][cite_XXX] and also says XXX must match the ids supplied, and the ids
    supplied are `<source_type>_<timestamp>_<n>`, so the model concatenates the
    two. components/markdown resolves both spellings.
    """
    for spec_q, a in _by_question(answers):
        referenced = set(CITATION_MARKER.findall(a["answer"]))
        if not referenced:
            continue
        known = {str(c.get("citation_id")) for c in a["citations"]}
        unknown = {m for m in referenced if not _resolves(m, known)}
        assert not unknown, f"{spec_q!r} cites unknown ids {unknown}; known {known}"


def test_answers_actually_cite_something(answers):
    """An answer with no marker at all leaves the reader no way to check it."""
    uncited = [q for q, a in _by_question(answers)
               if not CITATION_MARKER.search(a["answer"])]
    assert not uncited, f"answers carried no citation marker: {uncited}"


# ========================================================= stage 4: reporting
def test_search_progress_reports_step_durations(answers):
    """The frontend renders this string verbatim, so a missing timing here is a
    missing timing on screen."""
    for spec_q, a in _by_question(answers):
        assert DURATION.search(a["progress"]), (
            f"{spec_q!r} produced no step duration in its progress trace:\n"
            f"{a['progress'][:300]}"
        )


def test_search_progress_names_the_retrieval_step(answers):
    for spec_q, a in _by_question(answers):
        assert "searching user knowledge base" in a["progress"], (
            f"{spec_q!r} never reported a knowledge-base step:\n{a['progress'][:300]}"
        )


# These steps are mapped by frontend/src/pages/repository/index.tsx.
UI_KNOWN_STEPS = {
    "upload", "upload_pending", "upload_finish",
    "parse_pending", "parse_finish",
    "encode_pending", "encode_finish",
    "database_pending", "database_finish",
    "complete", "error",
}


def test_parse_steps_are_ones_the_upload_indicator_understands(parsed):
    """The parse stream reports steps, not timings -- the upload UI is a
    stepper, and per-step durations are a query-pipeline feature. What matters
    here is that the frontend recognises the step names it is sent."""
    emitted = {f.get("step") for f in parsed["frames"] if f.get("step")}
    assert emitted, "the parse stream reported no steps at all"
    unknown = emitted - UI_KNOWN_STEPS
    assert not unknown, (
        f"the parse emitted steps the upload indicator cannot map: {unknown}. "
        "mapSseStepToUiStep returns null for these, so the UI stops advancing "
        "while the parse carries on."
    )


def test_the_parse_reaches_the_step_that_finishes_the_indicator(parsed):
    emitted = {f.get("step") for f in parsed["frames"]}
    assert emitted & {"complete", "database_finish"}, (
        f"the indicator never reaches 'done'; steps were {sorted(x for x in emitted if x)}"
    )
