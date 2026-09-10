"""Does the pipeline produce answers that are any good?

    pytest test/journey/test_answer_quality_e2e.py -s

The rest of the journey proves the pipeline *runs*: it parses, it retrieves, it
cites honestly, it does not crash. None of that says the answers are correct.
This drives a golden QA set built from the document and scores three separate
things, because they fail in different ways and for different reasons:

    context      did the passage holding the answer reach the answer prompt?
    answer       does the prose state the fact?
    grounding    does a chunk the answer *cites* contain the fact?

Retrieval can succeed while the answer misses it. The answer can be right while
citing the wrong chunk -- right for the wrong reason, which reads perfectly and
is worse than being wrong. Scoring them apart is what tells you which half
regressed.

**Thresholds, not per-question assertions.** Every answer is an LLM call, so any
single question can vary between runs; a suite that fails on one phrasing wobble
teaches people to ignore it. The gate is the aggregate, and the per-question
table is printed so a regression is legible rather than a bare percentage.
"""
from __future__ import annotations

import json
import re

import pytest
from conftest import sse

pytestmark = pytest.mark.e2e

# Aggregate gates, with room for one wobble in eight. Which chunks survive the
# reranker's 0.1 relevancy floor varies between runs, so these are not 100%.
MIN_CONTEXT = 0.85
MIN_ANSWER = 0.85
MIN_GROUNDING = 0.85

CITATION_MARKER = re.compile(r"\[(?:doc|web)\]\[([^\]]+)\]")


def _norm(s: str) -> str:
    """Lowercase and collapse whitespace, so a line break inside a quoted
    phrase does not read as a missing fact."""
    return " ".join((s or "").lower().split())


def _contains(haystack: str, needle: str) -> bool:
    return _norm(needle) in _norm(haystack)


def _cited(answer: str, citations: list[dict]) -> list[dict]:
    """The citations the answer actually points at."""
    by_id = {str(c.get("citation_id")): c for c in citations}
    out: list[dict] = []
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


@pytest.fixture(scope="module")
def golden(repo_root) -> dict:
    path = repo_root / "test" / "journey" / "golden_qa.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scored(clean_user, parsed, golden, provider_available) -> list[dict]:
    """Ask every golden question once, and score each answer three ways."""
    rows = []
    for spec in golden["questions"]:
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
        ranked = " ".join(str(c.get("content_with_weight", "")) for c in citations)
        cited = _cited(answer, citations)

        gold = spec.get("gold")
        retrieval = _contains(ranked, gold) if gold else None
        grounding = (
            any(_contains(str(c.get("content_with_weight", "")), gold) for c in cited)
            if gold else None
        )

        ok = True
        for token in spec.get("expect_all", []):
            ok = ok and _contains(answer, token)
        if spec.get("expect_any"):
            ok = ok and any(_contains(answer, t) for t in spec["expect_any"])
        for token in spec.get("must_not_contain", []):
            ok = ok and not _contains(answer, token)

        rows.append({
            "id": spec["id"], "spec": spec, "answer": answer,
            "citations": citations, "cited": cited,
            "context": retrieval, "answer_ok": ok, "grounding": grounding,
        })

    _report(rows)
    return rows


def _report(rows: list[dict]) -> None:
    def mark(v):
        return " " if v is None else ("PASS" if v else "FAIL")

    print("\n\n  id    context  answer  grounding  question")
    print("  " + "-" * 74)
    for r in rows:
        print(f"  {r['id']:5} {mark(r['context']):>7}  {mark(r['answer_ok']):>6}"
              f"  {mark(r['grounding']):>9}  {r['spec']['question'][:40]}")
    for r in rows:
        if not r["answer_ok"]:
            print(f"\n  {r['id']} answer was:\n    {r['answer'][:300].strip()}")


def _rate(rows: list[dict], key: str) -> tuple[float, list[str]]:
    scored = [r for r in rows if r[key] is not None]
    failed = [r["id"] for r in scored if not r[key]]
    return (len(scored) - len(failed)) / max(1, len(scored)), failed


def test_the_passage_holding_the_answer_reaches_the_prompt(scored):
    """If the right chunk never reaches the prompt, nothing downstream can be
    right for the right reason."""
    rate, failed = _rate(scored, "context")
    assert rate >= MIN_CONTEXT, (
        f"context {rate:.0%} < {MIN_CONTEXT:.0%}; missed {failed}"
    )


def test_the_answers_state_the_facts(scored):
    rate, failed = _rate(scored, "answer_ok")
    assert rate >= MIN_ANSWER, (
        f"answer correctness {rate:.0%} < {MIN_ANSWER:.0%}; wrong on {failed}"
    )


def test_the_answers_cite_the_chunk_that_holds_the_fact(scored):
    """Right for the right reason. An answer can state the fact while citing a
    chunk that does not contain it, and that reads exactly like a good answer."""
    rate, failed = _rate(scored, "grounding")
    assert rate >= MIN_GROUNDING, (
        f"grounding {rate:.0%} < {MIN_GROUNDING:.0%}; ungrounded {failed}"
    )


def test_a_question_the_document_cannot_answer_is_not_invented(scored, golden):
    """Two failures at once.

    Fabrication: the document names no embedding model, default chunk size,
    vector database, or benchmark score. Requiring the answer to *acknowledge
    the gap* prevents those details from being invented with citations attached.

    Denial: saying no document was provided when one was ingested. That is false
    and it undermines every other answer the user has had, so the phrasings are
    forbidden outright.
    """
    for r in scored:
        if not r["spec"].get("unanswerable"):
            continue
        assert r["answer_ok"], (
            f"{r['id']} invented an answer the document does not contain:\n"
            f"{r['answer'][:400]}"
        )


def test_every_answer_cites_something(scored):
    """An uncited answer cannot be checked by a reader, however correct."""
    uncited = [r["id"] for r in scored
               if not r["spec"].get("unanswerable") and not r["cited"]]
    assert not uncited, f"answered with no resolvable citation: {uncited}"
