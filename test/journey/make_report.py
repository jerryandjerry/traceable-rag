#!/usr/bin/env python
"""Run the golden QA set and write REPORT.md: expected vs what the pipeline said.

    python test/journey/make_report.py

Needs the stack up and a funded provider. Registers a throwaway account, ingests
the RAGFlow fixture named by golden_qa.json, asks every question, and writes a
side-by-side report. The account is deleted afterwards.

The test asserts thresholds; this shows the answers, so a score can be read
rather than trusted.
"""
from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "fixtures" / "ragflow"
BASE = "http://localhost:8000"
PASSWORD = "PytestPass123!"
CITATION_MARKER = re.compile(r"\[(?:doc|web)\]\[([^\]]+)\]")


def norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def contains(haystack: str, needle: str) -> bool:
    return norm(needle) in norm(haystack)


def sse(response) -> list[dict]:
    out = []
    for raw in response.iter_lines():
        if not raw or not raw.startswith(b"data: "):
            continue
        body = raw[6:].decode("utf-8", "replace")
        if body == "[DONE]":
            break
        try:
            out.append(json.loads(body))
        except json.JSONDecodeError:
            continue
    return out


def cited(answer: str, citations: list[dict]) -> list[dict]:
    """The citations the answer actually points at, resolved the way the
    frontend resolves them."""
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


def cell(text: str, limit: int = 340) -> str:
    """One markdown table cell: no pipes, no newlines, trimmed."""
    flat = " ".join((text or "").split()).replace("|", "\\|")
    return (flat[:limit] + "…") if len(flat) > limit else (flat or "—")


def main() -> int:
    spec = json.loads((HERE / "golden_qa.json").read_text(encoding="utf-8"))
    pdf = FIXTURES / spec["document"]
    if not pdf.exists():
        print(f"missing {pdf}", file=sys.stderr)
        return 1

    s = requests.Session()
    user = f"report_{uuid.uuid4().hex[:8]}"
    s.post(f"{BASE}/register", json={"username": user, "password": PASSWORD}, timeout=60)
    token = s.post(f"{BASE}/login", json={"username": user, "password": PASSWORD},
                   timeout=60).json()["access_token"]
    s.headers["Authorization"] = f"Bearer {token}"
    print(f"account {user}")

    with pdf.open("rb") as fh:
        pid = s.post(f"{BASE}/start-processing",
                     files={"files": (pdf.name, fh, "application/pdf")},
                     timeout=300).json()["process_id"]
    for frame in sse(s.get(f"{BASE}/get-process-progress/{pid}", stream=True, timeout=1800)):
        if frame.get("step") == "complete":
            break
    chunks = s.get(f"{BASE}/document-chunks/{pdf.name}", timeout=120).json().get("chunks", [])
    print(f"ingested: {len(chunks)} chunks")

    rows = []
    for q in spec["questions"]:
        sid = s.post(f"{BASE}/create_session/", timeout=60).json()["session_id"]
        frames = sse(s.post(f"{BASE}/ai_search/?session_id={sid}",
                            json={"message": q["question"], "web_search": False,
                                  "deep_research": False},
                            stream=True, timeout=900))
        answer = "".join(str(f.get("content", "")) for f in frames
                         if f.get("role") == "assistant" and f.get("content"))
        citations = next((f["citations"] for f in reversed(frames) if f.get("citations")), [])
        used = cited(answer, citations)
        gold = q.get("gold")

        ok = True
        for t in q.get("expect_all", []):
            ok = ok and contains(answer, t)
        if q.get("expect_any"):
            ok = ok and any(contains(answer, t) for t in q["expect_any"])
        for t in q.get("must_not_contain", []):
            ok = ok and not contains(answer, t)

        rows.append({
            "q": q, "answer": answer, "citations": citations, "used": used,
            "answer_ok": ok,
            "context": contains(" ".join(str(c.get("content_with_weight", ""))
                                         for c in citations), gold) if gold else None,
            "grounding": any(contains(str(c.get("content_with_weight", "")), gold)
                             for c in used) if gold else None,
        })
        print(f"  {q['id']} done")

    (HERE / "REPORT.md").write_text(render(spec, rows, len(chunks)), encoding="utf-8")
    print(f"\nwrote {HERE / 'REPORT.md'}")

    s.delete(f"{BASE}/me", json={"password": PASSWORD, "confirm": "DELETE"}, timeout=600)
    return 0


def render(spec: dict, rows: list[dict], n_chunks: int) -> str:
    def mark(v):
        return "—" if v is None else ("✅" if v else "❌")

    def rate(key):
        got = [r for r in rows if r[key] is not None]
        return f"{sum(1 for r in got if r[key])}/{len(got)}" if got else "—"

    answered = [r for r in rows if not r["q"].get("unanswerable")]
    gaps = [r for r in rows if r["q"].get("unanswerable")]

    out = [
        "# Answer quality report",
        "",
        f"**Document** `{spec['document']}` — {spec['source']}, {n_chunks} chunks.",
        "",
        ("Generated by `python test/journey/make_report.py` against a live stack: a "
         "throwaway account, this document ingested into it, every question in "
         "`golden_qa.json` asked once. Answers vary between runs; this is one run."),
        "",
        "## Scores",
        "",
        "| | result | meaning |",
        "|---|---|---|",
        f"| context | {rate('context')} | the passage holding the answer reached the answer prompt |",
        f"| answer | {rate('answer_ok')} | the prose states the fact |",
        f"| grounding | {rate('grounding')} | a chunk the answer **cites** contains the fact |",
        "",
        ("Context can pass while grounding fails: the right passage reached the "
         "prompt, but the answer cited a different chunk. That reads perfectly and "
         "is worse than being wrong, which is why the two are scored apart."),
        "",
        "---",
        "",
        "## Questions the document answers",
        "",
    ]

    for r in answered:
        q = r["q"]
        out += [
            f"### {q['id']}. {q['question']}",
            "",
            f"*{q['why']}*",
            "",
            "| | |",
            "|---|---|",
            f"| **Expected clause** | {cell(q['gold'])} |",
            f"| **Answer must state** | {cell(', '.join(q.get('expect_all', []) + q.get('expect_any', [])))} |",
            f"| **Clause actually cited** | {cell(r['used'][0].get('content_with_weight') if r['used'] else '')} |",
            (f"| **Citations** | {len(r['citations'])} sent, {len(r['used'])} referenced "
             f"({', '.join(sorted({str(c.get('source_type')) for c in r['citations']})) or '—'}) |"),
            f"| **context / answer / grounding** | {mark(r['context'])} / {mark(r['answer_ok'])} / {mark(r['grounding'])} |",
            "",
            "**Pipeline answer**",
            "",
            "> " + (" ".join(r["answer"].split())[:700] or "*(empty)*"),
            "",
        ]

    out += [
        "---",
        "",
        "## Questions the document does not answer",
        "",
        ("These are the sharp end. A pipeline that will not say *your documents do "
         "not cover this* is more dangerous than one that retrieves badly. Each "
         "answer must admit the gap, and must never claim no document was provided "
         "— the user ingested one."),
        "",
    ]
    for r in gaps:
        q = r["q"]
        out += [
            f"### {q['id']}. {q['question']}",
            "",
            f"*{q['why']}*",
            "",
            "| | |",
            "|---|---|",
            "| **Expected** | admits the document does not cover it |",
            f"| **Must never say** | {cell(', '.join(q.get('must_not_contain', [])))} |",
            f"| **Verdict** | {mark(r['answer_ok'])} |",
            "",
            "**Pipeline answer**",
            "",
            "> " + (" ".join(r["answer"].split())[:700] or "*(empty)*"),
            "",
        ]

    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
