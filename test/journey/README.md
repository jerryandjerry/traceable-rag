# The journey

One document, end to end: a clean account, a parse, three questions, three
grounded answers — asserted through the API and again through a browser.

```bash
make test-journey                 # both halves
pytest test/journey               # backend only
cd frontend && node ../test/journey/journey.ui.cjs   # browser only
```

Needs the full stack: Postgres, Elasticsearch, the API on :8000, the frontend
on :5181, and a funded model provider.

## Why this exists when the endpoint suites are green

The suites in `test/backend` check that each endpoint answers. They cannot check
that the *system works*, and three defects lived comfortably underneath them:

- GraphRAG raised `KeyError` on every document and the caller swallowed it, so
  ingestion reported success with no graph behind it
- the model emits the intent `kb(all)` where the planner matched only the
  literal `kb(filter)`, so retrieval was skipped entirely
- GraphRAG was never planned at all

Every one of them produced a **plausible answer**. That is the whole problem: an
endpoint test cannot tell a grounded answer from a fluent guess, so the pipeline
scored green while being disconnected from its own knowledge base.

It found four more the moment it ran, all of them in the last few inches
between a correct API response and a reader's eyes:

- inline citation markers never resolved. `components/markdown` matched
  `/cite_(\d+)/` and then looked up `cite_001`, while the backend mints
  `knowledge_base_<ts>_001` — so every marker stayed on screen as literal
  `[doc][cite_...]` text. The unit spec did not catch it because it
  re-implemented the regex and asserted against the copy, and it had the
  mismatch written down as a "KNOWN GAP" test.
- the model spells that marker three different ways (`cite_003`,
  `cite_940127_003`, `cite_knowledge_base_940127_003`) because the prompt asks
  for `[doc][cite_XXX]` *and* says XXX must equal the id it was given. All three
  now resolve by trailing ordinal.
- nothing read `data-citation-id`, so a span styled `cursor: pointer` and
  titled "Click to view source" did nothing.
- the upload indicator did not recognise the `upload` step the backend actually
  sends, so its mapper returned null and the stepper sat still.

## The document

`test/fixtures/ragflow/Doc1.pdf` — the one-page RAGFlow benchmark document
"Purpose of RAGFlow," pinned from an immutable upstream commit and distributed
under Apache-2.0. The journey and parser baseline share this one copy.

It is a real PDF with embedded text and fonts, so the test exercises the same
DeepDoc PDF path as an uploaded document.

The questions quote the document rather than paraphrase it:

- *"turn raw documents into reliable context"*
- *"retrieves the most relevant passages and sends them to the model as context"*
- *"reduces hallucinations and improves traceability"*

## Whether the model already knows the document is beside the point

What the answer has to do is **cite a marker that resolves to the chunk holding
the fact**. So the assertions follow the marker:

1. the answer contains a citation marker
2. that marker resolves to a citation that was actually sent
3. that citation is a `knowledge_base` chunk from `Doc1.pdf`
4. that chunk's text contains the fact being claimed

An answer written from memory cites nothing and fails at step 1. An answer that
cites the wrong chunk fails at step 4, however well its prose reads. Neither
depends on the document being obscure.

## The golden QA set

`make_report.py` can generate `REPORT.md`: every question with its expected
clause, the clause the answer actually cited, and the answer itself. Run it with

```bash
python test/journey/make_report.py
```



`golden_qa.json` — twelve questions against `Doc1.pdf`. Eight it answers, four
it does not. `test_answer_quality_e2e.py` drives them and scores three
things separately, because they fail for different reasons:

| | question |
|---|---|
| **context** | did the passage holding the answer reach the answer prompt? |
| **answer** | does the prose state the fact? |
| **grounding** | does a chunk the answer *cites* contain the fact? |

Context can succeed while the answer misses it. The answer can be right while
citing a chunk that does not support it — right for the wrong reason, which
reads perfectly and is worse than being wrong. Scoring them apart tells you
which half regressed.

**The gate is the aggregate, not each question.** Every answer is an LLM call
and which chunks survive the reranker's 0.1 relevancy floor varies between runs,
so a per-question suite would fail on a phrasing wobble and teach people to
ignore it. The per-question table is printed so a regression is legible.

**The four unanswerable questions are the sharp end.** A pipeline that will not
say "your documents do not cover this" is more dangerous than one that retrieves
badly. They check that the answer admits the gap, and that it never claims no
document was provided when one was ingested — a false statement that undermines
every other answer the user has had.

## "A clean database" means a new account

Dropping the real database would take the developer's data with it. A fresh user
is isolated by construction: the Elasticsearch index is named after the user id,
graph files are per user, and every table is scoped by it. Emptiness is
asserted, not assumed, and the account is deleted afterwards — which also
exercises the delete path that removes documents, history, graph and index.

## What each half proves

**`test_journey_e2e.py`** — the account starts empty; the parse completes; the
document is listed; chunks are retrievable and keep the document's facts; a
graph was written; each question is answered with a fact from the document;
every citation is `knowledge_base`, carries its passage, contains the answer,
and names an id that exists; the progress trace carries per-step durations.

**`journey.ui.cjs`** — the same journey a person makes: registers through the
form, uploads through the picker, types each question into the composer. Every
assertion reads rendered text, because a correct API response still has to
survive markdown rendering, citation resolution and the progress component —
and it was exactly there that citation markers were being dropped as literal
`[doc][cite_...]` text.
