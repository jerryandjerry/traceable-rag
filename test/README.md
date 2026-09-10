# Tests outside the packages

```
test/
├── conftest.py     live-stack fixtures: base_url, live_backend, api, session_id
├── architecture/   the one-way dependency and the slot wiring; no stack
├── pipeline/       the real routers over HTTP with deterministic slots; no stack
├── backend/        pytest against the running API
├── frontend/       playwright against the running frontend
└── journey/        one document end to end, API and browser; see journey/README.md
```

`backend/`, `frontend/` and `journey/` talk to a running system; nothing there
is stubbed. `architecture/` and `pipeline/` need no stack.

Unit tests live with their projects — `backend/tests/unit/` and
`frontend/tests/unit/` — because a test that reaches a live service is not a
unit test, and the split is what lets CI run one layer without the other.

## Running

The E2E and journey commands need Postgres, Elasticsearch, the API on `:8000`,
and the frontend on `:5181`. Architecture, integration, unit, and golden checks
are local.

```bash
make test                # every tier, in order
make test-architecture   # no stack
make test-integration    # test/pipeline, no stack
make test-golden         # complete offline compatibility suite; real local OCR
make test-e2e            # test/backend + both browser scripts
make test-journey        # test/journey; also needs a funded provider
make test-answer-quality # golden-set pass rates; a measurement, not a gate

pytest test/backend      # API only, from the repo root
cd frontend && node ../test/frontend/ui.e2e.cjs        # browser only
cd frontend && node ../test/frontend/features.ui.cjs   # every feature a person can reach
```

The browser harness runs from `frontend/` so its screenshots and Playwright
install resolve, and it locates `playwright` explicitly — Node resolves modules
from the script's directory, not the cwd, and this script sits outside
`frontend/`.

## Environment

| Variable | Default |
|---|---|
| `E2E_BASE_URL` | `http://localhost:8000` |
| `E2E_WEB_URL` | `http://localhost:5181` |
| `E2E_USER` / `E2E_PASS` | Required by `ui.e2e.cjs` |
| `E2E_ANSWER_WAIT` | `120000` ms |

Supply an existing test account without committing its credentials:

```bash
E2E_USER='<test-user>' E2E_PASS='<test-password>' make test-e2e
```

The prefix avoids collision with the shell's `USER` variable.

## What is covered

**Architecture** — no import points upward or sideways into a peer slot, the
API never touches storage or a slot directly and calls no repository write,
every slot has a Protocol and a factory, and `QueryPipeline()` builds all of
them through those factories.

**Pipeline** — the trust boundary (401/404/422 before any header, revoked
tokens, policy-denied tools), the ingest job (ownership, stage order, truthful
outcomes, page coverage, upload limits, the tenant lock), account deletion
order, the context routes, and the stream lifecycle (cancellation, the turn
deadline, the answer final before `[DONE]`, sanitized errors, no provider
reached outside policy).

**API** — registration and login including duplicates and wrong passwords,
`/me`, session creation and listing, message history, document ingestion,
retrieval with grounded citations, account deletion, and cross-user isolation:
a second user must not read, delete, or write into another user's session, and
a deleted user's token is refused everywhere.

**Browser** — `ui.e2e.cjs`: login, the redirect for an unauthenticated visit,
the web search toggle, PDF upload, every route, and the graph viewer
initialising with real nodes and links. `features.ui.cjs`: every feature a
person can reach — register, ask from the home composer, follow up, the
progress trace, web search with the search step asserted, history after
reload, attach context, upload with the four-step row, View File chunks, a
knowledge question, graph nodes, sidebar delete, Delete File, 404, logout. Both
fail the run on any console error, failed request, or HTTP >= 400 seen at any
point.

## When the model provider is down

Tests that need a real completion or embedding declare `needs_provider` and
**skip** with the provider's own message. An account in arrears, an expired key
or a regional outage are not defects, and reporting them as failures teaches
people to ignore a red suite.

Everything else -- auth, sessions, history, account management, cross-user
isolation -- runs without a provider, because none of it retrieves anything.

## Not covered

`/deep_research/` is reserved but unimplemented. Its inert module is not
registered, the frontend control is hidden, and dormant
`ChatRequest.deep_research` plumbing remains for a future implementation.
