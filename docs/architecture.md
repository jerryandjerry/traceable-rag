# Traceable RAG architecture

Traceable RAG has query and ingestion pipelines, eight service slots, four
provider families, and three persistence technologies. This document defines
their stable boundaries, trust model, and storage ownership.

A **slot** is a directory containing:

```
<slot>/
├── base.py       the Protocol, typed with models/ contracts on both sides
└── factory.py    build_*(...) -> the configured Protocol implementation
```

Replacing an implementation changes its factory and selector; callers continue
to depend on the slot Protocol.

Protocols are structural: a module or object satisfies one by exposing the
declared operations; inheritance is not required. Representative operations:

| slot | the function that implements it |
|---|---|
| `intent/` | `analyze_query_intent` |
| `planner/` | `agent_plan` |
| `evaluator/` | `evaluate_context_sufficiency`, `is_sufficient`, `reflection` |
| `answer/` | `get_chat_completion`, `casual_chat_completion` |
| `executer/tools/` | `rag`, `web_search_answer`, `direct_llm_answer`, `graphrag` |

Providers are not slots. A slot is one step in a pipeline; an embedder is
something several steps call. They live in `providers/`, below `database/`,
because both storage engines embed query text themselves and so a store needs
an embedder, while a provider never needs a store. The VLM parser is the named
exception: its multimodal request shape is outside the text-only `LLM` Protocol,
so `service/parsers/vlm/processor.py` owns a dedicated OpenAI client.

---

## The layout

### The rule

One rule decides where everything goes, and a test enforces it
(`test/architecture/test_layering.py`):

```
api/                    HTTP: auth, job issuance, wire encoding
  pipeline/             the order of the steps, and the run's state
    service/<slot>/     one step each; peers, never importing each other
      database/         connections and persisted formats
        providers/      LLM, embedding, web search, rerank clients
          utils/        stateless helpers
          models/ exceptions/             pure data; config/ is an import leaf
```

A layer may import a lower layer at any distance. It may not import upward or
sideways into a peer slot. `pipeline/` may not call providers or vendor code
directly. `api/` normally calls pipelines or typed PostgreSQL repository reads;
two authenticated, tenant-checked filesystem exceptions remain: context routes
stage attachments under the owned session, and graph routes read the user's
graph files.

| | what belongs there |
|---|---|
| **`api/`** | HTTP adaptation, authorization, job issuance, typed PostgreSQL reads, scoped attachment/graph file I/O, wire encoding |
| **`pipeline/`** | the order of the steps, and the state one run accumulates |
| **`service/`** | the eight pipeline slots plus their shared provider-lifecycle boundary |
| **`database/`** | connections, persisted formats, tenant-scoped read and write primitives |
| **`providers/`** | stateful I/O adapters: the model, embedding, search and rerank clients |
| **`utils/`** | stateless, domain-neutral helpers |
| **`models/` `config/` `exceptions/`** | models and exceptions are pure data leaves; config is an import leaf whose settings load may create runtime directories and a development JWT key |

The provider/slot distinction and multimodal exception are defined above. The
approved `database/ -> providers/` edge lets Elasticsearch and graph stores
embed their own queries; no provider imports a store.

### The tree

```
visionagent/
├── api/                     HTTP boundary; scoped context/graph file exceptions
│   ├── body_limits.py       pre-parser raw-body and concurrent-upload admission fence
│   ├── deps.py              mints authorized jobs and the delete command; request-level policy boundary
│   ├── security.py          JWT verification, existence + revocation check, token minting, ownership
│   ├── sse.py               query-turn typed/legacy SSE helpers; route-specific frames stay in routes
│   └── routes/
├── pipeline/                orchestration; owns AgentState
│   ├── query.py             one turn: intent → plan → execute → rerank → judge → answer
│   ├── ingest.py            durable upload dispatch: stage → lease → parse → graph → index → Postgres; delete
│   ├── context.py           attach, read, remove, clear a session's context
│   ├── account.py           authenticate, register, change password, delete account
│   ├── session.py           create and delete a chat session
│   ├── startup.py           what runs once at boot (schema upgrades); lifespan supervises upload recovery
│   └── trace.py             the progress tree, measured not derived
├── service/                 the eight slots, each base.py + factory.py + implementations
│   ├── runtime.py           shared provider lifecycle boundary; not a slot
│   ├── intent/              question → Intent          INTENT_PARSER
│   ├── planner/             Intent → Plan              PLANNER
│   ├── executer/            Plan → ToolResult[]        EXECUTER
│   │   └── tools/           rag, graphrag, web, llm_direct
│   ├── rerank/              score, sort, cut to top_n  RERANK_PROVIDER
│   ├── evaluator/           enough context? refine     EVALUATOR
│   ├── answer/              context → answer frames    ANSWER
│   ├── parsers/             document → ParsedChunk[]   PARSER
│   └── vectorstore/         chunk → shape + write      CHUNKSTORE
├── providers/               model, embedding, web-search, and rerank adapters
│   ├── llm/  embedding/  websearch/  rerank/
├── database/                connections and persisted formats
│   ├── postgres/            engine, tables, repositories, durable upload queue/journal
│   ├── elasticsearch/       connection, hybrid search, chunk reads
│   ├── graph/               the three per-user files, backends, GraphRepository
│   └── session_context.py   attached-context JSON
├── utils/                   stateless helpers
├── models/  config/  exceptions/
└── vendor/                  RAGFlow, vendored parser and model files
```

### Where a change goes

| you are changing | it goes in |
|---|---|
| a new retrieval tool | `service/executer/tools/`, registered in `registry.py` |
| a different reranker or LLM | `providers/`, selected by its provider or slot factory |
| a new step in a turn | a slot in `service/`, called from `pipeline/query.py` |
| the order of the steps | `pipeline/query.py` only |
| what a stored chunk looks like | `service/vectorstore/` |
| how a chunk is fetched | `database/elasticsearch/` |
| a new endpoint | `api/routes/`, calling a pipeline |
| a new authorization rule | `api/deps.py`, `authorization_scope` |

### What the tests enforce

| test | asserts |
|---|---|
| `test/architecture/test_layering.py` | no upward or peer-slot imports; API database/search-client and pipeline/provider shortcuts are refused; only two named vendor shims reach first-party code |
| `test/architecture/test_slots.py` | every slot has a Protocol and a factory, the pipeline enters each through its root, `QueryPipeline()` builds all six through their factories, and orchestration state is constructed only in `pipeline/` |
| `make stale-imports` | forbidden legacy package paths do not survive in code, strings, or docs |
| `test/pipeline/` | the real routers over HTTP with deterministic slots, no stack required |

---

## The two rules

1. **Contract.** Trust-bearing state and primary slot boundaries use typed
   models; job and context models are frozen and reject extra fields. Explicit
   storage adapters remain around GraphRAG and ingest; the answer slot serializes
   typed chunks only at its SSE/PostgreSQL boundary.
2. **Protocol.** The component is reached through a `typing.Protocol` and
   constructed by a factory from config. Nothing upstream knows which
   implementation is behind it.

**A component's output does not depend on who consumes it.** No conditional
suppression: the stored chunk shape retains its analyzer fields, and current
dense retrieval always uses BM25 as its filter. A component whose behaviour
shifts based on its caller is not swappable.

---

## The eight slots

Each is `service/<slot>/` with a `base.py` Protocol and a `factory.py`.
`QueryPipeline` constructor-injects its six; ingestion resolves its parser and
chunk store through factories at use time. Attached-context extraction is a
documented exception that calls the dedicated `vlm_processor` directly.

| Slot | Protocol | Ships | Env |
|---|---|---|---|
| `intent/` | `IntentParser` | `llm`, `keywords` | `INTENT_PARSER` |
| `planner/` | `Planner` | `rule_based` | `PLANNER` |
| `executer/` | `Executer` | `concurrent` | `EXECUTER` |
| `rerank/` | `Reranker` | `dashscope` | `RERANK_PROVIDER` |
| `evaluator/` | `Evaluator` | `llm` | `EVALUATOR` |
| `answer/` | `AnswerGenerator` | `cited` (implemented by the chat module) | `ANSWER` |
| `parsers/` | `DocumentParser` | `deepdoc`, `vlm` | `PARSER` |
| `vectorstore/` | `ChunkStore` | `elasticsearch` | `CHUNKSTORE` |

Providers, which are not slots:

| Provider | Protocol | Ships | Env |
|---|---|---|---|
| `providers/llm/` | `LLM` | `dashscope`; `claude-cli` for local development/evaluation | `LLM_PROVIDER` |
| `providers/embedding/` | `Embedder` | `online`, `offline` | `EMBEDDING_PROVIDER` |
| `providers/websearch/` | `WebSearchProvider` | `serper`, `ddgs` | `WEB_SEARCH_PROVIDER` |
| `providers/rerank/` | the scoring call behind the rerank slot | `dashscope` | selected by the slot |

Configurable components inside a slot or store:

| Component | Lives in | Ships | Env |
|---|---|---|---|
| analyzer (text → search terms) | `service/vectorstore/elasticsearch/analyzer/` | `ragflow` | `ANALYZER` |
| retrieval tools | `service/executer/tools/` | `rag`, `graphrag`, `web_search_answer`, `direct_llm_answer` | registry, filtered by `ALLOWED_TOOLS` |

The analyzer factory and retrieval registry are active in production. Graph
extraction and the graph repository each have one live implementation and are
therefore not presented as configurable selectors.

### Two seams around web search, not one

`service/executer/tools/web/` exposes a `ToolResult` to the executor;
`providers/websearch/` hides whether DuckDuckGo or Serper performed the public
search.

The provider normalises field names at the boundary. Serper says
`title`/`link`/`snippet`, DuckDuckGo says `title`/`href`/`body`; both become
`WebResult(title, url, content)`, and `extra="forbid"` means a provider-specific
key raises instead of leaking upward.

Serper supplies related questions; DuckDuckGo returns an empty list for that
optional capability.

`ddgs` is the default because it needs no API key. `WEB_SEARCH_PROVIDER=serper`
switches back, and needs `SERPER_API_KEY`.

**Snippet ranking degrades rather than fails.** A provider returns approximately ten hits
(`WEB_SEARCH_RESULTS`) and the closest `WEB_SEARCH_TOP_K` are kept by embedding
similarity. When the embedder is unavailable that step is skipped and the
provider's ordering is used.

**Images and videos are web-tool output.** They travel on the authorized
`ToolResult`; the answer slot never calls a web provider. The frontend receives
`image_results.images[].thumbnailUrl` and `video_results.videos[].link`.

Media lookup is best-effort: failure yields empty media lists without discarding
text results.

### Not slots, and why

| Component | Reason |
|---|---|
| `service/executer/tools/registry.py` | the dispatcher, not a thing dispatched |
| Reflector | evaluation and reflection share one `Evaluation` and policy contract |
| Related-question generator, session namer | single-purpose answer helpers |
| Session context, conversation store, file storage | persistence concerns, not pipeline stages |
| Auth | a FastAPI dependency, not a pipeline stage |

---

### Local development/evaluation through the Claude CLI

The LLM is a provider rather than a slot. `LLM_PROVIDER=claude-cli` invokes the
locally authenticated `claude` binary for local development and evaluation; it
is not the production serving path.

It supplies LLM calls only. Embedding and reranking remain separately
configured; `EMBEDDING_PROVIDER=offline` provides local embeddings.

### Embedding providers

`online` uses the OpenAI SDK's embeddings API against configurable
`DASHSCOPE_BASE_URL`; the selected endpoint and model must implement that
compatible embeddings contract.
`offline` runs EmbeddingGemma-300m in-process, with no key and no network. The
`offline` extra (`cd backend && uv sync --locked --extra offline`) installs its
dependencies, but not the model weights. Provision the weights through
`EMBEDDING_MODEL_PATH` or at
`providers/embedding/models/embeddinggemma-300m/`.

They are **not interchangeable on an existing index**. `online` is configured
for 1024-wide vectors and `offline` produces 768, and Elasticsearch names the
field after the width — `q_1024_vec` against `q_768_vec` — so documents indexed
under one provider are invisible to queries made under the other. Switching
means re-indexing, and `EMBEDDING_DIM` (which `GRAPH_EMBEDDING_DIM` defaults to)
has to move with it.

The vendored hybrid search uses its BM25 query as the kNN filter, so dense
similarity re-scores BM25 matches rather than admitting independent vector-only
candidates. Final selection is performed by the rerank slot.

The CLI transport suppresses project/user instructions, memory, MCP servers,
and tools with environment flags, strict MCP configuration, disallowed tools,
and a temporary working directory. Because some controls are undocumented,
run `backend/scripts/check_claude_cli.py` after CLI upgrades.

`temperature` is accepted but ignored because the CLI has no equivalent. Each
call also pays process-startup cost, so this transport is intended for local
development rather than production serving.

## Ingestion

```
POST /start-processing
├─ shared pre-parser raw-body/concurrency fence → authenticate → parse and validate multipart files
├─ confirm the row and token version, mint IngestJob
└─ pipeline/ingest.py — plan ranges → create staging ticket → fsync staged files → queue/lease → worker

  parsers/                  PDF → chunks
  vectorstore/analyzer/     text → search terms
  providers/embedding/      text → vectors
  vectorstore/              chunks → Elasticsearch
  graphstore/service.py     chunks → entities and relations
  database/graph/           GraphML + graph-vector JSON
```

The same ASGI fence covers `/start-processing` and `/add_context`: it rejects an
oversized declared raw body without reading it, counts streamed bytes when
`Content-Length` is absent or false, and admits at most
`MAX_CONCURRENT_UPLOADS` bodies per API worker. Both routes authenticate before
parsing, accept only `files`, and enforce filename, count, per-file, and
aggregate-content limits while scanning request-owned spools. Knowledge names
are normalized to NFC and reject NUL, edge whitespace, and the reserved graph
separator; batch duplicates return 409 before staging.
The knowledge route completes an `IngestJob`; `pipeline/ingest.py` owns range
planning, tickets, durable bytes, dispatch, checkpoints, recovery, and metadata.

PostgreSQL is authoritative for ownership, status, cancellation, counters,
attempt budget, safe failure class, next-attempt time, leases, item checkpoints,
and the append-only progress journal. Before durable bytes, one transaction
takes sorted advisory locks for canonical `(user, document)` names, rejects an
existing document or active same-name job, then creates the non-dispatchable
`staging` ticket. Under a shared per-tenant staging lock, the pipeline rechecks
the ticket/account gate, fsyncs opaque files under
`STATE_DIR/upload_jobs/<process_id>/`, and changes the ticket to `queued`.
Every API worker supervises recovery. PostgreSQL serializes the live reservation
count across API replicas with the same node ID, enforces
`MAX_UPLOAD_WORKERS`, and increments `attempt_count` only when reserving a due
job. The child must activate that exact reservation and heartbeats its lease.
A child-start failure exhausts only a pristine all-pending ticket; if any
recovery checkpoint exists, the reservation is refunded and remains queued
until a worker can reconcile it.

The raw cap is `MAX_UPLOAD_TOTAL_BYTES` plus 1 MiB of framing. Defaults are two
admitted bodies, 10 files, 500 MiB per file, and 500 MiB aggregate content.
Large PDFs are split into page ranges covering every page. Each chunk ID is
SHA-256 over 8-byte-length-prefixed UTF-8 tenant, NFC document name,
`sequence:part_name`, zero-based parser ordinal, and content. Each item runs in
a spawned subprocess with one hard `PARSE_TIMEOUT_S` deadline across parse,
graph, embeddings, and Elasticsearch. Item, metadata, and compensation children
recheck the active lease owner under `tenant_lock` before mutation. A timeout
or ambiguous failure kills/reaps the item child, then uses a second bounded child to compensate the whole logical document
graph-first and Elasticsearch second. Only confirmed compensation may reset it
for `UPLOAD_RETRY_BACKOFF_S * 2^(attempt-1)` or terminal-fail at
`MAX_UPLOAD_ATTEMPTS`; a compensation failure leaves the lease and staging for
durable recovery. Cancellation during live item execution becomes terminal
after compensation regardless of budget. Lease expiry retains an ambiguous
`PROCESSING` checkpoint and backs off. Within budget, a non-cancelled replacement
compensates its logical document before reset and replay; cancellation instead
terminalizes after compensation. An exhausted lease gets one immediate
reconciliation incarnation before terminal failure. Already-finalized documents
remain authoritative; only partial documents with unfinalized effects are
compensated, and untouched future items become failed with zero counters. A
controlled requeue at a safe item boundary refunds its attempt. Metadata is
finalized immediately after a logical document's last sibling part and repeated
idempotently at recovery entry.

Durable staging survives API/worker restart. Terminal status is published only
after data is safe; staged bytes are then erased and `staging_cleaned` recorded.
`POST /cleanup-processes` retries erasure before deleting terminal rows. A
crash-left `staging` ticket expires to failed using PostgreSQL time. SSE resumes
from the journal ID, and account deletion requires staged-byte erasure before
removing the account.

**One tenant writer per user.** `tenant_lock(user_id)` is a cross-process file
lock. Each isolated item holds it across its lease/account recheck, parsing,
graph, and Elasticsearch work; isolated metadata finalization and compensation
take it separately and recheck the lease. Account, document, session, and attached-context mutations use the
same lock. Progress validates and renews the outer worker lease; cancellation
is polled during item execution and at checkpoints.
Account deletion takes the shared staging lock before closing the durable
admission gate and holds it through rollback or completion. While still holding
it, deletion cancels jobs and takes the tenant lock. It may then discard
in-flight checkpoints because it erases the complete tenant stores and staged
copies before deleting PostgreSQL last.
Document deletion canonicalizes the requested name, then rechecks the account
and resolves exactly one owned canonical row under the tenant lock; legacy NFD
spelling is carried unchanged through graph, Elasticsearch, and PostgreSQL
cleanup, while multiple legacy canonical matches fail with 409. Graph cleanup
removes that document's entries from each node/edge contribution ledger,
deletes empty aggregates, recomputes retained metadata, and re-embeds changed
vectors. Derived vectors carry exact `document_names_json` provenance. A
matching legacy aggregate or ambiguous vector without trustworthy provenance
fails closed and requires reindexing; deletion then retains the PostgreSQL row.
`docs/pipeline_offline_parsing.md` specifies deletion precisely.

## Query

```
POST /ai_search/
  ├─ authenticate: signature, then the users row still exists and its
  │  auth_version matches the token's
  ├─ verify session ownership; resolve policy (ALLOWED_TOOLS); mint QueryJob
  └─ pipeline/query.py — builds AgentState, derives ToolContext

  intent/     what is being asked
  planner/    which tools to run
  tools/      run them CONCURRENTLY, per-tool timeout, one failure confined
  rerank/     re-score and order
  evaluator/  enough? if not and rounds remain, reflect and repeat
  answer/     stream, with [doc][cite_XXX] / [web][cite_XXX]
```

The context is passed as an explicit keyword-only argument through the
pipeline, registry and every tool. It carries the same immutable identity values
across concurrent queries and reflection rounds; `QueryJob.context` constructs
a fresh value when accessed. No user, session, tenant, or authorization identity
is stored in module globals, singleton attributes, thread-locals or task-locals.
A non-authorizing request ID uses task-local context solely to correlate logs.

Query-time LLM, hosted-embedding, reranking, and public-web HTTP calls use
native async transports; GraphRAG awaits hosted query embeddings before its
synchronous graph/Elasticsearch work. `POST /add_context/` directly awaits the
VLM's AsyncOpenAI call. Cancelling a turn/request reaches those HTTP operations,
closes DashScope answer streams, and terminates/reaps a Claude CLI subprocess.
Explicit `asyncio.to_thread` bridges remain for PostgreSQL/history, graph
repository loading and file work, synchronous Elasticsearch, web HTML/JSON
parsing, the query LLM tool's session-context read, `POST /add_context/`
filesystem/JSON and VLM rendering, and offline sentence-transformers inference.
A cancelled caller abandons an already-running query-side thread result. Upload
items, metadata, and compensation run in killable spawned subprocesses; the
outer worker polls cancellation and reaps them. The retrieval loop is owned by the stream;
disconnect closes it, `TURN_TIMEOUT_S` bounds the turn, and `TOOL_TIMEOUT_S`
bounds each tool. The answer slot finalizes and persists the answer before
`[DONE]`; persistence failure produces an error frame instead.

No public web provider is reached unless web search is authorized. Media
ownership and best-effort behavior are defined under the web-search seams
above; the answer slot only forwards that tool output.

---

## Who owns the turn

The API authenticates, verifies session ownership, resolves what server policy
allows, validates the question, and mints one frozen `QueryJob`. That is the
whole trust boundary and it runs as FastAPI dependencies, so a bad request gets
a real 401/404/422 before any response header rather than a 200 carrying an SSE
error frame.

The job holds verified ids, authorized capabilities, and policy metadata, but
no bearer token, decoded claims, or password. Its run id correlates logs.

The pipeline builds `AgentState` from the job and is its only writer. Identity
reaches a tool as a `ToolContext` derived from the job, so it cannot drift
mid-turn.

Web search is a `WebSearchMode`, not a bool, and the job keeps both requested
and effective options so user preference remains distinct from policy denial.
DISABLED strips web search from a plan even when the intent
analyser asked for it. AUTO lets the system decide on both paths: the planner
on the knowledge path, and on the casual path the classifier's third answer,
`casual_web`, which runs a search before a simple question whose answer is
current. The planner is offered only tools that are both installed and
authorized, and `ToolRegistry.run` refuses anything outside the scope with a
`POLICY_DENIED` result and never invokes it. `allowed_tools` is a required
argument at every executer layer; there is no "not enforced" default.

Policy comes from the server. `ALLOWED_TOOLS` (default: all four) is parsed
into `ToolName` when `api/deps.py` imports and refuses to start on an unknown
name, so a misspelt deployment value cannot become "no tools" or "all tools" at
request time.

### Tokens, existence and revocation

`JWT_SECRET_KEY` is required (32+ characters). Without it the process refuses
to start -- unless `APP_ENV` is `development`, where a secret is generated once
and kept at `<STATE_DIR>/jwt_secret.dev` (by default under `backend/var/`,
gitignored, mode 0600) so a code reload does not sign everyone out, or `test`,
where it is random per process.
`make dev` and `make serve` set `APP_ENV=development` and reload on
`backend/src` only; a deployment sets the secret and leaves `APP_ENV` unset.
A token that does not verify -- stale secret, forged -- is a 401, not a 500.

A token carries the user's `auth_version`. On every route that requires
authentication, the users row is read and the token is refused if the row is
gone or the version has moved.
Changing the password bumps the version in the same commit and invalidates all
previously issued tokens; deleting the account removes the row, so its tokens
stop working immediately rather than at expiry.

Every HTTP response carries a validated or generated `X-Request-ID`. A pure
ASGI middleware keeps that value bound through streaming completion, and the
central logging formatter adds it to JSON records from first-party
`visionagent.*` loggers. Named event fields and `run_id` remain structured
dimensions. Their exception output keeps its type and source frames but omits
the raw message; routes expose stable public errors.
The request ID is ambient observability metadata only and never participates in
authentication, tenant selection, or tool authorization.

## Where state lives

By default, local runtime state is rooted at `backend/var/` (`STATE_DIR`).
`STORAGE_DIR` and `GRAPH_DIR` independently override the two child locations:

| Path | Holds |
|---|---|
| `STORAGE_DIR` (default `backend/var/uploads/`) | attached session files and session-context JSON |
| `GRAPH_DIR` (default `backend/var/graph/`) | `graph_*.graphml`, `vdb_*.json`, and `.lock_<user_id>` tenant locks |
| `STATE_DIR/upload_jobs/` | opaque, recoverable upload work items and per-tenant staging locks; bytes require a prior owner-bearing PostgreSQL ticket |

PostgreSQL stores users, sessions, messages, document metadata, and the durable
`upload_jobs` / `upload_job_items` / `upload_job_events` control plane;
Elasticsearch stores indexed chunks. Session-context raw files are atomically
published and JSON writes are atomic under the same per-tenant lock as
session/account deletion, with account liveness and session ownership rechecked
inside the lock. Single-file removal validates the entire persisted manifest,
unlinks only a direct child of the owned raw directory, fsyncs it, then updates
JSON; the old JSON remains a retry handle after an I/O failure. Whole-session
clear ignores manifest targets, safely erases the fixed owned raw directory
first (including crash orphans), fsyncs it, then removes JSON. Session and
account deletion use the same whole-directory privacy cleanup.

## Things that will bite you

**The Elasticsearch field names are the on-disk contract.** `RetrievedChunk.content`
is `content_with_weight` in the index. That rename must stop at
`service/vectorstore/elasticsearch/store.py`; if it reaches the index, every existing document
silently stops matching. Pinned by
`backend/tests/unit/test_golden.py::test_es_document_field_names_are_unchanged`.

**Index-time and query-time analysis must remain compatible**, or BM25 loses
matches without an error. Indexing uses the configured `Analyzer`; the vendored
query path uses `FulltextQueryer`/`rag_tokenizer`. Changing either side requires
a compatibility regression and may require reindexing.

**Embedding changes require coordinated reindexing.** The provider and
Elasticsearch width contract is specified under **Embedding providers** above.
Graph vectors are separate `vdb_*.json` files;
`backend/scripts/migrate_graph_embeddings.py` re-embeds them from stored source
text without LLM calls. Use its normal mode for a width change and `--force`
when changing provider or model without changing vector width. Elasticsearch
documents must be reindexed for either kind of embedding change.

**Parsing may be nondeterministic under load.** Treat chunk boundaries and
layout-derived metadata as parser output that must be regression-tested.

**Vendored RAGFlow resolves its assets from `__file__`.** Moving its path-bearing
directories or assets without preserving their relative layout breaks model
loading with a missing-file error at first parse, not an import error.

**The root `.gitattributes` routes the bundled opaque model assets through Git
LFS.** The broader metadata distributed with the HuggingFace download remains
ignored. A clean checkout needs `git lfs pull` before parsing or golden replay.

---

## Adding an implementation

```python
# backend/src/visionagent/providers/rerank/cohere.py
class CohereReranker:
    name = "cohere"

    async def score(self, *, query: str, texts: list[str]) -> list[float]: ...
```

```python
# backend/src/visionagent/service/rerank/factory.py
if chosen == "cohere":
    from visionagent.providers.rerank.cohere import CohereReranker
    return CohereReranker(**kwargs)
```

Then `RERANK_PROVIDER=cohere`. Write the tests to the same bar as the rest:
state the intended function, assert the **exact** expected output for given
input with the provider stubbed, and prove a fake can be substituted.

---

## Account management

| Endpoint | |
|---|---|
| `GET /me` | user id and username |
| `POST /me/password` | requires the **current** password, not just a session |
| `DELETE /me` | requires the password **and** `confirm: "DELETE"` |

`api/deps.py` verifies the password and confirmation, then issues a
`DeleteAccountCommand` without the credential. `pipeline/account.py` removes
the Elasticsearch index, graph files, attached-context files, messages,
sessions, document rows, and finally the user row. External stores are cleared
before the PostgreSQL account so failures can be retried. The cleanup
operations tolerate missing storage, but a completed deletion invalidates the
token, so a second HTTP call is unauthorized rather than a zero-result success.

Sessions are created and deleted through `pipeline/session.py`; the routes
never run SQL. `api/` reads PostgreSQL through typed repository methods, and
`test/architecture` fails on any SQLAlchemy, table, engine, raw-SQL or
Elasticsearch name inside it.
Session deletion holds the tenant lock, safely erases the fixed raw attachment
directory first, removes context JSON second, and deletes the owned PostgreSQL
session row last. A filesystem failure preserves both JSON and the row for retry.

## Not implemented

Declared interfaces and settings whose behavior is not implemented:

| Feature | State |
|---|---|
| **Deep Search / `/deep_research/`** | `api/routes/ai_search_mcp_rt.py` is an inert reserved module and is not mounted; the frontend control is hidden while its dormant state and `/ai_search/` request plumbing remain. |
| **`ChatRequest.deep_research`** | recorded in `QueryJob.requested`, forced to `False` in `effective`, and has no runtime branch |


---

## Known dead code

Accepted interfaces with no production caller:

| what | why it is dead |
|---|---|
| `ToolRegistry.describe()`, `.has()` | defined but not called by production code |
| `ChatRequest.chat_id`, `.attachments` | accepted by the request schema but not copied into `QueryJob` or read by the turn |
