# Traceable RAG

Traceable RAG is a full-stack, source-grounded knowledge search and chat
system. Replaceable pipeline slots make the architecture explicit, typed
progress events expose each turn as it runs, and citations connect generated
answers to retrieved documents and web sources.

## Architecture

The API layer authenticates and authorizes requests, then issues immutable job
contracts. Pipelines construct and own per-turn `AgentState`; services receive
narrow typed arguments or `ToolContext`, never request-scoped module globals.
The eight replaceable slots are intent, planner, executer, rerank, evaluator,
answer, parsers, and vectorstore. Each slot exposes a `base.py` Protocol and a
`factory.py` selected by configuration.

The primary dependency direction is
`api/ → pipeline/ → service/<slot>/ → database/ → providers/ → utils/`.
The API may perform typed repository reads; graph routes make tenant-checked
file reads, and `add_context` stores authenticated session attachments. The
answer slot persists completed messages. `database/` may use embedding
providers because both search stores embed their own queries.
`test/architecture` enforces the supported boundaries.

## 🎯 Overview

Traceable RAG enables users to:
- Upload PDF and DOCX knowledge-base documents for parsing and indexing
- Search tenant-scoped text and graph knowledge bases
- Add PDF or image files to a session for direct LLM context
- Combine authorized knowledge-base, graph, web, and session sources in chat
- Inspect document chunks and visualize knowledge graphs in 3D

**Core Architecture**: Professional turns run the orchestrator loop detailed
below. Casual turns bypass it; when web is authorized, a `casual_web` turn or a
forced-web casual turn makes one web-tool call. The RAG, GraphRAG, web-search,
and LLM tools execute selected calls.

## ✨ Key Features

### Orchestrator Agent
The query pipeline coordinates one immutable `QueryJob` and one mutable
`AgentState` per turn. For professional turns:
- **Intent Recognition**: Classifies `casual`, `casual_web`, or `professional` and identifies context types
- **Context Selection**: Resolves knowledge-base, graph, web, and session sources
- **Tool Selection**: Applies the job's allowlist and web-search policy
- **Action Plan Generation**: Uses the configured planner; the default is deterministic and rule-based
- **Tool Execution**: Runs independent calls concurrently with per-tool timeouts
- **Result Gathering**: Normalizes and reranks retrieved chunks
- **Context Evaluation**: May request another retrieval round, up to `MAX_ROUNDS`
- **Answer Generation**: Streams the answer; the answer slot persists the completed turn

### Sub-Agents (Implemented as Tools)
Specialized agents that execute specific retrieval and processing tasks:
- **RAG Agent** (`RAG` tool): Retrieves tenant-scoped chunks with vector and keyword search
- **GraphRAG Agent** (`GraphRAG` tool): Resolves graph entities and relationships back to source chunks
- **Web Search Agent** (`web_search` tool): Retrieves current information from the internet
- **LLM Agent** (`LLM` tool): Produces a direct model response, including cumulative session context when present

### Context Sources
Each source has a distinct persisted or provider-owned representation; the
canonical tool mapping appears in the table below:
- **User Knowledge Base**: Tenant-indexed Elasticsearch chunks
- **User Graph Knowledge Base**: Three tenant-scoped files under `GRAPH_DIR`
- **Web Content**: Normalized provider results with title, URL, and snippet text
- **Cumulative Session Attachments**: Extracted text appended to session-context JSON
- **Previous User Questions**: PostgreSQL question text; assistant answers are not prompt history

### Document Processing
- **Parsing**: DeepDoc supports PDF and DOCX; the VLM parser supports PDF, PNG, JPG, and JPEG
- **Large PDFs**: Estimated documents above 250,000 characters are split by page range
- **VLM bounds**: Renders and transcribes one source page at a time under one document deadline, page-count, pixel, source-image, and encoded-page limits
- **Images**: Extracted references remain encoded strings through storage and retrieval
- **Dual Indexing**: Builds graph data first, then indexes chunks in Elasticsearch
- **Progress**: Upload status and processing stages stream over SSE

### Knowledge Graph Construction (GraphRAG tool/Agent)
Construction runs during ingestion through the graph/vectorstore stage; the
GraphRAG retrieval tool only reads the resulting stores.
- **Entity Extraction**: Extracts typed entities from document chunks
- **Relationship Extraction**: Stores weighted connections with source chunk IDs
- **Contribution Ledger**: Makes retries idempotent and lets deletion recompute shared node and edge aggregates without removing other documents
- **Tenant Isolation**: Maintains one graph and two graph-vector files per user
- **Visualization**: Serves caller-owned GraphML to the 3D graph view

### User Experience
- **Repository Management**: Upload, list, inspect, and delete knowledge-base files
- **Chunk Display**: Shows chunks ordered by page and vertical position
- **Real-time Progress**: Shows typed workflow steps and evidence, with a legacy text fallback
- **Session Management**: Supports caller-owned sessions, messages, and attachments

## 🏗️ Architecture

### Agent System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                       │
│  - Repository Page (File Management)                          │
│  - Chat Interface (Agent Interaction)                       │
│  - Graph Visualization (3D Force Graph)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/SSE
┌──────────────────────────▼──────────────────────────────────┐
│                   Backend (FastAPI)                         │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │          Orchestrator Agent                          │    │
│  │  ┌──────────────────────────────────────────┐     │    │
│  │  │  1. Intent Recognition                     │     │    │
│  │  │  2. Context Selection                     │     │    │
│  │  │  3. Tool Selection                         │     │    │
│  │  │  4. Action Plan Generation                 │     │    │
│  │  │  5. Tool Execution                         │     │    │
│  │  │  6. Result Gathering & Reranking          │     │    │
│  │  │  7. Context Evaluation                    │     │    │
│  │  │  8. Final Answer Synthesis                 │     │    │
│  │  └──────────────────────────────────────────┘     │    │
│  └────────────────────────────────────────────────────┘    │
│                           │                                 │
│  ┌────────────────────────┴──────────────────────────┐    │
│  │              Tools (Sub-Agents)                    │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐        │    │
│  │  │   RAG    │  │ GraphRAG │  │   Web    │        │    │
│  │  │  (RAG)   │  │(GraphRAG)│  │  Search   │        │    │
│  │  │          │  │          │  │(web_search│        │    │
│  │  └──────────┘  └──────────┘  └──────────┘        │    │
│  │  ┌──────────┐                                    │    │
│  │  │   LLM    │                                    │    │
│  │  │  (LLM)   │                                    │    │
│  │  └──────────┘                                    │    │
│  └───────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │       File Upload & Processing                      │    │
│  │  (Builds Knowledge Bases for Tools)                 │    │
│  └────────────────────────────────────────────────────┘    │
└────────────┬──────────────┬──────────────┬─────────────────┘
             │              │              │
    ┌────────┴────────┐     │     ┌────────┴────────┐
    │  PostgreSQL     │     │     │ Elasticsearch    │
    │  - Users        │     │     │ - Chunks (RAG)   │
    │  - Sessions     │     │     │ - Embeddings     │
    │  - Messages     │     │     │ - Metadata       │
    │  - Knowledge    │     │     └──────────────────┘
    │    Base Info    │     │
    │  - Upload Jobs  │     │
    └─────────────────┘     │
                            │
                 ┌──────────┴──────────┐
                 │   GraphRAG Storage   │
                 │  - node_vdb         │
                 │  - edge_vdb         │
                 │  - NetworkX Graph   │
                 └─────────────────────┘
```

### Context-Tool Mapping

| Context Source | Tool (Sub-Agent) | Purpose |
|----------------|------------------|---------|
| User Knowledge Base | `RAG` tool (RAG Agent) | Retrieve tenant-scoped chunks with vector and keyword search |
| User Graph Knowledge Base | `GraphRAG` tool (GraphRAG Agent) | Resolve matching entities and relationships to source chunks |
| Web Content | `web_search` tool (Web Search Agent) | Retrieve current information from internet |
| Cumulative Session Attachments | `LLM` tool (LLM Agent) | Add extracted attachment content to a direct model response |
| Previous User Questions | Prompt context | Prior questions from the same session, not the full transcript |

### Technology Stack

#### Backend
- **FastAPI**: HTTP API and SSE streaming
- **Python 3.11+**: Backend runtime
- **Elasticsearch**: Document storage and hybrid search
- **PostgreSQL**: Users, sessions, messages, document metadata, and durable upload jobs
- **LightRAG-compatible local primitives**: nanoVectorDB-style vectors and NetworkX graphs
- **DeepDoc**: PDF and DOCX parsing with OCR
- **Uvicorn**: ASGI server

#### Frontend
- **React 18**: UI framework
- **TypeScript**: Type-safe JavaScript
- **Vite 8**: Build and development server
- **Delivery**: Route pages are lazy-loaded; `npm run build` enforces ceilings of 1,000,000 initial JavaScript bytes and 500,000 bytes per emitted chunk
- **Ant Design**: UI component library
- **React Router**: Client-side routing
- **Valtio**: State management
- **ahooks**: React hooks library
- **Force Graph 3D**: 3D graph visualization

### LLM Models & APIs

#### AI Provider: DashScope (Alibaba Cloud)
DashScope's OpenAI-compatible API is the production provider. A separately
documented `claude-cli` adapter is retained for local development and evaluation;
embedding, reranking, and web providers are selected independently.

**Models Used:**
- **`CHAT_MODEL`**: Intent, evaluation, reflection, and cited final answers
- **`CHAT_MODEL_TURBO`**: Casual answers, session naming, and graph extraction
- **`EMBEDDING_MODEL`**: Online query, chunk, entity, and relationship embeddings; offline embeddings use `EMBEDDING_MODEL_PATH`
- **`RERANK_MODEL`**: Relevance scoring after retrieval

**Provider Selection:**
```env
LLM_PROVIDER=dashscope
EMBEDDING_PROVIDER=online
RERANK_PROVIDER=dashscope
WEB_SEARCH_PROVIDER=ddgs
```

**Web Search Provider** (`WEB_SEARCH_PROVIDER`):
- **DuckDuckGo** (`ddgs`, the default): no API key
- **Serper API** (`serper`): needs a key
  ```env
  WEB_SEARCH_PROVIDER=serper
  SERPER_API_KEY=your_serper_api_key
  ```

**Model Usage by Agent:**
- **Orchestrator Agent**: Uses `CHAT_MODEL` for LLM decisions; the default planner maps intents to tools without an LLM
- **RAG tool** (RAG Agent): Uses the embedding provider for hybrid retrieval; the pipeline applies the reranker
- **GraphRAG tool** (GraphRAG Agent): Searches stored entity and relationship embeddings, then loads source chunks from Elasticsearch
- **Web Search tool** (Web Search Agent): Uses DuckDuckGo by default or Serper when configured
- **LLM tool** (LLM Agent): Uses `CHAT_MODEL` for a direct response, including extracted session attachments when present
- **Final Answer Generation**: Uses `CHAT_MODEL` for cited answers and `CHAT_MODEL_TURBO` for the casual path

## 📁 Project Structure

```
traceable-rag/
├── backend/src/visionagent/
│   ├── api/                     HTTP boundary, job issuance, SSE, and scoped attachment/graph file I/O
│   ├── pipeline/                query, ingest, context, account, session, startup, trace
│   ├── service/                 the eight slots: intent, planner, executer(+tools), rerank,
│   │                            evaluator, answer, parsers, vectorstore
│   │   └── runtime.py           shared provider lifecycle boundary; not a slot
│   ├── database/                postgres/, elasticsearch/, graph/ (+ tenant_lock), session_context.py
│   ├── providers/               llm, embedding, websearch, rerank clients
│   ├── utils/  models/  config/  exceptions/
│   └── vendor/ragflow/          vendored deepdoc + search engine
├── backend/tests/               unit/, golden/
├── frontend/src/
│   ├── pages/                   index, chat, repository, graph, auth
│   ├── components/  api/  router/  store/  layout/  utils/
├── frontend/tests/unit/
└── test/                        architecture, pipeline (HTTP, no stack), backend, frontend, journey (live)
```

## 🗄️ Database Architecture

### Five Persistence Targets

The system uses two database services and three per-user graph files.

1. **PostgreSQL**
   - **Purpose**: Users, sessions, messages, document metadata, and durable upload execution state
   - **Tables**: `users`, `sessions`, `messages`, `knowledgebases`, `upload_jobs`, `upload_job_items`, `upload_job_events`
   - **Access**: SQLAlchemy with the PostgreSQL driver

2. **Elasticsearch**
   - **Purpose**: Document chunks with embeddings for hybrid search
   - **Index per user**: `{user_id}`
   - **Fields**: `content_with_weight`, `q_<dim>_vec`, `docnm`, `page_num`, `ref_images`
   - **Operations**: Tenant-scoped bulk indexing, vector search, keyword search, and deletion

3. **Node Vector Store** (GraphRAG)
   - **Purpose**: Entity embeddings and metadata
   - **Format**: JSON file `vdb_nodes_{user_id}.json`
   - **Content**: Entity vectors for semantic similarity search

4. **Edge Vector Store** (GraphRAG)
   - **Purpose**: Relationship embeddings and metadata
   - **Format**: JSON file `vdb_edges_{user_id}.json`
   - **Content**: Relationship vectors for semantic similarity search

5. **NetworkX Graph** (GraphRAG)
   - **Purpose**: Knowledge graph structure
   - **Format**: GraphML file `graph_{user_id}.graphml`
   - **Content**: Nodes (entities) and edges (relationships) with metadata

## 🔄 Workflows

### 1. File Processing Workflow (Knowledge Base Building)

Each upload plans page ranges over request-scoped spools, then atomically
inserts an owner-bearing, non-dispatchable PostgreSQL `staging` ticket, item
checkpoints, and ordered progress journal after checking the user admission
gate under a database row lock. The shared upload/context ASGI fence rejects an
oversized declared body without reading it, counts streamed bodies during
parsing, and caps concurrent admissions. Authentication precedes multipart
parsing; the route accepts only `files` and enforces filenames, count,
per-file, and aggregate limits without joining streams in memory. Canonical NFC
document names are immutable: duplicates within a batch, an existing tenant
document, or a concurrent active upload return 409. Before any durable
application staging byte exists, the pipeline acquires a shared
per-tenant staging lock, rechecks that ticket and the account gate, writes and
fsyncs opaque work items under durable `STATE_DIR`, then atomically moves the
ticket to `queued` only if the account still accepts work. PostgreSQL
serializes the `MAX_UPLOAD_WORKERS` capacity decision across API processes on
one runtime node. A reserved worker activates only its own lease, then runs
each mutable item in a spawned subprocess that rechecks the lease owner and
account under `tenant_lock`; the parent kills and joins that subprocess on its
hard deadline, cancellation, shutdown, or lease loss. A second bounded,
lease-fenced child compensates ambiguous effects graph-first and Elasticsearch
second; only confirmed cleanup permits reset, retry, or terminal failure.
Attempts and safe failure classes are durable; retry uses
exponential backoff and terminates at `MAX_UPLOAD_ATTEMPTS`, while failed
compensation remains nonterminal for lease recovery. Lease expiry retains an
ambiguous `PROCESSING` checkpoint. Within budget, a non-cancelled replacement
compensates that logical document before reset and replay; cancellation instead
terminalizes after compensation, and an exhausted lease runs reconciliation
without replay. Metadata is separately isolated, bounded, and lease-fenced;
it is finalized immediately after each logical document's last sibling part
and rechecked at recovery entry. Any API worker can serve progress or recover a
lease. For cross-host recovery, all hosts must share PostgreSQL, `STATE_DIR`,
and any independently overridden `GRAPH_DIR`; cross-host context access also
requires shared `STORAGE_DIR`. Processes on one node must share
`VISIONAGENT_NODE_ID`.

The Step 2 detail below describes DeepDoc. VLM applies page, pixel,
encoded-size, and document-deadline bounds, retains only one rendered page at a
time, and emits at most one typed chunk per source page with its original page
number; blank responses are omitted.

```
Step 1: File Upload & Verification
├── User uploads files via /start-processing endpoint
├── Bound raw body bytes and concurrent admissions before multipart parsing
├── Authenticate, then enforce file-count, per-file, and aggregate content limits
├── Normalize logical names; reject batch, stored, or active canonical duplicates
├── Plan page-range splitting for large PDFs (>250K estimated characters)
├── Commit an owner-bearing, non-dispatchable PostgreSQL staging ticket
├── Under the tenant staging lock, recheck account/ticket and fsync opaque work items
├── Atomically move staging → queued only while the account accepts work
└── Enforce the node worker cap, reserve/increment an attempt, then start the child

Step 2: Document Parsing & Chunking
├── PDF processing with OCR and layout recognition
├── Text extraction and table detection
├── Merge sections using 32- and 1024-token thresholds
├── Group sections by semantic structure
└── Assign SHA-256 IDs from length-prefixed tenant + NFC document + sequence:part + ordinal + content

Step 3: GraphRAG (first, inside the isolated item's tenant_lock)
├── Extract entities and relationships using LLM
├── Merge with a per-source contribution ledger; carry exact document-name provenance into vectors
├── Keep aggregate/vector replay idempotent; re-extraction still calls the LLM
├── Propagate embedding failures without zero-vector rows
├── Persist in 3 tenant-scoped graph structures:
│   ├── node_vdb (entity embeddings)
│   ├── edge_vdb (relationship embeddings)
│   └── knowledge_graph (NetworkX structure)
└── Progress over SSE; a pre-publication graph failure is recorded and chunks are still indexed

Step 4: NormalRAG indexing (second)
├── Receive the normalized chunks from Step 2
├── Generate embeddings and index through the vectorstore slot
├── Write prepared content and image fields through the vectorstore slot
├── Store in Elasticsearch; fewer accepted than prepared is an error
└── Send progress updates via SSE to frontend

Step 5: Bounded PostgreSQL metadata finalization per logical document
├── Upsert immediately after that document's last sibling part becomes terminal
├── Repeat idempotently when a recovery worker starts
└── Set `error` when any stage for that document did not complete

Progress and cancellation are also PostgreSQL-backed. SSE streams the
append-only event journal by event id; a worker/API restart resumes unfinished
items from durable staging after the recorded backoff. Crash-left `staging`
tickets expire to `failed` and are erased before cleanup is acknowledged.
Terminal staged bytes are removed immediately when possible; owner cleanup
removes retained metadata only after erasure succeeds.
```

### 2. Agent Orchestrator Workflow

**Orchestrator Agent Steps:**
1. **Session and User Validation**: Authenticate user (token, and the users row still exists with a matching `auth_version`), validate session ownership, mint one immutable `QueryJob` — all as FastAPI dependencies, so a failure is a real 401/404/422
2. **Chat Scenario Recognition**: Detect "casual", "casual_web" or "professional" query type
3. **Workflow Path Selection**: Route to appropriate workflow based on scenario
4. **Query Intent Recognition** (professional only): Identify required context sources
5. **Action Plan Generation**: Map context types to authorized tools
6. **Tool Execution (1st Round)**: Execute independent calls concurrently
7. **Result Gathering & Reranking**: Normalize and rank tool results
8. **Context Evaluation**: Score whether the context is sufficient
9. **Additional Retrieval**: Repeat only when reflection yields refined queries and rounds remain
10. **Final Answer Synthesis**: Answer from gathered context and prior user questions; persist the turn before `[DONE]`

Steps 4–9 form the bounded loop in `pipeline/query.py`. `MAX_ROUNDS` defaults
to 2, `TOOL_TIMEOUT_S` bounds each tool, and `TURN_TIMEOUT_S` bounds the turn.

**Multi-Agent Execution Flow:**
```
User Query
    ↓
Orchestrator Agent: Intent Recognition
    ↓
Orchestrator Agent: Context Selection
    ↓
Orchestrator Agent: Tool Selection
    ├── "kb(filter)" → RAG tool (RAG Agent) → Elasticsearch
    ├── "kb(filter)" → GraphRAG tool (GraphRAG Agent) → Knowledge Graph + Elasticsearch
    ├── "web_search" → Web Search tool (Web Search Agent) → Internet
    └── "session_context" → LLM tool (LLM Agent) → Direct LLM
    ↓
Tools (Sub-Agents): Execute Tasks & Return Results
    ↓
Orchestrator Agent: Result Gathering & Reranking
    ↓
Orchestrator Agent: Context Evaluation
    ↓
Orchestrator Agent: Final Answer Generation
```

### 3. GraphRAG Tool Workflow

The GraphRAG tool reads the caller's graph stores and resolves matching graph
records to chunks in the caller's Elasticsearch index.

**GraphRAG Tool Steps:**
1. Query processing and validation
2. Keyword extraction (high-level vs low-level)
3. Hybrid mode search:
   - Entity-vector search over stored name-and-description records in `node_vdb`
   - Relationship-vector search over stored endpoint, keyword, and description records in `edge_vdb`
4. Chunk ID extraction from graph results (`source_id` field)
5. Elasticsearch text retrieval using chunk IDs (`es.mget()`)
6. Return text chunks to orchestrator agent

**Note**: The GraphRAG tool works as a sub-agent that:
- Derives every graph and Elasticsearch read from `ToolContext.user_id`
- Calls only reader methods, although `GraphRepository` is a write-capable type
- Returns an empty successful result only for an empty query/keywords or a
  genuine no-match/not-found result
- Returns an error `ToolResult` for graph, embedding, Elasticsearch, or stored-data failures

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.11** — `backend/pyproject.toml` has `requires-python = ">=3.11"`
  and the production image pins Python 3.11.16
- **uv** — installs the exact backend dependency set recorded in `backend/uv.lock`
- **Node.js `^22.12.0 || ^24.0.0 || >=26.0.0`** — matches the checked-in frontend engine contract
- **Git LFS** — retrieves the bundled DeepDoc ONNX and model assets
- **Docker** — the root Compose file provides PostgreSQL 15 and Elasticsearch 8.11.3

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/jerryandjerry/traceable-rag.git
cd traceable-rag
git lfs pull
```

2. **Create the virtual environment, install, fetch the NLTK corpora**
```bash
make install      # sync backend/.venv from backend/uv.lock, install NLTK data,
                  # then install frontend packages from package-lock.json
```
Backend dependencies are declared in `backend/pyproject.toml`. The `offline`
extra installs the local embedding dependencies, but not the model weights.
Provision the weights through `EMBEDDING_MODEL_PATH` or at
`backend/src/visionagent/providers/embedding/models/embeddinggemma-300m/`.

3. **Or by hand**
```bash
cd backend
uv sync --locked --extra dev --python 3.11
cd ..
```

4. **Download the NLTK corpora** (skip if `make install` completed)
```bash
backend/.venv/bin/python backend/scripts/setup_nltk.py
```
The script installs the tokenizer and WordNet data required by hybrid search.

5. **Environment variables**
Copy `backend/.env.example` to `backend/.env`, then set provider credentials and
models. Process environment variables override the file.

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/visionagent
ES_URL=http://localhost:9200
ES_PORT=9200
ELASTIC_PASSWORD=

DASHSCOPE_API_KEY=<your key>
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CHAT_MODEL=qwen-max
CHAT_MODEL_TURBO=qwen-turbo
EMBEDDING_MODEL=text-embedding-v3
RERANK_MODEL=gte-rerank
EMBEDDING_DIM=1024

SERPER_API_KEY=
```

`DASHSCOPE_BASE_URL`, both chat models, the embedding model and dimension, and
the rerank model are required at startup. `DASHSCOPE_API_KEY` is required when
the DashScope LLM/reranker, online embedder, or dedicated VLM client is called.
Production also requires a
`JWT_SECRET_KEY` of at least 32 characters. Development mode creates and reuses
`<STATE_DIR>/jwt_secret.dev` (default `backend/var/jwt_secret.dev`) when no
strong key is supplied.

Important optional controls include the tool allowlist, storage directories,
upload and question limits, retrieval sizes, round and timeout budgets, graph
search limits, and active component selectors. Defaults are defined in
`backend/src/visionagent/config/settings.py`; selected defaults are summarized
in `backend/.env.example`.
`ALLOWED_TOOLS` rejects unknown tool names, and wired implementation factories
raise when constructed with an unknown name. Graph extraction and graph storage
have one supported implementation each and are not exposed as selectors.

Changing the embedding provider, model, or width requires reindexing
Elasticsearch. Re-embed graph `vdb_*.json` files with
`backend/scripts/migrate_graph_embeddings.py`: normal mode for a width change
and `--force` for a same-width provider/model change.

When a container-oriented environment uses `host.docker.internal`, override it
for a native API process:
```bash
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/visionagent"
```

6. **Start Postgres and Elasticsearch**
```bash
docker compose up -d es01 postgres
```
The root Compose file starts PostgreSQL on `:5432` and Elasticsearch on
`:9200`, and applies `backend/init.sql` when PostgreSQL initializes.

For an existing database, apply the schema directly:
```bash
psql -h localhost -U postgres -d visionagent -f backend/init.sql
```

7. **Start it**
```bash
make dev          # API on :8000 and the frontend on :5181, ctrl-c stops both
```
`make serve` runs the API alone, `make web` the frontend alone.

Run Make targets from the repository root. `make dev` and `make serve` read
`backend/.env`; an exported `DATABASE_URL` takes precedence, and host runs
rewrite `host.docker.internal` to `localhost`. `make web` only starts the frontend.

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install dependencies**
```bash
npm ci
```

3. **Environment variables**
Create the ignored local file from its checked-in template:
```bash
cp .env.example .env
```
`frontend/.env`:
```env
VITE_API_BASE = /ai-search
VITE_API_PROXY = http://localhost:8000/
```
`VITE_API_BASE` is the path prefix the axios client sends to, and the Vite dev
proxy is keyed on that same value with `VITE_API_PROXY` as its target — it strips
the prefix before forwarding. Both names are required; there is no
`VITE_API_BASE_URL`.

4. **Start development server**
```bash
npm run dev
```

The frontend will be available at `http://localhost:5181` (set in
`vite.config.ts`).

## 📡 API Endpoints

`GET /openapi.json` exposes 26 application operations:

### Authentication
- `POST /register` - User registration
- `POST /login` - User login; returns a bearer token bound to the current `auth_version`
- `POST /logout` - Stateless acknowledgment; the client removes its token
- `GET /me` - Current user
- `POST /me/password` - Change password and revoke previously issued bearer tokens
- `DELETE /me` - Take the staging coordinator lock; close upload admission; cancel uploads; take the tenant lock; discard recovery checkpoints because the complete tenant stores are erased next; require staging erasure; then delete PostgreSQL data and the account last. Requires the password and `confirm: "DELETE"`; returns 409 if either lock is busy

### Session Management
- `POST /create_session/` - Create a new chat session
- `GET /get_sessions/` - List the caller's sessions
- `GET /get_messages/?session_id={id}` - Full history for one session
- `DELETE /delete_session/{session_id}` - Delete an owned session under its tenant lock: erase the fixed raw attachment directory first, remove context JSON second, and delete the PostgreSQL session/messages last; returns 409 while the lock is busy

  > `GET /sessions/` and `GET /sessions/{session_id}/` are not registered.
  > Use `/get_sessions/` and `/get_messages/`.

### File Management
- `POST /start-processing` - Upload and process files
  - Multipart form data with only the `files` field
  - Up to 10 files; each file and their combined content are capped at 500 MiB by default
  - Raw bodies are capped at the aggregate limit plus 1 MiB of multipart framing; each API worker admits two upload bodies by default and returns 429 with `Retry-After` when full
  - Names are normalized to NFC and must be 1-255 characters, NUL-free, free of leading/trailing whitespace, and free of the reserved graph separator ` | `; canonical duplicates in the batch, existing documents, and concurrent active same-name jobs return 409. Replacement requires deletion first
  - PostgreSQL admits at most two live ingest reservations per runtime node by default; item retries and pristine startup failures use at most three worker incarnations with exponential backoff, while a recovery-bearing spawn failure refunds its reservation and remains queued
  - Returns `process_id` for progress tracking
- `GET /process-status/{process_id}` - Current status, counters, attempt budget,
  safe last-failure class, next-attempt time, and retained event history
- `GET /get-process-progress/{process_id}` - SSE stream of processing progress
- `POST /kill-processing/{process_id}` - Persist cancellation; `staging`, `queued`, or unactivated jobs cancel immediately only while every item is still pending, otherwise remain recovery-pending until partial effects are reconciled and completed-document metadata is finalized
- `POST /cleanup-processes` - Erase terminal staging, acknowledge it, then remove completed, failed, and cancelled job records
- `GET /get_files/` - List the caller's files
- `DELETE /delete_file/?file_name={filename}` - Coordinate per-file deletion; returns 404 when absent and 409 if the tenant lock is busy or canonical legacy names are ambiguous
- `DELETE /delete-document/{file_name}` - Compatibility deletion endpoint; reports a missing document in its response and returns 409 if busy or ambiguous
- `GET /document-chunks/{file_name}` - Get chunks for a document

### Multi-Agent Orchestrator (AI Search)
- `POST /ai_search/?session_id={session_id}` - Multi-agent orchestrator workflow
  - Body fields: required `message`; optional `web_search`, `deep_research` (the frontend control is hidden and the server forces it off), `chat_id`, and `attachments` (the latter two are not copied into `QueryJob`)
  - `false` requests automatic web selection; `true` forces web search when policy permits it
  - Returns typed progress/evidence frames, legacy progress frames, answer frames, and a terminal frame over SSE

### Graph Visualization
- `GET /graphml/{filename}` - Get GraphML file for visualization
- `GET /graphml/` - List available GraphML files

### Context Management
- `POST /add_context/?session_id={id}` - Upload files into a session's context; shares the raw-body/admission fence and exact multipart limits with knowledge uploads, authenticates before parsing, atomically publishes each raw file, then separately atomically replaces context metadata
- `GET /get_context/{session_id}` - Read context length and attached-file metadata
- `DELETE /clear_context/{session_id}` - Durably erase the owned raw directory, including crash orphans, before removing its metadata
- `DELETE /remove_file/{session_id}?file_name={name}` - Durably erase matching confined raw bytes before updating metadata

> Every route that takes a `session_id` verifies ownership before accessing
> session data and returns 404 for either a missing or foreign session.

## 🎨 Frontend Features

### Repository Page

*Upload documents, monitor processing, and inspect page-ordered chunks.*

- **File List**: Display all uploaded files with metadata
- **File Upload**: The page prechecks up to 10 files and a 500 MiB per-file limit; the backend also enforces a 500 MiB aggregate limit
- **Progress Tracking**: Real-time progress indicators (Upload → Parse → Encode → Database)
- **Document Viewer**: Retractable right panel showing document chunks
  - Header: The panel title shows the document name; each chunk header shows its document and page
  - Content: Chunks sorted by `page_num`, then `top_int`
  - Full text display; returned image fields are not rendered by the current page
- **File Deletion**: Invokes the coordinated deletion pipeline and reports failures

### Graph Visualization Page

*Explore document-derived entities, relationships, and graph statistics.*

- **3D Force Graph**: Interactive visualization of knowledge graphs
- **Node Interaction**: Hover to see entity details
- **Edge Interaction**: Hover to see relationship details
- **Color Coding**: Nodes colored by document source
- **Info Panel**: Display graph statistics (nodes, links, documents)

### Chat Interface (Multi-Agent System)

![Chat answer with an inline citation and source panel](./docs/screenshots/frontend-chat.png)

*Ask questions with web search and inspect cited sources beside the answer.*

- **Adaptive Workflow**: Orchestrator agent automatically detects scenario and selects tools
- **Real-time Progress**: Renders typed `step` and `evidence` events during the turn, with legacy `workflow_progress` as a live-stream fallback
- **Workflow Progress Visualization**: Displays orchestrator steps with distinct pending, completed, failed, skipped, and timed-out markers
  - Shows main steps: understanding intent, action planning, initial search, result gathering, reviewing, complementary search, generating answer
  - Shows tool progress: searching user knowledge base (RAG tool), searching user graph base (GraphRAG tool), searching online (Web Search tool)
  - Displays timing information for each step
- **Typed Trace Status**: Preserves step status and evidence as separate fields in the live trace tree
- **Context-aware**: Uses authorized knowledge-base, graph, web, and session sources
- **Intelligent Routing**: Resolves tools from intent, request mode, and server policy

## 🔌 Frontend-Backend Communication

### Communication Protocols

#### 1. REST API (Standard Requests)
- **Protocol**: HTTP/HTTPS
- **Library**: Axios (frontend) with custom plugins
- **Use Cases**:
  - Authentication (login/register)
  - File listing and deletion
  - Session management
  - GraphML file retrieval
- **Request Format**: JSON where a body is defined; uploads use multipart form data
- **Response Format**: JSON except file and streaming endpoints

#### 2. Server-Sent Events (SSE) - Real-time Streaming
- **Protocol**: HTTP with `text/event-stream` content type
- **Backend**: FastAPI `StreamingResponse` with `media_type="text/event-stream"`
- **Frontend**: `ReadableStream` API with `getReader()` for parsing SSE events
- **Use Cases**:
  - **File Upload Progress**: Real-time progress updates during document processing
  - **Agent Workflow Progress**: Typed step/evidence trace, with a legacy summary fallback
  - **Chat Responses**: Streaming LLM responses for final answers
- **Event Format**:
  ```
  event: message
  data: {"role": "workflow_progress", "content": "Workflow Progress\n     ☒ understanding intent (1.40s)"}

  event: message
  data: {"documents": [...]}

  event: end
  data: [DONE]
  ```

### Workflow Progress Visualization

The chat workflow component
(`frontend/src/components/workflow-progress/index.tsx`) renders typed step and
evidence events as a tree. Legacy progress text remains a fallback for older
live streams. Persisted history restores no trace data.

**UI Elements:**
- **Status markers**: `☐` pending, `☒` completed, `✕` failed, `⊘` skipped, and `⏱` timed out
- **Timing**: Displays execution time for each step (e.g., `(1.40s)`)
- **Hierarchical Display**:
  - Main workflow steps (understanding intent, action planning, etc.)
  - Sub-steps (tool execution progress)
- **Real-time Updates**: Updates automatically as SSE events are received

Uncaught query-turn and answer-slot failures send sanitized errors with the turn's
`run_id` as the reference. Every response also carries `X-Request-ID`;
first-party `visionagent.*` logs are structured JSON correlated by that
non-authorizing ID, and their exception payloads omit raw exception messages.

**Example Display:**
```
Workflow Progress
     ☒ understanding intent (1.40s)
     ☒ action planning (0.00s)
     ☒ initial search (7.12s)
        ☒ searching user knowledge base (4.62s)
        ☒ searching user graph base (4.62s)
        ☒ searching online (6.93s)
     ☒ result gathering and reranking (0.00s)
     ☒ reviewing (0.00s) looks ok
     ☒ generating answer (13.77s)
```

## ⚙️ Configuration

### Backend Configuration

Key configuration files:
- `backend/src/visionagent/config/settings.py` - Application settings and LLM model configuration
- `backend/src/visionagent/vendor/ragflow/conf/mapping.json` - Elasticsearch mappings
- `backend/src/visionagent/service/executer/tools/graphrag/config.py` - GraphRAG retrieval configuration (reads `settings`)

### GraphRAG Configuration

```env
# Top-K values (service/executer/tools/graphrag/config.py reads these from settings)
GRAPH_ENTITY_TOP_K=5
GRAPH_RELATION_TOP_K=5
```

Both values limit current graph retrieval. Graph extraction and aggregate
updates run serially inside each tenant-locked ingestion workflow.

```python
# pipeline/ingest.py
CHAR_LIMIT = 250000  # PDF splitting threshold, estimated characters

# vendor/ragflow/rag/app/manual.py: 32 and 1024 are merge thresholds, not output bounds
```

### Elasticsearch Mappings

The system uses dynamic mappings with explicit keyword fields:
- `docnm`: Keyword (document name)
- `ref_images`: Keyword (reference images)
- `create_time`: Keyword (creation timestamp)

## 🧪 Testing

`make test` runs seven tiers in order. `test/README.md` defines their scope.

```bash
make test                  # everything
make test-unit             # unit layers only — no running stack needed
```

| Layer | Runner | Needs a running stack |
|---|---|---|
| `make stale-imports` | grep for pre-restructure package paths | no |
| `test/architecture/` — `make test-architecture` | pytest: the one-way dependency and the slot wiring | no |
| `backend/tests/unit/`, `frontend/tests/unit/` — `make test-unit` | pytest, vitest (jsdom) | no |
| `test/pipeline/` — `make test-integration` | pytest: the real routers over HTTP with fake slots | no |
| `backend/tests/unit/test_golden.py` — `make test-golden` | pytest against the schema-v3 baseline: exact current-platform real-PDF parser snapshots and shared analyzer/persisted-shape contracts | no |
| `test/backend/`, `test/frontend/` — `make test-e2e` | pytest, playwright (real Chromium) | yes |
| `test/journey/` — `make test-journey` | pytest + playwright: one document end to end | yes, and a funded provider |

`make test-answer-quality` separately enforces the golden-set thresholds; it is
not part of `make test`. `make golden-check` exactly replays the shared offline
contracts and the parser snapshot keyed by the current OS, architecture, and
Python major/minor; a missing platform snapshot fails closed. `make golden`
updates that platform entry while preserving the others. Neither command needs
application configuration or provider credentials.

Individually:

```bash
backend/.venv/bin/python -m pytest backend/tests/unit -v
cd frontend && npx vitest run
cd frontend && node ../test/frontend/ui.e2e.cjs
```

Focused regression coverage lives in
`backend/tests/unit/test_regressions.py` and
`test/backend/test_session_security_e2e.py`.

### Manual Testing

1. **File Upload Test**
   - Upload a supported document
   - Verify terminal job status and progress frames
   - Confirm the document row, indexed chunks, and graph output

2. **AI Search Test**
   - Create a session
   - Exercise casual, forced-web, and professional queries
   - Confirm caller isolation, citations, persistence, and `[DONE]`

3. **Graph Visualization Test**
   - Open the graph page after a successful extraction
   - Verify caller-owned nodes, edges, details, and statistics

## 📚 Documentation

- [Tests](./test/README.md) - What each test tier proves and how to run it
- [The journey suite](./test/journey/README.md) - One document end to end, and the golden question set
- [Architecture](./docs/architecture.md) - Runtime ownership, boundaries, and dependency rules
- [Query pipeline](./docs/pipeline_ai_search.md) - End-to-end turn flow and trace events
- [Ingestion pipeline](./docs/pipeline_offline_parsing.md) - Upload, parsing, indexing, and recovery

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Commit your changes (`git commit -m 'Describe the change'`)
4. Push the branch (`git push origin feature/my-change`)
5. Open a Pull Request

## 📝 License

No project-level license file is currently included.
Third-party components retain their upstream terms and attribution in
[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

## 🙏 Acknowledgments

- **LightRAG**: Graph extraction, storage, and retrieval foundations
- **RAGFlow / DeepDoc**: Document parsing and text analysis
- **Elasticsearch**: Hybrid search
- **FastAPI**: Backend framework
- **React**: Frontend framework

## 📧 Contact

No project contact is currently defined.

## 🔮 Future Enhancements

- Deep Research remains reserved, dormant future scope; its control is hidden

---

**Note**: `GET /openapi.json` is the runtime API contract; tests under `test/`
enforce architecture, wiring, HTTP behavior, UI behavior, and the live journey.

Copyright © 2026 Jerry Huang. All rights reserved.
