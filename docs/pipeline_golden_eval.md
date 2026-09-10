# Synthetic Evaluation Pipeline for Agentic RAG (Using Real Documents as Base)

> The repository implements exact analyzer/schema checks, a recorded parser
> compatibility baseline, and a twelve-question live answer-quality set.
> Sections 2 and 3 distinguish implemented coverage from proposed extensions.

## 1. Data Schema

### 1.1 Documents (Knowledge Base)

The schema-v3 compatibility baseline is
`backend/tests/golden/baseline.json`. It records:

- exact coarse and fine analyzer output for seven fixed strings;
- exact parser snapshots for three named, versioned RAGFlow fixtures in
  `test/fixtures/ragflow/`, grouped under `parse_by_platform` by operating
  system, architecture, and Python major/minor, including source and content
  SHA-256 identities;
- shared parser and tokenizer asset SHA-256 identities; and
- Elasticsearch non-vector fields, vector field name, and vector width.

Platform keys use forms such as `darwin-arm64-py311` and
`linux-x86_64-py311`. Each fixture runs in its own sanitized process with
deterministic language detection and fixed numeric thread counts. The current
platform must have an exact snapshot; missing fixtures, a missing platform key,
changed source or runtime-asset bytes, or malformed baseline data fail closed.

The live set uses `test/fixtures/ragflow/Doc1.pdf`, the Apache-2.0 RAGFlow
benchmark fixture "Purpose of RAGFlow."

### 1.2 Evaluation Example Schema

`test/journey/golden_qa.json` contains twelve examples. Every example
has `id`, `question`, and `why`.

Answerable examples define `gold` and use one or both answer matchers:

- `gold` — a source phrase required in retrieved and cited text;
- `expect_all` — answer substrings all required; and
- `expect_any` — acceptable alternatives.

Unanswerable examples define `unanswerable: true`,
`expect_any` gap language, and `must_not_contain` false
claims that no document was uploaded. Matching lowercases and collapses
whitespace; it is not semantic judging.

---

## 2. Synthetic Dataset Generation (Using Real Documents)

No generator exists in the repository. The sections below define proposed
extensions to the fixed compatibility baseline and source-backed live set.

### 2.1 Step 0: Prepare Knowledge Base

For the implemented live set:

1. Keep the pinned `test/fixtures/ragflow/Doc1.pdf` fixture available.
2. Create a fresh user.
3. Ingest the document through `POST /start-processing`.
4. Fail on any upload error before asking questions.
5. Delete the account after the run.

For a future synthetic set, preserve source document identity, chunk IDs,
chunk text, and metadata needed to reproduce ground truth.

---

## 2.2 Use Case 1 — Regulation Explanation

Eight implemented questions ask facts explicitly present in the fixed
document; four ask for facts absent from it.

### 2.2.1 Select Ground-Truth Chunks

The current set does not store ground-truth chunk IDs. Its ground truth is an
exact `gold` phrase from the source. A future retrieval benchmark may
define:

\[
R(q) = \{c_{i_1}, \ldots, c_{i_m}\}.
\]

That set must be reviewed against the exact parser baseline used to create it,
because parser changes can move chunk boundaries and IDs.

### 2.2.2 GPT-5 Query + Gold Answer Generation

Current questions and answer expectations are manually source-backed. No
GPT-5 generator is wired, and the runtime supports only the configured
`dashscope` or `claude-cli` LLM provider.

Any future generator must receive only selected source chunks, retain those
chunk IDs, and keep generated questions separate from manually approved
acceptance data.

---

## 2.3 Use Case 2 — Numeric Verification

Numeric scenario generation and numeric scoring are not implemented.

### 2.3.1 Extract or Define Numeric Rules from Real Documents

Future numeric rules must be transcribed into reviewed formulas and retain
their source chunk IDs. An LLM may rewrite prose but must not define the
numeric truth.

### 2.3.2 Generate Random Scenarios Programmatically

Scenarios must use a seeded generator and compute expected values and
compliance decisions in code so each case is reproducible.

### 2.3.3 GPT‑5 Writes Natural Query + Answer

A future LLM may turn a programmatic scenario into natural language. The
stored formula result remains authoritative, and the provider/model/version
must be recorded with the generated text.

---

## 2.4 Use Case 3 — Design Consultation

Design-scenario generation and issue scoring are not implemented.

### 2.4.1 Generate Design Scenarios Programmatically

Future scenarios must store input constraints, source rules, and a reviewed
`gold_issues` set. Violations must be computed, not inferred by the
answer model.

### 2.4.2 GPT‑5 Generates Query + Answer

Generated consultation questions must remain traceable to their scenario and
source rules. `required_tools` may use only registered tool names:
`RAG`, `GraphRAG`, `web_search`, and `LLM`.
No numeric-checking tool exists.

---

## 3. Evaluation Pipeline

The diagram is the canonical map of the implemented static and live paths.
Solid edges show the main flow; dotted edges also show implemented asynchronous
control and recovery. Only the explicitly styled `NOT IMPLEMENTED` nodes at the
end are proposals.

```mermaid
flowchart TD
  E0["Evaluation entry points"]

  E0 --> R0["make golden: refresh shared contracts and the current platform snapshot; preserve other platform snapshots"]
  R0 --> R1["Seven fixed analyzer strings"]
  R1 --> R2["Vendored tokenizer: record exact coarse and fine tokens"]
  R0 --> R3["Fixed manifest: three repository PDF fixtures"]
  R3 --> R4{"Every fixture and required runtime asset exists and is non-empty?"}
  R4 -->|"no"| SFAIL["FAIL: missing or invalid compatibility artifact"]
  R4 -->|"yes"| R5["Derive OS-architecture-Python key; run one clean worker per fixture with selectors removed and numeric threads fixed"]
  R5 --> R6["Deterministic DeepDoc language classification, OCR, layout, merge, and analysis"]
  R6 --> R7["Record exact hashes, tokens, pages, offsets, names, and image presence in the parse_by_platform entry for the current key"]
  R0 --> R8["Fake ParsedChunk through real document shaper with fixed analyzer, zero embedder, and rejecting writer"]
  R8 --> R9["Collect Elasticsearch fields, vector key, and fixed persisted width; no network or datastore"]
  R0 --> R10["Record full SHA-256 identities for required parser and tokenizer assets"]
  R2 --> R11["Validate schema-v3 candidate; preserve other platform entries; fsync temporary file; atomic replace baseline.json"]
  R7 --> R11
  R9 --> R11
  R10 --> R11

  E0 --> S0["make test-golden: complete offline pytest suite"]
  S0 --> S1{"schema-v3 baseline and current platform entry exist; shared and fixture manifests are exact?"}
  S1 -->|"no"| SFAIL
  S1 -->|"yes"| S2["Exact analyzer output and stemming assertions"]
  S1 -->|"yes"| S3["Real document shaper with local fakes and deliberately invalid application configuration"]
  S3 --> S4["Assert fields, vector key and width; assert no settings, provider, or datastore import"]
  S2 --> SRESULT{"All static assertions pass?"}
  S4 --> SRESULT
  SRESULT -->|"yes"| SPASS["PASS: deterministic analyzer and schema contract"]
  SRESULT -->|"no"| SFAIL

  S1 -->|"yes; slow"| S5["make golden-check in a subprocess"]
  E0 --> S6["make golden-check: canonical read-only offline replay"]
  S6 --> S5
  S5 --> S7["Regenerate shared contracts and the current platform parser snapshot"]
  S7 --> S8{"Shared contracts and selected schema-v3 platform snapshot equal the baseline exactly?"}
  S8 -->|"no"| SFAIL
  S8 -->|"yes"| SPARSEPASS["PASS: 3/3 fixtures and every recorded contract reproduce"]
  S1 -->|"yes; slow"| S11["Parse Doc1.pdf in three fresh workers"]
  S11 --> S14["Validate non-empty machine-readable snapshots and compare every field exactly"]
  S14 --> SREPRORESULT{"All three snapshots agree and equal the current platform baseline?"}
  SREPRORESULT -->|"yes"| SREPROPASS["PASS: clean-process parsing is reproducible"]
  SREPRORESULT -->|"no"| SFAIL

  E0 --> L0["make test-answer-quality: live answer-quality path"]
  L0 --> L1{"requests installed and GET openapi.json succeeds?"}
  L1 -->|"no"| LSKIP0["SKIP: live backend unavailable; no user created"]
  L1 -->|"yes"| L2["clean_user fixture: register, login, GET me; isolated user, index, graph, and rows"]
  L2 --> L3{"test/fixtures/ragflow/Doc1.pdf exists?"}
  L3 -->|"no"| LSKIP1["SKIP: fixed source PDF missing"]
  L3 -->|"yes"| I0{"ASGI declared/streamed raw-body cap and per-worker upload admission pass?"}
  I0 -->|"no"| LFAIL0["FAIL: upload or progress fixture raises before provider probe"]
  I0 -->|"yes"| I0P["POST start-processing; authenticate and mint IngestJob before parsing multipart into request-owned spools"]
  I0P --> I0A{"File count, per-file and aggregate content limits pass; spools are readable?"}
  I0A -->|"no"| LFAIL0
  I0A -->|"yes"| I0N{"NFC logical names are 1-255 characters, NUL-free,<br/>free of edge whitespace and reserved graph separator text,<br/>unique in the batch, tenant KB, and active jobs?"}
  I0N -->|"no"| LFAIL0
  I0N -->|"yes"| I0S["Under sorted per-name advisory locks, commit owner-bearing non-dispatchable STAGING job, item checkpoints, retry budget, and event rows before durable application bytes"]
  I0S -->|"error"| LFAIL0
  I0S --> I0F["Under the tenant staging lock, recheck ticket and account gate; stream and fsync opaque files or planned PDF page ranges"]
  I0F -->|"error"| LFAIL0
  I0F --> I0Q["Atomically recheck the account gate and move STAGING to QUEUED"]
  I0Q -->|"error or refused"| LFAIL0
  I0Q --> I0B["When next_attempt_at is due, take the worker-node advisory capacity lock; reserve at most MAX_UPLOAD_WORKERS live children with FOR UPDATE SKIP LOCKED; increment attempt_count"]
  I0Q -. "owner may later POST kill-processing" .-> IKILL["Persist cancellation before signalling; signal only a matching node, PID, and process-start fingerprint"]
  IKILL --> IKILLSTATE{"Every item still PENDING?"}
  IKILLSTATE -->|"yes: STAGING, QUEUED, or unactivated"| IKILLNOW["Commit terminal CANCELLED immediately; attempt staging erasure and acknowledgement"]
  IKILLSTATE -->|"no"| IKILLRECOVER{"Matching live local child?"}
  IKILLRECOVER -->|"yes: signal it"| ICOMP
  IKILLRECOVER -->|"no: queued recovery"| I0B
  IKILLRECOVER -->|"no: live elsewhere"| ILEASE
  I0B -->|"spawn succeeds"| I1["Child activates that exact reservation, records node/PID fingerprint, and starts the lease heartbeat"]
  I0B -->|"spawn fails"| ISPAWN{"Any non-PENDING recovery checkpoint?"}
  ISPAWN -->|"yes"| ISPAWNRECOVER["Record worker_spawn_failed, refund this reservation, and remain QUEUED until a worker can reconcile safely"]
  ISPAWNRECOVER --> I0B
  ISPAWN -->|"no"| ISPAWNBUDGET{"Pristine startup attempt budget exhausted?"}
  ISPAWNBUDGET -->|"no"| ISPAWNBACKOFF["Record worker_spawn_failed and queue after exponential backoff; no downstream effects exist"]
  ISPAWNBACKOFF --> I0B
  ISPAWNBUDGET -->|"yes"| I14F
  I1 --> IPREFLIGHT["Recovery preflight: from durable checkpoints, idempotently finalize every logical document whose sibling parts are terminal in a bounded child that rechecks the lease under tenant_lock"]
  IPREFLIGHT -->|"failure or timeout"| IRECOVER
  IPREFLIGHT -->|"success"| I1B{"Reservation attempt_count exceeds max_attempts?"}
  I1B -->|"yes: reconciliation incarnation"| IALLCOMP["In separate killable lease-fenced children under tenant_lock, compensate only partial logical documents with unfinalized effects, graph-first and then Elasticsearch; preserve finalized documents and zero compensated/untouched unfinished items"]
  IALLCOMP -->|"success; cancellation requested"| ICANCELLED
  IALLCOMP -->|"success; not cancelling"| I14F
  IALLCOMP -->|"failure or timeout"| IRECOVER["Keep job processing, checkpoints, and staging; stop heartbeat so lease recovery can retry"]
  I1B -->|"no"| IRECOVERED{"Any retained PROCESSING checkpoint?"}
  IRECOVERED -->|"no"| ICANCELENTRY{"Cancellation already requested?"}
  ICANCELENTRY -->|"yes"| ICANCELSAFE
  ICANCELENTRY -->|"no"| I1A["Spawn a clean item subprocess; acquire tenant_lock and recheck the active lease and account gate; parent enforces one hard PARSE_TIMEOUT_S deadline and polls cancellation, shutdown, and lease loss"]
  IRECOVERED -->|"yes"| IREPLAYCOMP["In bounded lease-fenced children under tenant_lock, compensate each affected logical document graph-first and then Elasticsearch"]
  IREPLAYCOMP -->|"failure or timeout"| IRECOVER
  IREPLAYCOMP -->|"success; cancellation requested"| ICANCELLED
  IREPLAYCOMP -->|"success; continue"| IREPLAYRESET["Reset every item of the compensated document to PENDING, then replay in this incarnation"]
  IREPLAYRESET --> I1A
  I1 -. "worker dies and lease expires" .-> ILEASE{"Recovery case"}
  ILEASE -->|"ordinary"| ILEASEBACKOFF["Retain the PROCESSING ambiguity marker; record worker_lease_expired and queue after backoff"]
  ILEASE -->|"cancellation pending"| ICANCELREPLAY["Retain processing checkpoint; queue after backoff for safe replay"]
  ILEASE -->|"budget exhausted, not cancelling"| IRECONQUEUE["Retain checkpoints; queue one immediate reconciliation incarnation"]
  ILEASEBACKOFF --> I0B
  ICANCELREPLAY --> I0B
  IRECONQUEUE --> I0B
  I1A --> I2{"Parser slot selected"}
  I2 -->|"DeepDoc default"| I2D["Parse PDF or DOCX with OCR, layout, tables, reading order, and chunk merge"]
  I2 -->|"VLM"| I2V{"VLM source"}
  I2V -->|"PDF"| I2VR["Validate 1-2000 pages; for each page enforce render, embedded-raster, and encoded-byte bounds; await transcription; preserve source page; release before the next"]
  I2V -->|"PNG/JPG/JPEG"| I2VI["Enforce source-byte and decoded-pixel bounds; verify and encode once; await transcription"]
  I2VR --> I2P
  I2VI --> I2P
  I2D --> I2P{"Parsing completed with at least one usable chunk?"}
  I1A -->|"deadline, cancellation, shutdown, lease loss, spawn/EOF/abnormal exit, or child error"| ICOMP["Hard-kill when needed and always join the item process; ambiguous mutations require fenced compensation"]
  I2 -->|"child error"| ICOMP
  ICOMP --> ICOMPR["In a separate bounded child, recheck the lease under tenant_lock, then remove graph contributions first and Elasticsearch second; graph refusal leaves search untouched"]
  ICOMPR -->|"lease lost, failure, or timeout"| IRECOVER
  ICOMPR -->|"success"| IOWNED{"Worker still owns the lease?"}
  IOWNED -->|"no"| IRECOVER
  IOWNED -->|"yes"| ICANCEL{"Cancellation or controlled shutdown?"}
  ICANCEL -->|"cancellation"| ICANCELLED["Preserve finalized documents; zero compensated and untouched unfinished items; publish terminal CANCELLED with no PROCESSING item"]
  ICANCEL -->|"shutdown"| ISHUTDOWNRETRY["Reset the compensated logical document, queue immediately, and refund this reservation's attempt"]
  ISHUTDOWNRETRY --> I0B
  ICANCEL -->|"neither"| IRETRY{"MAX_UPLOAD_ATTEMPTS exhausted?"}
  IRETRY -->|"no"| IBACKOFF["Reset every item for that logical document; queue after exponential backoff with safe failure class"]
  IBACKOFF --> I0B
  IRETRY -->|"yes"| I14F
  I2P -->|"no"| IERR["Record the stage error and continue worker finalization"]
  I2P -->|"yes"| I3["Assign SHA-256 chunk IDs over 8-byte-length-prefixed tenant, NFC logical document, sequence:part_name, zero-based ordinal, and content"]
  I3 --> I4["Graph write service inside the isolated item's tenant lock"]
  I4 --> I5["For each chunk with at least 10 words: NER-model LLM extracts entities and relationships"]
  I5 --> I6["Parse/group records; store canonical contributions_json by atomic source ID; derive exact document_names_json for vectors; merge replay leaves aggregates unchanged and identical VDB upserts skip embedding; preserve stored edge orientation"]
  I4 -->|"load, provider, extraction, or merge failure before publication"| I9["Record graph extraction failed; continue Elasticsearch indexing; the item and job later fail"]
  I5 -->|"provider or extraction failure"| I9
  I6 --> I6A["Embed new or changed node/edge aggregates; re-embed recomputed records; embedding failure propagates without a partial/zero row"]
  I6A -->|"embedding failure before publication"| I9
  I6A --> I7["GraphRepository temp-writes, fsyncs, and atomically replaces node vectors, edge vectors, then GraphML; fsync every parent directory"]
  I7 -->|"publication succeeds"| I10["ChunkStore slot: ElasticsearchChunkStore.index"]
  I7 -->|"publication fails after any durable prefix"| ICOMP
  I9 --> I10
  I10 --> I11["Embedding provider per chunk; analyzer only fills missing token fields; shape Elasticsearch documents"]
  I11 --> I12["Elasticsearch writer bulk-indexes into the user's index"]
  I10 -->|"uncaught embedding/index exception"| ICOMP
  I12 --> I12R{"Bulk call returned?"}
  I12R -->|"no"| ICOMP
  I12R -->|"yes"| I12A{"Every prepared chunk accepted?"}
  I12A -->|"no"| IERR
  I12A -->|"yes"| ICHECKPOINT["Commit terminal item checkpoint with durable counters and safe error"]
  IERR --> ICHECKPOINT
  ICHECKPOINT --> IDOCREADY{"Are all sibling parts of this logical document terminal?"}
  IDOCREADY -->|"no"| I14C
  IDOCREADY -->|"yes"| I13["Immediately run retry-safe Postgres knowledge-base upsert from durable checkpoints in a killable bounded subprocess that rechecks the lease under tenant_lock"]
  I13 --> I13A{"Metadata upsert completed?"}
  I13A -->|"no"| IMETARECOVER["Keep job processing and staging; stop heartbeat; lease recovery retries metadata without replaying terminal items"]
  IMETARECOVER -.-> ILEASE
  I13A -->|"yes"| I14C{"Cancellation requested?"}
  I14C -->|"yes"| ICANCELSAFE["In a lease-fenced child, compensate any partial logical document with unfinalized effects graph-first and then Elasticsearch; preserve finalized documents"]
  ICANCELSAFE -->|"failure or timeout"| IRECOVER
  ICANCELSAFE -->|"success"| ICANCELLED
  I14C -->|"no"| I14S{"Controlled shutdown at this safe boundary?"}
  I14S -->|"yes"| ISAFEREQUEUE["Queue unfinished work and refund this reservation's attempt"]
  ISAFEREQUEUE --> I0B
  I14S -->|"no"| INEXT{"Another unfinished item?"}
  INEXT -->|"yes"| I1A
  INEXT -->|"no"| IFINALMETA["Repeat the idempotent bounded metadata sweep to close any checkpoint/upsert crash gap"]
  IFINALMETA -->|"failure or timeout"| IRECOVER
  IFINALMETA -->|"success"| I14{"Worker status has any error?"}
  I14 -->|"yes"| I14F["Preserve finalized checkpoints; zero compensated or untouched unfinished items when applicable; commit terminal FAILED and error event"]
  I14F --> LFAIL0
  I14 -->|"no"| I15["Commit terminal completed status and progress event"]
  I15 --> L4["provider_available fixture: build_llm then await a probe completion"]
  I15 -.-> I15A["Worker or any supervisor erases shared staging and records staging_cleaned; job, item checkpoints, and journal remain until owner cleanup"]
  I14F -.-> I15A
  ICANCELLED -.-> I15A
  L4 --> L5{"LLM provider answers?"}
  L5 -->|"no"| LSKIP2["SKIP: provider unavailable after successful ingestion"]
  L5 -->|"yes"| L6["Load twelve manual cases from golden_qa.json: eight answerable and four unanswerable"]

  L6 --> Q0["For each case: create a separate empty chat session"]
  Q0 --> Q1["POST ai_search: web_search false maps to AUTO; deep_research false"]
  Q1 --> Q2["API authenticates, verifies session ownership, resolves allowed tools, and mints QueryJob"]
  Q2 --> Q3["QueryPipeline constructs per-turn AgentState from the immutable job"]
  Q3 --> Q4["Intent slot: classify chat scenario"]
  Q4 -->|"casual or casual_web"| QC["Casual path: optional authorized web_search for casual_web, then Answer slot; no retrieval loop"]
  Q4 -->|"professional"| Q5["Intent slot: analyze_query_intent returns knowledge scenario, context intents, and graph keywords"]
  Q5 --> Q6["Planner slot: intersect intent, AUTO web choice, session context, registry, and authorization"]
  Q6 --> Q7["Executer slot: dispatch the planned subset concurrently; recheck authorization; default 120 seconds per tool"]
  Q7 --> Q7A["Frozen ToolRegistry maps the four authorized tool names to implementations"]
  Q7A --> T1["RAG tool: user's Elasticsearch hybrid BM25 and kNN retrieval"]
  Q7A --> T2["GraphRAG tool: entity and relationship vector search, source chunk IDs, then Elasticsearch chunk lookup"]
  Q7A --> T3["web_search tool: authorized web provider plus snippet rerank, related questions, images, and videos"]
  Q7A --> T4["LLM tool: session context plus provider completion returned as current-context evidence"]
  T1 --> Q8["Accumulate ToolResults across rounds; keep partial results; deduplicate chunks by ID"]
  T2 --> Q8
  T3 --> Q8
  T4 --> Q8
  Q8 --> Q9["Reranker slot: provider scores candidates; keep score at least 0.1 and top five by default"]
  Q9 --> Q10["Evaluator slot: empty context returns 0.0; otherwise the LLM scores context, then evaluator policy applies the above-0.5 default threshold"]
  Q10 --> Q11{"Insufficient and another round remains? Default maximum is two rounds"}
  Q11 -->|"yes"| Q12["Evaluator reflection: request up to three KB or web queries as prompt guidance, parse any returned count, and enforce web policy"]
  Q12 --> Q13{"Any refined query remains?"}
  Q13 -->|"yes"| Q5
  Q13 -->|"no"| Q14["Answer slot: ranked context, citation IDs, session history, and gap instruction to LLM"]
  Q11 -->|"no"| Q14
  QC --> Q15["Stream answer SSE and persist the turn"]
  Q14 --> Q15
  Q15 --> Q16{"All twelve cases completed?"}
  Q16 -->|"no"| Q0
  Q16 -->|"yes"| M0["Normalize answer text; take last non-empty citations; resolve doc and web citation markers"]

  M0 --> M1["Context metric, eight answerable: returned citation text contains gold"]
  M0 --> M2["Answer metric, all twelve: every expect_all, one expect_any, and no must_not_contain"]
  M0 --> M3["Grounding metric, eight answerable: a citation actually used by the answer contains gold"]
  M1 --> M4["Aggregate gates: context at least 85 percent or 7 of 8; answer at least 85 percent or 11 of 12; grounding at least 85 percent or 7 of 8"]
  M2 --> M4
  M3 --> M4
  M4 --> M5["Hard gates: each unanswerable case passes answer rules; every answerable case uses a resolvable citation"]
  M5 --> M6{"Every aggregate and hard gate passes?"}
  M6 -->|"yes"| LPASS["PASS: live answer-quality suite"]
  M6 -->|"no"| LFAIL1["FAIL: assertions identify failed case IDs; wrong-answer rows also print answer text"]

  LSKIP1 --> CLEAN["clean_user teardown, if created: DELETE me takes the staging coordinator lock, closes admission, cancels uploads, then takes tenant_lock; it may discard recovery checkpoints because complete tenant stores are erased; staging erasure precedes index, graph, context, and PostgreSQL account removal"]
  LFAIL0 --> CLEAN
  LSKIP2 --> CLEAN
  LPASS --> CLEAN
  LFAIL1 --> CLEAN

  E0 -.-> F0["NOT IMPLEMENTED: synthetic dataset generation"]
  F0 -.-> F1["Future source-chunk IDs, Recall at k, Precision at k, and required-tool metrics"]
  F0 -.-> F2["Future seeded numeric scenarios, code-computed truth, and numeric scoring"]
  F0 -.-> F3["Future design scenarios, reviewed gold issues, and issue scoring"]
  style F0 stroke-dasharray: 5 5
  style F1 stroke-dasharray: 5 5
  style F2 stroke-dasharray: 5 5
  style F3 stroke-dasharray: 5 5
```

### 3.1 Retrieval Metrics

The implemented context check joins all returned citation text and asks
whether it contains `gold`. It does not compute Recall@k or Precision@k
because `relevant_chunk_ids` do not exist.

A future chunk-ID benchmark may compute:

\[
\operatorname{Recall@k}(q)=\frac{|A_k(q)\cap R(q)|}{|R(q)|},
\qquad
\operatorname{Precision@k}(q)=\frac{|A_k(q)\cap R(q)|}{k}.
\]

---

### 3.2 Agent Behavior Metrics

Tool recall, precision, and sequence success are not implemented. The live
quality test does not assert which tool produced a citation and accepts both
`[doc][ID]` and `[web][ID]` markers.

If a future dataset adds `required_tools`, compute:

\[
\operatorname{ToolRecall}=
\frac{|T_{\mathrm{pred}}\cap T_{\mathrm{gold}}|}{|T_{\mathrm{gold}}|},
\qquad
\operatorname{ToolPrecision}=
\frac{|T_{\mathrm{pred}}\cap T_{\mathrm{gold}}|}{|T_{\mathrm{pred}}|}.
\]

---

### 3.3 Answer Quality Metrics

The implemented suite scores context, answer text, and grounding separately.
There is no LLM judge.

#### 3.3.1 Regulation Explanation

- **Context:** concatenated returned citation text contains `gold`.
- **Answer:** normalized text satisfies `expect_all`,
  `expect_any`, and `must_not_contain`.
- **Grounding:** a marker used in the answer resolves to a citation passage
  containing `gold`.

The minimum passing counts are 7/8 for context, 11/12 for answer text, and 7/8
for grounding. Every answerable response must use a resolvable citation.

#### 3.3.2 Numeric Verification

Not implemented. Future evaluation must compare model output with
`numeric_ground_truth` for compliance correctness and absolute or
relative value error.

#### 3.3.3 Design Consultation

Not implemented. Future evaluation may compare extracted issues with the
reviewed `gold_issues` set:

\[
\operatorname{IssueRecall}=
\frac{|I_{\mathrm{pred}}\cap I_{\mathrm{gold}}|}{|I_{\mathrm{gold}}|},
\qquad
\operatorname{IssuePrecision}=
\frac{|I_{\mathrm{pred}}\cap I_{\mathrm{gold}}|}{|I_{\mathrm{pred}}|}.
\]

---

## 4. Final Evaluation Summary

Run the implemented checks from the repository root:

- `make test-golden`
- `cd backend && .venv/bin/python -m pytest tests/unit/test_golden.py -m "not slow"`
- `make golden-check`
- `cd backend && .venv/bin/python -m pytest tests/unit/test_golden.py -m slow`
- `make test-answer-quality`

Compatibility acceptance requires the schema-v3 baseline, all three fixed PDF
fixtures, the shared parser and tokenizer assets, exact analyzer and
Elasticsearch-shape checks, and exact equality with the parser snapshot selected
by the current OS, architecture, and Python major/minor. `Doc1.pdf` is also snapshotted across
three clean workers. Missing or changed inputs and missing platform entries
fail; no compatibility path skips them.

The live quality run applies the thresholds, unanswerable string rules, and
per-answer citation requirement specified in §3.3.1.

Re-record with `make golden` only after reviewing the behavior change. It
refreshes the shared contracts, inserts or replaces the current
`parse_by_platform` entry, preserves other platform entries, validates the
candidate, and atomically replaces the baseline. Recording uses a fixed local
zero embedder and rejecting writer and needs no application configuration,
datastore, or provider credential. Generate the descriptive, non-gating live report with
`backend/.venv/bin/python test/journey/make_report.py`.

---

Keep future generated datasets separately versioned so these fixed
compatibility and answer-quality artifacts remain reproducible.
