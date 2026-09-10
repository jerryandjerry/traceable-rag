# Traceable RAG Ingestion Pipeline

Document ingestion builds tenant-scoped text and graph indexes from an
authorized upload.

## Architecture Overview

### 1. **NormalRAG System** (Existing)

- Parse documents into typed chunks.
- Analyze, embed, and write chunks to the user's Elasticsearch index.
- Record document metadata in PostgreSQL and stream progress through SSE.

### 2. **GraphRAG System** (LightRAG Integration)

- Extract entities and relationships before NormalRAG indexing.
- Store node vectors, edge vectors, and NetworkX GraphML under `GRAPH_DIR`.
- Run each mutable document item in a killable subprocess that holds
  `tenant_lock(user_id)` across parsing, graph mutation, and Elasticsearch;
  item, metadata, and compensation children recheck the active lease under
  that lock before mutation.
- Record pre-publication graph failures while continuing to index searchable
  chunks; compensate and retry ambiguous publication failures.

## File Processing Workflow

```mermaid
flowchart TD
    Client["Client sends multipart files"] --> TransportFence{"Shared UploadBodyLimitMiddleware<br/>/start-processing and /add_context<br/>Per-API-worker admission below MAX_CONCURRENT_UPLOADS?"}
    TransportFence -->|full| BusyReject["Return 429 with Retry-After<br/>without reading the body"]
    TransportFence -->|admitted| DeclaredSize{"Declared Content-Length above<br/>MAX_UPLOAD_TOTAL_BYTES + 1 MiB?"}
    DeclaredSize -->|yes| DeclaredReject["Return 413 without reading the body"]
    DeclaredSize -->|no or absent| RouteChoice{"Upload surface"}
    RouteChoice -->|knowledge document| Route["POST /start-processing"]
    RouteChoice -->|session attachment| ContextRoute["POST /add_context/?session_id={session_id}"]
    ContextRoute --> ContextAuth["Authenticate JWT, check current user row,<br/>and verify session ownership before request.form"]
    ContextAuth -->|invalid or foreign| AuthReject
    ContextAuth -->|owned| ContextForm["Parse counted multipart spools; only files;<br/>validate count, per-file, aggregate, safe 1-255-character names,<br/>and the sanitized 255-byte filesystem bound"]
    ContextForm -->|invalid| ShapeReject
    ContextForm -->|valid| ContextLock["Acquire async tenant_lock; recheck account gate and session ownership"]
    ContextLock -->|unavailable| ContextBusy["Return 409; publish no context"]
    ContextLock -->|live| ContextRaw["Sanitize/uniquify name; stream to 0600 temp;<br/>fsync; atomically hard-link into fixed owned session directory"]
    ContextRaw --> ContextVlm["VLM processor: one bounded page at a time<br/>under one PARSE_TIMEOUT_S document budget"]
    ContextVlm --> ContextJson["Atomically temp-write, fsync, and replace session JSON<br/>while retaining the matching raw path as deletion manifest"]
    ContextJson --> ContextDone["Return per-file status and timing"]
    ContextRaw -->|copy or extraction/commit failure| ContextRollback["Durably unlink unpublished/uncommitted raw file"]
    ContextVlm -->|failed| ContextRollback
    ContextJson -->|failed| ContextRollback

    RemoveContext["DELETE /remove_file/{session_id}?file_name={name}<br/>authenticate, own session, acquire tenant_lock"] --> ValidateManifest{"Every manifest entry is a canonical direct child<br/>of the fixed owned session raw directory?"}
    ValidateManifest -->|no| ContextDeleteFail["Fail closed without unlinking any target"]
    ValidateManifest -->|yes| RawFirst["Unlink matching raw entries and fsync first;<br/>missing raw entries are retry-safe"]
    RawFirst --> ContextMetadata["Atomically rewrite JSON without the file"]
    ClearContext["DELETE /clear_context/{session_id}<br/>authenticate, own session, acquire tenant_lock"] --> ClearRaw["Ignore manifest targets; safely erase and fsync<br/>the fixed owned raw directory without following symlinks"]
    ClearRaw --> ClearJson["Remove and fsync session JSON second"]
    ClearRaw -->|I/O failure| ContextDeleteFail
    RawFirst -->|I/O failure| ContextDeleteFail
    Route --> Auth["API dependencies authenticate JWT<br/>check token and user row<br/>build AuthorizationScope before request.form"]
    Auth --> Permit{"Upload permitted<br/>and optional session owned?"}
    Permit -->|no| AuthReject["Return 401, 403, or 404"]
    Permit -->|yes| DraftJob["Mint immutable IngestJob<br/>run_id, user_id, authorization, session_id"]
    DraftJob --> ParseForm["Parse multipart through a counted receive wrapper<br/>into request-owned seekable spools<br/>files field only; no FastAPI File dependency"]
    ParseForm -->|streamed raw cap exceeded| StreamReject["Abort parsing and return 413"]
    ParseForm -->|malformed multipart| FormReject["Return 400"]
    ParseForm --> UploadShape{"Only the files field; at least one file part;<br/>count at most MAX_UPLOAD_FILES?"}
    UploadShape -->|no| ShapeReject["Return 422 for shape<br/>or 413 for count"]
    UploadShape -->|yes| Names{"Normalize every logical name to NFC;<br/>1-255 Unicode characters, no NUL, no edge whitespace,<br/>and no reserved graph separator text?"}
    Names -->|no| NameReject["Return 422"]
    Names -->|yes| BatchUnique{"Canonical names unique within this batch?"}
    BatchUnique -->|no| ConflictReject["Return 409 before STAGING or durable bytes"]
    BatchUnique -->|yes| ReadUpload["Scan and rewind every spool in 1 MiB pieces<br/>without joining file bytes"]
    ReadUpload --> Limits{"Each file at most MAX_UPLOAD_BYTES<br/>and aggregate content at most MAX_UPLOAD_TOTAL_BYTES?"}
    Limits -->|no| LimitReject["Return 413"]
    Limits -->|yes| Job["Complete IngestJob.file_names"]
    Job --> Start["Run pipeline.ingest.start_processing in an asyncio worker thread<br/>shield and await the staging handoff on request cancellation<br/>so request-owned spools remain open through durable staging"]

    Start --> Stage["stage_files plans work over seekable request spools"]
    Stage --> IsPdf{"PDF?"}
    IsPdf -->|no| OneItem["Plan one direct-copy work item<br/>original name and part name"]
    IsPdf -->|yes| SampleTemp["Inspect PDF directly from the request spool"]
    SampleTemp --> Estimate["Count pages<br/>sample all pages up to 5, otherwise random 5<br/>estimate pages times maximum sampled characters"]
    Estimate --> TooLarge{"Estimate above CHAR_LIMIT 250,000?"}
    TooLarge -->|no| KeepPdf["Emit original PDF as one work item"]
    TooLarge -->|yes| SplitPdf["Split into ordered page-range parts<br/>retain original filename for every part"]
    KeepPdf --> SampleCleanup["Finish request-spool planning"]
    SplitPdf --> SampleCleanup
    SampleCleanup --> Staged["PreparedUpload work-item plan<br/>source stream or PDF page range; no split byte blobs"]
    OneItem --> Staged
    Stage -->|error| StartError["Return HTTP 500<br/>no worker is started"]

    Staged --> DurableTx["One PostgreSQL transaction before durable application bytes<br/>lock users row FOR SHARE; take sorted advisory locks per tenant and canonical name<br/>reject an existing document or active same-name job<br/>insert owner-bearing STAGING job, pending items, retry budget, and initial events"]
    DurableTx -->|account or name conflict| TicketConflict["Return 409<br/>no durable application bytes or worker"]
    DurableTx -->|database error| TicketError["Return 500<br/>no durable application bytes or worker"]
    DurableTx --> StageLock["Acquire shared per-tenant upload staging lock"]
    StageLock --> StageFence{"Re-read owner ticket and account gate<br/>still STAGING, live, and not cancelled?"}
    StageFence -->|no| StageAbort["Erase any partial durable bytes before lock release<br/>fail the ticket if it is still STAGING and acknowledge terminal cleanup<br/>return 409 for unavailable account or 500 otherwise"]
    StageFence -->|yes| DurableFiles["Stream originals or one PDF page range at a time to shared STATE_DIR/upload_jobs/process_id<br/>opaque 0600 names; fsync every file and directory<br/>staging_node_id is provenance only"]
    DurableFiles -->|write or fsync error| StageAbort
    DurableFiles --> QueueTx{"Atomically recheck account gate<br/>and move STAGING to QUEUED?"}
    QueueTx -->|no| StageAbort
    QueueTx -->|yes| Reserve{"Under one PostgreSQL advisory capacity lock,<br/>fewer than MAX_UPLOAD_WORKERS live reservations on this worker_node?<br/>Reserve row with FOR UPDATE SKIP LOCKED and increment attempt_count"}
    Reserve -->|yes| Dispatch["Spawn a clean multiprocessing child<br/>with process_id, lease owner, and worker node id"]
    Reserve -->|no; capacity full, backoff pending, or row unavailable| Response["Return process_id; supervisor retains queued work"]
    Dispatch --> Response
    Dispatch -->|child started| Activate{"Child activates the matching<br/>owner reservation?"}
    Dispatch -->|start failed| ReleaseReservation["Record worker_spawn_failed;<br/>queue with exponential backoff, or fail at exhausted budget<br/>only when every item is still pending"]
    ReleaseReservation --> Response
    ReleaseReservation --> Supervisor
    Activate -->|no| ExitDuplicate["Exit without reading staged data"]
    Activate -->|yes| Worker["process_files_worker<br/>reservation already carries worker_node_id; child records PID and PID-start fingerprint<br/>heartbeat thread renews the lease during long calls<br/>node id scopes signalling, never staging access"]
    Supervisor["Every API worker: upload supervisor"] --> ExpireStage["Expire at most 100 crash-left STAGING tickets<br/>using PostgreSQL time and FOR UPDATE SKIP LOCKED<br/>atomically mark them failed"]
    ExpireStage --> Recover{"For at most 100 expired worker leases:<br/>what recovery is required?"}
    Recover -->|ordinary crash| LeaseBackoff["Retain any PROCESSING checkpoint as an ambiguity marker;<br/>queue after exponential backoff"]
    Recover -->|cancellation pending| CancelReplay["Retain the processing checkpoint;<br/>queue after backoff for safe replay and cancellation finalization"]
    Recover -->|budget exhausted, not cancelling| ExhaustedQueue["Retain checkpoints and staging;<br/>queue one immediate reconciliation incarnation"]
    LeaseBackoff --> Reconcile["Erase and acknowledge terminal staging;<br/>remove legacy no-row staging after one-hour grace"]
    CancelReplay --> Reconcile
    ExhaustedQueue --> Reconcile
    Reconcile --> ReserveAny["Reserve one queued row<br/>with FOR UPDATE SKIP LOCKED"]
    ReserveAny --> RecoverDispatch["Spawn one recovery child"]
    RecoverDispatch --> Activate
    Worker -.-> ToolBoundary["Tool boundary<br/>No executor or query tools run during ingest<br/>GraphRAGService writes; it is not the GraphRAG retrieval tool"]
    Worker --> RecoveryMetadata["Recovery preflight: from durable checkpoints,<br/>idempotently finalize every logical document whose sibling parts are terminal<br/>in a bounded child that rechecks the lease under tenant_lock"]
    RecoveryMetadata -->|failure or timeout| RecoveryWait
    RecoveryMetadata -->|success| RetryBudget{"attempt_count at most max_attempts?"}
    RetryBudget -->|no| ExhaustedReconcile["In separate killable lease-fenced children under tenant_lock,<br/>compensate only partial logical documents with unfinalized effects,<br/>graph-first and then Elasticsearch"]
    ExhaustedReconcile -->|compensation fails or times out| RecoveryWait
    ExhaustedReconcile -->|all compensation succeeds| ExhaustedPublish["Preserve finalized documents; mark compensated and untouched<br/>unfinished items failed with zero counters; publish FAILED,<br/>or CANCELLED when cancellation was requested"]
    ExhaustedPublish --> ReleaseLock
    RetryBudget -->|yes| RecoveredCheckpoint{"Any retained PROCESSING checkpoint?"}
    RecoveredCheckpoint -->|no| EntryCancel{"Cancellation already requested?"}
    EntryCancel -->|yes| CancelReconcile
    EntryCancel -->|no| NextItem{"Next staged work item?"}
    RecoveredCheckpoint -->|yes| RecoveredCompensate["In bounded lease-fenced children under tenant_lock,<br/>compensate each affected logical document<br/>graph-first and then Elasticsearch"]
    RecoveredCompensate -->|failure or timeout| RecoveryWait
    RecoveredCompensate -->|success; cancellation requested| CancelledTerminal
    RecoveredCompensate -->|success; continue| RecoveredReset["Reset every item of each compensated document to pending;<br/>replay in this worker incarnation"]
    RecoveredReset --> NextItem

    NextItem -->|yes; shutdown already set| Requeued
    NextItem -->|yes; continue| Account{"Owned lease still current and user row<br/>exists with deletion_requested false?"}
    Account -->|no| Gone["Account is unavailable; do not recreate stores or rows"]
    Account -->|yes| ItemClaim{"Atomically claim the next pending item<br/>under the same lease?"}
    ItemClaim -->|no; cancelled| CancelStop["Stop before starting another item"]
    ItemClaim -->|no; lease lost| RecoveryWait
    CancelStop --> Reload
    ItemClaim -->|yes| Ready["Append ordered upload_finish event<br/>and renew the worker lease"]
    Ready --> StagedPath["Resolve opaque staged path<br/>and verify it remains inside the job directory"]
    StagedPath -->|missing or invalid before isolation| FailedCheckpoint["Commit a failed item checkpoint and safe run-id error;<br/>no downstream compensation is needed"]
    FailedCheckpoint --> ReadyMetadata
    StagedPath --> Isolate["Spawn one clean item subprocess<br/>outer worker enforces one hard PARSE_TIMEOUT_S wall-clock deadline<br/>and polls cancellation, shutdown, and lease ownership"]
    Isolate --> TenantLock["Item subprocess acquires tenant_lock user_id<br/>and rechecks the active lease and account deletion gate"]
    TenantLock -->|lease lost| RecoveryWait
    TenantLock -->|account unavailable| KillItem
    TenantLock -->|live| Run["execute_insert_process_sync<br/>creates one IngestRun and asyncio event loop"]
    Run -->|uncaught item exception| KillItem
    Isolate -->|deadline, cancellation, shutdown, lease loss,<br/>spawn failure, EOF, abnormal exit, or child error| KillItem["Hard-kill when needed and always join the item subprocess;<br/>ambiguous mutations require fenced compensation"]
    KillItem --> Compensate["In a separate bounded child, recheck the active lease under tenant_lock,<br/>then delete graph contributions first and Elasticsearch second;<br/>graph refusal leaves search data untouched"]
    Compensate -->|lease lost, failure, or timeout| RecoveryWait["Do not reset checkpoints or publish terminal state;<br/>retain staging for durable recovery"]
    RecoveryWait -.-> Supervisor
    Compensate -->|success| LeaseOwned{"Worker still owns the lease?"}
    LeaseOwned -->|no| RecoveryWait
    LeaseOwned -->|yes| CancelAfterItem{"Cancellation or controlled shutdown?"}
    CancelAfterItem -->|cancellation| CancelledTerminal
    CancelAfterItem -->|shutdown| ShutdownRetry["Reset the compensated logical document;<br/>queue immediately and refund this reservation's attempt"]
    ShutdownRetry -.-> Supervisor
    CancelAfterItem -->|neither| RetryItem{"Attempt budget exhausted?"}
    RetryItem -->|no| Backoff["Reset every item for that logical document to pending<br/>record safe failure class; queue with exponential backoff"]
    Backoff -.-> Supervisor
    RetryItem -->|yes| DeadLetter["Preserve finalized documents; mark the compensated document<br/>and untouched future items failed with zero counters;<br/>commit terminal FAILED"]
    DeadLetter --> ReleaseLock

    Run --> ParserSlot["Service slot PARSER<br/>build_parser from PARSER setting"]
    ParserSlot --> ParserChoice{"Configured parser"}
    ParserChoice -->|deepdoc| DeepDoc["DeepDocParser<br/>PDF or DOCX<br/>OCR, layout, tables, reading order and chunk merge"]
    ParserChoice -->|vlm| Vlm["VLMParser<br/>PDF or PNG/JPG/JPEG<br/>one PARSE_TIMEOUT_S document budget"]
    DeepDoc --> ParserResult{"Parser completed?"}
    Vlm --> VlmSource{"PDF?"}
    VlmSource -->|yes| VlmCount{"Page count is 1..2000?"}
    VlmCount -->|no| ParserResult
    VlmCount -->|yes| VlmPage["Open only the next page<br/>check render pixels and embedded-raster pixels<br/>render and enforce encoded-page bytes"]
    VlmPage --> VlmCall["Directly await qwen-vl-plus<br/>preserve source page number; failure becomes placeholder"]
    VlmCall --> VlmRelease["Release encoded page and native document objects"]
    VlmRelease -->|more pages| VlmPage
    VlmRelease -->|done| ParserResult
    VlmSource -->|image| VlmImage["Check source bytes and decoded pixels<br/>verify and encode one image"]
    VlmImage --> VlmCall
    ParserResult -->|child error or no result| KillItem
    ParserResult -->|yes| Legacy["Validated ParsedChunk list"]
    Legacy --> HasText{"Any chunks?"}
    HasText -->|no| EmptyRun["Return IngestRun.error<br/>no text could be extracted"]
    HasText -->|yes| ChunkIds["Assign SHA-256 chunk ID over 8-byte-length-prefixed UTF-8<br/>tenant, NFC logical document, sequence:part_name,<br/>zero-based parser ordinal, and content"]
    ChunkIds --> Normalize["Retain typed chunks<br/>normalized positions and base64 images"]

    Normalize --> GraphRepo["Vectorstore graph component inside the item subprocess lock<br/>construct GraphRAGService and GraphRepository<br/>load per-user graph and vector files directly"]
    GraphRepo --> GraphNext{"Next typed ParsedChunk?"}
    GraphNext -->|yes| Eligible{"Text present and at least 10 whitespace words?"}
    Eligible -->|no| GraphSkip["Skip graph extraction for this chunk"]
    GraphSkip --> GraphNext
    Eligible -->|yes| Prompt["GraphRAG service builds the LightRAG prompt"]
    Prompt --> Ner["LLM provider call<br/>model is ner_model from CHAT_MODEL_TURBO<br/>one call per eligible chunk"]
    Ner --> Records["Parse entity and relationship records<br/>attach chunk ID and original filename"]
    Ner -->|provider error| GraphFailed
    Records -->|parse error| GraphFailed
    Records --> GraphNext
    GraphNext -->|no| GroupGraph["Group entities by name<br/>group relations by source-target pair"]
    GroupGraph --> ReplayGuard["Maintain canonical contributions_json keyed by atomic source chunk ID;<br/>derive lossless document_names_json for vectors; retain stored edge orientation;<br/>merge replay leaves aggregates unchanged and identical VDB upserts skip embedding"]
    ReplayGuard --> MergeNodes["For each entity, sequentially merge<br/>and upsert its NetworkX node"]
    MergeNodes -->|merge error| GraphFailed
    MergeNodes --> NodeVector["Upsert new or changed node nanoVectorDB record<br/>EMBEDDING_PROVIDER embeds its aggregate content<br/>embedding failure propagates without appending a partial row"]
    NodeVector -->|error| GraphFailed
    NodeVector --> MergeEdges["For each relationship, sequentially merge<br/>and upsert its NetworkX edge and missing nodes"]
    MergeEdges -->|merge error| GraphFailed
    MergeEdges --> EdgeVector["Upsert new or changed edge nanoVectorDB record<br/>EMBEDDING_PROVIDER embeds its aggregate content<br/>embedding failure propagates without appending a partial row"]
    EdgeVector -->|error| GraphFailed
    EdgeVector --> SaveNode["Temp-write, fsync, and atomically replace vdb_nodes_USER_ID.json"]
    SaveNode --> SaveEdge["Atomically replace and fsync vdb_edges_USER_ID.json"]
    SaveEdge --> SaveGraph["Atomically replace and fsync graph_USER_ID.graphml and parent directory"]
    SaveNode -->|publication error| GraphAmbiguous["Raise GraphPersistenceError<br/>the three-file snapshot may have a durable prefix"]
    SaveEdge -->|publication error| GraphAmbiguous
    SaveGraph -->|publication error| GraphAmbiguous
    GraphAmbiguous --> KillItem
    SaveGraph --> GraphOutcome{"Graph stage outcome"}
    GraphOutcome -->|success| GraphCounts["Set IngestRun entity_count and relation_count"]
    GraphOutcome -->|false result| GraphEmpty["Set IngestRun.error<br/>graph extraction produced no entities or relations"]
    GraphRepo -->|load error before publication| GraphFailed["Set IngestRun.error to graph extraction failed<br/>retain the item tenant lock through Elasticsearch"]

    GraphCounts --> ChunkSlot["Service slot CHUNKSTORE<br/>build_chunkstore from CHUNKSTORE setting<br/>live implementation is ElasticsearchChunkStore"]
    GraphEmpty --> ChunkSlot
    GraphFailed --> ChunkSlot
    ChunkSlot --> IndexNext{"Next normalized ParsedChunk?"}
    IndexNext -->|yes| TextEmbed["EMBEDDING_PROVIDER embeds chunk content"]
    TextEmbed --> Analyze["ANALYZER fills normal and fine search terms<br/>and document-name terms when absent"]
    Analyze --> Shape["Shape stable Elasticsearch document<br/>tenant chunk ID, document ID, metadata, images and q_dimension_vec"]
    Shape --> IndexNext
    IndexNext -->|no| Bulk["database.ElasticsearchStore.index<br/>bulk upsert to user_id index by deterministic chunk ID"]
    Bulk --> IndexResult{"Bulk call completed?"}
    IndexResult -->|exception or child failure| KillItem
    IndexResult -->|yes| ShortWrite{"Accepted every prepared chunk?"}
    ShortWrite -->|no| ShortError["Append short-index error to IngestRun<br/>keep accepted chunk count"]
    ShortWrite -->|yes| RunDone["Return IngestRun"]
    ShortError --> RunDone

    EmptyRun --> ItemCheckpoint["Commit item terminal checkpoint<br/>indexed count, elapsed time and safe error<br/>recompute durable job counters"]
    RunDone --> ItemCheckpoint
    ItemCheckpoint --> ReadyMetadata{"Are all sibling parts of this logical document terminal?"}
    ReadyMetadata -->|no| NextItem
    ReadyMetadata -->|yes| Pg["Immediately spawn isolated metadata finalization:<br/>retry-safe PostgreSQL knowledgebases upsert from durable checkpoints<br/>after rechecking the lease under tenant_lock; hard PARSE_TIMEOUT_S deadline"]
    Pg --> PgResult{"Metadata write completed?"}
    PgResult -->|no| PgRetry["Leave job processing and retain staging;<br/>expired-lease recovery repeats metadata finalization<br/>without replaying terminal items"]
    PgRetry -.-> Supervisor
    PgResult -->|yes| NextItem
    NextItem -->|no| Reload["Reload all durable item checkpoints and repeat the<br/>idempotent metadata sweep to close any checkpoint/upsert gap"]
    Reload --> FinalMetadata["For each fully terminal logical document,<br/>run bounded retry-safe metadata finalization"]
    FinalMetadata -->|failure or timeout| PgRetry
    FinalMetadata -->|success| CancelDecision
    Gone --> SkipRows["Do not recreate PostgreSQL document rows"]
    SkipRows --> CancelDecision
    CancelDecision{"Cancellation requested?"} -->|yes| CancelReconcile["In a lease-fenced child, compensate any partial logical document<br/>graph-first and then Elasticsearch; preserve finalized documents"]
    CancelReconcile -->|failure or timeout| RecoveryWait
    CancelReconcile -->|success| CancelledTerminal["Mark compensated and untouched unfinished items failed with zero counters;<br/>atomically publish CANCELLED; no PROCESSING checkpoint remains"]
    CancelDecision -->|no| ShutdownDecision{"Worker shutdown signal?"}
    ShutdownDecision -->|yes, at a safe item boundary| Requeued["Return owned job to queued; refund this reservation's attempt;<br/>retain staging and unfinished items"]
    Requeued -.-> Supervisor
    ShutdownDecision -->|no| Terminal{"Any recorded error or unavailable account?"}
    Terminal -->|yes| Failed["Atomically set upload_jobs.status failed<br/>append terminal error event<br/>parsed chunks may still be searchable"]
    Terminal -->|no| Unfinished{"Any unfinished item?"}
    Unfinished -->|yes| Requeued
    Unfinished -->|no| Completed["Atomically set upload_jobs.status completed<br/>append terminal complete event"]
    Failed --> ReleaseLock["Stop heartbeat and finish this worker incarnation"]
    Completed --> ReleaseLock
    CancelledTerminal --> ReleaseLock
    PgRetry --> ReleaseLock
    Requeued --> ReleaseLock
    ReleaseLock --> TerminalState{"Job terminal?"}
    TerminalState -->|yes| RemoveStage["Worker or any supervisor erases shared staged bytes"]
    RemoveStage --> StageAck["Commit staging_cleaned acknowledgement<br/>retain observable job, items, and journal"]
    TerminalState -->|no| KeepStage["Retain shared staging for recovery"]
    KeepStage -.-> Supervisor

    Worker -.-> Progress["Every progress callback validates and renews the lease<br/>appends parse, graph, encode, database, or error event<br/>cancellation is polled during isolated work and at item boundaries"]
    Progress --> Journal["PostgreSQL upload_job_events<br/>monotonic event id is the SSE cursor"]
    Failed --> Observe["Owner-only progress APIs"]
    Completed --> Observe
    CancelledTerminal --> Observe
    StageAck --> Observe
    CancelStop --> Observe
    Response --> Observe
    Observe --> Snapshot["GET /process-status<br/>reads PostgreSQL from any API worker<br/>includes attempt_count, max_attempts, last_failure_class, next_attempt_at;<br/>excludes lease, worker identity, and staged paths"]
    Observe --> Sse["GET /get-process-progress<br/>polls events after the last event id<br/>streams every event once per connection"]
    Observe --> Kill["POST /kill-processing<br/>persist cancellation before any signal"]
    Kill --> CancelKind{"STAGING, QUEUED, or unactivated<br/>and every item still PENDING?"}
    CancelKind -->|yes| CancelledNow["Commit terminal cancelled immediately"]
    CancelledNow --> RemoveStage
    CancelKind -->|no| CancelPending["Keep cancellation pending; job is active or recovery-queued<br/>signal only matching worker_node_id plus PID-start fingerprint<br/>otherwise recover after lease expiry"]
    CancelPending -.-> Supervisor
    Observe --> Cleanup["POST /cleanup-processes<br/>delete only terminal rows with staging_cleaned<br/>cascade the item checkpoints and event journal"]

    DeleteRequest["DELETE /delete_file or /delete-document<br/>authenticate caller and canonicalize filename to NFC"] --> DeleteLock["Acquire tenant_lock with 30-second bound"]
    DeleteLock --> DeleteExists{"Account still active, and how many owned rows<br/>match after canonical NFC normalization?"}
    DeleteExists -->|none| DeleteMissing["Return 404 or compatibility missing result"]
    DeleteExists -->|multiple legacy spellings| DeleteAmbiguous["Return 409; mutate no store"]
    DeleteExists -->|exactly one| DeleteLedger["Use its stored spelling for every store;<br/>do graph records have trustworthy contribution provenance?"]
    DeleteLedger -->|target-bearing legacy aggregate or shared orphan vector| DeleteLegacy["Fail closed; retain PostgreSQL row and Elasticsearch chunks<br/>reindex graph before retry"]
    DeleteLedger -->|yes| DeleteGraph["Remove target contributions; delete empty records;<br/>recompute retained node/edge aggregates and degrees;<br/>preserve edge orientation; update/delete and re-embed exact VDB records"]
    DeleteGraph --> DeleteSaveNode["Temp-write, fsync, and atomically replace the node-vector file"]
    DeleteSaveNode --> DeleteSaveEdge["Temp-write, fsync, and atomically replace the edge-vector file"]
    DeleteSaveEdge --> DeleteSaveGraph["Temp-write, fsync, and atomically replace GraphML;<br/>fsync each parent directory"]
    DeleteSaveGraph --> DeleteES["Delete document chunks from the tenant Elasticsearch index"]
    DeleteES --> DeletePG["Delete PostgreSQL document row last"]
    DeleteLock -->|timeout| DeleteBusy["Return 409; retain row for retry"]
    DeleteGraph -->|error| DeleteFailure["Propagate stable error; retain row for retry"]
    DeleteSaveNode -->|error| DeleteFailure
    DeleteSaveEdge -->|error| DeleteFailure
    DeleteSaveGraph -->|error| DeleteFailure
    DeleteES -->|error| DeleteFailure
```

### **Step 1: File Upload & Verification**

- `UploadBodyLimitMiddleware` covers both `/start-processing` and
  `/add_context`. It rejects a declared raw body above
  `MAX_UPLOAD_TOTAL_BYTES` plus 1 MiB of multipart framing without reading it;
  for a missing or dishonest `Content-Length`, its receive wrapper counts raw
  chunks and aborts multipart parsing at the same cap. Each API worker admits
  `MAX_CONCURRENT_UPLOADS` bodies (two by default); overflow returns 429 with
  `Retry-After` without reading the body.
- Both routes authenticate and verify any session before `request.form()`; no
  `File(...)` dependency spools an unauthenticated body. They accept only the
  `files` multipart field and enforce `MAX_UPLOAD_FILES` (10),
  `MAX_UPLOAD_BYTES` per file, and `MAX_UPLOAD_TOTAL_BYTES` across file content;
  both byte limits default to 500 MiB. They scan and rewind request spools in
  1 MiB pieces without joining file bytes. Context names are safe path
  components of 1-255 characters whose sanitized form is at most 255 UTF-8
  bytes; knowledge-name rules are below.
- Knowledge-document names are Unicode NFC, 1-255 characters, NUL-free,
  free of leading/trailing whitespace, and cannot contain the graph field
  separator text ` | `.
  A duplicate canonical name in one batch returns 409 before a ticket exists.
  The staging transaction serializes canonical `(user, name)` admission and
  also returns 409 when that name already exists or has an active job. Uploads
  are immutable: replacement requires successful deletion first.
- It completes an immutable `IngestJob` and calls
  `pipeline.ingest.start_processing()` off the event loop.
- The pipeline first commits an owner-bearing, non-dispatchable `STAGING` job,
  its item checkpoints, and initial events. Under the per-tenant staging lock,
  it rechecks the ticket and account gate, streams and fsyncs opaque files to
  shared `STATE_DIR`, then atomically moves `STAGING` to `QUEUED`; PostgreSQL
  never stores raw file bytes.
- The API parent reserves the committed row before spawning; the child only
  activates that reservation. A PostgreSQL advisory lock makes the live
  reservation count linearizable across API replicas with the same
  `VISIONAGENT_NODE_ID`, enforcing `MAX_UPLOAD_WORKERS` (two by default).
  `FOR UPDATE SKIP LOCKED` prevents duplicate dispatch.
- Every reservation increments `attempt_count`. A spawn failure has no
  downstream side effect and either backs off or fails at
  `MAX_UPLOAD_ATTEMPTS` (three by default) only for a pristine, all-pending
  ticket. A spawn failure for a recovery-bearing job refunds the reservation
  and leaves it queued until a worker can reconcile safely. A timeout or other
  isolated-item failure first runs bounded graph-first, then Elasticsearch
  compensation; only confirmed cleanup may reset the logical document and back
  off or publish terminal failure. Failed compensation leaves the lease, checkpoints, and
  staging intact for durable recovery. Ordinary lease expiry retains any
  `PROCESSING` checkpoint, queues after the same exponential backoff, and makes
  the replacement compensate that logical document before reset and replay. An
  exhausted lease queues one immediate reconciliation incarnation. That worker
  preserves already-finalized documents, compensates only partial documents
  with unfinalized effects, and zeroes compensated and untouched unfinished
  items before terminal publication. A safe-boundary worker requeue refunds its
  attempt.
- Each document item runs in a spawned subprocess. The parent enforces one hard
  `PARSE_TIMEOUT_S` wall-clock budget across parsing, graph extraction,
  embeddings, and Elasticsearch, and kills and joins that subprocess on
  timeout, cancellation, shutdown, or lease loss. Each item, metadata, and
  compensation child rechecks the active `(process_id, owner)` lease while
  holding `tenant_lock` before mutation. PostgreSQL metadata is finalized in
  bounded isolation immediately after the last sibling part of each logical
  document; every recovery worker first repeats the idempotent finalization for
  already-complete documents.
- `STAGING`, `QUEUED`, or unactivated cancellation is immediate only while
  every item is still `PENDING`. Active or recovery-bearing cancellation is
  persisted, the matching local worker is
  signalled only when node ID, PID, and process-start fingerprint agree, and
  isolated work is killed and compensated before terminal acknowledgement.
  Cancellation during live item execution becomes `cancelled` after confirmed
  compensation regardless of its attempt count.
- Terminal status commits before shared staging is erased and
  `staging_cleaned` is recorded. Only then may owner cleanup delete the job,
  item checkpoints, and journal. A crash-left `STAGING` ticket expires to
  `failed` using database time and is erased; legacy staging with no job row is
  removed after a one-hour grace period.

```
1.2 PDF SPLITTING LOGIC
  - Apply only to PDF uploads.
  - Read all pages when count <= 5; otherwise sample 5 random pages.
  - estimated_chars = total_pages * maximum sampled-page character count.
  - Split when estimated_chars > CHAR_LIMIT (250,000).
  - num_parts = floor(estimated_chars / CHAR_LIMIT) + 1.
  - pages_per_part = ceil(total_pages / num_parts).
  - Emit every source page exactly once with pypdf.
  - Keep the original filename for graph, index, and PostgreSQL metadata.
```

### **Step 2: Document Parsing & Chunking**

`PARSER` selects `deepdoc` (default) or `vlm`. Both
implement `DocumentParser` and return `ParsedChunk[]`.

```
DeepDoc PDF
  OCR/layout/tables/positions
  -> outline or title/bullet section levels
  -> page/top/left reading order
  -> merge below 32 tokens, or below 1,024 in the same group
  -> tokenize_table() + tokenize_chunks()

DeepDoc DOCX
  paragraphs/tables -> tokenized chunks

VLM PDF/image
  page count check -> render one page -> pixel/encoded-byte checks
  -> qwen-vl-plus transcription -> release page -> next page
  -> one nonblank chunk carrying its original source-page number

ParsedChunk
  id, content, content_tokens, content_tokens_fine, doc_name_tokens,
  page_nums, top_offsets, ref_images, image
```

DeepDoc declares PDF and DOCX support. VLM declares PDF, PNG, JPG, and JPEG
support and requires DashScope. Empty DeepDoc chunks fail parsing. VLM applies
one `PARSE_TIMEOUT_S` budget to the document, accepts at most 2,000 PDF pages,
bounds rendered pages at 40 million pixels, embedded source rasters at 100
million pixels, and encoded pages/source images at 25 MiB. At most one rendered
PDF page is live; cancellation stops before the next render and closes the
provider client. A failed page call becomes an `[Error extracting text]` chunk;
a blank page response is omitted without shifting later page numbers.

### **Step 3: Two Processing Stages, in Order**

The same chunks run through GraphRAG first and NormalRAG second. PostgreSQL
metadata is finalized per logical document immediately after its last sibling
part, with the same idempotent sweep at recovery entry.

#### **GraphRAG Stage** (runs first, under the tenant lock)

Chunks below ten whitespace-separated words are skipped. Other chunks receive
one serial extraction call using the NER model configured by
`CHAT_MODEL_TURBO`.

```python
# the isolated item subprocess already owns tenant_lock(user_id)
run = execute_insert_process_sync(
    staged_path,
    original_filename,
    user_id,
    progress,
    run_id=job.identity.run_id,
    tenant_lock_held=True,
    part_identity=f"{item.sequence}:{item.part_name}",
)
```

The LightRAG prompt yields entities, relationships, descriptions, keywords,
and weights; the pipeline attaches one source chunk ID and logical document
name to each extracted record. Fresh aggregates persist canonical
`contributions_json` in GraphML; derived vectors carry canonical
`document_names_json` arrays of exact document names. Re-extraction still calls the LLM, but a
replayed source does not change the aggregate and an identical vector upsert
does not call the embedder. New contributions retain the stored undirected-edge
orientation and recompute the aggregate and its vector. Provider, extraction,
merge, or embedding failures before publication become `graph extraction failed`; no zero-vector
fallback or partial vector row is appended, and Elasticsearch indexing
continues so the document remains searchable while its item and batch fail.
Each graph file is temp-written, fsynced, atomically replaced, and its parent
directory fsynced. A publication failure raises `GraphPersistenceError`: the
isolated item is compensated and retried because a crash can leave a durable
prefix of the node-vector, edge-vector, GraphML save order. Replay and
compensation converge it.
Entity and relation extraction, merge, and upsert run serially inside the
tenant-locked ingestion workflow.

#### **NormalRAG Stage** (runs second)

`ElasticsearchChunkStore` owns analysis, embedding, document shape,
and the database write.

```
chunk_id = SHA256(
  length_prefix(user_id) + length_prefix(NFC logical filename) +
  length_prefix(sequence + ":" + part_name) + length_prefix(zero-based ordinal) +
  length_prefix(content)
)
doc_id   = xxhash64(original_filename + user_id)

Stored fields:
  id, chunk_id, content_with_weight, content_ltks, content_sm_ltks,
  important_kwd, important_tks, question_kwd, question_tks,
  create_time, create_timestamp_flt, page_num, top_int, kb_id,
  docnm_tks, doc_id, docnm, ref_images, image, q_<dimension>_vec
```

Each `length_prefix` is an unsigned 8-byte big-endian byte length followed by
the field's UTF-8 bytes. The public durable worker therefore reproduces the
same ID for a retry while separating documents, parts, and ordinals that carry
identical text.

`content_with_weight` and `q_<dimension>_vec` are persistent
contracts. A short bulk write is an error. Graph failure does not prevent
indexing, but the batch is marked failed.

## Database Architecture

### **Five Persistence Targets:**

This section counts five logical stores across three storage systems:

1. **Elasticsearch** — chunk text, analyzer fields, embeddings, positions,
   and images.
2. **PostgreSQL** — users, sessions, messages, document metadata, and durable
   upload job, item-checkpoint, and event-journal rows.
3. **nanoVectorDB_node** — entity vectors and source metadata.
4. **nanoVectorDB_edge** — relationship vectors and source metadata.
5. **NetworkX** — the GraphML entity/relationship graph.

Items 3-5 are files under `GRAPH_DIR`. Canonical chunk text remains
in Elasticsearch.

## LightRAG Integration Details

`GraphRAGService` uses the LightRAG prompt and merge helpers with
project-owned persistence and failure handling.

### **Entity Processing Pipeline:**

```python
for entity_name, records in entity_groups.items():
    entity = await _merge_nodes_then_upsert(
        entity_name, records, knowledge_graph, config, None, None, None
    )
    await node_vdb.upsert({
        compute_mdhash_id(entity_name, prefix="ent-"):
            _node_vector_metadata(entity_name, entity)
    })
```

The helper carries exact `document_names_json` provenance and embeds the merged
entity name and description when the aggregate is new or changed. A pure
source-ID replay performs no vector write; an embedding failure propagates.

#### **4.2.2 [lightrag/operate.py] _merge_nodes_then_upsert()** - Entities are merged with existing ones, descriptions are combined, and the final entity is inserted into the knowledge graph.

`_merge_nodes_then_upsert()` is implemented in
`service/vectorstore/graphstore/lightrag_utils.py` and returns the merged
metadata used by the vector upsert above.

### **Relationship Processing Pipeline:**

```python
for edge_key, records in relationship_groups.items():
    edge = await _merge_edges_then_upsert(
        edge_key[0], edge_key[1], records,
        knowledge_graph, config, None, None, None, []
    )
    if edge:
        await edge_vdb.upsert({
            edge_vector_id(edge["src_id"], edge["tgt_id"]):
                _edge_vector_metadata(edge)
        })
```

The helper carries exact `document_names_json` provenance and embeds both
endpoints, keywords, and description when the aggregate is new or changed. A
pure source-ID replay performs no vector write; an embedding failure propagates.

#### **4.3.1 [lightrag/operate.py] _merge_edges_then_upsert()** - Relationships are merged with existing ones, missing entities are created, and the final relationship is inserted into the knowledge graph.

`_merge_edges_then_upsert()` lives beside the node helper in
`service/vectorstore/graphstore/lightrag_utils.py`; its returned metadata feeds
the vector upsert above.

## How to Use

### **1. Start the Backend:**

```bash
make serve
```

PostgreSQL, Elasticsearch, model settings, and selected provider credentials
must be configured.

### **2. Upload Files:**

```bash
curl -X POST "http://localhost:8000/start-processing" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "files=@your_document.pdf"
```

The response contains `process_id`.

### **3. Monitor Processing:**

Use `GET /get-process-progress/<process_id>` for SSE or
`GET /process-status/<process_id>` for a snapshot. Both enforce
ownership.

### **4. GraphRAG Output (Expected):**

```
starting graph extraction; run_id=<id> chunk_count=<n>
GraphRAG skipped short chunk; chunk_id=<id> words=<n> minimum=10
graph extraction completed; run_id=<id> entity_count=<n> relation_count=<n>
```

While retained, process status reports the terminal batch result;
`knowledgebases.error` persists each document's failure.

## Key Features

### **LightRAG Integration:**

- Extraction, merge, graph/vector persistence, and tenant-scoped progress
  reporting are active.

### **Runtime Invariants:**

- Parser output and Elasticsearch fields remain stable; graph writes remain
  tenant-locked; partial failures remain visible.

### **Architecture Properties:**

- Stable chunk IDs connect isolated text and graph stores without ambient
  request state; pre-publication graph failures do not discard searchable text,
  while ambiguous publication failures enter compensation and bounded retry.

## Testing

Use deterministic pipeline tests for orchestration and the live journey for
real stores and providers.

### **1. File Upload Test:**

```bash
backend/.venv/bin/python -m pytest test/pipeline/test_ingest_pipeline.py
make test-journey
```

These cover job propagation, order, partial outcomes, page coverage, upload
limits, graph locking, and one live document journey.

### **2. File Deletion Test:**

```bash
backend/.venv/bin/python -m pytest \
  test/pipeline/test_ingest_pipeline.py -k delete
```

Deletion runs graph, Elasticsearch, then PostgreSQL. Failures retain the row
for retry. Graph deletion removes the document's lossless ledger entries,
deletes empty nodes/edges and their exact vectors, recomputes retained node
type/description/source/doc and edge description/keywords/weight/source/doc,
and re-embeds changed aggregates while preserving stored edge orientation.
A target-bearing legacy aggregate or shared orphan vector without provenance
fails closed before persistence and requires graph reindexing.

### **3. Data Inspection:**

- `backend/var/graph/vdb_nodes_<user_id>.json`
- `backend/var/graph/vdb_edges_<user_id>.json`
- `backend/var/graph/graph_<user_id>.graphml`

`GRAPH_DIR` overrides the directory.

### **4. System Requirements:**

- Python 3.11 and the installed backend package.
- PostgreSQL and Elasticsearch for live ingestion.
- Production replicas share PostgreSQL, a durable `STATE_DIR`, and any
  independently overridden `GRAPH_DIR`. Set a unique `VISIONAGENT_NODE_ID` per
  runtime node; the fallback hashes host and machine identity, and the value
  only fences local PID signalling.
- Required model settings and selected-provider credentials.
- A production JWT secret of at least 32 characters; `make dev` and
  `make serve` explicitly use development mode.

## Acceptance Criteria

### **Implemented Paths:**

- Supported parser inputs traverse the authorized upload, PostgreSQL metadata,
  graph, and Elasticsearch paths.

### **Runtime Guarantees:**

- Server limits and page coverage are enforced; store failures remain visible;
  graph replacement remains atomic and tenant-locked.

### **Supported Capabilities:**

- Runtime selection covers both parsers, per-user stores, and retryable
  cross-store deletion.

### **🎯 System Output Example:**

```
process-status.status: failed
process-status.total_chunks_inserted: 12
knowledgebases.error: graph extraction failed
```

A pre-publication graph failure may coexist with searchable Elasticsearch
chunks; a graph publication failure is compensated before retry.

### **Production Acceptance:**

Production acceptance requires stale-import, architecture, unit, integration,
E2E, journey, and golden checks in the target environment. Every API/worker
host must share PostgreSQL, `STATE_DIR`, and any independently overridden
`GRAPH_DIR`; replicas sharing a runtime node must also share
`VISIONAGENT_NODE_ID` so its worker cap is global there. VLM page-error
placeholders remain searchable, and pre-ledger graphs must be reindexed before
deleting a document that participates in a legacy aggregate.
