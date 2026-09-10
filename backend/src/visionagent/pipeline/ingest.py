"""The ingest pipeline: a document from upload to indexed and graphed.

parse -> chunk -> extract entities and relations into the graph -> embed and
index in Elasticsearch -> record the document in Postgres, reporting progress
at each stage.

The upload route reads multipart data and mints an ``IngestJob``;
``start_processing`` owns all subsequent orchestration.

Writes go through the vectorstore slot's ChunkStore contract, never through
the vendored Elasticsearch client: ingest decides *what* is stored, the slot
decides how a document is shaped, and database/ owns the connection.

Every file yields an ``IngestRun``. A document whose graph provider or
extraction failed remains indexed and searchable but is reported as partial.
A graph persistence failure has ambiguous commit state, so it aborts into
durable compensation and retry.

Upload jobs, item checkpoints and progress events are durable PostgreSQL rows.
Staged bytes live under the configured state directory.  Workers claim a
lease, heartbeat it and resume only unfinished items after a process or API
restart; no API-worker global is authoritative for status or cancellation.
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import io
import logging
import math
import multiprocessing
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import unicodedata
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

from visionagent.config.settings import settings
from visionagent.database.elasticsearch import chunks_of_document
from visionagent.database.graph import GRAPH_FIELD_SEP, GraphPersistenceError, tenant_lock
from visionagent.database.knowledgebase_operations import insert_knowledgebase
from visionagent.database.postgres.repositories import (
    KnowledgeBaseRepository,
    UserRepository,
)
from visionagent.database.postgres.upload_jobs import (
    UploadAccountUnavailable as UploadAccountUnavailable,
)
from visionagent.database.postgres.upload_jobs import (
    UploadDocumentConflict as UploadDocumentConflict,
)
from visionagent.database.postgres.upload_jobs import (
    UploadJobRepository,
    UploadJobStore,
)
from visionagent.models import (
    IngestJob,
    IngestRun,
    ParsedChunk,
    StagedUpload,
    UploadItemState,
    UploadJobRecord,
    UploadJobState,
    UploadProgressEvent,
)
from visionagent.service.parsers import build_parser
from visionagent.service.vectorstore import ChunkStore, GraphRAGService, build_chunkstore

_ITEM_PROCESS_CONTEXT = multiprocessing.get_context("spawn")

logger = logging.getLogger(__name__)

_STORE: ChunkStore | None = None
_STORE_LOCK = threading.Lock()
_JOB_REPOSITORY: UploadJobStore | None = None
_JOB_REPOSITORY_LOCK = threading.Lock()
# Forking an ASGI worker after it has started threads can inherit locked
# runtimes, DB pools and provider clients. Spawn imports a clean interpreter
# and receives only the durable process/owner identifiers.
_UPLOAD_PROCESS_FACTORY: Any = multiprocessing.get_context("spawn").Process
_LOCAL_UPLOAD_PROCESSES: dict[int, Any] = {}
_LOCAL_UPLOAD_PROCESSES_LOCK = threading.Lock()
_UPLOAD_SHUTDOWN_REQUESTED = threading.Event()

# A worker renews at one third of the lease.  If it disappears, another API
# worker may claim the job after this bounded interval.
UPLOAD_LEASE_SECONDS = 45.0
UPLOAD_STARTUP_LEASE_SECONDS = 180.0
UPLOAD_HEARTBEAT_SECONDS = 15.0
UPLOAD_MAX_HEARTBEAT_FAILURES = 2
UPLOAD_SUPERVISOR_SECONDS = 10.0
UPLOAD_ORPHAN_GRACE_SECONDS = 3600.0
UPLOAD_SHUTDOWN_GRACE_SECONDS = 5.0


_NODE_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}")


def _validated_node_id(value: str) -> str:
    if not _NODE_ID_PATTERN.fullmatch(value):
        raise RuntimeError(
            "VISIONAGENT_NODE_ID must be 1-128 letters, digits, dot, underscore, colon or dash"
        )
    return value


def _worker_node_id() -> str:
    """Stable identity shared by API workers and children on one OS node.

    This must not derive from shared ``STATE_DIR``: staging is cluster-visible,
    while a PID is meaningful only on its own runtime node. Production
    deployments should set a stable unique value explicitly. The fallback
    hashes host identity so infrastructure names never enter PostgreSQL.
    """
    configured = os.getenv("VISIONAGENT_NODE_ID", "").strip()
    if configured:
        return _validated_node_id(configured)
    hostname = socket.gethostname().strip()
    if not hostname:
        raise RuntimeError("VISIONAGENT_NODE_ID is required when host identity is unavailable")
    try:
        machine_id = Path("/etc/machine-id").read_text(encoding="ascii").strip()
    except OSError:
        machine_id = ""
    hardware_id = f"{uuid.getnode():012x}"
    return uuid.uuid5(
        uuid.NAMESPACE_DNS, f"{hostname}:{machine_id}:{hardware_id}"
    ).hex


_UPLOAD_NODE_ID = _worker_node_id()


def enable_upload_workers() -> None:
    """Open local dispatch before the application starts its supervisor."""
    _UPLOAD_SHUTDOWN_REQUESTED.clear()


def _reap_upload_workers() -> int:
    """Join exited local children and release their OS/process resources."""
    with _LOCAL_UPLOAD_PROCESSES_LOCK:
        processes = tuple(_LOCAL_UPLOAD_PROCESSES.items())
    reaped = 0
    for key, process in processes:
        try:
            if process.is_alive():
                continue
            process.join(timeout=0)
            close = getattr(process, "close", None)
            if callable(close):
                close()
        except Exception:
            logger.exception("upload child reaping failed")
            continue
        with _LOCAL_UPLOAD_PROCESSES_LOCK:
            if _LOCAL_UPLOAD_PROCESSES.get(key) is process:
                del _LOCAL_UPLOAD_PROCESSES[key]
                reaped += 1
    return reaped


def shutdown_upload_workers(
    *, timeout_s: float = UPLOAD_SHUTDOWN_GRACE_SECONDS
) -> int:
    """Bound local child shutdown without making process memory authoritative.

    SIGTERM lets a worker checkpoint its active item and return the durable row
    to the queue. Children that outlive the grace window are killed; their
    unexpired database lease prevents concurrent execution until recovery.
    """
    # Close dispatch first. _launch_upload_worker checks this while holding the
    # same registry lock used for the snapshot below, so a concurrent
    # supervisor cannot start a child between the snapshot and shutdown.
    _UPLOAD_SHUTDOWN_REQUESTED.set()
    _reap_upload_workers()
    with _LOCAL_UPLOAD_PROCESSES_LOCK:
        processes = tuple(_LOCAL_UPLOAD_PROCESSES.values())
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except Exception:
            logger.exception("upload child termination failed")

    deadline = time.monotonic() + max(0.0, timeout_s)
    for process in processes:
        try:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        except Exception:
            logger.exception("upload child graceful join failed")

    for process in processes:
        try:
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        except Exception:
            logger.exception("upload child forced shutdown failed")
    return _reap_upload_workers()


def _store() -> ChunkStore:
    """One store for the module; it owns the analyzer and the embedder.

    Built under a lock: two first callers on different threads built two
    stores, each with its own embedding client.
    """
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = build_chunkstore()
    return _STORE


def _jobs() -> UploadJobStore:
    """The stateless durable repository, constructed once per process."""
    global _JOB_REPOSITORY
    if _JOB_REPOSITORY is None:
        with _JOB_REPOSITORY_LOCK:
            if _JOB_REPOSITORY is None:
                _JOB_REPOSITORY = UploadJobRepository()
    return _JOB_REPOSITORY


def dummy(prog: Any=None, msg: str="") -> None:
    pass


def parse(file_name: str, file_path: str | Path, callback: Any = None) -> list[ParsedChunk]:
    """Parse a document into the pipeline's validated chunk contract."""
    return build_parser().parse(
        file_path=Path(file_path), file_name=file_name, progress=callback or dummy
    )

async def execute_insert_process(file_path: Any, file_name: Any, user_id: Any, callback: Any=None,
                                 *, run_id: str | None = None,
                                 tenant_lock_held: bool = False,
                                 part_identity: str | None = None) -> IngestRun:
    """Parse, index, and graph one tenant-owned document.

    Returns the ``IngestRun`` for this file. ``error`` is set when a stage did not
    complete; a graph provider/extraction failure does not stop the chunks
    from being indexed, because a searchable document without a graph is
    worth more than no document, but it is reported rather than swallowed.
    A graph persistence failure raises because some repository files may have
    committed and must be reconciled before indexing can continue.
    """
    supplied_name = str(file_name)
    normalized_name = normalize_document_name(supplied_name)
    if supplied_name != normalized_name:
        raise UploadNameInvalid("document names must use NFC Unicode normalization")
    file_name = normalized_name
    run = IngestRun(
        run_id=run_id or uuid.uuid4().hex,
        user_id=str(user_id),
        file_name=normalized_name,
    )

    # 2. DOCUMENT PARSING & CHUNKING (chunk())
    if callback:
        callback(0.1, "Starting document parsing")
    logger.debug(
        "starting document parse",
        extra={"run_id": run.run_id, "file_suffix": _safe_suffix(str(file_name))},
    )
    try:
        documents = parse(str(file_name), file_path, callback=callback)
        logger.debug(
            "document parse completed",
            extra={"run_id": run.run_id, "raw_chunk_count": len(documents or [])},
        )
    except Exception:
        logger.exception("document parse failed", extra={"run_id": run.run_id})
        raise

    if callback:
        callback(0.3, "Parsing complete, starting chunk processing...")

    if not documents:
        run.error = "no text could be extracted from the document"
        return run

    # Both stores use one stable, tenant-scoped chunk identity. The parser
    # cannot assign it because a parser deliberately has no tenant identity.
    stable_part = part_identity or Path(file_path).name

    def persisted_chunk_id(chunk: ParsedChunk, ordinal: int) -> str:
        digest = hashlib.sha256()
        for component in (
            run.user_id,
            run.file_name,
            stable_part,
            str(ordinal),
            chunk.content,
        ):
            encoded = component.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    run.chunks = [
        chunk.model_copy(
            update={"id": persisted_chunk_id(chunk, ordinal)}
        )
        for ordinal, chunk in enumerate(documents)
    ]

    # Extract graph entities and relationships before chunk indexing.
    # Provider/extraction failure is recorded and chunks remain searchable.
    # Persistence ambiguity is rethrown for durable compensation. Under the
    # tenant lock, the graph is read, changed, and written back whole so
    # concurrent tenant writes serialize.
    try:
        logger.info(
            "starting graph extraction",
            extra={"run_id": run.run_id, "chunk_count": len(documents)},
        )
        guard = nullcontext() if tenant_lock_held else tenant_lock(str(user_id))
        with guard:
            graphrag_result = await GraphRAGService(run.user_id).process_chunks_for_graphrag(
                run.chunks, str(file_name), callback=callback
            )
        if graphrag_result.get("success"):
            run.entity_count = int(graphrag_result.get("entities", 0))
            run.relation_count = int(graphrag_result.get("relationships", 0))
            logger.info(
                "graph extraction completed",
                extra={
                    "run_id": run.run_id,
                    "entity_count": run.entity_count,
                    "relation_count": run.relation_count,
                },
            )
        else:
            run.error = "graph extraction produced no entities or relations"
            logger.warning("graph extraction returned no result", extra={"run_id": run.run_id})
    except GraphPersistenceError:
        # GraphRepository.save() may already have atomically published one or
        # two derived files.  Continuing into Elasticsearch would make a
        # partially committed document terminal.  The isolated-item boundary
        # turns this into durable graph-first compensation and a bounded retry.
        logger.exception("graph persistence failed", extra={"run_id": run.run_id})
        raise
    except Exception:
        logger.exception("graph extraction failed", extra={"run_id": run.run_id})
        run.error = "graph extraction failed"

    # 4. CHUNK PROCESSING & EMBEDDING GENERATION, then DATABASE INSERTION,
    #    both inside the store's index(): the embedding is part of what a
    #    stored document is. The progress hook keeps the per-chunk updates
    #    the upload indicator reads on the outer 0.5..0.8 band.
    if callback:
        callback(0.5, "Processing chunks and generating embeddings")

    def _progress(done: int, total: int, chunk_id: str) -> None:
        if not callback:
            return
        callback(msg=f"Chunk {done}/{total} id: {chunk_id}")
        callback(0.5 + (done / total) * 0.3, f"Processed {done}/{total} chunks")
        if done == total:
            callback(0.8, "Processing complete, starting database insertion")
            callback(0.9, "Inserting into database")

    logger.info(
        "starting chunk indexing",
        extra={"run_id": run.run_id, "chunk_count": len(run.chunks)},
    )
    run.indexed_count = _store().index(
        chunks=run.chunks, index_name=str(user_id), doc_name=str(file_name), progress=_progress
    )
    if run.indexed_count != len(run.chunks):
        # The writer reports how many the bulk request accepted. Fewer than
        # prepared is a partial index, and it was silently a success.
        short = f"indexed {run.indexed_count} of {len(run.chunks)} chunks"
        run.error = f"{run.error}; {short}" if run.error else short

    if callback:
        callback(1.0, "File processing completed successfully")
    logger.info(
        "chunk indexing completed",
        extra={"run_id": run.run_id, "indexed_count": run.indexed_count},
    )
    return run


def execute_insert_process_sync(file_path: Any, file_name: Any, user_id: Any, callback: Any=None,
                                *, run_id: str | None = None,
                                tenant_lock_held: bool = False,
                                part_identity: str | None = None) -> IngestRun:
    """Run ingestion synchronously and propagate the async flow's failures.

    The worker records a failed file explicitly; this boundary does not retry
    partial writes without stage-aware compensation.
    """
    return asyncio.run(
        execute_insert_process(
            file_path,
            file_name,
            user_id,
            callback,
            run_id=run_id,
            tenant_lock_held=tenant_lock_held,
            part_identity=part_identity,
        )
    )


# ------------------------------------------------------- the background job

# Split a PDF whose estimated text exceeds this many characters into parts,
# each parsed as its own file under the original document's name.
CHAR_LIMIT = 250000


@dataclass(frozen=True, slots=True)
class PreparedUpload:
    """A request-spooled source plus the durable work item it will become.

    Split PDF items refer to a page range in the original request stream; they
    do not contain another copy of those pages. The stream is consumed before
    ``start_processing`` returns and is never handed to the durable worker.
    """

    original_name: str
    part_name: str
    source: BinaryIO
    start_page: int | None = None
    end_page: int | None = None

    def __post_init__(self) -> None:
        has_start = self.start_page is not None
        has_end = self.end_page is not None
        if has_start != has_end:
            raise ValueError("a PDF page range needs both endpoints")
        if has_start and (
            self.start_page is None
            or self.end_page is None
            or self.start_page < 0
            or self.end_page <= self.start_page
        ):
            raise ValueError("invalid PDF page range")


LegacyStagedData = tuple[str, str, bytes | BinaryIO]
StagedData = PreparedUpload | LegacyStagedData
MAX_DURABLE_FILE_NAME_CHARS = 255


class UploadNameInvalid(ValueError):
    """An upload name cannot be represented by the durable schema."""


def validate_durable_file_name(name: str) -> None:
    """Match PostgreSQL VARCHAR(255): at most 255 Unicode characters."""
    if not name or len(name) > MAX_DURABLE_FILE_NAME_CHARS:
        raise UploadNameInvalid(
            f"file names must contain 1-{MAX_DURABLE_FILE_NAME_CHARS} characters"
        )
    if "\x00" in name:
        raise UploadNameInvalid("file names must not contain NUL characters")


def canonical_document_name(name: str) -> str:
    """NFC identity used to resolve both current and legacy stored names."""
    normalized = unicodedata.normalize("NFC", name)
    validate_durable_file_name(normalized)
    return normalized


def validate_graph_document_name(name: str) -> None:
    """Reject names that cannot be represented losslessly in graph displays."""
    validate_durable_file_name(name)
    if name != name.strip():
        raise UploadNameInvalid("file names must not begin or end with whitespace")
    if GRAPH_FIELD_SEP in name:
        raise UploadNameInvalid(
            f"file names must not contain the reserved text {GRAPH_FIELD_SEP!r}"
        )


def normalize_document_name(name: str) -> str:
    """Canonical graph-safe logical name for all new document admission."""
    normalized = canonical_document_name(name)
    validate_graph_document_name(normalized)
    return normalized


def _upload_jobs_root() -> Path:
    root = settings.state_dir / "upload_jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _job_directory(process_id: str) -> Path:
    """Resolve a server-generated job directory without accepting traversal."""
    root = _upload_jobs_root().resolve()
    candidate = (root / process_id).resolve()
    if candidate.parent != root:
        raise ValueError("invalid upload process id")
    return candidate


@contextmanager
def upload_staging_lock(
    user_id: str, *, timeout_s: float | None = None
) -> Iterator[None]:
    """Serialize staging writers and account erasure for one tenant.

    The lock lives on the same shared durable volume as the staged bytes, so
    every API replica observes it.  Its opaque name does not disclose a user
    identifier in a filesystem listing.  This is deliberately separate from
    the tenant-store lock: staging a bounded request must not wait behind a
    long graph extraction, while account deletion acquires both fences in a
    fixed order (staging, then tenant stores).
    """
    digest = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()
    path = _upload_jobs_root() / f".tenant-staging-{digest}.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("another operation holds upload staging") from None
                time.sleep(0.05)
        yield
    finally:
        try:
            if locked:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    # Closing the descriptor below also releases flock. Do not
                    # turn a committed queued ticket into an HTTP failure.
                    logger.exception("upload staging lock release failed")
        finally:
            try:
                os.close(descriptor)
            except OSError:
                logger.exception("upload staging lock descriptor close failed")


def _safe_suffix(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,12}", suffix) else ".bin"


def _staged_names(file_data: StagedData) -> tuple[str, str]:
    if isinstance(file_data, PreparedUpload):
        return file_data.original_name, file_data.part_name
    original_name, part_name, _content = file_data
    return original_name, part_name


def _source_stream(source: bytes | BinaryIO) -> BinaryIO:
    if isinstance(source, bytes):
        return io.BytesIO(source)
    return source


def plan_staged_files(
    files_data: Sequence[StagedData],
) -> tuple[StagedUpload, ...]:
    """Allocate opaque storage keys without writing request bytes."""
    staged: list[StagedUpload] = []
    for sequence, file_data in enumerate(files_data):
        original_name, part_name = _staged_names(file_data)
        if normalize_document_name(original_name) != original_name:
            raise UploadNameInvalid("document names must use NFC Unicode normalization")
        validate_durable_file_name(part_name)
        staged.append(
            StagedUpload(
                sequence=sequence,
                original_name=original_name,
                part_name=part_name,
                storage_key=(
                    f"{sequence:06d}-{uuid.uuid4().hex}{_safe_suffix(part_name)}"
                ),
            )
        )
    return tuple(staged)


def _write_staged_data(
    destination: BinaryIO,
    file_data: StagedData,
    pdf_readers: dict[int, Any],
) -> None:
    """Copy one prepared item without materializing its whole content."""
    if isinstance(file_data, PreparedUpload):
        stream = file_data.source
        if file_data.start_page is not None and file_data.end_page is not None:
            from pypdf import PdfReader, PdfWriter

            reader_key = id(stream)
            reader = pdf_readers.get(reader_key)
            if reader is None:
                stream.seek(0)
                reader = PdfReader(stream)
                pdf_readers[reader_key] = reader
            writer = PdfWriter()
            for page_number in range(file_data.start_page, file_data.end_page):
                writer.add_page(reader.pages[page_number])
            writer.write(destination)
            return
    else:
        _original_name, _part_name, source = file_data
        stream = _source_stream(source)

    stream.seek(0)
    shutil.copyfileobj(stream, destination, length=1 << 20)


def persist_staged_files(
    process_id: str,
    files_data: Sequence[StagedData],
    *,
    planned: tuple[StagedUpload, ...] | None = None,
) -> tuple[StagedUpload, ...]:
    """Write recoverable work items under opaque server-generated names."""
    staged = planned if planned is not None else plan_staged_files(files_data)
    if len(staged) != len(files_data):
        raise ValueError("staging plan does not match upload items")
    for sequence, (file_data, item) in enumerate(zip(files_data, staged, strict=True)):
        original_name, part_name = _staged_names(file_data)
        if (
            item.sequence != sequence
            or item.original_name != original_name
            or item.part_name != part_name
            or item.status is not UploadItemState.PENDING
        ):
            raise ValueError("staging plan does not match upload items")

    directory = _job_directory(process_id)
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    pdf_readers: dict[int, Any] = {}
    try:
        for file_data, item in zip(files_data, staged, strict=True):
            path = directory / item.storage_key
            with path.open("xb") as handle:
                _write_staged_data(handle, file_data, pdf_readers)
                handle.flush()
                os.fsync(handle.fileno())
            path.chmod(0o600)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return staged
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def _staged_path(process_id: str, item: StagedUpload) -> Path:
    directory = _job_directory(process_id)
    path = (directory / item.storage_key).resolve()
    if path.parent != directory.resolve():
        raise ValueError("invalid staged upload path")
    return path


def _remove_staged_files(process_id: str) -> None:
    directory = _job_directory(process_id)
    if directory.exists():
        shutil.rmtree(directory)


def _cleanup_one_terminal_staging(
    process_id: str, *, staging_lock_held: bool = False
) -> bool:
    """Erase and durably acknowledge one copy from shared staging."""
    repository = _jobs()
    record = repository.get(process_id, include_items=False, include_progress=False)
    if record is None or record.staging_cleaned:
        return True
    if not record.status.terminal:
        return False
    # A STAGING ticket can be cancelled while its request is blocked on this
    # lock. Re-read after acquisition so cleanup never races a late mkdir/write.
    guard = nullcontext() if staging_lock_held else upload_staging_lock(record.user_id)
    with guard:
        record = repository.get(
            process_id, include_items=False, include_progress=False
        )
        if record is None or record.staging_cleaned:
            return True
        if not record.status.terminal:
            return False
        try:
            _remove_staged_files(process_id)
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception(
                "terminal upload staging removal failed",
                extra={"process_id": process_id},
            )
            return False
        if repository.mark_staging_cleaned(process_id):
            return True
    # Another cleanup may have acknowledged and deleted the row between our
    # snapshot and update; that is success, not an erasure failure.
    latest = repository.get(process_id, include_items=False, include_progress=False)
    return latest is None or latest.staging_cleaned


def cleanup_terminal_staging() -> int:
    """Reconcile terminal staging left by a crash after the status commit."""
    repository = _jobs()
    cleaned = 0
    for process_id in repository.terminal_staging_ids():
        try:
            cleaned += int(_cleanup_one_terminal_staging(process_id))
        except Exception:
            logger.exception(
                "terminal upload staging reconciliation failed",
                extra={"process_id": process_id},
            )
    return cleaned


def cleanup_orphan_staging(*, min_age_seconds: float = UPLOAD_ORPHAN_GRACE_SECONDS) -> int:
    """Remove legacy/unexpected staging that has no durable PostgreSQL job.

    New requests commit an owner-bearing STAGING ticket before writing bytes.
    The age grace protects any directory left by an older build or external
    repair while the durable-row check is performed.
    """
    root = _upload_jobs_root()
    cutoff = time.time() - min_age_seconds
    candidates: list[Path] = []
    for path in root.iterdir():
        if not re.fullmatch(r"[0-9a-f]{32}", path.name):
            continue
        try:
            # Do not follow a symlink planted in writable state and never let
            # a concurrent worker cleanup abort reconciliation of other jobs.
            if path.is_symlink() or not path.is_dir() or path.stat().st_mtime > cutoff:
                continue
        except FileNotFoundError:
            continue
        candidates.append(path)
    existing = _jobs().existing_ids(path.name for path in candidates)
    removed = 0
    for path in candidates:
        if path.name in existing:
            continue
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            continue
        except OSError:
            logger.exception(
                "orphan upload staging removal failed",
                extra={"process_id": path.name},
            )
        else:
            removed += 1
    return removed


def stage_files(
    uploads: Sequence[tuple[str, bytes | BinaryIO]],
) -> list[PreparedUpload]:
    """Plan work items over seekable, request-scoped upload streams.

    A large PDF is split into page ranges so that parsing stays within the
    worker's memory; every part carries the original name so the parts are
    recorded as one document. No split byte blobs are created: ranges are
    written one at a time only after the owner-bearing STAGING ticket commits.
    """
    files_data: list[PreparedUpload] = []
    normalized_uploads = [(normalize_document_name(name), source) for name, source in uploads]
    names = [name for name, _source in normalized_uploads]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise UploadDocumentConflict(
            f"duplicate document name in upload batch: {duplicate_names[0]!r}"
        )
    for filename, source in normalized_uploads:
        stream = _source_stream(source)
        if not filename.lower().endswith('.pdf'):
            files_data.append(PreparedUpload(filename, filename, stream))
            continue

        # Inspection operates on FastAPI's request spool. Persistent
        # application bytes must not exist until the owner-bearing STAGING row
        # has committed.
        import random

        import pdfplumber

        stream.seek(0)
        with pdfplumber.open(cast(Any, stream)) as pdf:
            total_pages = len(pdf.pages)
            logger.debug("inspected PDF", extra={"page_count": total_pages})

            if total_pages <= 5:
                sample_pages = list(range(total_pages))
            else:
                sample_pages = random.sample(range(total_pages), 5)

            char_counts = []
            for page_idx in sample_pages:
                page_text = pdf.pages[page_idx].extract_text() or ""
                char_counts.append(len(page_text))

            max_chars_per_page = max(char_counts) if char_counts else 0
            estimated_total_chars = total_pages * max_chars_per_page
            logger.debug(
                "estimated PDF text size",
                extra={
                    "estimated_characters": estimated_total_chars,
                    "max_characters_per_page": max_chars_per_page,
                },
            )
        stream.seek(0)

        if estimated_total_chars > CHAR_LIMIT:
            num_parts = (estimated_total_chars // CHAR_LIMIT) + 1
            # Rounded up, so num_parts parts of this size cover every page.
            pages_per_part = max(1, math.ceil(total_pages / num_parts))
            logger.info(
                "splitting large PDF",
                extra={
                    "part_count": num_parts,
                    "pages_per_part": pages_per_part,
                    "page_count": total_pages,
                },
            )
            for part_idx in range(num_parts):
                start_page = part_idx * pages_per_part
                end_page = min(start_page + pages_per_part, total_pages)
                if start_page >= total_pages:
                    break

                base_name, extension = os.path.splitext(filename)
                chunk_filename = f"part_{part_idx:03d}_{base_name}{extension}"
                files_data.append(
                    PreparedUpload(
                        filename,
                        chunk_filename,
                        stream,
                        start_page=start_page,
                        end_page=end_page,
                    )
                )
                logger.debug(
                    "created PDF part",
                    extra={
                        "part_index": part_idx,
                        "file_suffix": extension.lower(),
                        "start_page": start_page + 1,
                        "end_page": end_page,
                    },
                )
        else:
            logger.debug(
                "PDF does not require splitting", extra={"page_count": total_pages}
            )
            files_data.append(PreparedUpload(filename, filename, stream))
    return files_data


class UploadCancelled(RuntimeError):
    """The durable job was cancelled or this worker lost its lease."""


class UploadLeaseLost(UploadCancelled):
    """A stale reservation is fenced before it can mutate tenant stores."""


class UploadRetryable(RuntimeError):
    """A post-side-effect checkpoint failed and must be retried durably."""


class ItemExecutionTimeout(RuntimeError):
    """The killable per-item execution budget expired."""


class ItemExecutionFailure(RuntimeError):
    """An isolated item failed or exited ambiguously."""


def _compensate_logical_document(
    user_id: str,
    file_name: str,
    *,
    process_id: str,
    owner: str,
) -> None:
    """Remove partial effects in a self-timed child which owns the fence."""
    parent, child = _ITEM_PROCESS_CONTEXT.Pipe(duplex=False)
    process = _ITEM_PROCESS_CONTEXT.Process(
        target=_isolated_compensation_main,
        args=(
            child,
            user_id,
            file_name,
            settings.parse_timeout_s,
            process_id,
            owner,
        ),
        daemon=True,
    )
    try:
        process.start()
    except (OSError, RuntimeError) as error:
        parent.close()
        child.close()
        raise ItemExecutionFailure("could not start compensation isolation") from error
    child.close()
    try:
        if not parent.poll(settings.parse_timeout_s + 1):
            raise ItemExecutionTimeout("document compensation deadline exceeded")
        try:
            failure = parent.recv()
        except (EOFError, OSError) as error:
            if process.exitcode == 124:
                raise ItemExecutionTimeout(
                    "document compensation self-timed out"
                ) from error
            raise ItemExecutionFailure(
                "compensation child exited without confirmation"
            ) from error
        if failure == UploadLeaseLost.__name__:
            raise UploadLeaseLost(process_id)
        if failure is not None:
            raise ItemExecutionFailure(f"document compensation failed: {failure}")
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        parent.close()


def _isolated_compensation_main(
    connection: Any,
    user_id: str,
    file_name: str,
    timeout_s: float,
    process_id: str,
    owner: str,
) -> None:
    try:
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, lambda _signum, _frame: os._exit(124))
            signal.setitimer(signal.ITIMER_REAL, timeout_s)
        with tenant_lock(user_id):
            if not _jobs().owns_active_lease(process_id, owner):
                raise UploadLeaseLost(process_id)
            if not UserRepository().accepts_upload_work(user_id):
                # Account deletion closes the durable gate before taking this
                # fence and owns complete store erasure. Running graph cleanup
                # afterward could recreate empty per-user graph files.
                connection.send(None)
                return
            # Graph deletion is the fail-closed side: legacy aggregates may
            # not have enough contribution identity to repair safely. Do it
            # before Elasticsearch so refusal leaves search data untouched.
            # A later ES failure is convergent: contribution-ledger graph
            # deletion is idempotent, so the durable retry can rerun both.
            graph = asyncio.run(GraphRAGService(user_id).delete_file_data(file_name))
            if not graph.get("success"):
                raise RuntimeError("graph compensation failed")
            _store().delete_document(doc_name=file_name, index_name=str(user_id))
        connection.send(None)
    except BaseException as error:
        connection.send(type(error).__name__)
    finally:
        connection.close()


def _isolated_item_main(
    connection: Any,
    staged_path: str,
    original_name: str,
    user_id: str,
    run_id: str,
    timeout_s: float,
    part_identity: str,
    process_id: str,
    owner: str,
) -> None:
    """Run all mutable item stages in a process the supervisor can kill."""
    try:
        if hasattr(signal, "SIGALRM"):
            signal.signal(
                signal.SIGALRM,
                lambda _signum, _frame: os._exit(124),
            )
            signal.setitimer(signal.ITIMER_REAL, timeout_s)
        with tenant_lock(user_id):
            if not _jobs().owns_active_lease(process_id, owner):
                raise UploadLeaseLost(process_id)
            if not UserRepository().accepts_upload_work(user_id):
                raise UploadAccountUnavailable("account is unavailable for upload")
            run = execute_insert_process_sync(
                Path(staged_path),
                original_name,
                user_id,
                lambda progress=None, message=None, **_kw: connection.send(
                    ("progress", progress, message)
                ),
                run_id=run_id,
                tenant_lock_held=True,
                part_identity=part_identity,
            )
        connection.send(("result", run.model_dump(mode="json")))
    except BaseException as error:
        connection.send(("error", type(error).__name__))
    finally:
        connection.close()


_ITEM_EXECUTION_TARGET = _isolated_item_main


def _execute_item_with_deadline(
    *,
    staged_path: Path,
    original_name: str,
    user_id: str,
    run_id: str,
    process_id: str,
    owner: str,
    progress_callback: Any,
    cancelled: Any,
    part_identity: str | None = None,
) -> IngestRun:
    """Execute an item in killable isolation and return only after it exits."""
    parent, child = _ITEM_PROCESS_CONTEXT.Pipe(duplex=False)
    process = _ITEM_PROCESS_CONTEXT.Process(
        target=_ITEM_EXECUTION_TARGET,
        args=(
            child,
            str(staged_path),
            original_name,
            user_id,
            run_id,
            settings.parse_timeout_s,
            part_identity or staged_path.name,
            process_id,
            owner,
        ),
        daemon=True,
    )
    try:
        process.start()
    except (OSError, RuntimeError) as error:
        parent.close()
        child.close()
        raise ItemExecutionFailure("could not start item isolation") from error
    child.close()
    deadline = time.monotonic() + settings.parse_timeout_s
    try:
        while True:
            if cancelled():
                raise UploadCancelled(run_id)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ItemExecutionTimeout("item execution deadline exceeded")
            if parent.poll(min(0.1, remaining)):
                try:
                    message = parent.recv()
                except (EOFError, OSError) as error:
                    if process.exitcode == 124:
                        raise ItemExecutionTimeout(
                            "item child self-timed out"
                        ) from error
                    raise ItemExecutionFailure(
                        "item child exited without a result"
                    ) from error
                if not isinstance(message, tuple) or not message:
                    raise ItemExecutionFailure("item child returned a malformed result")
                if message[0] == "progress":
                    progress_callback(message[1], message[2])
                elif message[0] == "result":
                    return IngestRun.model_validate(message[1])
                elif len(message) > 1 and message[1] == UploadLeaseLost.__name__:
                    raise UploadLeaseLost(process_id)
                else:
                    raise ItemExecutionFailure(
                        f"isolated item failed: {message[1]}"
                    )
            elif not process.is_alive():
                if process.exitcode == 124:
                    raise ItemExecutionTimeout("item child self-timed out")
                raise ItemExecutionFailure(
                    "isolated item process exited without a result"
                )
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5.0)
        parent.close()


def _isolated_metadata_main(
    connection: Any,
    arguments: tuple[Any, ...],
    timeout_s: float,
    process_id: str,
    owner: str,
) -> None:
    try:
        user_id = str(arguments[0])
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, lambda _signum, _frame: os._exit(124))
            signal.setitimer(signal.ITIMER_REAL, timeout_s)
        with tenant_lock(user_id):
            if not _jobs().owns_active_lease(process_id, owner):
                raise UploadLeaseLost(process_id)
            if not UserRepository().accepts_upload_work(user_id):
                raise UploadAccountUnavailable("account is unavailable for upload")
            insert_knowledgebase(*arguments)
        connection.send(None)
    except BaseException as error:
        connection.send(type(error).__name__)
    finally:
        connection.close()


def _finalize_metadata_with_deadline(
    arguments: tuple[Any, ...],
    cancelled: Any = lambda: False,
    *,
    process_id: str,
    owner: str,
) -> None:
    """Bound the final PostgreSQL checkpoint with the same hard item budget."""
    parent, child = _ITEM_PROCESS_CONTEXT.Pipe(duplex=False)
    process = _ITEM_PROCESS_CONTEXT.Process(
        target=_isolated_metadata_main,
        args=(child, arguments, settings.parse_timeout_s, process_id, owner),
        daemon=True,
    )
    try:
        process.start()
    except (OSError, RuntimeError) as error:
        parent.close()
        child.close()
        raise ItemExecutionFailure("could not start metadata isolation") from error
    child.close()
    try:
        deadline = time.monotonic() + settings.parse_timeout_s
        while not parent.poll(0.1):
            if cancelled():
                raise UploadCancelled("metadata finalization cancelled")
            if time.monotonic() >= deadline:
                raise ItemExecutionTimeout("metadata finalization deadline exceeded")
        try:
            failure_class = parent.recv()
        except (EOFError, OSError) as error:
            if process.exitcode == 124:
                raise ItemExecutionTimeout("metadata child self-timed out") from error
            raise ItemExecutionFailure(
                "metadata child exited without confirmation"
            ) from error
        if failure_class == UploadLeaseLost.__name__:
            raise UploadLeaseLost(process_id)
        if failure_class is not None:
            raise RuntimeError(f"metadata finalization failed: {failure_class}")
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5.0)
        parent.close()


def _process_start_token(pid: int) -> str | None:
    """Stable OS fingerprint used to avoid signalling a recycled PID."""
    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        # The command name may contain spaces and parentheses. Fields after
        # the final ')' begin at process-state (field 3); starttime is field 22.
        remainder = proc_stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return f"linux:{remainder[19]}"
    except (OSError, IndexError):
        pass
    try:
        result = subprocess.run(
            ("ps", "-o", "lstart=", "-p", str(pid)),
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        started = result.stdout.strip()
        return f"ps:{started}" if started else None
    except (OSError, subprocess.SubprocessError):
        return None


def _heartbeat_upload_job(
    process_id: str,
    owner: str,
    stop: threading.Event,
    lost: threading.Event,
    terminate_on_loss: bool,
) -> None:
    """Renew the worker lease independently of long parser/provider calls."""
    failures = 0
    while not stop.wait(UPLOAD_HEARTBEAT_SECONDS):
        try:
            if not _jobs().heartbeat(process_id, owner, UPLOAD_LEASE_SECONDS):
                lost.set()
                if terminate_on_loss:
                    os.kill(os.getpid(), signal.SIGTERM)
                return
            failures = 0
        except Exception:
            failures += 1
            logger.exception(
                "upload heartbeat failed",
                extra={"process_id": process_id},
            )
            # Continuing after a prolonged database outage risks two workers
            # owning the same job once the lease expires. Fail closed first.
            if failures >= UPLOAD_MAX_HEARTBEAT_FAILURES:
                lost.set()
                if terminate_on_loss:
                    os.kill(os.getpid(), signal.SIGTERM)
                return


def process_files_worker(process_id: str, owner: str, worker_node_id: str) -> None:
    """
    Worker function to process files in background process

    Runs under the IngestJob the API minted: the user id, the run id and the
    authorization all come from the ticket. Every part produces an IngestRun;
    a part whose stage failed is reported as an error step for its file and
    recorded on the document's row, and the batch finishes as "failed" so the
    caller cannot mistake it for a clean run.
    """
    if worker_node_id != _UPLOAD_NODE_ID:
        logger.error("upload reservation belongs to a different runtime node")
        return
    repository = _jobs()
    # SQLAlchemy pools must not hand a forked child a connection inherited
    # from its parent. The in-memory test repository has no engine to reset.
    if isinstance(repository, UploadJobRepository):
        from visionagent.database.postgres.engine import engine

        engine.dispose(close=False)

    worker_pid = os.getpid()
    if not repository.activate(
        process_id,
        owner,
        worker_pid,
        _process_start_token(worker_pid),
        UPLOAD_LEASE_SECONDS,
    ):
        return
    record = repository.get(process_id, include_items=True, include_progress=False)
    if record is None:
        return
    job = record.job
    user_id = job.identity.user_id
    owns_main_thread = threading.current_thread() is threading.main_thread()
    lost_lease = threading.Event()
    shutdown_signal_received = threading.Event()
    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_upload_job,
        args=(process_id, owner, stop_heartbeat, lost_lease, owns_main_thread),
        name=f"upload-heartbeat-{process_id}",
        daemon=True,
    )
    heartbeat.start()

    def report(event: dict[str, Any]) -> None:
        if lost_lease.is_set() or not repository.append_progress(
            process_id, owner, event, UPLOAD_LEASE_SECONDS
        ):
            raise UploadCancelled(process_id)

    previous_sigterm: Any = None
    if owns_main_thread:
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def _cancel_on_signal(_signum: int, _frame: Any) -> None:
            # Both API cancellation and host shutdown are cooperative: do not
            # start another item, but checkpoint and expose an item already in
            # progress. Lease loss is different; continuing then would allow
            # two owners to mutate the tenant concurrently.
            if lost_lease.is_set():
                raise UploadCancelled(process_id)
            shutdown_signal_received.set()

        signal.signal(signal.SIGTERM, _cancel_on_signal)

    try:
        def owned_snapshot() -> UploadJobRecord:
            snapshot = repository.get(
                process_id, include_items=True, include_progress=False
            )
            if (
                snapshot is None
                or snapshot.status is not UploadJobState.PROCESSING
                or snapshot.lease_owner != owner
                or lost_lease.is_set()
            ):
                raise UploadCancelled(process_id)
            return snapshot

        # Account deletion can cancel and clear a worker which activated but
        # was blocked on this lock. It must not proceed using the stale record
        # loaded before the lock acquisition.
        record = owned_snapshot()
        account_gone = False
        # A short fence synchronizes the gate check with account deletion;
        # the parent releases it before spawning the isolated item child.
        with tenant_lock(user_id):
            if not UserRepository().accepts_upload_work(user_id):
                account_gone = True

        metadata_finalized: set[str] = set()

        def finalize_ready_documents(
            snapshot: UploadJobRecord | None = None,
        ) -> UploadJobRecord:
            """Durably publish every fully checkpointed logical document.

            This runs after each part and again on worker recovery. The
            knowledge-base upsert is idempotent, so a crash between the item
            checkpoint and this call converges without replaying or deleting
            an already-complete document.
            """
            current = snapshot or owned_snapshot()
            if account_gone:
                return current
            document_names = sorted({item.original_name for item in current.items})
            for document_name in document_names:
                if document_name in metadata_finalized:
                    continue
                current = owned_snapshot()
                parts = [
                    item
                    for item in current.items
                    if item.original_name == document_name
                ]
                if not parts or any(not item.status.terminal for item in parts):
                    continue
                error = next((item.error for item in parts if item.error), None)
                try:
                    _finalize_metadata_with_deadline(
                        (
                            user_id,
                            document_name,
                            sum(item.indexed_count for item in parts),
                            sum(item.process_time for item in parts),
                            error,
                        ),
                        # API cancellation and host shutdown still require a
                        # completed document's metadata before its checkpoint
                        # can survive. Only loss of ownership stops the upsert.
                        cancelled=lost_lease.is_set,
                        process_id=process_id,
                        owner=owner,
                    )
                except UploadCancelled:
                    raise
                except Exception as metadata_error:
                    logger.exception(
                        "knowledge-base metadata finalization failed",
                        extra={
                            "run_id": job.identity.run_id,
                            "file_suffix": _safe_suffix(document_name),
                        },
                    )
                    raise UploadRetryable(
                        "knowledge-base metadata finalization failed"
                    ) from metadata_error
                metadata_finalized.add(document_name)
            return owned_snapshot()

        def documents_with_unfinalized_effects(
            snapshot: UploadJobRecord,
        ) -> tuple[str, ...]:
            """Return partial logical documents that may own store effects."""
            names: list[str] = []
            for document_name in sorted({
                item.original_name for item in snapshot.items
            }):
                parts = [
                    item
                    for item in snapshot.items
                    if item.original_name == document_name
                ]
                if (
                    any(not item.status.terminal for item in parts)
                    and any(item.status is not UploadItemState.PENDING for item in parts)
                ):
                    names.append(document_name)
            return tuple(names)

        # Recovery may start after all parts of document A were checkpointed
        # but before its metadata upsert. Close that gap before cancellation,
        # exhaustion, or work on document B can become visible.
        record = finalize_ready_documents(record)

        recovered_documents = sorted({
            item.original_name
            for item in record.items
            if item.status is UploadItemState.PROCESSING
        })
        if (
            record.attempt_count > record.max_attempts
            and any(not item.status.terminal for item in record.items)
        ):
            terminal_compensations = documents_with_unfinalized_effects(record)
            for document_name in terminal_compensations:
                _compensate_logical_document(
                    user_id,
                    document_name,
                    process_id=process_id,
                    owner=owner,
                )
            terminal_status = (
                UploadJobState.CANCELLED
                if record.cancellation_requested
                else UploadJobState.FAILED
            )
            transitioned = repository.finish_reconciled(
                process_id,
                owner,
                terminal_status,
                terminal_compensations,
                {
                    "role": "upload_progress",
                    "step": (
                        "cancelled"
                        if terminal_status is UploadJobState.CANCELLED
                        else "error"
                    ),
                    "message": (
                        "Cancelled upload effects were reconciled"
                        if terminal_status is UploadJobState.CANCELLED
                        else "Upload stopped after repeated worker failures"
                    ),
                    "failure_class": (
                        "item_cancelled"
                        if terminal_status is UploadJobState.CANCELLED
                        else "worker_lease_expired"
                    ),
                },
            )
            if not transitioned:
                raise UploadCancelled(process_id)
            return
        if recovered_documents:
            # Lease expiry proves only that the old worker stopped
            # heartbeating, not where its child stopped. Graph/search may be
            # committed even though the item checkpoint still says
            # processing, so delete those effects before any replay.
            for document_name in recovered_documents:
                _compensate_logical_document(
                    user_id,
                    document_name,
                    process_id=process_id,
                    owner=owner,
                )
            record = owned_snapshot()
            if record.cancellation_requested:
                if not repository.finish_reconciled(
                    process_id,
                    owner,
                    UploadJobState.CANCELLED,
                    recovered_documents,
                    {
                        "role": "upload_progress",
                        "step": "cancelled",
                        "message": "Cancelled upload effects were reconciled",
                        "failure_class": "item_cancelled",
                    },
                ):
                    raise UploadCancelled(process_id)
                return
            for document_name in recovered_documents:
                if not repository.reset_reconciled_document(
                    process_id, owner, document_name
                ):
                    snapshot = owned_snapshot()
                    if snapshot.cancellation_requested and repository.finish_reconciled(
                        process_id,
                        owner,
                        UploadJobState.CANCELLED,
                        recovered_documents,
                        {
                            "role": "upload_progress",
                            "step": "cancelled",
                            "message": "Cancelled upload effects were reconciled",
                            "failure_class": "item_cancelled",
                        },
                    ):
                        return
                    raise UploadCancelled(process_id)
            record = owned_snapshot()
        cancelling_on_entry = record.cancellation_requested
        resumed_sequence = next(
            (
                item.sequence
                for item in record.items
                if item.status is UploadItemState.PROCESSING
            ),
            None,
        )

        # Which step the granular progress updates belong to. The fractions
        # arriving at progress_callback are on two different scales: deepdoc
        # counts OCR pages on its own 0..0.67, while the chunk loop reports on
        # the outer 0.5..0.8. Knowing the step is what lets each be normalised
        # to its own 0-100. Reset per file below.
        stage = {"name": "upload"}

        for item in record.items:
            if item.status.terminal:
                continue
            # SIGTERM may arrive while this worker waits for the tenant lock.
            # It must not start a fresh item after shutdown has already begun.
            # A prior worker's processing checkpoint stays durable for replay.
            if shutdown_signal_received.is_set():
                break
            # A recovered cancellation may replay the one item that was
            # already in flight, but it never starts another item.
            if cancelling_on_entry and item.sequence != resumed_sequence:
                break
            original_name, file_name = item.original_name, item.part_name

            # The account may have been deleted since the upload was
            # authorized. Deletion holds the tenant lock while it clears the
            # stores; a worker that then recreated an index or a row would
            # resurrect a deleted user's data.
            if not UserRepository().accepts_upload_work(user_id):
                account_gone = True
                report({
                    "role": "upload_progress",
                    "step": "error",
                    "message": "The account no longer exists; processing stopped",
                })
                break

            if not repository.begin_item(process_id, owner, item.sequence):
                # Cancellation can commit between the loop's snapshot and the
                # atomic begin. Pending work is refused by the repository; if
                # we still own the job, continue to normal cancellation
                # finalization instead of waiting for the lease to expire.
                if owned_snapshot().cancellation_requested:
                    break
                raise UploadCancelled(process_id)

            try:
                # Send file processing start message
                progress_msg = {
                    "role": "upload_progress",
                    "step": "upload_finish",
                    "message": f"File {file_name} ready for processing"
                }
                report(progress_msg)

                staged_path = _staged_path(process_id, item)
                if not staged_path.is_file():
                    raise FileNotFoundError("staged upload is missing")
                start = time.time()
                # Process the file
                stage["name"] = "upload"

                def progress_callback(progress: Any=None, message: Any=None, msg: Any=None, prog: Any=None) -> None:
                        if msg is not None:
                            message = msg
                        if prog is not None:
                            progress = prog

                        # Map progress to steps
                        step = None
                        percent = None
                        if message and "Starting document parsing" in message:
                            step, percent, stage["name"] = "parse_pending", 0.0, "parse"
                        elif message and "Parsing complete" in message:
                            step, percent, stage["name"] = "parse_finish", 100.0, "graph"
                        elif message and "Processing chunks and generating embeddings" in message:
                            step, percent, stage["name"] = "encode_pending", 50.0, "encode"
                        elif message and "Processing complete" in message:
                            step, percent, stage["name"] = "encode_finish", 100.0, "done"
                        elif message and "Inserting into database" in message:
                            step, percent, stage["name"] = "database_pending", 0.0, "database"
                        elif message and "File processing completed successfully" in message:
                            step, percent, stage["name"] = "database_finish", 100.0, "done"
                        elif progress is not None:
                            # A granular update from inside whichever step runs.
                            # Encode covers two jobs: graph extraction runs one
                            # LLM call per chunk on 0.3..0.5 and is the bulk of
                            # the wall clock, then the embeddings run on
                            # 0.5..0.8. They share the step's 0-100 half each,
                            # so the number moves throughout rather than sitting
                            # at 0 until the embeddings start.
                            if stage["name"] == "parse":
                                step, percent = "parse_pending", progress / 0.67 * 100
                            elif stage["name"] == "graph":
                                step, percent = "encode_pending", (progress - 0.3) / 0.2 * 50
                            elif stage["name"] == "encode":
                                step, percent = "encode_pending", 50 + (progress - 0.5) / 0.3 * 50

                        if step:
                            progress_msg = {
                                "role": "upload_progress",
                                "step": step,
                                # Keep the provider's granular message in the
                                # durable event journal; SSE resumes by event
                                # id and therefore preserves repeated steps.
                                "message": message,
                                "percent": None if percent is None else round(max(0.0, min(100.0, percent))),
                            }
                            report(progress_msg)

                # Execute the file processing directly from durable staging.
                try:
                    run = _execute_item_with_deadline(
                        staged_path=staged_path,
                        original_name=original_name,
                        user_id=user_id,
                        run_id=job.identity.run_id,
                        process_id=process_id,
                        owner=owner,
                        progress_callback=progress_callback,
                        cancelled=lambda: lost_lease.is_set()
                        or shutdown_signal_received.is_set()
                        or (
                            (snapshot := repository.get(
                                process_id,
                                include_items=False,
                                include_progress=False,
                            )) is None
                            or snapshot.cancellation_requested
                            or snapshot.lease_owner != owner
                        ),
                        part_identity=f"{item.sequence}:{item.part_name}",
                    )
                except (
                    UploadCancelled,
                    ItemExecutionTimeout,
                    ItemExecutionFailure,
                ):
                    raise
                except Exception as execution_error:
                    raise ItemExecutionFailure(
                        "isolated item result was ambiguous"
                    ) from execution_error

                duration = time.time() - start
                try:
                    item_finished = repository.finish_item(
                        process_id,
                        owner,
                        item.sequence,
                        indexed_count=run.indexed_count,
                        process_time=duration,
                        error=run.error,
                    )
                except Exception as checkpoint_error:
                    raise ItemExecutionFailure(
                        "item checkpoint outcome was ambiguous"
                    ) from checkpoint_error
                if not item_finished:
                    raise UploadCancelled(process_id)
                finalize_ready_documents()
                if run.error:
                    report({
                        "role": "upload_progress",
                        "step": "error",
                        "message": f"{original_name}: {run.error}",
                    })

            except (ItemExecutionTimeout, ItemExecutionFailure) as item_failure:
                snapshot = owned_snapshot()
                # The isolated child may have committed graph/search effects
                # before it timed out or died. Rewind durable checkpoints only
                # after the entire logical document has been compensated,
                # including on non-final attempts.
                compensated_documents: tuple[str, ...] = ()
                if original_name not in metadata_finalized:
                    _compensate_logical_document(
                        user_id,
                        original_name,
                        process_id=process_id,
                        owner=owner,
                    )
                    compensated_documents = (original_name,)
                snapshot = owned_snapshot()
                failure_class = (
                    "item_timeout"
                    if isinstance(item_failure, ItemExecutionTimeout)
                    else "item_execution_failed"
                )
                if (
                    snapshot.cancellation_requested
                    or snapshot.attempt_count >= snapshot.max_attempts
                ):
                    terminal_status = (
                        UploadJobState.CANCELLED
                        if snapshot.cancellation_requested
                        else UploadJobState.FAILED
                    )
                    if not repository.finish_reconciled(
                        process_id,
                        owner,
                        terminal_status,
                        compensated_documents,
                        {
                            "role": "upload_progress",
                            "step": (
                                "cancelled"
                                if terminal_status is UploadJobState.CANCELLED
                                else "error"
                            ),
                            "message": (
                                "Cancelled upload effects were reconciled"
                                if terminal_status is UploadJobState.CANCELLED
                                else (
                                    "Upload stopped after repeated item failures. "
                                    f"Reference: {job.identity.run_id}"
                                )
                            ),
                            "failure_class": failure_class,
                        },
                    ):
                        raise UploadCancelled(process_id) from None
                    return
                if compensated_documents:
                    if not repository.retry_document(process_id, owner, original_name, {
                        "role": "upload_progress",
                        "step": "recovery",
                        "message": "Interrupted file processing will be retried",
                        "failure_class": failure_class,
                    }):
                        raise UploadCancelled(process_id) from None
                raise UploadRetryable(process_id) from None
            except UploadRetryable:
                raise
            except UploadCancelled:
                cancelled_compensated_documents: tuple[str, ...] = ()
                if original_name not in metadata_finalized:
                    _compensate_logical_document(
                        user_id,
                        original_name,
                        process_id=process_id,
                        owner=owner,
                    )
                    cancelled_compensated_documents = (original_name,)
                snapshot = owned_snapshot()
                if snapshot.cancellation_requested:
                    if not repository.finish_reconciled(
                        process_id,
                        owner,
                        UploadJobState.CANCELLED,
                        cancelled_compensated_documents,
                        {
                            "role": "upload_progress",
                            "step": "cancelled",
                            "message": "Interrupted file effects were reconciled",
                            "failure_class": "item_cancelled",
                        },
                    ):
                        raise UploadCancelled(process_id) from None
                    return
                if shutdown_signal_received.is_set():
                    if cancelled_compensated_documents:
                        transitioned = repository.retry_document(
                            process_id,
                            owner,
                            original_name,
                            {
                                "role": "upload_progress",
                                "step": "recovery",
                                "message": (
                                    "Worker stopped; compensated file processing is queued"
                                ),
                                "failure_class": "worker_shutdown",
                            },
                            refund_attempt=True,
                        )
                    else:
                        transitioned = repository.requeue_owned(
                            process_id,
                            owner,
                            {
                                "role": "upload_progress",
                                "step": "recovery",
                                "message": "Worker stopped; remaining processing is queued",
                            },
                        )
                    if not transitioned:
                        raise UploadCancelled(process_id) from None
                    raise UploadRetryable(process_id) from None
                raise
            except Exception:
                # The traceback goes to the log under the run id; the client
                # sees a stable message. Exception text names hosts, paths and
                # SQL, and this frame is rendered on the upload row.
                logger.exception(
                    "upload item processing failed",
                    extra={
                        "run_id": job.identity.run_id,
                        "item_sequence": item.sequence,
                        "file_suffix": _safe_suffix(file_name),
                    },
                )
                error_msg = {
                    "role": "upload_progress",
                    "step": "error",
                    "message": f"Error processing {file_name}. Reference: {job.identity.run_id}"
                }
                if not repository.finish_item(
                    process_id,
                    owner,
                    item.sequence,
                    indexed_count=0,
                    process_time=0.0,
                    error="processing failed",
                ):
                    raise UploadCancelled(process_id) from None
                finalize_ready_documents()
                report(error_msg)

            # Cancellation is observed only at this safe boundary: graph,
            # chunks and the durable item checkpoint now agree.
            record = owned_snapshot()
            if record.cancellation_requested or shutdown_signal_received.is_set():
                break

        # Re-read durable checkpoints and close any crash gap between a
        # document's final item and its metadata upsert.
        record = finalize_ready_documents()
        errors = {
            item.original_name: item.error for item in record.items
            if item.error
        }

        # The process is completed only when every file completed fully.
        if record.cancellation_requested:
            compensated_documents = documents_with_unfinalized_effects(record)
            for document_name in compensated_documents:
                _compensate_logical_document(
                    user_id,
                    document_name,
                    process_id=process_id,
                    owner=owner,
                )
            if not repository.finish_reconciled(
                process_id,
                owner,
                UploadJobState.CANCELLED,
                compensated_documents,
                {
                    "role": "upload_progress",
                    "step": "cancelled",
                    "message": "File processing was cancelled after finalizing active data",
                    "failure_class": "item_cancelled",
                },
            ):
                raise UploadCancelled(process_id)
        elif shutdown_signal_received.is_set():
            if not repository.requeue_owned(process_id, owner, {
                    "role": "upload_progress",
                    "step": "recovery",
                    "message": "Worker stopped; remaining file processing is queued",
            }):
                raise UploadCancelled(process_id)
        elif account_gone or errors:
            failed_count = (
                len(errors)
                if not account_gone
                else len({item.original_name for item in record.items}) or 1
            )
            if not repository.finish(process_id, owner, UploadJobState.FAILED, {
                "role": "upload_progress",
                "step": "error",
                "message": f"{failed_count} document(s) did not fully complete",
            }):
                raise UploadCancelled(process_id)
        elif any(not item.status.terminal for item in record.items):
            # Defensive backstop: no code path may report completion while a
            # durable work item still needs execution.
            if not repository.requeue_owned(process_id, owner, {
                    "role": "upload_progress",
                    "step": "recovery",
                    "message": "Unfinished file processing remains queued",
            }):
                raise UploadCancelled(process_id)
        else:
            if not repository.finish(process_id, owner, UploadJobState.COMPLETED, {
                "role": "upload_progress",
                "step": "complete",
                "message": "File processing completed successfully"
            }):
                raise UploadCancelled(process_id)

    except UploadCancelled:
        logger.info(
            "upload worker stopped after cancellation or lease loss",
            extra={"process_id": process_id},
        )
    except UploadRetryable:
        logger.warning(
            "upload worker left for durable finalization retry",
            extra={"process_id": process_id},
        )
    except (ItemExecutionTimeout, ItemExecutionFailure):
        # Reconciliation itself is externally mutable. Never publish a
        # terminal job until its isolated child confirms cleanup completed;
        # leave the lease to expire into the durable recovery path.
        logger.warning(
            "upload reconciliation left for durable retry",
            extra={"process_id": process_id},
        )
    except Exception:
        logger.exception(
            "upload worker failed", extra={"run_id": job.identity.run_id}
        )
        # A repository call can fail after the isolated child has committed
        # graph/search effects but before the item checkpoint is observable to
        # this process. Never guess that such a job is safe to terminalize.
        # Lease recovery preserves the processing marker and compensates the
        # logical document before replay or terminal failure.
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=2.0)
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        try:
            _cleanup_one_terminal_staging(process_id)
        except Exception:
            logger.exception(
                "terminal upload staging reconciliation failed",
                extra={"process_id": process_id},
            )


def start_processing(
    job: IngestJob,
    uploads: Sequence[tuple[str, BinaryIO]],
) -> str:
    """Persist an owner ticket, stage bytes, then dispatch the durable job.

    Returns the process id the progress endpoints are polled with. The route
    has validated and rewound the multipart spools and minted the complete
    ticket by the time this runs. All streams are copied into fenced durable
    staging before this function returns; no request object reaches a worker.
    """
    process_id = uuid.uuid4().hex
    files_data = stage_files(uploads)
    staged = plan_staged_files(files_data)
    repository = _jobs()
    repository.create_staging(
        process_id,
        job,
        staged,
        (
            {
                "role": "upload_progress",
                "step": "upload",
                "message": "Starting file processing...",
            },
        ),
        staging_node_id=_UPLOAD_NODE_ID,
    )
    try:
        with upload_staging_lock(job.identity.user_id):
            try:
                ticket = repository.get(
                    process_id, include_items=False, include_progress=False
                )
                if (
                    ticket is None
                    or ticket.user_id != job.identity.user_id
                    or ticket.status is not UploadJobState.STAGING
                    or ticket.cancellation_requested
                    or not UserRepository().accepts_upload_work(job.identity.user_id)
                ):
                    raise UploadAccountUnavailable(
                        "account is unavailable for upload"
                    )
                persist_staged_files(process_id, files_data, planned=staged)
                if not repository.finalize_staging(
                    process_id, job.identity.user_id
                ):
                    raise UploadAccountUnavailable(
                        "account is unavailable for upload"
                    )
            except BaseException:
                # Account deletion may have cancelled the ticket while this
                # request held the staging lock. Remove any bytes before the
                # lock is released, then leave a terminal durable audit row.
                try:
                    _remove_staged_files(process_id)
                except FileNotFoundError:
                    pass
                try:
                    repository.fail_staging(
                        process_id,
                        job.identity.user_id,
                        {
                            "role": "upload_progress",
                            "step": "error",
                            "message": "Upload staging did not complete",
                        },
                    )
                except Exception:
                    logger.exception(
                        "upload staging failure could not be recorded",
                        extra={"process_id": process_id},
                    )
                raise
    except BaseException:
        # Once terminal, acknowledge cleanup and remove the retry handle only
        # after bytes are known absent. A concurrent account deletion may have
        # already done both, which is also success.
        try:
            _cleanup_one_terminal_staging(process_id)
        except Exception:
            logger.exception(
                "failed upload staging reconciliation failed",
                extra={"process_id": process_id},
            )
        raise

    # Reserve the committed row before dispatch. If process creation fails the
    # reservation is returned to the queue; a supervisor can retry it.
    try:
        _dispatch_upload_worker(process_id=process_id)
    except Exception:
        logger.exception(
            "initial upload dispatch failed; durable job remains queued",
            extra={"process_id": process_id},
        )
    return process_id


def _launch_upload_worker(process_id: str, owner: str, worker_node_id: str) -> int | None:
    _reap_upload_workers()
    process = _UPLOAD_PROCESS_FACTORY(
        target=process_files_worker,
        args=(process_id, owner, worker_node_id),
    )
    with _LOCAL_UPLOAD_PROCESSES_LOCK:
        if _UPLOAD_SHUTDOWN_REQUESTED.is_set():
            raise RuntimeError("upload worker dispatch is shutting down")
        if hasattr(process, "daemon"):
            process.daemon = False
        process.start()
        required = ("is_alive", "join", "terminate", "kill")
        if all(callable(getattr(process, name, None)) for name in required):
            _LOCAL_UPLOAD_PROCESSES[id(process)] = process
    return int(process.pid) if process.pid is not None else None


def _dispatch_upload_worker(*, process_id: str | None = None) -> bool:
    """Reserve one row, then spawn exactly one child for that reservation."""
    repository = _jobs()
    owner = uuid.uuid4().hex
    reserved = repository.reserve_next(
        owner,
        _UPLOAD_NODE_ID,
        UPLOAD_STARTUP_LEASE_SECONDS,
        process_id=process_id,
        max_workers=settings.max_upload_workers,
    )
    if reserved is None:
        return False
    try:
        worker_pid = _launch_upload_worker(reserved, owner, _UPLOAD_NODE_ID)
    except BaseException:
        # No child exists when Process.start() raises. If PostgreSQL is also
        # unavailable, the short reservation lease still recovers it.
        try:
            repository.release_reservation(reserved, owner)
        except Exception:
            logger.exception(
                "upload reservation release failed",
                extra={"process_id": reserved},
            )
        raise
    logger.info(
        "upload worker started",
        extra={"process_id": reserved, "worker_pid": worker_pid},
    )
    return True


def recover_upload_jobs() -> int:
    """Dispatch queued jobs and re-dispatch workers whose leases expired.

    Every API worker may run this. ``reserve_next`` serializes the capacity
    decision for this node and uses ``SKIP LOCKED`` before process creation, so
    concurrent supervisors respect the shared node limit and each launch at
    most one distinct job.
    """
    if _UPLOAD_SHUTDOWN_REQUESTED.is_set():
        return 0
    repository = _jobs()
    _reap_upload_workers()
    repository.expire_staging(max_age_seconds=UPLOAD_ORPHAN_GRACE_SECONDS)
    repository.requeue_expired()
    cleanup_terminal_staging()
    cleanup_orphan_staging()
    try:
        return int(_dispatch_upload_worker())
    except Exception:
        logger.exception("could not dispatch queued upload")
        return 0


async def supervise_upload_jobs() -> None:
    """Continuously recover queued/orphaned jobs for this API process."""
    while True:
        try:
            await asyncio.to_thread(recover_upload_jobs)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("upload job supervisor pass failed")
        await asyncio.sleep(UPLOAD_SUPERVISOR_SECONDS)


def upload_job(process_id: str, *, progress: bool = True) -> UploadJobRecord | None:
    return _jobs().get(process_id, include_items=False, include_progress=progress)


def upload_events_after(process_id: str, after_id: int) -> tuple[UploadProgressEvent, ...]:
    return _jobs().events_after(process_id, after_id)


def cancel_processing(
    process_id: str, user_id: str, *, cleanup_staging: bool = True
) -> bool:
    """Persist cancellation, then interrupt the currently leased worker."""
    found, _changed, worker_pid, start_token, worker_node_id = _jobs().request_cancel(
        process_id, user_id
    )
    if not found:
        return False
    if (
        worker_pid is not None
        and worker_pid != os.getpid()
        and worker_node_id == _UPLOAD_NODE_ID
        and start_token is not None
        and _process_start_token(worker_pid) == start_token
    ):
        try:
            os.kill(worker_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            logger.exception(
                "upload worker signal failed",
                extra={"worker_pid": worker_pid, "process_id": process_id},
            )
    # Any replica can attest that bytes are gone from shared durable staging.
    # Usually the worker's finally block wins; otherwise a supervisor
    # reconciles the terminal row after a crash.
    if cleanup_staging:
        try:
            _cleanup_one_terminal_staging(process_id)
        except Exception:
            logger.exception(
                "cancelled upload staging removal failed",
                extra={"process_id": process_id},
            )
    return True


def cleanup_upload_jobs(
    user_id: str,
    *,
    require_all: bool = False,
    staging_lock_held: bool = False,
) -> int:
    repository = _jobs()
    process_ids = repository.terminal_ids_for_user(user_id)
    failures: list[str] = []
    cleaned = 0
    for process_id in process_ids:
        try:
            staging_cleaned = _cleanup_one_terminal_staging(
                process_id, staging_lock_held=staging_lock_held
            )
        except Exception:
            logger.exception(
                "upload staging removal failed",
                extra={"process_id": process_id},
            )
            failures.append(process_id)
            continue
        if not staging_cleaned:
            # The shared staged copy must be erased and acknowledged before
            # this durable retry/status handle may be deleted.
            failures.append(process_id)
            continue
        if repository.delete_terminal(process_id, user_id):
            cleaned += 1
    if require_all and failures:
        raise RuntimeError("could not erase all durable upload staging")
    return cleaned


def cancel_user_uploads(user_id: str, *, cleanup_staging: bool = True) -> int:
    """Cancel every queued/running upload before an account is erased."""
    cancelled = 0
    for process_id in _jobs().active_ids_for_user(user_id):
        if cancel_processing(
            process_id, user_id, cleanup_staging=cleanup_staging
        ):
            cancelled += 1
    return cancelled


def settle_user_upload_cancellations(user_id: str) -> int:
    """Finalize only cancellations that have no in-flight item checkpoint."""
    return _jobs().settle_cancellations_for_user(user_id)


def discard_user_uploads_for_account_deletion(user_id: str) -> int:
    """Abandon partial checkpoints before the locked tenant-store erasure."""
    return _jobs().discard_cancellations_for_user(user_id)


def document_chunks(user_id: str, file_name: str) -> list[dict[str, Any]]:
    """Every stored chunk of one of this user's documents, in reading order."""
    return chunks_of_document(user_id, file_name)


# ------------------------------------------------------------------ deletion

class DocumentNotFound(LookupError):
    """No such document for this user."""


class DocumentNameAmbiguous(RuntimeError):
    """Legacy rows collide after canonical Unicode normalization."""


async def delete_document(user_id: str, file_name: str) -> dict[str, Any]:
    """Remove one document from every store that holds a piece of it.

    The Postgres row is confirmed first and removed last. While it stands the
    document is listed and the operation can be retried; a store that refuses
    -- chunks or graph -- is raised, not logged, so the caller learns the
    document is still there rather than finding it gone from the list with
    its data left behind.
    """
    canonical_name = canonical_document_name(file_name)
    removed: dict[str, Any] = {"chunks": 0, "entities": 0, "relations": 0}
    documents = KnowledgeBaseRepository()

    # Under the tenant lock so a re-ingest of the same name cannot interleave
    # with this delete. The account gate and row are deliberately resolved
    # only after taking the lock: a request that waited behind account erasure
    # must not construct a graph repository and recreate empty tenant files.
    # Bounded: a request must not queue for an hour behind an extraction; the
    # route answers "busy" instead.
    with tenant_lock(str(user_id), timeout_s=30.0):
        if not UserRepository().accepts_upload_work(str(user_id)):
            raise DocumentNotFound(canonical_name)
        stored_names = documents.names_matching_canonical(
            str(user_id), canonical_name
        )
        if not stored_names:
            raise DocumentNotFound(canonical_name)
        if len(stored_names) != 1:
            raise DocumentNameAmbiguous(canonical_name)
        stored_name = stored_names[0]

        # Graph goes first because a legacy aggregate may fail closed and
        # require reindexing. In that case no searchable chunks are removed.
        # A later ES failure is retryable: replaying graph deletion is a no-op.
        graph = await GraphRAGService(user_id).delete_file_data(stored_name)
        if not graph.get("success"):
            raise RuntimeError(
                f"graph cleanup failed for {stored_name}: {graph.get('error')}"
            )
        removed["entities"] = graph.get("entities_deleted", 0)
        removed["relations"] = graph.get("relationships_deleted", 0)
        removed["chunks"] = _store().delete_document(
            doc_name=stored_name, index_name=str(user_id)
        )
        if not documents.delete_exact(str(user_id), stored_name):
            raise RuntimeError("document metadata changed during deletion")

    return removed
