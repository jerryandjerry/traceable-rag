"""PostgreSQL-backed upload job queue and progress journal.

PostgreSQL is the authority for ownership, state, cancellation and ordered
progress.  Worker processes keep no status dictionary of their own.  A lease
lets another API process recover an interrupted job without allowing two
workers to execute it at once; per-item checkpoints make that recovery resume
at the first unfinished staged file.
"""
from __future__ import annotations

import json
import threading
import unicodedata
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import text

from visionagent.config.settings import settings
from visionagent.database.postgres.engine import SessionLocal
from visionagent.models import (
    IngestJob,
    StagedUpload,
    UploadItemState,
    UploadJobRecord,
    UploadJobState,
    UploadProgressEvent,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _json_value(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


class UploadAccountUnavailable(RuntimeError):
    """The account vanished or entered its durable deletion gate."""


class UploadDocumentConflict(RuntimeError):
    """An immutable logical document name is already owned or being staged."""


class UploadJobStore(Protocol):
    """Persistence contract consumed by the ingest orchestrator."""

    def create(self, process_id: str, job: IngestJob, items: Iterable[StagedUpload],
               initial_events: Iterable[dict[str, Any]], *, staging_node_id: str) -> None: ...
    def create_staging(
        self,
        process_id: str,
        job: IngestJob,
        items: Iterable[StagedUpload],
        initial_events: Iterable[dict[str, Any]],
        *,
        staging_node_id: str,
    ) -> None: ...
    def finalize_staging(self, process_id: str, user_id: str) -> bool: ...
    def fail_staging(self, process_id: str, user_id: str, event: dict[str, Any]) -> bool: ...
    def expire_staging(self, *, max_age_seconds: float, limit: int = 100) -> int: ...
    def get(self, process_id: str, *, include_items: bool = True,
            include_progress: bool = True) -> UploadJobRecord | None: ...
    def events_after(self, process_id: str, after_id: int) -> tuple[UploadProgressEvent, ...]: ...
    def reserve_next(self, owner: str, worker_node_id: str, lease_seconds: float,
                     *, process_id: str | None = None,
                     max_workers: int | None = None) -> str | None: ...
    def activate(self, process_id: str, owner: str, worker_pid: int,
                 worker_start_token: str | None, lease_seconds: float) -> bool: ...
    def release_reservation(self, process_id: str, owner: str) -> bool: ...
    def requeue_owned(self, process_id: str, owner: str,
                      event: dict[str, Any]) -> bool: ...
    def retry_document(self, process_id: str, owner: str, original_name: str,
                       event: dict[str, Any], *, refund_attempt: bool = False) -> bool: ...
    def reset_reconciled_document(
        self, process_id: str, owner: str, original_name: str
    ) -> bool: ...
    def finish_reconciled(
        self,
        process_id: str,
        owner: str,
        status: UploadJobState,
        compensated_documents: Iterable[str],
        event: dict[str, Any],
    ) -> bool: ...
    def owns_active_lease(self, process_id: str, owner: str) -> bool: ...
    def heartbeat(self, process_id: str, owner: str, lease_seconds: float) -> bool: ...
    def append_progress(self, process_id: str, owner: str, event: dict[str, Any],
                        lease_seconds: float) -> bool: ...
    def begin_item(self, process_id: str, owner: str, sequence: int) -> bool: ...
    def finish_item(self, process_id: str, owner: str, sequence: int, *, indexed_count: int,
                    process_time: float, error: str | None) -> bool: ...
    def finish(self, process_id: str, owner: str, status: UploadJobState,
               event: dict[str, Any]) -> bool: ...
    def finish_cancelled(self, process_id: str, owner: str,
                         event: dict[str, Any]) -> bool: ...
    def request_cancel(
        self, process_id: str, user_id: str
    ) -> tuple[bool, bool, int | None, str | None, str | None]: ...
    def settle_cancellations_for_user(self, user_id: str) -> int: ...
    def discard_cancellations_for_user(self, user_id: str) -> int: ...
    def requeue_expired(self, *, limit: int = 100) -> int: ...
    def terminal_staging_ids(self) -> list[str]: ...
    def mark_staging_cleaned(self, process_id: str) -> bool: ...
    def terminal_ids_for_user(self, user_id: str) -> list[str]: ...
    def delete_terminal(self, process_id: str, user_id: str) -> bool: ...
    def existing_ids(self, process_ids: Iterable[str]) -> set[str]: ...
    def active_ids_for_user(self, user_id: str) -> list[str]: ...


class UploadJobRepository:
    """Durable queue operations; every mutation is one database transaction."""

    def __init__(self, session_factory: Any = None) -> None:
        self._sessions = session_factory or SessionLocal

    @staticmethod
    def _insert_event(db: Any, process_id: str, event: dict[str, Any]) -> None:
        db.execute(
            text(
                "INSERT INTO upload_job_events (process_id, role, step, message, percent) "
                "VALUES (:pid, :role, :step, :message, :percent)"
            ),
            {
                "pid": process_id,
                "role": event.get("role", "upload_progress"),
                "step": event["step"],
                "message": event.get("message"),
                "percent": event.get("percent"),
            },
        )

    def create(
        self,
        process_id: str,
        job: IngestJob,
        items: Iterable[StagedUpload],
        initial_events: Iterable[dict[str, Any]],
        *,
        staging_node_id: str,
    ) -> None:
        """Compatibility helper that creates and immediately queues a ticket."""
        self.create_staging(
            process_id,
            job,
            items,
            initial_events,
            staging_node_id=staging_node_id,
        )
        if not self.finalize_staging(process_id, job.identity.user_id):
            raise RuntimeError("upload staging ticket could not be queued")

    def create_staging(
        self,
        process_id: str,
        job: IngestJob,
        items: Iterable[StagedUpload],
        initial_events: Iterable[dict[str, Any]],
        *,
        staging_node_id: str,
    ) -> None:
        """Create the owner-bearing, non-dispatchable ticket before bytes exist."""
        staged = tuple(items)
        db = self._sessions()
        try:
            # Serialize upload acceptance with the account-deletion gate. FOR
            # SHARE conflicts with the gate's UPDATE: either this transaction
            # commits a visible job first, or it observes the gate and refuses.
            account = db.execute(
                text(
                    "SELECT deletion_requested FROM users "
                    "WHERE id = :user_id FOR SHARE"
                ),
                {"user_id": int(job.identity.user_id)},
            ).fetchone()
            if account is None or bool(account.deletion_requested):
                raise UploadAccountUnavailable("account is unavailable for upload")
            # A transaction-scoped advisory lock makes the check-and-create
            # linearizable across API workers. Sort names so multi-file
            # batches cannot deadlock one another in opposite orders.
            document_names = sorted(
                {unicodedata.normalize("NFC", item.original_name) for item in staged}
            )
            for file_name in document_names:
                db.execute(
                    text("SELECT pg_advisory_xact_lock(hashtextextended(:claim, 0))"),
                    {"claim": f"{job.identity.user_id}\0{file_name}"},
                )
            conflict = db.execute(
                text(
                    "SELECT file_name FROM knowledgebases "
                    "WHERE user_id = :user_id "
                    "AND normalize(file_name, NFC) = ANY(:file_names) "
                    "UNION ALL "
                    "SELECT item.original_name FROM upload_job_items item "
                    "JOIN upload_jobs job ON job.process_id = item.process_id "
                    "WHERE job.user_id = :user_id "
                    "AND job.status IN ('staging', 'queued', 'processing') "
                    "AND normalize(item.original_name, NFC) = ANY(:file_names) LIMIT 1"
                ),
                {"user_id": job.identity.user_id, "file_names": document_names},
            ).fetchone()
            if conflict is not None:
                raise UploadDocumentConflict(
                    f"document {str(conflict.file_name)!r} already exists or is active"
                )
            db.execute(
                text(
                    "INSERT INTO upload_jobs "
                    "(process_id, run_id, user_id, job_payload, status, total_files, "
                    "max_attempts, staging_node_id) VALUES "
                    "(:pid, :run_id, :user_id, CAST(:job AS JSONB), 'staging', :total, "
                    ":max_attempts, :staging_node_id)"
                ),
                {
                    "pid": process_id,
                    "run_id": job.identity.run_id,
                    "user_id": job.identity.user_id,
                    "job": json.dumps(job.model_dump(mode="json")),
                    "total": len(staged),
                    "max_attempts": settings.max_upload_attempts,
                    "staging_node_id": staging_node_id,
                },
            )
            for item in staged:
                db.execute(
                    text(
                        "INSERT INTO upload_job_items "
                        "(process_id, sequence, original_name, part_name, storage_key, status) "
                        "VALUES (:pid, :sequence, :original, :part, :storage, 'pending')"
                    ),
                    {
                        "pid": process_id,
                        "sequence": item.sequence,
                        "original": item.original_name,
                        "part": item.part_name,
                        "storage": item.storage_key,
                    },
                )
            for event in initial_events:
                self._insert_event(db, process_id, event)
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def reset_reconciled_document(
        self, process_id: str, owner: str, original_name: str
    ) -> bool:
        """Reset a crash-ambiguous document after confirmed compensation."""
        db = self._sessions()
        try:
            job = db.execute(
                text(
                    "SELECT process_id FROM upload_jobs WHERE process_id=:pid "
                    "AND lease_owner=:owner AND status='processing' AND "
                    "cancellation_requested=FALSE FOR UPDATE"
                ),
                {"pid": process_id, "owner": owner},
            ).fetchone()
            if job is None:
                db.rollback()
                return False
            db.execute(
                text(
                    "UPDATE upload_job_items SET status='pending', indexed_count=0, "
                    "process_time=0, error=NULL, updated_at=NOW() WHERE "
                    "process_id=:pid AND original_name=:name"
                ),
                {"pid": process_id, "name": original_name},
            )
            db.execute(
                text(
                    "UPDATE upload_jobs SET processed_files=(SELECT COUNT(*) FROM "
                    "upload_job_items WHERE process_id=:pid AND status IN "
                    "('completed','failed')), total_chunks_inserted=(SELECT "
                    "COALESCE(SUM(indexed_count),0) FROM upload_job_items WHERE "
                    "process_id=:pid), updated_at=NOW() WHERE process_id=:pid "
                    "AND lease_owner=:owner AND status='processing'"
                ),
                {"pid": process_id, "owner": owner},
            )
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def finalize_staging(self, process_id: str, user_id: str) -> bool:
        """Queue a persisted ticket only while its account still accepts work."""
        db = self._sessions()
        try:
            # This lock serializes the final queue transition with account
            # deletion's gate update. The winner commits first; the loser then
            # observes the authoritative gate state.
            account = db.execute(
                text(
                    "SELECT deletion_requested FROM users "
                    "WHERE id = :user_id FOR SHARE"
                ),
                {"user_id": int(user_id)},
            ).fetchone()
            if account is None or bool(account.deletion_requested):
                raise UploadAccountUnavailable("account is unavailable for upload")
            row = db.execute(
                text(
                    "UPDATE upload_jobs SET status = 'queued', updated_at = NOW() "
                    "WHERE process_id = :pid AND user_id = :user_id "
                    "AND status = 'staging' AND cancellation_requested = FALSE "
                    "RETURNING process_id"
                ),
                {"pid": process_id, "user_id": user_id},
            ).fetchone()
            db.commit()
            return row is not None
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def fail_staging(
        self, process_id: str, user_id: str, event: dict[str, Any]
    ) -> bool:
        """Make a staging failure and its progress event durable together."""
        db = self._sessions()
        try:
            row = db.execute(
                text(
                    "UPDATE upload_jobs SET status = 'failed', lease_owner = NULL, "
                    "worker_pid = NULL, worker_start_token = NULL, worker_node_id = NULL, "
                    "lease_expires_at = NULL, updated_at = NOW() "
                    "WHERE process_id = :pid AND user_id = :user_id "
                    "AND status = 'staging' RETURNING process_id"
                ),
                {"pid": process_id, "user_id": user_id},
            ).fetchone()
            if row is not None:
                db.execute(
                    text(
                        "UPDATE upload_job_items SET status='failed', indexed_count=0, "
                        "process_time=0, error='upload staging failed', updated_at=NOW() "
                        "WHERE process_id=:pid AND status IN ('pending','processing')"
                    ),
                    {"pid": process_id},
                )
                db.execute(
                    text(
                        "UPDATE upload_jobs SET processed_files=(SELECT COUNT(*) FROM "
                        "upload_job_items WHERE process_id=:pid AND status IN "
                        "('completed','failed')), total_chunks_inserted=(SELECT "
                        "COALESCE(SUM(indexed_count),0) FROM upload_job_items WHERE "
                        "process_id=:pid) WHERE process_id=:pid"
                    ),
                    {"pid": process_id},
                )
                self._insert_event(db, process_id, event)
            db.commit()
            return row is not None
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def expire_staging(self, *, max_age_seconds: float, limit: int = 100) -> int:
        """Fail a bounded batch of crash-left staging tickets using DB time."""
        if max_age_seconds < 0:
            raise ValueError("max_age_seconds must be non-negative")
        if limit <= 0:
            return 0
        db = self._sessions()
        try:
            rows = db.execute(
                text(
                    "SELECT process_id FROM upload_jobs WHERE status = 'staging' "
                    "AND created_at < NOW() - (:max_age_seconds * INTERVAL '1 second') "
                    "ORDER BY created_at, process_id FOR UPDATE SKIP LOCKED LIMIT :limit"
                ),
                {"max_age_seconds": max_age_seconds, "limit": limit},
            ).fetchall()
            process_ids = [str(row.process_id) for row in rows]
            for process_id in process_ids:
                db.execute(
                    text(
                        "UPDATE upload_jobs SET status = 'failed', lease_owner = NULL, "
                        "worker_pid = NULL, worker_start_token = NULL, worker_node_id = NULL, "
                        "lease_expires_at = NULL, updated_at = NOW() "
                        "WHERE process_id = :pid AND status = 'staging'"
                    ),
                    {"pid": process_id},
                )
                db.execute(
                    text(
                        "UPDATE upload_job_items SET status='failed', indexed_count=0, "
                        "process_time=0, error='upload staging expired', updated_at=NOW() "
                        "WHERE process_id=:pid AND status IN ('pending','processing')"
                    ),
                    {"pid": process_id},
                )
                db.execute(
                    text(
                        "UPDATE upload_jobs SET processed_files=(SELECT COUNT(*) FROM "
                        "upload_job_items WHERE process_id=:pid AND status IN "
                        "('completed','failed')), total_chunks_inserted=(SELECT "
                        "COALESCE(SUM(indexed_count),0) FROM upload_job_items WHERE "
                        "process_id=:pid) WHERE process_id=:pid"
                    ),
                    {"pid": process_id},
                )
                self._insert_event(
                    db,
                    process_id,
                    {
                        "role": "upload_progress",
                        "step": "error",
                        "message": "Upload staging expired before queueing",
                    },
                )
            db.commit()
            return len(process_ids)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _items(self, db: Any, process_id: str) -> tuple[StagedUpload, ...]:
        rows = db.execute(
            text(
                "SELECT sequence, original_name, part_name, storage_key, status, "
                "indexed_count, process_time, error FROM upload_job_items "
                "WHERE process_id = :pid ORDER BY sequence"
            ),
            {"pid": process_id},
        ).fetchall()
        return tuple(
            StagedUpload(
                sequence=int(row.sequence),
                original_name=str(row.original_name),
                part_name=str(row.part_name),
                storage_key=str(row.storage_key),
                status=UploadItemState(str(row.status)),
                indexed_count=int(row.indexed_count or 0),
                process_time=float(row.process_time or 0.0),
                error=str(row.error) if row.error is not None else None,
            )
            for row in rows
        )

    def _events(
        self, db: Any, process_id: str, *, after_id: int = 0
    ) -> tuple[UploadProgressEvent, ...]:
        rows = db.execute(
            text(
                "SELECT id, role, step, message, percent, created_at "
                "FROM upload_job_events WHERE process_id = :pid AND id > :after "
                "ORDER BY id"
            ),
            {"pid": process_id, "after": after_id},
        ).fetchall()
        return tuple(
            UploadProgressEvent(
                id=int(row.id),
                role=str(row.role),
                step=str(row.step),
                message=str(row.message) if row.message is not None else None,
                percent=int(row.percent) if row.percent is not None else None,
                created_at=row.created_at,
            )
            for row in rows
        )

    def get(
        self, process_id: str, *, include_items: bool = True, include_progress: bool = True
    ) -> UploadJobRecord | None:
        db = self._sessions()
        try:
            row = db.execute(
                text(
                    "SELECT process_id, job_payload, status, cancellation_requested, "
                    "total_files, processed_files, total_chunks_inserted, worker_pid, "
                    "worker_start_token, staging_node_id, staging_cleaned, worker_node_id, "
                    "lease_owner, lease_expires_at, attempt_count, max_attempts, "
                    "last_failure_class, next_attempt_at, created_at, updated_at "
                    "FROM upload_jobs WHERE process_id = :pid"
                ),
                {"pid": process_id},
            ).fetchone()
            if row is None:
                return None
            return UploadJobRecord(
                process_id=str(row.process_id),
                job=IngestJob.model_validate(_json_value(row.job_payload)),
                status=UploadJobState(str(row.status)),
                cancellation_requested=bool(row.cancellation_requested),
                total_files=int(row.total_files),
                processed_files=int(row.processed_files or 0),
                total_chunks_inserted=int(row.total_chunks_inserted or 0),
                worker_pid=int(row.worker_pid) if row.worker_pid is not None else None,
                worker_start_token=(
                    str(row.worker_start_token) if row.worker_start_token is not None else None
                ),
                staging_node_id=(
                    str(row.staging_node_id) if row.staging_node_id is not None else None
                ),
                staging_cleaned=bool(row.staging_cleaned),
                worker_node_id=(
                    str(row.worker_node_id) if row.worker_node_id is not None else None
                ),
                lease_owner=str(row.lease_owner) if row.lease_owner is not None else None,
                lease_expires_at=row.lease_expires_at,
                attempt_count=int(row.attempt_count or 0),
                max_attempts=int(row.max_attempts),
                last_failure_class=(
                    str(row.last_failure_class) if row.last_failure_class else None
                ),
                next_attempt_at=row.next_attempt_at,
                created_at=row.created_at,
                updated_at=row.updated_at,
                items=self._items(db, process_id) if include_items else (),
                progress=self._events(db, process_id) if include_progress else (),
            )
        finally:
            db.close()

    def events_after(self, process_id: str, after_id: int) -> tuple[UploadProgressEvent, ...]:
        db = self._sessions()
        try:
            return self._events(db, process_id, after_id=after_id)
        finally:
            db.close()

    def reserve_next(
        self,
        owner: str,
        worker_node_id: str,
        lease_seconds: float,
        *,
        process_id: str | None = None,
        max_workers: int | None = None,
    ) -> str | None:
        """Atomically reserve one queued job before a child is spawned.

        ``SKIP LOCKED`` lets every API process poll the same queue without all
        of them dispatching the same backlog. A preferred id is used only by
        the request that just created that job; supervisors reserve the oldest
        available row.
        """
        db = self._sessions()
        try:
            if max_workers is not None:
                if max_workers <= 0:
                    raise ValueError("max_workers must be positive")
                # Every API process sharing this node id takes the same
                # transaction-scoped lock. The live-count and reservation
                # update are therefore one serialized capacity decision;
                # reservations count before a child PID exists.
                db.execute(
                    text(
                        "SELECT pg_advisory_xact_lock(1447110735, hashtext(:node_id))"
                    ),
                    {"node_id": worker_node_id},
                )
                capacity = db.execute(
                    text(
                        "SELECT COUNT(*) AS live_workers FROM upload_jobs "
                        "WHERE status = 'processing' AND worker_node_id = :node_id "
                        "AND lease_expires_at > NOW()"
                    ),
                    {"node_id": worker_node_id},
                ).fetchone()
                if capacity is not None and int(capacity.live_workers) >= max_workers:
                    db.commit()
                    return None
            predicate = "AND process_id = :preferred " if process_id is not None else ""
            parameters: dict[str, object] = {
                "owner": owner,
                "node_id": worker_node_id,
                "lease_seconds": lease_seconds,
            }
            if process_id is not None:
                parameters["preferred"] = process_id
            row = db.execute(
                text(
                    "WITH candidate AS ("
                    " SELECT process_id FROM upload_jobs WHERE status = 'queued' "
                    " AND (next_attempt_at IS NULL OR next_attempt_at <= NOW()) "
                    f" {predicate}"
                    " ORDER BY created_at, process_id FOR UPDATE SKIP LOCKED LIMIT 1"
                    ") UPDATE upload_jobs AS job SET status = 'processing', "
                    "attempt_count = attempt_count + 1, next_attempt_at = NULL, "
                    "lease_owner = :owner, worker_node_id = :node_id, worker_pid = NULL, "
                    "worker_start_token = NULL, "
                    "lease_expires_at = NOW() + (:lease_seconds * INTERVAL '1 second'), "
                    "updated_at = NOW() FROM candidate "
                    "WHERE job.process_id = candidate.process_id RETURNING job.process_id"
                ),
                parameters,
            ).fetchone()
            db.commit()
            return str(row.process_id) if row is not None else None
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def activate(
        self,
        process_id: str,
        owner: str,
        worker_pid: int,
        worker_start_token: str | None,
        lease_seconds: float,
    ) -> bool:
        """Attach the child identity to a reservation it already owns."""
        db = self._sessions()
        try:
            row = db.execute(
                text(
                    "UPDATE upload_jobs SET worker_pid = :worker_pid, "
                    "worker_start_token = :start_token, "
                    "lease_expires_at = NOW() + (:lease_seconds * INTERVAL '1 second'), "
                    "updated_at = NOW() WHERE process_id = :pid "
                    "AND lease_owner = :owner AND status = 'processing' "
                    "RETURNING process_id"
                ),
                {
                    "pid": process_id,
                    "owner": owner,
                    "worker_pid": worker_pid,
                    "start_token": worker_start_token,
                    "lease_seconds": lease_seconds,
                },
            ).fetchone()
            db.commit()
            return row is not None
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def release_reservation(self, process_id: str, owner: str) -> bool:
        """Return an unstarted reservation to the queue after spawn failure."""
        db = self._sessions()
        try:
            job = db.execute(
                text(
                    "SELECT attempt_count, max_attempts, EXISTS (SELECT 1 FROM "
                    "upload_job_items item WHERE item.process_id=:pid AND "
                    "item.status!='pending') AS requires_reconciliation FROM "
                    "upload_jobs WHERE process_id=:pid AND lease_owner=:owner "
                    "AND status='processing' AND worker_pid IS NULL FOR UPDATE"
                ),
                {"pid": process_id, "owner": owner},
            ).fetchone()
            if job is None:
                db.rollback()
                return False
            requires_reconciliation = bool(job.requires_reconciliation)
            exhausted = (
                int(job.attempt_count) >= int(job.max_attempts)
                and not requires_reconciliation
            )
            attempt_count = max(
                0,
                int(job.attempt_count) - (1 if requires_reconciliation else 0),
            )
            db.execute(
                text(
                    "UPDATE upload_jobs SET status=:status, attempt_count=:attempt_count, "
                    "lease_owner=NULL, worker_node_id=NULL, worker_pid=NULL, "
                    "worker_start_token=NULL, lease_expires_at=NULL, "
                    "last_failure_class='worker_spawn_failed', next_attempt_at=:next, "
                    "updated_at=NOW() WHERE process_id=:pid AND lease_owner=:owner "
                    "AND status='processing' AND worker_pid IS NULL"
                ),
                {
                    "pid": process_id,
                    "owner": owner,
                    "status": "failed" if exhausted else "queued",
                    "attempt_count": attempt_count,
                    "next": None if exhausted else _utcnow() + timedelta(
                        seconds=settings.upload_retry_backoff_s
                        * (2 ** max(0, attempt_count - 1))
                    ),
                },
            )
            if exhausted:
                db.execute(
                    text(
                        "UPDATE upload_job_items SET status='failed', indexed_count=0, "
                        "process_time=0, error='worker startup attempt budget exhausted', "
                        "updated_at=NOW() WHERE process_id=:pid AND status IN "
                        "('pending','processing')"
                    ),
                    {"pid": process_id},
                )
                db.execute(
                    text(
                        "UPDATE upload_jobs SET processed_files=(SELECT COUNT(*) FROM "
                        "upload_job_items WHERE process_id=:pid AND status IN "
                        "('completed','failed')), total_chunks_inserted=(SELECT "
                        "COALESCE(SUM(indexed_count),0) FROM upload_job_items WHERE "
                        "process_id=:pid) WHERE process_id=:pid"
                    ),
                    {"pid": process_id},
                )
                self._insert_event(db, process_id, {
                    "step": "error",
                    "message": "Upload stopped after repeated worker startup failures",
                })
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def requeue_owned(
        self, process_id: str, owner: str, event: dict[str, Any]
    ) -> bool:
        """Relinquish an activated worker at a durable item boundary.

        The worker calls this only while it is not executing an item. A
        ``processing`` item can still be present after crash recovery when the
        replacement receives SIGTERM before replay; preserve that checkpoint
        so cancellation cannot skip required finalization.
        """
        db = self._sessions()
        try:
            row = db.execute(
                text(
                    "UPDATE upload_jobs AS job SET status = 'queued', lease_owner = NULL, "
                    "worker_node_id = NULL, worker_pid = NULL, worker_start_token = NULL, "
                    "lease_expires_at = NULL, attempt_count = GREATEST(0, attempt_count - 1), "
                    "next_attempt_at = NULL, updated_at = NOW() "
                    "WHERE job.process_id = :pid AND job.lease_owner = :owner "
                    "AND job.status = 'processing' "
                    "RETURNING job.process_id"
                ),
                {
                    "pid": process_id,
                    "owner": owner,
                },
            ).fetchone()
            if row is not None:
                self._insert_event(db, process_id, event)
            db.commit()
            return row is not None
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def retry_document(
        self,
        process_id: str,
        owner: str,
        original_name: str,
        event: dict[str, Any],
        *,
        refund_attempt: bool = False,
    ) -> bool:
        """Reset one compensated document for a later, non-terminal attempt.

        A normal ambiguous item failure consumes its attempt. A cooperative
        host shutdown does not, because the worker deliberately stopped the
        isolated child and compensated its effects; ``refund_attempt`` makes
        that distinction durable.
        """
        db = self._sessions()
        try:
            job = db.execute(
                text(
                    "SELECT attempt_count, max_attempts, cancellation_requested FROM upload_jobs "
                    "WHERE process_id=:pid AND lease_owner=:owner AND status='processing' "
                    "FOR UPDATE"
                ),
                {"pid": process_id, "owner": owner},
            ).fetchone()
            if job is None:
                db.rollback()
                return False
            next_attempt_count = max(
                0, int(job.attempt_count) - (1 if refund_attempt else 0)
            )
            if bool(job.cancellation_requested) or next_attempt_count >= int(job.max_attempts):
                db.rollback()
                return False
            db.execute(text(
                "UPDATE upload_job_items SET status='pending', indexed_count=0, "
                "process_time=0, error=NULL, updated_at=NOW() WHERE process_id=:pid "
                "AND original_name=:name"
            ), {"pid": process_id, "name": original_name})
            db.execute(text(
                "UPDATE upload_jobs SET status='queued', lease_owner=NULL, worker_node_id=NULL, "
                    "worker_pid=NULL, worker_start_token=NULL, lease_expires_at=NULL, "
                "attempt_count=:attempt_count, "
                "processed_files=(SELECT COUNT(*) FROM upload_job_items WHERE process_id=:pid "
                "AND status IN ('completed','failed')), total_chunks_inserted=(SELECT "
                "COALESCE(SUM(indexed_count),0) FROM upload_job_items WHERE process_id=:pid), "
                "last_failure_class=:failure, next_attempt_at=:next, updated_at=NOW() "
                "WHERE process_id=:pid"
            ), {
                "pid": process_id,
                "attempt_count": next_attempt_count,
                "failure": str(event.get("failure_class", "item_ambiguous"))[:64],
                "next": None if refund_attempt else _utcnow() + timedelta(
                    seconds=settings.upload_retry_backoff_s
                    * (2 ** max(0, next_attempt_count - 1))
                ),
            })
            self._insert_event(db, process_id, event)
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def finish_reconciled(
        self,
        process_id: str,
        owner: str,
        status: UploadJobState,
        compensated_documents: Iterable[str],
        event: dict[str, Any],
    ) -> bool:
        """Atomically publish a terminal job after external compensation.

        Completed items from other logical documents remain authoritative.
        Every part of a compensated document, plus every untouched future
        item, becomes failed with zero counters before the job can become
        terminal. Thus a preserved completed checkpoint always belongs to a
        document whose metadata was finalized by the worker first.
        """
        if status not in {UploadJobState.FAILED, UploadJobState.CANCELLED}:
            raise ValueError("reconciliation may finish only as failed or cancelled")
        document_names = tuple(dict.fromkeys(compensated_documents))
        db = self._sessions()
        try:
            job = db.execute(
                text(
                    "SELECT cancellation_requested FROM upload_jobs "
                    "WHERE process_id=:pid AND lease_owner=:owner "
                    "AND status='processing' FOR UPDATE"
                ),
                {"pid": process_id, "owner": owner},
            ).fetchone()
            expected_cancellation = status is UploadJobState.CANCELLED
            if job is None or bool(job.cancellation_requested) != expected_cancellation:
                db.rollback()
                return False
            db.execute(
                text(
                    "UPDATE upload_job_items SET status='failed', indexed_count=0, "
                    "process_time=0, error=CASE WHEN original_name = "
                    "ANY(CAST(:documents AS varchar[])) THEN :compensated_error "
                    "ELSE :unstarted_error END, updated_at=NOW() "
                    "WHERE process_id=:pid AND (original_name = "
                    "ANY(CAST(:documents AS varchar[])) "
                    "OR status IN ('pending','processing'))"
                ),
                {
                    "pid": process_id,
                    "documents": list(document_names),
                    "compensated_error": (
                        "cancelled after compensation"
                        if expected_cancellation
                        else "attempt budget exhausted after compensation"
                    ),
                    "unstarted_error": (
                        "cancelled before processing"
                        if expected_cancellation
                        else "attempt budget exhausted before processing"
                    ),
                },
            )
            row = db.execute(
                text(
                    "UPDATE upload_jobs SET status=:status, lease_owner=NULL, "
                    "worker_node_id=NULL, worker_pid=NULL, worker_start_token=NULL, "
                    "lease_expires_at=NULL, processed_files=(SELECT COUNT(*) FROM "
                    "upload_job_items WHERE process_id=:pid AND status IN "
                    "('completed','failed')), total_chunks_inserted=(SELECT "
                    "COALESCE(SUM(indexed_count),0) FROM upload_job_items WHERE "
                    "process_id=:pid), last_failure_class=:failure, "
                    "next_attempt_at=NULL, updated_at=NOW() WHERE process_id=:pid "
                    "AND lease_owner=:owner AND status='processing' RETURNING process_id"
                ),
                {
                    "pid": process_id,
                    "owner": owner,
                    "status": status.value,
                    "failure": str(event.get("failure_class", "item_ambiguous"))[:64],
                },
            ).fetchone()
            if row is None:
                db.rollback()
                return False
            self._insert_event(db, process_id, event)
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def owns_active_lease(self, process_id: str, owner: str) -> bool:
        """Fence external mutations by the current unexpired reservation."""
        db = self._sessions()
        try:
            row = db.execute(
                text(
                    "SELECT 1 FROM upload_jobs WHERE process_id = :pid "
                    "AND lease_owner = :owner AND status = 'processing' "
                    "AND lease_expires_at > NOW()"
                ),
                {"pid": process_id, "owner": owner},
            ).fetchone()
            return row is not None
        finally:
            db.close()

    def heartbeat(self, process_id: str, owner: str, lease_seconds: float) -> bool:
        db = self._sessions()
        try:
            row = db.execute(
                text(
                    "UPDATE upload_jobs SET lease_expires_at = "
                    "NOW() + (:lease_seconds * INTERVAL '1 second'), updated_at = NOW() "
                    "WHERE process_id = :pid AND lease_owner = :owner "
                    "AND status = 'processing' "
                    "RETURNING process_id"
                ),
                {
                    "pid": process_id,
                    "owner": owner,
                    "lease_seconds": lease_seconds,
                },
            ).fetchone()
            db.commit()
            return row is not None
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def append_progress(
        self,
        process_id: str,
        owner: str,
        event: dict[str, Any],
        lease_seconds: float,
    ) -> bool:
        """Append only while this worker owns the active job."""
        db = self._sessions()
        try:
            active = db.execute(
                text(
                    "UPDATE upload_jobs SET lease_expires_at = "
                    "NOW() + (:lease_seconds * INTERVAL '1 second'), updated_at = NOW() "
                    "WHERE process_id = :pid AND lease_owner = :owner "
                    "AND status = 'processing' "
                    "RETURNING process_id"
                ),
                {
                    "pid": process_id,
                    "owner": owner,
                    "lease_seconds": lease_seconds,
                },
            ).fetchone()
            if active is not None:
                self._insert_event(db, process_id, event)
            db.commit()
            return active is not None
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def begin_item(self, process_id: str, owner: str, sequence: int) -> bool:
        db = self._sessions()
        try:
            row = db.execute(
                text(
                    "UPDATE upload_job_items AS item "
                    "SET status = 'processing', updated_at = NOW() "
                    "WHERE item.process_id = :pid AND item.sequence = :sequence "
                    "AND item.status IN ('pending', 'processing') "
                    "AND EXISTS (SELECT 1 FROM upload_jobs j WHERE j.process_id = :pid "
                    "AND j.lease_owner = :owner AND j.status = 'processing' "
                    "AND (item.status = 'processing' "
                    "OR j.cancellation_requested = FALSE) "
                    ") RETURNING sequence"
                ),
                {"pid": process_id, "owner": owner, "sequence": sequence},
            ).fetchone()
            db.commit()
            return row is not None
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def finish_item(
        self,
        process_id: str,
        owner: str,
        sequence: int,
        *,
        indexed_count: int,
        process_time: float,
        error: str | None,
    ) -> bool:
        db = self._sessions()
        try:
            row = db.execute(
                text(
                    "UPDATE upload_job_items SET status = :status, indexed_count = :chunks, "
                    "process_time = :duration, error = :error, updated_at = NOW() "
                    "WHERE process_id = :pid AND sequence = :sequence AND status = 'processing' "
                    "AND EXISTS (SELECT 1 FROM upload_jobs j WHERE j.process_id = :pid "
                    "AND j.lease_owner = :owner AND j.status = 'processing' "
                    ") RETURNING sequence"
                ),
                {
                    "pid": process_id,
                    "owner": owner,
                    "sequence": sequence,
                    "status": "failed" if error else "completed",
                    "chunks": indexed_count,
                    "duration": process_time,
                    "error": error,
                },
            ).fetchone()
            if row is not None:
                db.execute(
                    text(
                        "UPDATE upload_jobs SET "
                        "processed_files = (SELECT COUNT(*) FROM upload_job_items "
                        " WHERE process_id = :pid AND status IN ('completed', 'failed')), "
                        "total_chunks_inserted = (SELECT COALESCE(SUM(indexed_count), 0) "
                        " FROM upload_job_items WHERE process_id = :pid), updated_at = NOW() "
                        "WHERE process_id = :pid AND lease_owner = :owner"
                    ),
                    {"pid": process_id, "owner": owner},
                )
            db.commit()
            return row is not None
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def finish(
        self,
        process_id: str,
        owner: str,
        status: UploadJobState,
        event: dict[str, Any],
    ) -> bool:
        if status not in {UploadJobState.COMPLETED, UploadJobState.FAILED}:
            raise ValueError("a worker may finish only as completed or failed")
        db = self._sessions()
        try:
            job = db.execute(
                text(
                    "SELECT cancellation_requested, EXISTS (SELECT 1 FROM "
                    "upload_job_items item WHERE item.process_id=:pid AND "
                    "item.status='processing') AS has_processing, EXISTS (SELECT 1 "
                    "FROM upload_job_items item WHERE item.process_id=:pid AND "
                    "item.status!='completed') AS has_incomplete FROM upload_jobs "
                    "WHERE process_id=:pid AND lease_owner=:owner "
                    "AND status='processing' FOR UPDATE"
                ),
                {"pid": process_id, "owner": owner},
            ).fetchone()
            if (
                job is None
                or bool(job.cancellation_requested)
                or bool(job.has_processing)
                or (
                    status is UploadJobState.COMPLETED
                    and bool(job.has_incomplete)
                )
            ):
                db.rollback()
                return False
            if status is UploadJobState.FAILED:
                db.execute(
                    text(
                        "UPDATE upload_job_items SET status='failed', indexed_count=0, "
                        "process_time=0, error='processing stopped before item began', "
                        "updated_at=NOW() WHERE process_id=:pid AND status='pending'"
                    ),
                    {"pid": process_id},
                )
            row = db.execute(
                text(
                    "UPDATE upload_jobs SET status=:status, lease_owner=NULL, "
                    "worker_pid=NULL, worker_start_token=NULL, worker_node_id=NULL, "
                    "lease_expires_at=NULL, processed_files=(SELECT COUNT(*) FROM "
                    "upload_job_items WHERE process_id=:pid AND status IN "
                    "('completed','failed')), total_chunks_inserted=(SELECT "
                    "COALESCE(SUM(indexed_count),0) FROM upload_job_items WHERE "
                    "process_id=:pid), updated_at=NOW() WHERE process_id=:pid "
                    "AND lease_owner=:owner AND status='processing' RETURNING process_id"
                ),
                {"pid": process_id, "owner": owner, "status": status.value},
            ).fetchone()
            if row is None:
                db.rollback()
                return False
            self._insert_event(db, process_id, event)
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def finish_cancelled(
        self, process_id: str, owner: str, event: dict[str, Any]
    ) -> bool:
        """Finish a cooperative cancellation after its in-flight item is safe."""
        db = self._sessions()
        try:
            job = db.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM upload_job_items item WHERE "
                    "item.process_id=:pid AND item.status='processing') AS "
                    "has_processing FROM upload_jobs WHERE process_id=:pid AND "
                    "lease_owner=:owner AND status='processing' AND "
                    "cancellation_requested=TRUE FOR UPDATE"
                ),
                {"pid": process_id, "owner": owner},
            ).fetchone()
            if job is None or bool(job.has_processing):
                db.rollback()
                return False
            db.execute(
                text(
                    "UPDATE upload_job_items SET status='failed', indexed_count=0, "
                    "process_time=0, error='cancelled before processing', "
                    "updated_at=NOW() WHERE process_id=:pid AND status='pending'"
                ),
                {"pid": process_id},
            )
            row = db.execute(
                text(
                    "UPDATE upload_jobs SET status='cancelled', lease_owner=NULL, "
                    "worker_pid=NULL, worker_start_token=NULL, worker_node_id=NULL, "
                    "lease_expires_at=NULL, processed_files=(SELECT COUNT(*) FROM "
                    "upload_job_items WHERE process_id=:pid AND status IN "
                    "('completed','failed')), total_chunks_inserted=(SELECT "
                    "COALESCE(SUM(indexed_count),0) FROM upload_job_items WHERE "
                    "process_id=:pid), updated_at=NOW() WHERE process_id=:pid AND "
                    "lease_owner=:owner AND status='processing' RETURNING process_id"
                ),
                {"pid": process_id, "owner": owner},
            ).fetchone()
            if row is None:
                db.rollback()
                return False
            self._insert_event(db, process_id, event)
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def request_cancel(
        self, process_id: str, user_id: str
    ) -> tuple[bool, bool, int | None, str | None, str | None]:
        """Return found, changed, and a live worker's node/PID fingerprint."""
        db = self._sessions()
        try:
            row = db.execute(
                text(
                    "SELECT status, cancellation_requested, worker_pid, "
                    "worker_start_token, worker_node_id, "
                    "(lease_expires_at IS NOT NULL AND lease_expires_at > NOW()) AS lease_live, "
                    "EXISTS (SELECT 1 FROM upload_job_items item "
                    " WHERE item.process_id = upload_jobs.process_id "
                    " AND item.status = 'processing') AS has_inflight, "
                    "EXISTS (SELECT 1 FROM upload_job_items item "
                    " WHERE item.process_id = upload_jobs.process_id "
                    " AND item.status IN ('completed','failed')) AS has_checkpoint "
                    "FROM upload_jobs "
                    "WHERE process_id = :pid AND user_id = :user_id FOR UPDATE"
                ),
                {"pid": process_id, "user_id": user_id},
            ).fetchone()
            if row is None:
                db.commit()
                return False, False, None, None, None
            state = UploadJobState(str(row.status))
            if state.terminal:
                db.commit()
                return True, False, None, None, None
            changed = not bool(row.cancellation_requested)
            # A staging ticket, never-started queued job, or unactivated
            # reservation has no downstream-store side effects and can become
            # terminal now. A recovered queued job may retain an in-flight
            # checkpoint; it must replay and finalize that item like an active
            # worker.
            immediate = (
                state in {UploadJobState.STAGING, UploadJobState.QUEUED}
                or row.worker_pid is None
            ) and not bool(row.has_inflight) and not bool(row.has_checkpoint)
            if changed:
                if immediate:
                    db.execute(
                        text(
                            "UPDATE upload_jobs SET status = 'cancelled', "
                            "cancellation_requested = TRUE, lease_owner = NULL, "
                            "worker_pid = NULL, worker_start_token = NULL, "
                            "worker_node_id = NULL, lease_expires_at = NULL, "
                            "updated_at = NOW() WHERE process_id = :pid"
                        ),
                        {"pid": process_id},
                    )
                    db.execute(
                        text(
                            "UPDATE upload_job_items SET status='failed', "
                            "indexed_count=0, process_time=0, "
                            "error='cancelled before processing', updated_at=NOW() "
                            "WHERE process_id=:pid AND status IN "
                            "('pending','processing')"
                        ),
                        {"pid": process_id},
                    )
                    db.execute(
                        text(
                            "UPDATE upload_jobs SET processed_files=(SELECT COUNT(*) "
                            "FROM upload_job_items WHERE process_id=:pid AND status IN "
                            "('completed','failed')), total_chunks_inserted=(SELECT "
                            "COALESCE(SUM(indexed_count),0) FROM upload_job_items "
                            "WHERE process_id=:pid) WHERE process_id=:pid"
                        ),
                        {"pid": process_id},
                    )
                else:
                    db.execute(
                        text(
                            "UPDATE upload_jobs SET cancellation_requested = TRUE, "
                            "next_attempt_at=CASE WHEN status='queued' THEN NULL "
                            "ELSE next_attempt_at END, "
                            "updated_at = NOW() WHERE process_id = :pid"
                        ),
                        {"pid": process_id},
                    )
                self._insert_event(
                    db,
                    process_id,
                    {
                        "role": "upload_progress",
                        "step": "cancelled" if immediate else "cancellation_pending",
                        "message": (
                            "File processing was cancelled"
                            if immediate
                            else "Cancellation requested; finalizing the active file"
                        ),
                    },
                )
            db.commit()
            live = bool(row.lease_live)
            return (
                True,
                changed,
                int(row.worker_pid) if not immediate and live and row.worker_pid else None,
                str(row.worker_start_token)
                if not immediate and live and row.worker_start_token
                else None,
                str(row.worker_node_id)
                if not immediate and live and row.worker_node_id
                else None,
            )
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def settle_cancellations_for_user(self, user_id: str) -> int:
        """Make requested cancellations terminal while the tenant lock is held.

        Only jobs with no in-flight item can be acknowledged here. A crashed
        in-flight item may already have graph or search-index effects; it must
        pass through lease recovery and metadata finalization instead of being
        silently made terminal.
        """
        db = self._sessions()
        try:
            rows = db.execute(
                text(
                    "UPDATE upload_jobs SET status = 'cancelled', lease_owner = NULL, "
                    "worker_pid = NULL, worker_start_token = NULL, worker_node_id = NULL, "
                    "lease_expires_at = NULL, updated_at = NOW() "
                    "WHERE user_id = :user_id AND status = 'processing' "
                    "AND cancellation_requested = TRUE "
                    "AND NOT EXISTS (SELECT 1 FROM upload_job_items item "
                    " WHERE item.process_id = upload_jobs.process_id "
                    " AND item.status = 'processing') RETURNING process_id"
                ),
                {"user_id": user_id},
            ).fetchall()
            for row in rows:
                db.execute(
                    text(
                        "UPDATE upload_job_items SET status='failed', indexed_count=0, "
                        "process_time=0, error='cancelled before processing', "
                        "updated_at=NOW() WHERE process_id=:pid AND status='pending'"
                    ),
                    {"pid": str(row.process_id)},
                )
                db.execute(
                    text(
                        "UPDATE upload_jobs SET processed_files=(SELECT COUNT(*) FROM "
                        "upload_job_items WHERE process_id=:pid AND status IN "
                        "('completed','failed')), total_chunks_inserted=(SELECT "
                        "COALESCE(SUM(indexed_count),0) FROM upload_job_items WHERE "
                        "process_id=:pid) WHERE process_id=:pid"
                    ),
                    {"pid": str(row.process_id)},
                )
                self._insert_event(
                    db,
                    str(row.process_id),
                    {
                        "role": "upload_progress",
                        "step": "cancelled",
                        "message": "File processing was cancelled",
                    },
                )
            db.commit()
            return len(rows)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def discard_cancellations_for_user(self, user_id: str) -> int:
        """Compensate cancelled work immediately for account erasure.

        The caller must hold the tenant write lock and must subsequently erase
        every tenant store. This deliberately differs from ordinary
        cancellation recovery: partial item effects are being destroyed, not
        exposed as a document, so an in-flight checkpoint can be abandoned.
        """
        db = self._sessions()
        try:
            rows = db.execute(
                text(
                    "SELECT process_id FROM upload_jobs WHERE user_id = :user_id "
                    "AND status IN ('staging', 'queued', 'processing') "
                    "AND cancellation_requested = TRUE "
                    "FOR UPDATE"
                ),
                {"user_id": user_id},
            ).fetchall()
            process_ids = [str(row.process_id) for row in rows]
            for process_id in process_ids:
                db.execute(
                    text(
                        "UPDATE upload_job_items SET status = 'failed', "
                        "indexed_count=0, process_time=0, "
                        "error = 'cancelled for account deletion', updated_at = NOW() "
                        "WHERE process_id = :pid AND status IN ('pending','processing')"
                    ),
                    {"pid": process_id},
                )
                db.execute(
                    text(
                        "UPDATE upload_jobs SET status = 'cancelled', lease_owner = NULL, "
                        "worker_pid = NULL, worker_start_token = NULL, worker_node_id = NULL, "
                        "lease_expires_at = NULL, updated_at = NOW() "
                        "WHERE process_id = :pid"
                    ),
                    {"pid": process_id},
                )
                db.execute(
                    text(
                        "UPDATE upload_jobs SET processed_files=(SELECT COUNT(*) FROM "
                        "upload_job_items WHERE process_id=:pid AND status IN "
                        "('completed','failed')), total_chunks_inserted=(SELECT "
                        "COALESCE(SUM(indexed_count),0) FROM upload_job_items WHERE "
                        "process_id=:pid) WHERE process_id=:pid"
                    ),
                    {"pid": process_id},
                )
                self._insert_event(
                    db,
                    process_id,
                    {
                        "role": "upload_progress",
                        "step": "cancelled",
                        "message": "File processing was cancelled for account deletion",
                    },
                )
            db.commit()
            return len(process_ids)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def requeue_expired(self, *, limit: int = 100) -> int:
        """Requeue a bounded batch of expired leases for later reservation."""
        db = self._sessions()
        try:
            expired = db.execute(
                text(
                    "SELECT process_id, cancellation_requested, attempt_count, max_attempts "
                    "FROM upload_jobs "
                    "WHERE status = 'processing' "
                    "AND lease_expires_at < NOW() ORDER BY lease_expires_at "
                    "FOR UPDATE SKIP LOCKED LIMIT :limit"
                ),
                {"limit": limit},
            ).fetchall()
            expired_ids = [str(row.process_id) for row in expired]
            for expired_row in expired:
                process_id = str(expired_row.process_id)
                exhausted = int(expired_row.attempt_count) >= int(expired_row.max_attempts)
                if exhausted and not bool(expired_row.cancellation_requested):
                    db.execute(
                        text(
                            "UPDATE upload_jobs SET status = 'queued', lease_owner = NULL, "
                            "worker_pid = NULL, worker_start_token = NULL, worker_node_id = NULL, "
                            "lease_expires_at = NULL, last_failure_class = 'worker_lease_expired', "
                            "next_attempt_at = NULL, updated_at = NOW() WHERE process_id = :pid"
                        ),
                        {"pid": process_id},
                    )
                    self._insert_event(db, process_id, {
                        "role": "upload_progress",
                        "step": "recovery",
                        "message": "Reconciling an exhausted upload before terminal failure",
                    })
                    continue
                # A crashed child may have committed graph/search effects
                # before its item checkpoint. Preserve ``processing`` as the
                # durable ambiguity marker; the replacement compensates the
                # whole logical document before it resets and replays it.
                db.execute(
                    text(
                        "UPDATE upload_jobs SET status = 'queued', lease_owner = NULL, "
                        "worker_pid = NULL, worker_start_token = NULL, worker_node_id = NULL, "
                        "lease_expires_at = NULL, last_failure_class = 'worker_lease_expired', "
                        "next_attempt_at = NOW() + (:backoff * INTERVAL '1 second'), "
                        "updated_at = NOW() "
                        "WHERE process_id = :pid"
                    ),
                    {
                        "pid": process_id,
                        "backoff": settings.upload_retry_backoff_s
                        * (2 ** max(0, int(expired_row.attempt_count) - 1)),
                    },
                )
                self._insert_event(
                    db,
                    process_id,
                    {
                        "role": "upload_progress",
                        "step": "recovery",
                        "message": "Resuming interrupted file processing",
                    },
                )
            db.commit()
            return len(expired_ids)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def terminal_ids_for_user(self, user_id: str) -> list[str]:
        db = self._sessions()
        try:
            rows = db.execute(
                text(
                    "SELECT process_id FROM upload_jobs WHERE user_id = :user_id "
                    "AND status IN ('completed', 'failed', 'cancelled') ORDER BY created_at"
                ),
                {"user_id": user_id},
            ).fetchall()
            return [str(row.process_id) for row in rows]
        finally:
            db.close()

    def terminal_staging_ids(self) -> list[str]:
        """Terminal jobs whose shared staged bytes need reconciliation."""
        db = self._sessions()
        try:
            rows = db.execute(
                text(
                    "SELECT process_id FROM upload_jobs WHERE staging_cleaned = FALSE "
                    "AND status IN ('completed', 'failed', 'cancelled') ORDER BY created_at"
                )
            ).fetchall()
            return [str(row.process_id) for row in rows]
        finally:
            db.close()

    def mark_staging_cleaned(self, process_id: str) -> bool:
        """Acknowledge erasure from the shared durable staging volume."""
        db = self._sessions()
        try:
            row = db.execute(
                text(
                    "UPDATE upload_jobs SET staging_cleaned = TRUE, updated_at = NOW() "
                    "WHERE process_id = :pid "
                    "AND status IN ('completed', 'failed', 'cancelled') RETURNING process_id"
                ),
                {"pid": process_id},
            ).fetchone()
            db.commit()
            return row is not None
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def delete_terminal(self, process_id: str, user_id: str) -> bool:
        """Delete the retry handle only after its staging is confirmed absent."""
        db = self._sessions()
        try:
            row = db.execute(
                text(
                    "DELETE FROM upload_jobs WHERE process_id = :pid AND user_id = :user_id "
                    "AND status IN ('completed', 'failed', 'cancelled') "
                    "AND staging_cleaned = TRUE RETURNING process_id"
                ),
                {"pid": process_id, "user_id": user_id},
            ).fetchone()
            db.commit()
            return row is not None
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def existing_ids(self, process_ids: Iterable[str]) -> set[str]:
        ids = tuple(process_ids)
        if not ids:
            return set()
        db = self._sessions()
        try:
            rows = db.execute(
                text("SELECT process_id FROM upload_jobs WHERE process_id = ANY(:ids)"),
                {"ids": list(ids)},
            ).fetchall()
            return {str(row.process_id) for row in rows}
        finally:
            db.close()

    def active_ids_for_user(self, user_id: str) -> list[str]:
        db = self._sessions()
        try:
            rows = db.execute(
                text(
                    "SELECT process_id FROM upload_jobs WHERE user_id = :user_id "
                    "AND status IN ('staging', 'queued', 'processing') ORDER BY created_at"
                ),
                {"user_id": user_id},
            ).fetchall()
            return [str(row.process_id) for row in rows]
        finally:
            db.close()


class InMemoryUploadJobRepository:
    """Contract-equivalent repository for isolated tests, never a fallback."""

    def __init__(self) -> None:
        self._records: dict[str, UploadJobRecord] = {}
        self._next_event = 1
        self._lock = threading.RLock()

    def _event(self, value: dict[str, Any]) -> UploadProgressEvent:
        event = UploadProgressEvent(
            id=self._next_event,
            role=value.get("role", "upload_progress"),
            step=value["step"],
            message=value.get("message"),
            percent=value.get("percent"),
            created_at=_utcnow(),
        )
        self._next_event += 1
        return event

    def create(self, process_id: str, job: IngestJob, items: Iterable[StagedUpload],
               initial_events: Iterable[dict[str, Any]], *, staging_node_id: str) -> None:
        """Compatibility helper that creates and immediately queues a ticket."""
        self.create_staging(
            process_id,
            job,
            items,
            initial_events,
            staging_node_id=staging_node_id,
        )
        if not self.finalize_staging(process_id, job.identity.user_id):
            raise RuntimeError("upload staging ticket could not be queued")

    def create_staging(
        self,
        process_id: str,
        job: IngestJob,
        items: Iterable[StagedUpload],
        initial_events: Iterable[dict[str, Any]],
        *,
        staging_node_id: str,
    ) -> None:
        with self._lock:
            staged = tuple(items)
            document_names = {
                unicodedata.normalize("NFC", item.original_name) for item in staged
            }
            if any(
                record.user_id == job.identity.user_id
                and record.status in {
                    UploadJobState.STAGING,
                    UploadJobState.QUEUED,
                    UploadJobState.PROCESSING,
                    UploadJobState.COMPLETED,
                }
                and any(
                    unicodedata.normalize("NFC", item.original_name) in document_names
                    for item in record.items
                )
                for record in self._records.values()
            ):
                raise UploadDocumentConflict("document already exists or is active")
            self._records[process_id] = UploadJobRecord(
                process_id=process_id,
                job=job,
                status=UploadJobState.STAGING,
                total_files=len(staged),
                max_attempts=settings.max_upload_attempts,
                staging_node_id=staging_node_id,
                items=staged,
                progress=tuple(self._event(e) for e in initial_events),
                created_at=_utcnow(),
                updated_at=_utcnow(),
            )

    def finalize_staging(self, process_id: str, user_id: str) -> bool:
        with self._lock:
            record = self._records.get(process_id)
            if (
                record is None
                or record.user_id != user_id
                or record.status is not UploadJobState.STAGING
                or record.cancellation_requested
            ):
                return False
            self._records[process_id] = record.model_copy(
                update={"status": UploadJobState.QUEUED, "updated_at": _utcnow()}
            )
            return True

    def fail_staging(
        self, process_id: str, user_id: str, event: dict[str, Any]
    ) -> bool:
        with self._lock:
            record = self._records.get(process_id)
            if (
                record is None
                or record.user_id != user_id
                or record.status is not UploadJobState.STAGING
            ):
                return False
            items = tuple(
                item.model_copy(update={
                    "status": UploadItemState.FAILED,
                    "indexed_count": 0,
                    "process_time": 0.0,
                    "error": "upload staging failed",
                }) if not item.status.terminal else item
                for item in record.items
            )
            self._records[process_id] = record.model_copy(update={
                "status": UploadJobState.FAILED,
                "items": items,
                "processed_files": sum(item.status.terminal for item in items),
                "total_chunks_inserted": sum(item.indexed_count for item in items),
                "progress": (*record.progress, self._event(event)),
                "lease_owner": None,
                "worker_pid": None,
                "worker_start_token": None,
                "worker_node_id": None,
                "lease_expires_at": None,
                "updated_at": _utcnow(),
            })
            return True

    def reset_reconciled_document(
        self, process_id: str, owner: str, original_name: str
    ) -> bool:
        with self._lock:
            record = self._records.get(process_id)
            if (
                record is None
                or record.status is not UploadJobState.PROCESSING
                or record.lease_owner != owner
                or record.cancellation_requested
            ):
                return False
            items = tuple(
                item.model_copy(update={
                    "status": UploadItemState.PENDING,
                    "indexed_count": 0,
                    "process_time": 0.0,
                    "error": None,
                }) if item.original_name == original_name else item
                for item in record.items
            )
            self._records[process_id] = record.model_copy(update={
                "items": items,
                "processed_files": sum(item.status.terminal for item in items),
                "total_chunks_inserted": sum(item.indexed_count for item in items),
                "updated_at": _utcnow(),
            })
            return True

    def expire_staging(self, *, max_age_seconds: float, limit: int = 100) -> int:
        if max_age_seconds < 0:
            raise ValueError("max_age_seconds must be non-negative")
        if limit <= 0:
            return 0
        with self._lock:
            cutoff = _utcnow() - timedelta(seconds=max_age_seconds)
            expired = sorted(
                (
                    (process_id, record)
                    for process_id, record in self._records.items()
                    if record.status is UploadJobState.STAGING
                    and record.created_at is not None
                    and record.created_at < cutoff
                ),
                key=lambda pair: (pair[1].created_at or _utcnow(), pair[0]),
            )[:limit]
            for process_id, record in expired:
                items = tuple(
                    item.model_copy(update={
                        "status": UploadItemState.FAILED,
                        "indexed_count": 0,
                        "process_time": 0.0,
                        "error": "upload staging expired",
                    }) if not item.status.terminal else item
                    for item in record.items
                )
                self._records[process_id] = record.model_copy(update={
                    "status": UploadJobState.FAILED,
                    "items": items,
                    "processed_files": sum(item.status.terminal for item in items),
                    "total_chunks_inserted": sum(item.indexed_count for item in items),
                    "progress": (*record.progress, self._event({
                        "role": "upload_progress",
                        "step": "error",
                        "message": "Upload staging expired before queueing",
                    })),
                    "lease_owner": None,
                    "worker_pid": None,
                    "worker_start_token": None,
                    "worker_node_id": None,
                    "lease_expires_at": None,
                    "updated_at": _utcnow(),
                })
            return len(expired)

    def get(self, process_id: str, *, include_items: bool = True,
            include_progress: bool = True) -> UploadJobRecord | None:
        with self._lock:
            record = self._records.get(process_id)
            if record is None:
                return None
            return record.model_copy(update={
                "items": record.items if include_items else (),
                "progress": record.progress if include_progress else (),
            }, deep=True)

    def events_after(self, process_id: str, after_id: int) -> tuple[UploadProgressEvent, ...]:
        with self._lock:
            record = self._records.get(process_id)
            return () if record is None else tuple(e for e in record.progress if e.id > after_id)

    def reserve_next(
        self,
        owner: str,
        worker_node_id: str,
        lease_seconds: float,
        *,
        process_id: str | None = None,
        max_workers: int | None = None,
    ) -> str | None:
        with self._lock:
            if max_workers is not None:
                if max_workers <= 0:
                    raise ValueError("max_workers must be positive")
                now = _utcnow()
                live_workers = sum(
                    1
                    for record in self._records.values()
                    if record.status is UploadJobState.PROCESSING
                    and record.worker_node_id == worker_node_id
                    and record.lease_expires_at is not None
                    and record.lease_expires_at > now
                )
                if live_workers >= max_workers:
                    return None
            candidates = sorted(
                (
                    (pid, record)
                    for pid, record in self._records.items()
                    if record.status is UploadJobState.QUEUED
                    and (record.next_attempt_at is None or record.next_attempt_at <= _utcnow())
                    and (process_id is None or pid == process_id)
                ),
                key=lambda pair: (pair[1].created_at or _utcnow(), pair[0]),
            )
            if not candidates:
                return None
            selected, record = candidates[0]
            self._records[selected] = record.model_copy(update={
                "status": UploadJobState.PROCESSING,
                "attempt_count": record.attempt_count + 1,
                "next_attempt_at": None,
                "lease_owner": owner,
                "worker_node_id": worker_node_id,
                "worker_pid": None,
                "worker_start_token": None,
                "lease_expires_at": _utcnow() + timedelta(seconds=lease_seconds),
                "updated_at": _utcnow(),
            })
            return selected

    def activate(
        self,
        process_id: str,
        owner: str,
        worker_pid: int,
        worker_start_token: str | None,
        lease_seconds: float,
    ) -> bool:
        with self._lock:
            record = self._records.get(process_id)
            if record is None or record.status is not UploadJobState.PROCESSING \
                    or record.lease_owner != owner:
                return False
            self._records[process_id] = record.model_copy(update={
                "worker_pid": worker_pid,
                "worker_start_token": worker_start_token,
                "lease_expires_at": _utcnow() + timedelta(seconds=lease_seconds),
                "updated_at": _utcnow(),
            })
            return True

    def release_reservation(self, process_id: str, owner: str) -> bool:
        with self._lock:
            record = self._records.get(process_id)
            if (
                record is None
                or record.status is not UploadJobState.PROCESSING
                or record.lease_owner != owner
                or record.worker_pid is not None
            ):
                return False
            requires_reconciliation = any(
                item.status is not UploadItemState.PENDING for item in record.items
            )
            exhausted = (
                record.attempt_count >= record.max_attempts
                and not requires_reconciliation
            )
            attempt_count = max(
                0,
                record.attempt_count - (1 if requires_reconciliation else 0),
            )
            items = record.items
            if exhausted:
                items = tuple(
                    item.model_copy(update={
                        "status": UploadItemState.FAILED,
                        "indexed_count": 0,
                        "process_time": 0.0,
                        "error": "worker startup attempt budget exhausted",
                    }) if not item.status.terminal else item
                    for item in items
                )
            self._records[process_id] = record.model_copy(update={
                "status": UploadJobState.FAILED if exhausted else UploadJobState.QUEUED,
                "items": items,
                "processed_files": sum(item.status.terminal for item in items),
                "total_chunks_inserted": sum(item.indexed_count for item in items),
                "last_failure_class": "worker_spawn_failed",
                "attempt_count": attempt_count,
                "progress": record.progress + ((self._event({
                    "step": "error",
                    "message": "Upload stopped after repeated worker startup failures",
                }),) if exhausted else ()),
                "next_attempt_at": None if exhausted else _utcnow() + timedelta(
                    seconds=settings.upload_retry_backoff_s
                    * (2 ** max(0, attempt_count - 1))
                ),
                "lease_owner": None,
                "worker_node_id": None,
                "worker_pid": None,
                "worker_start_token": None,
                "lease_expires_at": None,
                "updated_at": _utcnow(),
            })
            return True

    def requeue_owned(
        self, process_id: str, owner: str, event: dict[str, Any]
    ) -> bool:
        with self._lock:
            record = self._records.get(process_id)
            if (
                record is None
                or record.status is not UploadJobState.PROCESSING
                or record.lease_owner != owner
            ):
                return False
            self._records[process_id] = record.model_copy(update={
                "status": UploadJobState.QUEUED,
                "progress": (*record.progress, self._event(event)),
                "attempt_count": max(0, record.attempt_count - 1),
                "next_attempt_at": None,
                "lease_owner": None,
                "worker_node_id": None,
                "worker_pid": None,
                "worker_start_token": None,
                "lease_expires_at": None,
                "updated_at": _utcnow(),
            })
            return True

    def retry_document(
        self,
        process_id: str,
        owner: str,
        original_name: str,
        event: dict[str, Any],
        *,
        refund_attempt: bool = False,
    ) -> bool:
        with self._lock:
            record = self._records.get(process_id)
            if (
                record is None
                or record.status is not UploadJobState.PROCESSING
                or record.lease_owner != owner
            ):
                return False
            next_attempt_count = max(
                0, record.attempt_count - (1 if refund_attempt else 0)
            )
            if record.cancellation_requested or next_attempt_count >= record.max_attempts:
                return False
            items = tuple(
                item.model_copy(update={
                    "status": UploadItemState.PENDING,
                    "indexed_count": 0,
                    "process_time": 0.0,
                    "error": None,
                }) if item.original_name == original_name else item
                for item in record.items
            )
            self._records[process_id] = record.model_copy(update={
                "status": UploadJobState.QUEUED,
                "items": items,
                "processed_files": sum(item.status.terminal for item in items),
                "total_chunks_inserted": sum(item.indexed_count for item in items),
                "progress": (*record.progress, self._event(event)),
                "last_failure_class": str(event.get("failure_class", "item_ambiguous"))[:64],
                "attempt_count": next_attempt_count,
                "next_attempt_at": None if refund_attempt else _utcnow() + timedelta(
                    seconds=settings.upload_retry_backoff_s
                    * (2 ** max(0, next_attempt_count - 1))
                ),
                "lease_owner": None,
                "worker_node_id": None,
                "worker_pid": None,
                "worker_start_token": None,
                "lease_expires_at": None,
                "updated_at": _utcnow(),
            })
            return True

    def finish_reconciled(
        self,
        process_id: str,
        owner: str,
        status: UploadJobState,
        compensated_documents: Iterable[str],
        event: dict[str, Any],
    ) -> bool:
        if status not in {UploadJobState.FAILED, UploadJobState.CANCELLED}:
            raise ValueError("reconciliation may finish only as failed or cancelled")
        with self._lock:
            record = self._records.get(process_id)
            expected_cancellation = status is UploadJobState.CANCELLED
            if (
                record is None
                or record.status is not UploadJobState.PROCESSING
                or record.lease_owner != owner
                or record.cancellation_requested != expected_cancellation
            ):
                return False
            document_names = set(compensated_documents)
            items = tuple(
                item.model_copy(update={
                    "status": UploadItemState.FAILED,
                    "indexed_count": 0,
                    "process_time": 0.0,
                    "error": (
                        "cancelled after compensation"
                        if expected_cancellation
                        else "attempt budget exhausted after compensation"
                    ) if item.original_name in document_names else (
                        "cancelled before processing"
                        if expected_cancellation
                        else "attempt budget exhausted before processing"
                    ),
                })
                if item.original_name in document_names or not item.status.terminal
                else item
                for item in record.items
            )
            self._records[process_id] = record.model_copy(update={
                "status": status,
                "items": items,
                "processed_files": sum(item.status.terminal for item in items),
                "total_chunks_inserted": sum(item.indexed_count for item in items),
                "progress": (*record.progress, self._event(event)),
                "last_failure_class": str(event.get("failure_class", "item_ambiguous"))[:64],
                "next_attempt_at": None,
                "lease_owner": None,
                "worker_node_id": None,
                "worker_pid": None,
                "worker_start_token": None,
                "lease_expires_at": None,
                "updated_at": _utcnow(),
            })
            return True

    def owns_active_lease(self, process_id: str, owner: str) -> bool:
        with self._lock:
            record = self._records.get(process_id)
            return bool(
                record is not None
                and record.lease_owner == owner
                and record.status is UploadJobState.PROCESSING
                and record.lease_expires_at is not None
                and record.lease_expires_at > _utcnow()
            )

    def heartbeat(self, process_id: str, owner: str, lease_seconds: float) -> bool:
        with self._lock:
            record = self._records.get(process_id)
            if record is None or record.lease_owner != owner \
                    or record.status is not UploadJobState.PROCESSING:
                return False
            self._records[process_id] = record.model_copy(update={
                "lease_expires_at": _utcnow() + timedelta(seconds=lease_seconds),
                "updated_at": _utcnow(),
            })
            return True

    def append_progress(self, process_id: str, owner: str, event: dict[str, Any],
                        lease_seconds: float) -> bool:
        with self._lock:
            if not self.heartbeat(process_id, owner, lease_seconds):
                return False
            record = self._records[process_id]
            self._records[process_id] = record.model_copy(
                update={"progress": (*record.progress, self._event(event))}
            )
            return True

    def begin_item(self, process_id: str, owner: str, sequence: int) -> bool:
        with self._lock:
            record = self._records.get(process_id)
            if record is None or record.lease_owner != owner \
                    or record.status is not UploadJobState.PROCESSING:
                return False
            changed = False
            items = []
            for item in record.items:
                if (
                    item.sequence == sequence
                    and not item.status.terminal
                    and (
                        item.status is UploadItemState.PROCESSING
                        or not record.cancellation_requested
                    )
                ):
                    item = item.model_copy(update={"status": UploadItemState.PROCESSING})
                    changed = True
                items.append(item)
            if changed:
                self._records[process_id] = record.model_copy(update={"items": tuple(items)})
            return changed

    def finish_item(self, process_id: str, owner: str, sequence: int, *, indexed_count: int,
                    process_time: float, error: str | None) -> bool:
        with self._lock:
            record = self._records.get(process_id)
            if record is None or record.lease_owner != owner \
                    or record.status is not UploadJobState.PROCESSING:
                return False
            changed = False
            items = []
            for item in record.items:
                if item.sequence == sequence and item.status is UploadItemState.PROCESSING:
                    item = item.model_copy(update={
                        "status": UploadItemState.FAILED if error else UploadItemState.COMPLETED,
                        "indexed_count": indexed_count,
                        "process_time": process_time,
                        "error": error,
                    })
                    changed = True
                items.append(item)
            if changed:
                terminal = [i for i in items if i.status.terminal]
                self._records[process_id] = record.model_copy(update={
                    "items": tuple(items),
                    "processed_files": len(terminal),
                    "total_chunks_inserted": sum(i.indexed_count for i in items),
                    "updated_at": _utcnow(),
                })
            return changed

    def finish(self, process_id: str, owner: str, status: UploadJobState,
               event: dict[str, Any]) -> bool:
        if status not in {UploadJobState.COMPLETED, UploadJobState.FAILED}:
            raise ValueError("a worker may finish only as completed or failed")
        with self._lock:
            record = self._records.get(process_id)
            if record is None or record.lease_owner != owner \
                    or record.status is not UploadJobState.PROCESSING \
                    or record.cancellation_requested \
                    or any(
                        item.status is UploadItemState.PROCESSING
                        for item in record.items
                    ) \
                    or (
                        status is UploadJobState.COMPLETED
                        and any(
                            item.status is not UploadItemState.COMPLETED
                            for item in record.items
                        )
                    ):
                return False
            items = record.items
            if status is UploadJobState.FAILED:
                items = tuple(
                    item.model_copy(update={
                        "status": UploadItemState.FAILED,
                        "indexed_count": 0,
                        "process_time": 0.0,
                        "error": "processing stopped before item began",
                    })
                    if item.status is UploadItemState.PENDING else item
                    for item in items
                )
            self._records[process_id] = record.model_copy(update={
                "status": status,
                "items": items,
                "processed_files": sum(item.status.terminal for item in items),
                "total_chunks_inserted": sum(item.indexed_count for item in items),
                "progress": (*record.progress, self._event(event)),
                "lease_owner": None,
                "worker_pid": None,
                "worker_start_token": None,
                "worker_node_id": None,
                "lease_expires_at": None,
                "updated_at": _utcnow(),
            })
            return True

    def finish_cancelled(
        self, process_id: str, owner: str, event: dict[str, Any]
    ) -> bool:
        with self._lock:
            record = self._records.get(process_id)
            if (
                record is None
                or record.lease_owner != owner
                or record.status is not UploadJobState.PROCESSING
                or not record.cancellation_requested
                or any(
                    item.status is UploadItemState.PROCESSING
                    for item in record.items
                )
            ):
                return False
            items = tuple(
                item.model_copy(update={
                    "status": UploadItemState.FAILED,
                    "indexed_count": 0,
                    "process_time": 0.0,
                    "error": "cancelled before processing",
                })
                if item.status is UploadItemState.PENDING else item
                for item in record.items
            )
            self._records[process_id] = record.model_copy(update={
                "status": UploadJobState.CANCELLED,
                "items": items,
                "processed_files": sum(item.status.terminal for item in items),
                "total_chunks_inserted": sum(item.indexed_count for item in items),
                "progress": (*record.progress, self._event(event)),
                "lease_owner": None,
                "worker_pid": None,
                "worker_start_token": None,
                "worker_node_id": None,
                "lease_expires_at": None,
                "updated_at": _utcnow(),
            })
            return True

    def request_cancel(
        self, process_id: str, user_id: str
    ) -> tuple[bool, bool, int | None, str | None, str | None]:
        with self._lock:
            record = self._records.get(process_id)
            if record is None or record.user_id != user_id:
                return False, False, None, None, None
            if record.status.terminal:
                return True, False, None, None, None
            live = record.lease_expires_at is not None and record.lease_expires_at > _utcnow()
            changed = not record.cancellation_requested
            has_inflight = any(
                item.status is UploadItemState.PROCESSING for item in record.items
            )
            has_checkpoint = any(item.status.terminal for item in record.items)
            immediate = (
                record.status in {UploadJobState.STAGING, UploadJobState.QUEUED}
                or record.worker_pid is None
            ) and not has_inflight and not has_checkpoint
            if changed:
                items = record.items
                if immediate:
                    items = tuple(
                        item.model_copy(update={
                            "status": UploadItemState.FAILED,
                            "indexed_count": 0,
                            "process_time": 0.0,
                            "error": "cancelled before processing",
                        }) if not item.status.terminal else item
                        for item in items
                    )
                update: dict[str, Any] = {
                    "status": UploadJobState.CANCELLED if immediate else record.status,
                    "cancellation_requested": True,
                    "items": items,
                    "processed_files": sum(item.status.terminal for item in items),
                    "total_chunks_inserted": sum(item.indexed_count for item in items),
                    "next_attempt_at": (
                        None
                        if record.status is UploadJobState.QUEUED and not immediate
                        else record.next_attempt_at
                    ),
                    "progress": (*record.progress, self._event({
                        "step": "cancelled" if immediate else "cancellation_pending",
                        "message": (
                            "File processing was cancelled"
                            if immediate
                            else "Cancellation requested; finalizing the active file"
                        ),
                    })),
                    "updated_at": _utcnow(),
                }
                if immediate:
                    update.update({
                        "lease_owner": None,
                        "worker_pid": None,
                        "worker_start_token": None,
                        "worker_node_id": None,
                        "lease_expires_at": None,
                    })
                self._records[process_id] = record.model_copy(update=update)
            return (
                True,
                changed,
                record.worker_pid if not immediate and live else None,
                record.worker_start_token if not immediate and live else None,
                record.worker_node_id if not immediate and live else None,
            )

    def settle_cancellations_for_user(self, user_id: str) -> int:
        with self._lock:
            settled = 0
            for process_id, record in tuple(self._records.items()):
                if (
                    record.user_id != user_id
                    or record.status is not UploadJobState.PROCESSING
                    or not record.cancellation_requested
                    or any(
                        item.status is UploadItemState.PROCESSING
                        for item in record.items
                    )
                ):
                    continue
                items = tuple(
                    item.model_copy(update={
                        "status": UploadItemState.FAILED,
                        "indexed_count": 0,
                        "process_time": 0.0,
                        "error": "cancelled before processing",
                    }) if not item.status.terminal else item
                    for item in record.items
                )
                self._records[process_id] = record.model_copy(update={
                    "status": UploadJobState.CANCELLED,
                    "items": items,
                    "processed_files": sum(item.status.terminal for item in items),
                    "total_chunks_inserted": sum(item.indexed_count for item in items),
                    "progress": (*record.progress, self._event({
                        "step": "cancelled",
                        "message": "File processing was cancelled",
                    })),
                    "lease_owner": None,
                    "worker_pid": None,
                    "worker_start_token": None,
                    "worker_node_id": None,
                    "lease_expires_at": None,
                    "updated_at": _utcnow(),
                })
                settled += 1
            return settled

    def discard_cancellations_for_user(self, user_id: str) -> int:
        with self._lock:
            discarded = 0
            for process_id, record in tuple(self._records.items()):
                if (
                    record.user_id != user_id
                    or record.status
                    not in {
                        UploadJobState.STAGING,
                        UploadJobState.QUEUED,
                        UploadJobState.PROCESSING,
                    }
                    or not record.cancellation_requested
                ):
                    continue
                items = tuple(
                    item.model_copy(update={
                        "status": UploadItemState.FAILED,
                        "indexed_count": 0,
                        "process_time": 0.0,
                        "error": "cancelled for account deletion",
                    })
                    if not item.status.terminal else item
                    for item in record.items
                )
                self._records[process_id] = record.model_copy(update={
                    "status": UploadJobState.CANCELLED,
                    "items": items,
                    "processed_files": sum(item.status.terminal for item in items),
                    "total_chunks_inserted": sum(item.indexed_count for item in items),
                    "progress": (*record.progress, self._event({
                        "step": "cancelled",
                        "message": "File processing was cancelled for account deletion",
                    })),
                    "lease_owner": None,
                    "worker_pid": None,
                    "worker_start_token": None,
                    "worker_node_id": None,
                    "lease_expires_at": None,
                    "updated_at": _utcnow(),
                })
                discarded += 1
            return discarded

    def requeue_expired(self, *, limit: int = 100) -> int:
        with self._lock:
            now = _utcnow()
            expired = [
                (process_id, record)
                for process_id, record in self._records.items()
                if record.status is UploadJobState.PROCESSING
                and record.lease_expires_at is not None
                and record.lease_expires_at < now
            ][:limit]
            for process_id, record in expired:
                if (
                    record.attempt_count >= record.max_attempts
                    and not record.cancellation_requested
                ):
                    self._records[process_id] = record.model_copy(update={
                        "status": UploadJobState.QUEUED,
                        "lease_owner": None,
                        "worker_pid": None,
                        "worker_start_token": None,
                        "worker_node_id": None,
                        "lease_expires_at": None,
                        "last_failure_class": "worker_lease_expired",
                        "next_attempt_at": None,
                        "progress": (*record.progress, self._event({
                            "step": "recovery",
                            "message": "Reconciling an exhausted upload before terminal failure",
                        })),
                        "updated_at": now,
                    })
                    continue
                self._records[process_id] = record.model_copy(update={
                    "status": UploadJobState.QUEUED,
                    "lease_owner": None,
                    "worker_pid": None,
                    "worker_start_token": None,
                    "worker_node_id": None,
                    "lease_expires_at": None,
                    "last_failure_class": "worker_lease_expired",
                    "next_attempt_at": now + timedelta(
                        seconds=settings.upload_retry_backoff_s
                        * (2 ** max(0, record.attempt_count - 1))
                    ),
                    "progress": (*record.progress, self._event({
                        "step": "recovery",
                        "message": "Resuming interrupted file processing",
                    })),
                })
            return len(expired)

    def terminal_ids_for_user(self, user_id: str) -> list[str]:
        with self._lock:
            return [pid for pid, record in self._records.items()
                    if record.user_id == user_id and record.status.terminal]

    def terminal_staging_ids(self) -> list[str]:
        with self._lock:
            return [
                process_id
                for process_id, record in self._records.items()
                if record.status.terminal and not record.staging_cleaned
            ]

    def mark_staging_cleaned(self, process_id: str) -> bool:
        with self._lock:
            record = self._records.get(process_id)
            if (
                record is None
                or not record.status.terminal
            ):
                return False
            self._records[process_id] = record.model_copy(
                update={"staging_cleaned": True, "updated_at": _utcnow()}
            )
            return True

    def delete_terminal(self, process_id: str, user_id: str) -> bool:
        with self._lock:
            record = self._records.get(process_id)
            if (
                record is None
                or record.user_id != user_id
                or not record.status.terminal
                or not record.staging_cleaned
            ):
                return False
            del self._records[process_id]
            return True

    def existing_ids(self, process_ids: Iterable[str]) -> set[str]:
        with self._lock:
            return set(process_ids).intersection(self._records)

    def active_ids_for_user(self, user_id: str) -> list[str]:
        with self._lock:
            return [
                process_id
                for process_id, record in self._records.items()
                if record.user_id == user_id and not record.status.terminal
            ]
