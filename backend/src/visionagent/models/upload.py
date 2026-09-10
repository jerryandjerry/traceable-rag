"""Durable upload-job contracts.

The HTTP request that accepted an upload is short lived; parsing and indexing
are not.  These values are the persisted hand-off between API workers and the
background ingest worker.  They deliberately contain no bearer token or raw
file bytes.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from visionagent.models.context import IngestJob


class UploadJobState(StrEnum):
    STAGING = "staging"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED, self.CANCELLED}


class UploadItemState(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED}


class StagedUpload(BaseModel):
    """One on-disk work item, possibly one part of a split PDF."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=0)
    original_name: str = Field(min_length=1)
    part_name: str = Field(min_length=1)
    storage_key: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    status: UploadItemState = UploadItemState.PENDING
    indexed_count: int = Field(default=0, ge=0)
    process_time: float = Field(default=0.0, ge=0.0)
    error: str | None = None


class UploadProgressEvent(BaseModel):
    """One ordered SSE event.  ``id`` is the durable stream cursor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(ge=1)
    role: str = "upload_progress"
    step: str = Field(min_length=1)
    message: str | None = None
    percent: int | None = Field(default=None, ge=0, le=100)
    created_at: datetime | None = None

    def wire_dict(self) -> dict[str, object]:
        """The legacy-compatible event body plus a stable cursor."""
        return self.model_dump(mode="json", exclude_none=True)


class UploadJobRecord(BaseModel):
    """The authoritative snapshot reconstructed from PostgreSQL."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    process_id: str = Field(min_length=1)
    job: IngestJob
    status: UploadJobState
    cancellation_requested: bool = False
    total_files: int = Field(ge=0)
    processed_files: int = Field(default=0, ge=0)
    total_chunks_inserted: int = Field(default=0, ge=0)
    worker_pid: int | None = Field(default=None, ge=1)
    worker_start_token: str | None = None
    staging_node_id: str | None = None
    staging_cleaned: bool = False
    worker_node_id: str | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    last_failure_class: str | None = None
    next_attempt_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    items: tuple[StagedUpload, ...] = ()
    progress: tuple[UploadProgressEvent, ...] = ()

    @property
    def user_id(self) -> str:
        return self.job.identity.user_id

    @property
    def run_id(self) -> str:
        return self.job.identity.run_id

    def public_dict(self) -> dict[str, object]:
        """Status returned by the API; leases and staged paths stay private."""
        return {
            "user_id": self.user_id,
            "run_id": self.run_id,
            "status": self.status.value,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "total_chunks_inserted": self.total_chunks_inserted,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "last_failure_class": self.last_failure_class,
            "next_attempt_at": (
                self.next_attempt_at.isoformat() if self.next_attempt_at else None
            ),
            "progress": [event.wire_dict() for event in self.progress],
        }
