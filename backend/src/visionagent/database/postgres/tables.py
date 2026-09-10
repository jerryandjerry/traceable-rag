"""SQLAlchemy tables.

These are an implementation detail of one component (the conversation store),
not domain vocabulary. The domain vocabulary is Pydantic and lives in
`visionagent.models`.

The authoritative schema is `backend/init.sql`; `Base.metadata.create_all()` is
a convenience for local setup only.
"""
from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

# A single metadata registry keeps every table visible to create_all().
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(100), nullable=False)
    # Bumped on every password change. Tokens carry the value at issue time
    # and stop verifying once it moves, so a stolen token dies with the
    # password it was issued under.
    auth_version = Column(Integer, nullable=False, server_default="0")
    # Set before account deletion touches any external store. Upload creation
    # locks this row and refuses work while the gate is set, closing the window
    # where a newly accepted job could recreate data after deletion.
    deletion_requested = Column(Boolean, nullable=False, server_default="false")


class Session(Base):
    __tablename__ = "sessions"

    session_id = Column(String(16), primary_key=True)
    session_name = Column(String(255), nullable=False)
    user_id = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now())


class Message(Base):
    __tablename__ = "messages"

    message_id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    session_id = Column(
        String(16),
        ForeignKey(
            "sessions.session_id",
            ondelete="CASCADE",
            name="fk_messages_session",
        ),
        nullable=False,
    )
    user_question = Column(Text, nullable=False)
    model_answer = Column(Text, nullable=False)
    documents = Column(Text)
    recommended_questions = Column(Text)
    think = Column(Text)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now())


class KnowledgeBase(Base):
    __tablename__ = "knowledgebases"
    __table_args__ = (
        UniqueConstraint("user_id", "file_name", name="uq_kb_user_file"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    file_name = Column(String(255), nullable=False)
    total_chunks = Column(Integer, nullable=True)
    process_time = Column(Float, nullable=True)
    # Set when any ingest stage leaves the document partially processed.
    error = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default="CURRENT_TIMESTAMP")
    updated_at = Column(TIMESTAMP, nullable=False, server_default="CURRENT_TIMESTAMP")


class UploadJob(Base):
    """Durable queue state for an upload batch."""

    __tablename__ = "upload_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('staging', 'queued', 'processing', 'completed', 'failed', 'cancelled')",
            name="ck_upload_jobs_status",
        ),
    )

    process_id = Column(String(96), primary_key=True)
    run_id = Column(String(96), nullable=False)
    user_id = Column(String(255), nullable=False)
    job_payload = Column(JSONB, nullable=False)
    status = Column(String(16), nullable=False, server_default="staging")
    cancellation_requested = Column(Boolean, nullable=False, server_default="false")
    total_files = Column(Integer, nullable=False)
    processed_files = Column(Integer, nullable=False, server_default="0")
    total_chunks_inserted = Column(Integer, nullable=False, server_default="0")
    worker_pid = Column(Integer, nullable=True)
    worker_start_token = Column(String(128), nullable=True)
    staging_node_id = Column(String(128), nullable=False)
    staging_cleaned = Column(Boolean, nullable=False, server_default="false")
    worker_node_id = Column(String(128), nullable=True)
    lease_owner = Column(String(64), nullable=True)
    lease_expires_at = Column(TIMESTAMP(timezone=True), nullable=True)
    attempt_count = Column(Integer, nullable=False, server_default="0")
    max_attempts = Column(Integer, nullable=False, server_default="3")
    last_failure_class = Column(String(64), nullable=True)
    next_attempt_at = Column(TIMESTAMP(timezone=True), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())


class UploadJobItem(Base):
    """A resumable checkpoint for one staged file or split-PDF part."""

    __tablename__ = "upload_job_items"
    __table_args__ = (
        UniqueConstraint("process_id", "sequence", name="uq_upload_job_item_sequence"),
        CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed')",
            name="ck_upload_job_items_status",
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    process_id = Column(
        String(96), ForeignKey("upload_jobs.process_id", ondelete="CASCADE"), nullable=False
    )
    sequence = Column(Integer, nullable=False)
    original_name = Column(String(255), nullable=False)
    part_name = Column(String(255), nullable=False)
    storage_key = Column(String(255), nullable=False)
    status = Column(String(16), nullable=False, server_default="pending")
    indexed_count = Column(Integer, nullable=False, server_default="0")
    process_time = Column(Float, nullable=False, server_default="0")
    error = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())


class UploadJobEvent(Base):
    """Append-only progress stream; the primary key is the SSE cursor."""

    __tablename__ = "upload_job_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    process_id = Column(
        String(96), ForeignKey("upload_jobs.process_id", ondelete="CASCADE"), nullable=False
    )
    role = Column(String(64), nullable=False, server_default="upload_progress")
    step = Column(String(64), nullable=False)
    message = Column(Text, nullable=True)
    percent = Column(Integer, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
