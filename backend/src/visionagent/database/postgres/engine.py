import os
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Initialize the application's environment-file policy before reading DATABASE_URL.
import visionagent.config.settings  # noqa: F401
from visionagent.database.postgres.tables import Base

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Fail at startup with an actionable configuration error.
    raise RuntimeError("Missing required environment variable: DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Any:
    """Yield one database session and close it afterward."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db() -> None:
    """Create tables that do not already exist."""
    Base.metadata.create_all(bind=engine)


# Idempotent startup upgrades mirror ``backend/init.sql`` for existing volumes.
_SCHEMA_UPGRADES = (
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_version INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS deletion_requested BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE knowledgebases ADD COLUMN IF NOT EXISTS error TEXT",
    """DO $$ BEGIN
        IF EXISTS (
            SELECT 1 FROM knowledgebases GROUP BY user_id, file_name HAVING COUNT(*) > 1
        ) THEN
            RAISE EXCEPTION 'knowledgebases contains duplicate (user_id, file_name) rows; deduplicate before startup';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid = 'knowledgebases'::regclass
              AND conname = 'uq_kb_user_file'
        ) THEN
            ALTER TABLE knowledgebases ADD CONSTRAINT uq_kb_user_file
                UNIQUE (user_id, file_name);
        END IF;
    END $$""",
    # NOT VALID preserves existing orphan rows while enforcing new writes.
    """DO $$ BEGIN
        ALTER TABLE messages ADD CONSTRAINT fk_messages_session
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            ON DELETE CASCADE NOT VALID;
    EXCEPTION WHEN duplicate_object THEN NULL; END $$""",
    """CREATE TABLE IF NOT EXISTS upload_jobs (
        process_id VARCHAR(96) PRIMARY KEY,
        run_id VARCHAR(96) NOT NULL,
        user_id VARCHAR(255) NOT NULL,
        job_payload JSONB NOT NULL,
        status VARCHAR(16) NOT NULL DEFAULT 'staging',
        cancellation_requested BOOLEAN NOT NULL DEFAULT FALSE,
        total_files INTEGER NOT NULL,
        processed_files INTEGER NOT NULL DEFAULT 0,
        total_chunks_inserted INTEGER NOT NULL DEFAULT 0,
        worker_pid INTEGER,
        worker_start_token VARCHAR(128),
        staging_node_id VARCHAR(128) NOT NULL,
        staging_cleaned BOOLEAN NOT NULL DEFAULT FALSE,
        worker_node_id VARCHAR(128),
        lease_owner VARCHAR(64),
        lease_expires_at TIMESTAMPTZ,
        attempt_count INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL DEFAULT 3,
        last_failure_class VARCHAR(64),
        next_attempt_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT ck_upload_jobs_status CHECK (
            status IN ('staging', 'queued', 'processing', 'completed', 'failed', 'cancelled')
        )
    )""",
    # Normalize generated or explicit constraint names to the current state set.
    """DO $$ BEGIN
        IF EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid = 'upload_jobs'::regclass
              AND conname = 'upload_jobs_status_check'
        ) THEN
            ALTER TABLE upload_jobs DROP CONSTRAINT upload_jobs_status_check;
        END IF;
        IF EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid = 'upload_jobs'::regclass
              AND conname = 'ck_upload_jobs_status'
              AND POSITION('staging' IN pg_get_constraintdef(oid)) = 0
        ) THEN
            ALTER TABLE upload_jobs DROP CONSTRAINT ck_upload_jobs_status;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid = 'upload_jobs'::regclass
              AND conname = 'ck_upload_jobs_status'
        ) THEN
            ALTER TABLE upload_jobs ADD CONSTRAINT ck_upload_jobs_status CHECK (
                status IN ('staging', 'queued', 'processing', 'completed', 'failed', 'cancelled')
            );
        END IF;
    END $$""",
    "ALTER TABLE upload_jobs ALTER COLUMN status SET DEFAULT 'staging'",
    "ALTER TABLE upload_jobs ADD COLUMN IF NOT EXISTS worker_start_token VARCHAR(128)",
    "ALTER TABLE upload_jobs ADD COLUMN IF NOT EXISTS staging_node_id VARCHAR(128)",
    "ALTER TABLE upload_jobs ADD COLUMN IF NOT EXISTS staging_cleaned "
    "BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE upload_jobs ADD COLUMN IF NOT EXISTS worker_node_id VARCHAR(128)",
    "ALTER TABLE upload_jobs ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE upload_jobs ADD COLUMN IF NOT EXISTS max_attempts INTEGER NOT NULL DEFAULT 3",
    "ALTER TABLE upload_jobs ADD COLUMN IF NOT EXISTS last_failure_class VARCHAR(64)",
    "ALTER TABLE upload_jobs ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ",
    "CREATE INDEX IF NOT EXISTS idx_upload_jobs_user_status ON upload_jobs(user_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_upload_jobs_dispatch ON upload_jobs(status, lease_expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_upload_jobs_retry_dispatch "
    "ON upload_jobs(status, next_attempt_at, created_at)",
    """CREATE TABLE IF NOT EXISTS upload_job_items (
        id BIGSERIAL PRIMARY KEY,
        process_id VARCHAR(96) NOT NULL REFERENCES upload_jobs(process_id) ON DELETE CASCADE,
        sequence INTEGER NOT NULL,
        original_name VARCHAR(255) NOT NULL,
        part_name VARCHAR(255) NOT NULL,
        storage_key VARCHAR(255) NOT NULL,
        status VARCHAR(16) NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
        indexed_count INTEGER NOT NULL DEFAULT 0,
        process_time DOUBLE PRECISION NOT NULL DEFAULT 0,
        error TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_upload_job_item_sequence UNIQUE (process_id, sequence)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_upload_job_items_pending ON upload_job_items(process_id, status)",
    """CREATE TABLE IF NOT EXISTS upload_job_events (
        id BIGSERIAL PRIMARY KEY,
        process_id VARCHAR(96) NOT NULL REFERENCES upload_jobs(process_id) ON DELETE CASCADE,
        role VARCHAR(64) NOT NULL DEFAULT 'upload_progress',
        step VARCHAR(64) NOT NULL,
        message TEXT,
        percent INTEGER CHECK (percent IS NULL OR (percent >= 0 AND percent <= 100)),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_upload_job_events_cursor ON upload_job_events(process_id, id)",
)


def ensure_schema() -> None:
    """Apply the idempotent tables and column additions the code depends on."""
    with engine.begin() as conn:
        for statement in _SCHEMA_UPGRADES:
            conn.execute(text(statement))
