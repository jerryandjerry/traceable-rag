
CREATE EXTENSION IF NOT EXISTS pgcrypto;
-- Users and authentication state.
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(100) NOT NULL,
    auth_version INTEGER NOT NULL DEFAULT 0,  -- bumped on password change; tokens carry it
    deletion_requested BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_version INTEGER NOT NULL DEFAULT 0;
ALTER TABLE users ADD COLUMN IF NOT EXISTS deletion_requested BOOLEAN NOT NULL DEFAULT FALSE;

-- Chat sessions.
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(16) PRIMARY KEY,
    session_name VARCHAR(255) NOT NULL,  
    user_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Session indexes.
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at  ON sessions(created_at);

-- Chat messages.
CREATE TABLE IF NOT EXISTS messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(16) NOT NULL,
    user_question TEXT NOT NULL,
    model_answer TEXT NOT NULL,
    documents TEXT,  -- serialized citation metadata
    recommended_questions TEXT,  
    think TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_messages_session FOREIGN KEY (session_id)
        REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- Message indexes.
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
-- Preserve existing orphan rows while enforcing the relation for new writes.
DO $$ BEGIN
    ALTER TABLE messages ADD CONSTRAINT fk_messages_session
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        ON DELETE CASCADE NOT VALID;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Ingested-document metadata.
CREATE TABLE IF NOT EXISTS knowledgebases (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    total_chunks INT NULL,
    process_time DOUBLE PRECISION NULL,
    error TEXT NULL,  -- set when a stage of ingest failed; the document is partial
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Idempotent column additions for existing databases.
ALTER TABLE knowledgebases ADD COLUMN IF NOT EXISTS total_chunks INT;
ALTER TABLE knowledgebases ADD COLUMN IF NOT EXISTS process_time DOUBLE PRECISION;
ALTER TABLE knowledgebases ADD COLUMN IF NOT EXISTS error TEXT;

-- Knowledge-base indexes and uniqueness.
CREATE INDEX IF NOT EXISTS idx_knowledgebases_user_id ON knowledgebases(user_id);
CREATE INDEX IF NOT EXISTS idx_knowledgebases_created_at ON knowledgebases(created_at);
DO $$ BEGIN
    ALTER TABLE knowledgebases ADD CONSTRAINT uq_kb_user_file UNIQUE (user_id, file_name);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Durable upload queue. PostgreSQL, rather than an API worker's memory, owns
-- job identity, progress, cancellation and recovery leases.
CREATE TABLE IF NOT EXISTS upload_jobs (
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
);
ALTER TABLE upload_jobs ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE upload_jobs ADD COLUMN IF NOT EXISTS max_attempts INTEGER NOT NULL DEFAULT 3;
ALTER TABLE upload_jobs ADD COLUMN IF NOT EXISTS last_failure_class VARCHAR(64);
ALTER TABLE upload_jobs ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ;
-- Normalize generated or explicit constraint names to the current state set.
DO $$ BEGIN
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
END $$;
ALTER TABLE upload_jobs ALTER COLUMN status SET DEFAULT 'staging';
CREATE INDEX IF NOT EXISTS idx_upload_jobs_user_status ON upload_jobs(user_id, status);
CREATE INDEX IF NOT EXISTS idx_upload_jobs_dispatch ON upload_jobs(status, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_upload_jobs_retry_dispatch ON upload_jobs(status, next_attempt_at, created_at);
CREATE TABLE IF NOT EXISTS upload_job_items (
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
);
CREATE INDEX IF NOT EXISTS idx_upload_job_items_pending ON upload_job_items(process_id, status);

CREATE TABLE IF NOT EXISTS upload_job_events (
    id BIGSERIAL PRIMARY KEY,
    process_id VARCHAR(96) NOT NULL REFERENCES upload_jobs(process_id) ON DELETE CASCADE,
    role VARCHAR(64) NOT NULL DEFAULT 'upload_progress',
    step VARCHAR(64) NOT NULL,
    message TEXT,
    percent INTEGER CHECK (percent IS NULL OR (percent >= 0 AND percent <= 100)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_upload_job_events_cursor ON upload_job_events(process_id, id);
