import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from visionagent.database.postgres.engine import get_db
from visionagent.exceptions.database import DatabaseError

logger = logging.getLogger(__name__)


def insert_knowledgebase(user_id: str, file_name: str, total_chunks: int | None = None,
                         process_time: float | None = None, error: str | None = None) -> None:
    """Upsert one tenant's knowledge-base document record.

    ``error`` records why ingest did not fully complete. It is always written,
        so a re-ingest that succeeds clears the previous run's error.
    """
    db = next(get_db())
    try:
        has_metrics = total_chunks is not None or process_time is not None
        db.execute(
            text(
                """
                INSERT INTO knowledgebases
                    (user_id, file_name, total_chunks, process_time, error, updated_at)
                VALUES (
                    :user_id,
                    :file_name,
                    CASE WHEN :has_metrics THEN COALESCE(:total_chunks, 0) ELSE NULL END,
                    CASE WHEN :has_metrics THEN COALESCE(:process_time, 0) ELSE NULL END,
                    CASE WHEN :has_metrics THEN :error ELSE NULL END,
                    CURRENT_TIMESTAMP
                )
                ON CONFLICT (user_id, file_name) DO UPDATE SET
                    total_chunks = CASE WHEN :has_metrics
                        THEN COALESCE(EXCLUDED.total_chunks, knowledgebases.total_chunks)
                        ELSE knowledgebases.total_chunks END,
                    process_time = CASE WHEN :has_metrics
                        THEN COALESCE(EXCLUDED.process_time, knowledgebases.process_time)
                        ELSE knowledgebases.process_time END,
                    error = CASE WHEN :has_metrics
                        THEN EXCLUDED.error ELSE knowledgebases.error END,
                    updated_at = CURRENT_TIMESTAMP
                """
            ),
            {
                "user_id": user_id,
                "file_name": file_name,
                "total_chunks": total_chunks,
                "process_time": process_time,
                "error": error,
                "has_metrics": has_metrics,
            },
        )
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise RuntimeError(f"Failed to insert into knowledgebases: {e}") from e
    finally:
        db.close()
        logger.debug("knowledge-base upsert attempt finished")

def verify_user_knowledgebase(user_id: str) -> bool:
    """Return whether the tenant owns at least one knowledge-base record."""
    db = next(get_db())
    try:
        query_result = db.execute(
            text("SELECT id FROM knowledgebases WHERE user_id = :user_id LIMIT 1"),
            {"user_id": user_id}
        ).fetchone()

        if query_result:
            return True
        else:
            return False
    except SQLAlchemyError as e:
        raise DatabaseError(f"Database operation failed: {e}") from e
    finally:
        db.close()

def get_user_history_questions(session_id: str) -> Any:
    """Return the questions previously stored for a session."""
    db = next(get_db())
    try:
        messages_data = db.execute(
            text("SELECT user_question FROM messages WHERE session_id = :session_id"),
            {"session_id": session_id}
        ).fetchall()

        history_questions = []
        for message in messages_data:
            history_questions.append(message.user_question)

        return history_questions
    except SQLAlchemyError as e:
        raise RuntimeError(f"Failed to fetch history questions: {e}") from e
    finally:
        db.close()
