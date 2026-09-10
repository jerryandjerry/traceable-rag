"""What the process does once, before it answers its first request.

A workflow like the others, run by the application lifespan rather than by a
route. The schema upgrade is a write to Postgres, and api/ does not write to
storage; it asks a pipeline to.
"""
from __future__ import annotations

from visionagent.database.postgres.engine import ensure_schema


def prepare_storage() -> None:
    """Bring the database to the shape the code reads.

    Idempotent schema changes only: older volumes gain `auth_version`,
    `knowledgebases.error`, and the durable upload queue/journal tables; a
    database already at that shape is untouched.
    """
    ensure_schema()
