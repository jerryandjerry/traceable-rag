"""Errors the storage layer raises.

database/ must not raise HTTPException. A transport status code is the API's
decision, and a lower layer that picks one has decided how every future caller
-- a worker, a script, another pipeline -- reports its failure.
"""
from __future__ import annotations


class DatabaseError(RuntimeError):
    """A storage operation failed."""
