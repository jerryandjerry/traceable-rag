"""Per-user graph repository over GraphML and two JSON vector indexes.

Retrieval tools use its read methods; the graph indexing slot uses its write
methods. Both remain independent of the three-file persistence layout.
"""
from __future__ import annotations

import asyncio
import fcntl
import os
import tempfile
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

from visionagent.config.settings import settings
from visionagent.database.graph.identity import canonical_edge_vector_ids
from visionagent.database.graph.lightrag_storage import NetworkXGraphStorage, nanoVectorDB


class GraphPersistenceError(RuntimeError):
    """A graph snapshot may have been partly published and must be reconciled."""


def atomic_write(path: str | Path, write: Any) -> None:
    """Write through a temporary file in the same directory, then rename.

    Graph files are loaded, mutated in memory and written back whole. A plain
    open(path, 'w') truncates first, so a reader arriving mid-write sees an
    empty or half-written file and a concurrent ingest can lose the other's
    update. os.replace is atomic on the same filesystem, so a reader sees
    either the old file or the new one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    os.close(fd)
    try:
        write(tmp)
        # The writer has closed the file. Flush its completed bytes before
        # publishing the name, then flush the directory entry after replace.
        with open(tmp, "rb") as completed:
            os.fsync(completed.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@contextmanager
def tenant_lock(user_id: str, *, timeout_s: float | None = None) -> Iterator[None]:
    """One writer per user across processes, for as long as the block runs.

    The graph is loaded whole, changed in memory and written back whole;
    atomic replacement protects readers from a half-written file but not two
    writers from each other -- the second save discarded the first's
    document. The ingest worker is a separate process, so the lock is a file
    lock under the graph directory rather than a threading primitive. Ingest
    holds it through all tenant work and terminalization; document, session,
    context, and account mutations use the same fence, so deletion cannot
    interleave with a write authorized just before it began.

    `timeout_s` bounds the wait; None blocks. A request that would otherwise
    wait behind a long extraction raises TimeoutError instead.
    """
    path = settings.graph_dir / f".lock_{user_id}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        if timeout_s is None:
            fcntl.flock(fh, fcntl.LOCK_EX)
        else:
            deadline = time.monotonic() + timeout_s
            while True:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"another operation holds user {user_id}'s stores"
                        ) from None
                    time.sleep(0.2)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


@asynccontextmanager
async def async_tenant_lock(
    user_id: str,
    *,
    timeout_s: float | None = None,
) -> AsyncIterator[None]:
    """Cancellation-safe async form of the per-tenant writer lock.

    ``flock(LOCK_NB)`` never parks the event-loop thread. Contention yields via
    ``asyncio.sleep`` so cancellation is observed immediately, and the file
    descriptor is unlocked and closed in ``finally`` on every exit path.
    """
    path = settings.graph_dir / f".lock_{user_id}"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    locked = False
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"another operation holds user {user_id}'s stores"
                    ) from None
                await asyncio.sleep(0.05)
        yield
    finally:
        try:
            if locked:
                fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()


class GraphRepository:
    """The three per-user files, opened once.

    Scoped to one user by construction: every path is derived from the user id
    handed to __init__, so a caller cannot ask it for someone else's graph.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = str(user_id)
        working_dir = str(settings.graph_dir)
        self.working_dir = working_dir

        self.node_vdb = nanoVectorDB(
            embedding_dim=settings.graph_embedding_dimensions,
            storage_file=f"{working_dir}/vdb_nodes_{self.user_id}.json",
            namespace=f"nodes_{self.user_id}",
        )
        self.edge_vdb = nanoVectorDB(
            embedding_dim=settings.graph_embedding_dimensions,
            storage_file=f"{working_dir}/vdb_edges_{self.user_id}.json",
            namespace=f"edges_{self.user_id}",
        )
        # Canonicalize persisted edge-vector ids before reads or mutations.
        # Endpoint metadata makes this lossless; malformed or duplicate pairs
        # fail closed in the identity helper.
        self.edge_vdb.ids, _changed = canonical_edge_vector_ids(
            self.edge_vdb.ids,
            self.edge_vdb.metadatas,
        )
        self.knowledge_graph = NetworkXGraphStorage(
            namespace=self.user_id,
            working_dir=working_dir,
        )

    # ------------------------------------------------------------- reading

    async def query_entities(self, text: str, top_k: int) -> list[dict[str, Any]]:
        """Entities nearest to `text`."""
        return list(await self.node_vdb.query(text, top_k=top_k))

    async def query_relations(self, text: str, top_k: int) -> list[dict[str, Any]]:
        """Relations nearest to `text`."""
        return list(await self.edge_vdb.query(text, top_k=top_k))

    async def get_nodes(self, names: list[str]) -> dict[str, Any]:
        return dict(await self.knowledge_graph.get_nodes_batch(names))

    async def get_edges(self, pairs: list[dict[str, str]]) -> dict[Any, Any]:
        """Relations for {"src": ..., "tgt": ...} pairs, keyed by (src, tgt)."""
        return dict(await self.knowledge_graph.get_edges_batch(pairs))

    async def aclose(self) -> None:
        """Close async transports owned by both vector stores."""
        results = await asyncio.gather(
            self.node_vdb.aclose(),
            self.edge_vdb.aclose(),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    # ------------------------------------------------------------- writing

    def save(self) -> None:
        """Persist a convergent snapshot or raise a typed ambiguity signal.

        Each file is atomically replaced, but three filesystem names cannot be
        committed as one operation.  The vector indices are derived data and
        are published before the GraphML contribution ledger.  Therefore a
        retry or compensation can always converge from either the old ledger
        (discarding orphan vectors) or the new ledger (replaying deletion).
        Callers must treat *any* failure here as potentially partly committed.
        """
        publications = (
            (
                "node vectors",
                f"{self.working_dir}/vdb_nodes_{self.user_id}.json",
                self.node_vdb.save,
            ),
            (
                "edge vectors",
                f"{self.working_dir}/vdb_edges_{self.user_id}.json",
                self.edge_vdb.save,
            ),
            (
                "graph ledger",
                f"{self.working_dir}/graph_{self.user_id}.graphml",
                self.knowledge_graph.save,
            ),
        )
        for stage, path, writer in publications:
            try:
                atomic_write(path, writer)
            except Exception as error:
                raise GraphPersistenceError(
                    f"graph snapshot publication failed at {stage}"
                ) from error
