"""Step timing and the progress tree.

The orchestrator owns both. A component is timed by the step that wraps it, so
a swapped-in implementation that has never heard of tracing is timed
identically -- and components stay callable standalone, because they emit leaf
detail through an optional callback that defaults to nothing.

Durations use `time.perf_counter()`:

  * it is **monotonic**, so an NTP correction cannot produce a negative
    duration the way `time.time()` can
  * it measures **elapsed wall time, not CPU time**, so it keeps counting while
    a coroutine is parked on await or a thread is blocked on a socket -- which
    is the end-to-end number the tree should show

A parent's duration is measured directly, never derived from its children. The
gap between a parent and its slowest child is the signal that concurrency is or
is not working; computing the parent would conceal exactly that.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count

from visionagent.models import Evidence, EvidenceKind, StepStatus, TraceStep

# The currently open step, per task. A plain stack on the Tracer is wrong under
# concurrency: asyncio.gather runs the children simultaneously, so each one
# would see whichever sibling happened to open last as its parent, and the
# tree came out as a chain instead of a fan. asyncio copies the context when a
# task is created, so every child inherits the parent that was open at
# dispatch, and its own writes stay local to it.
_CURRENT_STEP: ContextVar[str | None] = ContextVar("va_current_step", default=None)



class Tracer:
    """Builds a flat, patchable step tree for one run."""

    def __init__(self, on_change: Callable[[TraceStep], None] | None = None) -> None:
        # Called whenever a step opens or closes, so a caller can stream the
        # tree as it happens rather than only at the end.
        self._on_change = on_change
        self._t0 = time.perf_counter()
        self._ids = count(1)
        self.steps: dict[str, TraceStep] = {}
        self.evidence: list[Evidence] = []

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def _next_id(self) -> str:
        return f"s{next(self._ids)}"

    @contextmanager
    def step(self, label: str, *, note: str | None = None) -> Iterator[TraceStep]:
        """Open a step, time it, close it. Nests under whatever is open."""
        node = TraceStep(
            id=self._next_id(),
            parent_id=_CURRENT_STEP.get(),
            label=label,
            status=StepStatus.RUNNING,
            started_at=round(self.elapsed, 4),
            note=note,
        )
        self.steps[node.id] = node
        token = _CURRENT_STEP.set(node.id)
        self._notify(node)
        started = time.perf_counter()
        try:
            yield node
        except Exception:
            node.status = StepStatus.FAILED
            raise
        else:
            if node.status is StepStatus.RUNNING:
                node.status = StepStatus.DONE
        finally:
            # Measured, not derived. max(children) would hide queueing, GIL
            # contention and to_thread handoff -- the things worth seeing.
            node.duration_s = round(time.perf_counter() - started, 4)
            _CURRENT_STEP.reset(token)
            self._notify(node)

    def _notify(self, node: TraceStep) -> None:
        """Hand the listener a snapshot, never the live node.

        The listener queues what it is given and drains later. Handed the
        node itself, a fast step's "opened" and "closed" notifications were
        two references to one object, and by the time the queue drained both
        read DONE -- the RUNNING transition never reached the client.
        """
        if self._on_change is not None:
            self._on_change(node.model_copy(deep=True))

    def emit(self, kind: EvidenceKind | str, label: str, *,
             chunk_id: str | None = None, thumbnail: str | None = None) -> Evidence:
        """Record a leaf detail under the currently open step."""
        current = _CURRENT_STEP.get()
        if current is None:
            raise RuntimeError("emit() called with no open step")
        item = Evidence(
            step_id=current,
            kind=EvidenceKind(kind),
            label=label,
            chunk_id=chunk_id,
            thumbnail=thumbnail,
        )
        self.evidence.append(item)
        return item

    def emitter(self) -> Callable[..., Evidence]:
        """A callable a component can be handed, or not.

        Optional so a component stays usable standalone: a caller outside this
        pipeline never constructs one.
        """
        return self.emit

    def tree(self) -> list[TraceStep]:
        return list(self.steps.values())
