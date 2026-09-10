"""Typed progress and evidence frames emitted by the query pipeline.

The API sends these frames alongside a legacy text rendering. The current chat
page prefers the typed trace and uses the text form as a compatibility fallback.
Keeping evidence separate permits chunk previews without embedding images in a
text blob.

The tree is flat with `parent_id` rather than nested children, because SSE
sends deltas. Over a 50-second run one step is patched at a time; the tree is
never resent.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class EvidenceKind(StrEnum):
    DOC = "doc"
    URL = "url"
    NODE = "node"
    CHUNK = "chunk"


class TraceStep(BaseModel):
    """One node of the progress tree.

    Durations are measured with `time.perf_counter()`, which is monotonic (so a
    clock adjustment cannot produce a negative duration) and measures elapsed
    wall time rather than CPU time (so it keeps counting while a coroutine is
    parked on await). A parent's duration is measured directly, never derived
    from its children -- the gap between a parent and its slowest child is the
    signal that concurrency is or is not working.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    parent_id: str | None = None
    label: str = Field(min_length=1)
    status: StepStatus = StepStatus.PENDING
    started_at: float = Field(default=0.0, ge=0.0)
    """Offset from the start of the run, in seconds."""

    duration_s: float | None = Field(default=None, ge=0.0)
    note: str | None = None
    """e.g. "looks ok" / "needs improvement" on the reviewing step."""


class Evidence(BaseModel):
    """A leaf detail under a step: what was read, what was found."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    kind: EvidenceKind
    label: str = Field(min_length=1)
    chunk_id: str | None = None
    thumbnail: str | None = None
    """Base64 preview of the matched chunk. Sent as its own frame so it never
    delays answer tokens."""
