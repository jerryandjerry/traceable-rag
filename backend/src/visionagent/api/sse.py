"""SSE frames for the progress trace.

The API serializes pipeline trace models as typed ``step`` and ``evidence``
events. A step's first frame carries the complete node; subsequent frames patch
its status, duration, and optional note. A text-form ``workflow_progress``
message is emitted alongside the typed events.
"""
from __future__ import annotations

import json

from visionagent.models import Evidence, TraceStep


def frame(event: str, payload: object) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def step_frame(step: TraceStep, *, full: bool = False) -> str:
    """A step's first frame carries everything; later ones only what changed."""
    if full:
        data = step.model_dump()
    else:
        data = {
            "id": step.id,
            "status": step.status.value,
            "duration_s": step.duration_s,
        }
        if step.note:
            data["note"] = step.note
    return frame("step", data)


def evidence_frame(item: Evidence) -> str:
    """Its own frame so a thumbnail never delays an answer token."""
    return frame("evidence", item.model_dump())


def token_frame(content: str, reasoning: str = "") -> str:
    return frame("token", {"text": content, "reasoning": reasoning})


def legacy_progress_frame(steps: list[TraceStep]) -> str:
    """Render the text-form workflow message sent with typed step events."""
    by_parent: dict[str | None, list[TraceStep]] = {}
    for s in steps:
        by_parent.setdefault(s.parent_id, []).append(s)

    lines = ["Workflow Progress"]

    def render(parent: str | None, indent: str) -> None:
        for s in by_parent.get(parent, []):
            mark = "☒" if s.duration_s is not None else "☐"
            timing = f" ({s.duration_s:.2f}s)" if s.duration_s is not None else ""
            note = f" {s.note}" if s.note else ""
            lines.append(f"{indent}{mark} {s.label}{timing}{note}")
            render(s.id, indent + "   ")

    render(None, "     ")
    return frame("message", {"role": "workflow_progress", "content": "\n".join(lines)})
