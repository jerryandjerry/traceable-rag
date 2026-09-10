"""Tests for the SSE trace frames.

Intended function:
    step_frame(full=True)  -> exactly the whole step
    step_frame()           -> exactly id + what changed, so the tree is patched
                              rather than resent
    evidence_frame         -> its own frame, so a thumbnail never delays a token
    legacy_progress_frame  -> the pre-rendered workflow-progress fallback
"""
from __future__ import annotations

import json

from visionagent.api.sse import (
    evidence_frame,
    legacy_progress_frame,
    step_frame,
    token_frame,
)
from visionagent.models import Evidence, StepStatus, TraceStep


def parse(raw: str) -> tuple[str, dict | str]:
    head, body = raw.split("\ndata: ", 1)
    payload = body.rstrip("\n")
    return head.removeprefix("event: "), (
        payload if payload == "[DONE]" else json.loads(payload)
    )


def step(**kw) -> TraceStep:
    return TraceStep(id="s1", label="initial search", **kw)


def test_every_frame_is_well_formed_sse():
    for raw in (step_frame(step()), evidence_frame(
        Evidence(step_id="s1", kind="doc", label="curb.pdf")), token_frame("hi")):
        assert raw.startswith("event: ")
        assert "\ndata: " in raw
        assert raw.endswith("\n\n")


def test_a_first_frame_carries_the_whole_step():
    event, data = parse(step_frame(
        step(parent_id="s0", status=StepStatus.RUNNING, started_at=6.23), full=True))
    assert event == "step"
    assert data == {
        "id": "s1", "parent_id": "s0", "label": "initial search",
        "status": "running", "started_at": 6.23, "duration_s": None, "note": None,
    }


def test_a_later_frame_carries_only_what_changed():
    """The tree is patched, not resent: over a 50s run that is the difference
    between one small frame and the whole tree, repeatedly."""
    event, data = parse(step_frame(step(status=StepStatus.DONE, duration_s=4.62)))
    assert event == "step"
    assert data == {"id": "s1", "status": "done", "duration_s": 4.62}
    assert "label" not in data and "parent_id" not in data


def test_a_note_rides_along_when_there_is_one():
    _, data = parse(step_frame(step(status=StepStatus.DONE, duration_s=1.0,
                                    note="needs improvement")))
    assert data["note"] == "needs improvement"


def test_evidence_is_its_own_frame_and_carries_the_thumbnail():
    """The whole reason the contract changed: an image cannot go in a text blob."""
    event, data = parse(evidence_frame(Evidence(
        step_id="s3", kind="chunk", label="Curb Design Guide.pdf",
        chunk_id="c1", thumbnail="data:image/webp;base64,AAA")))
    assert event == "evidence"
    assert data == {
        "step_id": "s3", "kind": "chunk", "label": "Curb Design Guide.pdf",
        "chunk_id": "c1", "thumbnail": "data:image/webp;base64,AAA",
    }


def test_token_frames_carry_both_channels():
    _, data = parse(token_frame("Curb ", "reasoning here"))
    assert data == {"text": "Curb ", "reasoning": "reasoning here"}


def test_legacy_frame_renders_the_workflow_progress_fallback():
    """Text-only clients receive the same hierarchy as typed trace clients."""
    steps = [
        TraceStep(id="s1", label="understanding intent", duration_s=1.4),
        TraceStep(id="s2", label="initial search", duration_s=7.02),
        TraceStep(id="s3", parent_id="s2", label="searching user knowledge base",
                  duration_s=4.62),
        TraceStep(id="s4", parent_id="s2", label="searching online", duration_s=6.93),
        TraceStep(id="s5", label="reviewing", duration_s=0.5, note="looks ok"),
    ]
    event, data = parse(legacy_progress_frame(steps))
    assert event == "message"
    assert data["role"] == "workflow_progress"
    assert data["content"].splitlines() == [
        "Workflow Progress",
        "     ☒ understanding intent (1.40s)",
        "     ☒ initial search (7.02s)",
        "        ☒ searching user knowledge base (4.62s)",
        "        ☒ searching online (6.93s)",
        "     ☒ reviewing (0.50s) looks ok",
    ]


def test_legacy_frame_marks_unfinished_steps_as_pending():
    _, data = parse(legacy_progress_frame([
        TraceStep(id="s1", label="initial search"),
    ]))
    assert data["content"].splitlines()[1] == "     ☐ initial search"
