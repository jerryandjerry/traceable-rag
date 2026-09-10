import json
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

# Initialize the deterministic environment-file policy before application imports.
import visionagent.config.settings  # noqa: F401
from visionagent.api.deps import AuthorizedQueryJob, CurrentUser
from visionagent.api.observability import get_api_logger, translated_http_error
from visionagent.api.sse import evidence_frame, legacy_progress_frame, step_frame
from visionagent.models import Evidence, TraceStep
from visionagent.pipeline.query import QueryPipeline
from visionagent.pipeline.session import (
    AccountWriteUnavailable,
)
from visionagent.pipeline.session import (
    create_session as create_session_workflow,
)

logger = get_api_logger(__name__)

_BUILD = threading.Lock()


def get_pipeline(request: Request) -> QueryPipeline:
    """The process's one pipeline, built during the application lifespan.

    A dependency rather than a bare call so a test can substitute a pipeline
    built from deterministic slots. Overriding the module global instead would
    leak between tests and would not exercise the route's own wiring.

    Safe as a singleton only because it keeps no per-turn state: everything a
    request needs lives in the AgentState the pipeline builds from its job.
    An app mounted without the lifespan (a bare FastAPI() in a test) builds it
    here, under a lock -- FastAPI resolves sync dependencies in worker threads,
    and an unguarded lazy global was constructed twice under concurrent cold
    start.
    """
    state = request.app.state
    pipeline = getattr(state, "pipeline", None)
    if pipeline is None:
        with _BUILD:
            pipeline = getattr(state, "pipeline", None)
            if pipeline is None:
                pipeline = state.pipeline = QueryPipeline()
    assert isinstance(pipeline, QueryPipeline)
    return pipeline


router = APIRouter()
@router.post("/create_session/")
async def create_session(user_id: CurrentUser) -> dict[str, Any]:
    try:
        # The session must exist before it can be used: /ai_search/ and the
        # context routes verify ownership against this row.
        session_id = create_session_workflow(user_id)
    except AccountWriteUnavailable as exc:
        raise HTTPException(
            status_code=409,
            detail="Account deletion is in progress; session was not created",
        ) from exc
    except Exception as db_error:
        raise translated_http_error(
            db_error,
            operation="create_session",
            detail="Failed to create session",
        ) from db_error

    return {
        "session_id": session_id,
        "status": "success",
        "message": "Session created successfully"
    }
@router.post("/ai_search/")
async def ai_search(
    job: AuthorizedQueryJob,
    pipeline: QueryPipeline = Depends(get_pipeline),
) -> StreamingResponse:
    # The dependency authenticated the caller, verified session ownership,
    # resolved what policy allows and validated the question, all before this
    # body runs. What arrives is an immutable ticket; the pipeline builds the
    # turn's state from it.

    async def enhanced_workflow_generator() -> Any:
        try:
            # Everything else -- scenario recognition, the casual/professional
            # branch, the retrieval loop and the answer -- is pipeline/query.py.
            # This route turns its events into SSE bytes; it does not know how
            # retrieval works, and the pipeline does not know about SSE.
            tracer_steps: dict[str, TraceStep] = {}

            async for kind, payload in pipeline.stream(job):
                if kind == "step":
                    assert isinstance(payload, TraceStep)
                    first = payload.id not in tracer_steps
                    tracer_steps[payload.id] = payload
                    yield step_frame(payload, full=first)
                    # Emit the text-form workflow representation as well.
                    yield legacy_progress_frame(list(tracer_steps.values()))
                elif kind == "evidence":
                    assert isinstance(payload, Evidence)
                    yield evidence_frame(payload)
                elif kind == "frame":
                    # The answer slot's own SSE frames, forwarded verbatim.
                    assert isinstance(payload, str)
                    yield payload

        except Exception as exc:
            # Logs keep the exception type and source frames, never its raw
            # text (which may name providers, hosts, paths or SQL). The run id
            # is what a user quotes and what locates the correlated event.
            logger.exception(
                "query_stream_failed",
                exc,
                operation="ai_search",
                run_id=job.identity.run_id,
                session_id=job.session_id,
                user_id=job.identity.user_id,
            )
            error_message = {
                "role": "error",
                "run_id": job.identity.run_id,
                "content": f"The request failed. Reference: {job.identity.run_id}",
            }
            yield f"event: error\ndata: {json.dumps(error_message)}\n\n"

    return StreamingResponse(
        enhanced_workflow_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
