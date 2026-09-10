import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from visionagent.api.body_limits import UploadBodyLimitMiddleware
from visionagent.api.observability import RequestContextMiddleware
from visionagent.api.routes import (
    add_context_rt,
    ai_search_rt,
    file_upload_rt,
    graphml_rt,
    history_rt,
    user_rt,
)
from visionagent.pipeline.ingest import (
    enable_upload_workers,
    shutdown_upload_workers,
    supervise_upload_jobs,
)
from visionagent.pipeline.query import QueryPipeline
from visionagent.pipeline.startup import prepare_storage
from visionagent.utils.observability import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and close process-owned services.

    Storage and schema preparation precede construction of the shared,
    stateless query pipeline and durable upload supervisor.
    """
    # The server may install its handlers after importing this module. Elect
    # one application JSON sink now so gunicorn/uvicorn handlers cannot double
    # every application record.
    configure_logging()
    prepare_storage()
    app.state.pipeline = QueryPipeline()
    enable_upload_workers()
    upload_supervisor = asyncio.create_task(
        supervise_upload_jobs(), name="upload-job-supervisor"
    )
    try:
        yield
    finally:
        upload_supervisor.cancel()
        with suppress(asyncio.CancelledError):
            await upload_supervisor
        await asyncio.to_thread(shutdown_upload_workers)
        await app.state.pipeline.aclose()


app = FastAPI(title="Traceable RAG", lifespan=lifespan)
# Starlette wraps user middleware in reverse registration order. Register the
# upload fence first so it catches receive-side overflow before the outer
# request/error boundary records the resulting 413.
app.add_middleware(UploadBodyLimitMiddleware)
app.add_middleware(RequestContextMiddleware)

app.include_router(user_rt.router)
app.include_router(history_rt.router)
app.include_router(ai_search_rt.router)
app.include_router(add_context_rt.router)
app.include_router(file_upload_rt.router)
app.include_router(graphml_rt.router)

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
