"""HTTP endpoints for document upload and durable progress streaming.

This layer reads multipart data, mints the ``IngestJob``, and translates worker
status into SSE frames. ``pipeline.ingest`` owns splitting, worker lifecycle,
parse/graph/index stages, and the PostgreSQL write.
"""
import asyncio
import json
from typing import Any, BinaryIO

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from visionagent.api.deps import AuthorizedIngestJob
from visionagent.api.observability import get_api_logger, translated_http_error
from visionagent.api.security import access_security, get_current_user_id
from visionagent.config.settings import settings
from visionagent.pipeline import ingest
from visionagent.pipeline.ingest import DocumentNameAmbiguous, DocumentNotFound

router = APIRouter()
logger = get_api_logger(__name__)


_UPLOAD_REQUEST_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["files"],
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string", "format": "binary"},
                        }
                    },
                }
            }
        },
    }
}


async def _start_file_processing(
    job: AuthorizedIngestJob,
    files: list[StarletteUploadFile],
) -> dict[str, Any]:
    """Validate parsed files, durably stage them, and return the process id."""
    # Authenticated and authorized before this body runs. What is left for
    # the route is to validate the request spool, within the server's limits,
    # and complete the ticket with the names. The frontend's own count and
    # size checks are conveniences; a direct caller meets these.
    if len(files) > settings.max_upload_files:
        raise HTTPException(
            status_code=413,
            detail=f"at most {settings.max_upload_files} files per upload",
        )
    uploads: list[tuple[str, BinaryIO]] = []
    aggregate = 0
    for file in files:
        name = file.filename or "upload"
        try:
            name = ingest.normalize_document_name(name)
        except ingest.UploadNameInvalid as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        total = 0
        try:
            while chunk := await file.read(1 << 20):
                total += len(chunk)
                aggregate += len(chunk)
                if total > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{name} exceeds {settings.max_upload_bytes} bytes",
                    )
                if aggregate > settings.max_upload_total_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "upload exceeds "
                            f"{settings.max_upload_total_bytes} aggregate bytes"
                        ),
                    )
            # The request FormData owns this seekable spooled stream and
            # closes it on context exit. start_processing synchronously
            # consumes it inside the worker thread before that exit.
            await file.seek(0)
        except HTTPException:
            raise
        except Exception as e:
            raise translated_http_error(
                e,
                operation="read_upload",
                detail=f"Error reading file {name}",
                status_code=400,
                file_name=name,
            ) from e
        # UploadFile.filename is Optional: a multipart part without one used
        # to reach .lower() and os.path.splitext() and raise mid-upload.
        uploads.append((name, file.file))

    complete = job.model_copy(update={"file_names": tuple(name for name, _ in uploads)})
    try:
        # Inspection and durable staging are file work, not loop work. Shield
        # the thread handoff so cancellation cannot let FastAPI close its
        # request-owned spools while that thread is still consuming them.
        staging_task = asyncio.create_task(
            asyncio.to_thread(ingest.start_processing, complete, uploads)
        )
        try:
            process_id = await asyncio.shield(staging_task)
        except asyncio.CancelledError:
            try:
                await staging_task
            except BaseException:
                # start_processing already reconciles/logs a failed STAGING
                # ticket. Preserve the request cancellation after ownership
                # of every UploadFile can safely return to FastAPI.
                pass
            raise
    except ingest.UploadAccountUnavailable as e:
        raise HTTPException(
            status_code=409,
            detail="Account deletion is in progress; upload was not accepted",
        ) from e
    except ingest.UploadDocumentConflict as e:
        raise HTTPException(
            status_code=409,
            detail="A document with this name already exists or is being processed",
        ) from e
    except ingest.UploadNameInvalid as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise translated_http_error(
            e,
            operation="start_file_processing",
            detail="Failed to start file processing",
            run_id=complete.identity.run_id,
            user_id=complete.identity.user_id,
        ) from e

    return {"process_id": process_id}


@router.post("/start-processing", openapi_extra=_UPLOAD_REQUEST_BODY)
async def start_file_processing(
    request: Request,
    job: AuthorizedIngestJob,
) -> dict[str, Any]:
    """Authenticate first, then parse a bounded multipart upload."""
    try:
        # No File(...) parameter on purpose: FastAPI would parse and spool the
        # complete multipart body before resolving the authentication job.
        # The outer ASGI fence has already admitted and byte-bounded this body.
        async with request.form(
            max_files=settings.max_upload_files,
            max_fields=settings.max_upload_files,
        ) as form:
            if any(key != "files" for key, _value in form.multi_items()):
                raise HTTPException(
                    status_code=422,
                    detail="only the files multipart field is accepted",
                )
            values = form.getlist("files")
            if not values:
                raise HTTPException(status_code=422, detail="files field is required")
            if not all(isinstance(value, StarletteUploadFile) for value in values):
                raise HTTPException(
                    status_code=422,
                    detail="files must be multipart file parts",
                )
            files = [value for value in values if isinstance(value, StarletteUploadFile)]
            return await _start_file_processing(job, files)
    except StarletteHTTPException as error:
        if error.status_code == 400 and str(error.detail).startswith("Too many files"):
            raise HTTPException(
                status_code=413,
                detail=f"at most {settings.max_upload_files} files per upload",
            ) from error
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
            headers=error.headers,
        ) from error


async def _owned_process(process_id: str, credentials: Any, *, progress: bool = False) -> Any:
    """The durable job, or 401/404: an upload is visible to its owner only."""
    user_id = get_current_user_id(credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Access denied")
    record = await asyncio.to_thread(ingest.upload_job, process_id, progress=progress)
    if record is None:
        raise HTTPException(status_code=404, detail="Process not found")
    if record.user_id != user_id:
        raise HTTPException(status_code=401, detail="Access denied")
    return record


@router.get("/get-process-progress/{process_id}")
async def get_process_progress(
    process_id: str,
    request: Request,
    credentials: Any = Depends(access_security),
) -> StreamingResponse:
    """Stream the durable progress journal in event-id order."""
    await _owned_process(process_id, credentials)

    last_event_id = request.headers.get("last-event-id", "0")
    cursor_at_connect = int(last_event_id) if last_event_id.isdigit() else 0

    async def generate_progress_stream() -> Any:
        cursor = cursor_at_connect
        try:
            while True:
                events = await asyncio.to_thread(
                    ingest.upload_events_after, process_id, cursor
                )
                for event in events:
                    cursor = event.id
                    yield (
                        f"id: {event.id}\n"
                        f"event: message\n"
                        f"data: {json.dumps(event.wire_dict())}\n\n"
                    )

                record = await asyncio.to_thread(
                    ingest.upload_job, process_id, progress=False
                )
                if record is None:
                    logger.info(
                        "upload_progress_stream_ended",
                        process_id=process_id,
                        reason="process_removed",
                    )
                    break
                if record.status.terminal:
                    # The terminal event and status commit atomically. Since
                    # events were fetched first, one final pass closes the
                    # race where the commit lands between these two reads.
                    final_events = await asyncio.to_thread(
                        ingest.upload_events_after, process_id, cursor
                    )
                    for event in final_events:
                        cursor = event.id
                        yield (
                            f"id: {event.id}\n"
                            f"event: message\n"
                            f"data: {json.dumps(event.wire_dict())}\n\n"
                        )
                    break
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception(
                "upload_progress_stream_failed",
                error,
                process_id=process_id,
            )
            message = {
                "role": "upload_progress",
                "step": "error",
                "message": "Progress stream failed; reconnect to resume",
            }
            yield f"event: message\ndata: {json.dumps(message)}\n\n"

    return StreamingResponse(
        generate_progress_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.post("/kill-processing/{process_id}")
async def kill_file_processing(
    process_id: str,
    credentials: Any = Depends(access_security),
) -> dict[str, Any]:
    """Persist cancellation; active work acknowledges it at a safe boundary."""
    record = await _owned_process(process_id, credentials)
    try:
        cancelled = await asyncio.to_thread(
            ingest.cancel_processing, process_id, record.user_id
        )
        if not cancelled:
            raise HTTPException(status_code=404, detail="Process not found")
        current = await asyncio.to_thread(ingest.upload_job, process_id, progress=False)
        current_status = current.status.value if current is not None else "removed"
        process_status = (
            current_status
            if current is None or current.status.terminal
            else "cancellation_pending"
        )
        if process_status == "cancelled":
            message = "Process cancelled successfully"
        elif process_status == "cancellation_pending":
            message = "Cancellation requested; the active file is being finalized"
        else:
            message = (
                f"Process was already {process_status}; cancellation was not applied"
            )
        logger.info(
            "upload_cancellation_accepted",
            process_id=process_id,
            process_status=process_status,
        )
        return {
            "status": "success",
            "process_status": process_status,
            "message": message,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise translated_http_error(
            e,
            operation="kill_file_processing",
            detail="Upload cancelled",
            process_id=process_id,
        ) from e

@router.get("/process-status/{process_id}")
async def get_process_status(
    process_id: str,
    credentials: Any = Depends(access_security),
) -> Any:
    """Return a durable snapshot including the ordered progress journal."""
    try:
        record = await _owned_process(process_id, credentials, progress=True)
        return record.public_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise translated_http_error(
            e,
            operation="get_process_status",
            detail="Failed to get process status",
            process_id=process_id,
        ) from e

@router.post("/cleanup-processes")
async def cleanup_completed_processes(
    credentials: Any = Depends(access_security),
) -> dict[str, Any]:
    """Delete this user's terminal job journal and staged files."""
    try:
        user_id = get_current_user_id(credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Access denied")
        cleaned_count = await asyncio.to_thread(ingest.cleanup_upload_jobs, user_id)

        logger.info(
            "upload_processes_cleaned",
            cleaned_count=cleaned_count,
            user_id=user_id,
        )
        return {"cleaned_count": cleaned_count}

    except HTTPException:
        raise
    except Exception as e:
        raise translated_http_error(
            e,
            operation="cleanup_processes",
            detail="Failed to cleanup processes",
        ) from e

@router.delete("/delete-document/{file_name}")
async def delete_document(
    file_name: str,
    credentials: Any = Depends(access_security),
) -> dict[str, Any]:
    """
    Delete a document and its associated chunks from the knowledge base.
    """
    try:
        user_id = get_current_user_id(credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Access denied")

        # Both delete endpoints use the same cross-store pipeline operation.
        try:
            await ingest.delete_document(user_id, file_name)
        except DocumentNotFound:
            return {"message": "Document not found"}
        except DocumentNameAmbiguous as error:
            raise HTTPException(
                status_code=409,
                detail="Conflicting legacy document names require reindexing",
            ) from error
        except TimeoutError as e:
            raise HTTPException(status_code=409, detail="An upload is still running; try again") from e
        return {"message": "Document deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise translated_http_error(
            e,
            operation="delete_document",
            detail="Failed to delete document",
            file_name=file_name,
        ) from e


@router.get("/document-chunks/{file_name}")
async def get_document_chunks(
    file_name: str,
    credentials: Any = Depends(access_security),
) -> dict[str, Any]:
    """
    Get all chunks for a specific document, sorted by page_num then top_int
    """
    try:
        user_id = get_current_user_id(credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Access denied")

        return {"chunks": ingest.document_chunks(user_id, file_name)}

    except HTTPException:
        raise
    except Exception as e:
        raise translated_http_error(
            e,
            operation="get_document_chunks",
            detail="Failed to get document chunks",
            file_name=file_name,
        ) from e
