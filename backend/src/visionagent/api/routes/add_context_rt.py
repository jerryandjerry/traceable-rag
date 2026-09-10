import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from visionagent.api.observability import get_api_logger, translated_http_error
from visionagent.api.security import access_security, get_current_user_id, verify_session_owner
from visionagent.config.settings import settings

# The module, not its functions: three route handlers below carry the same
# names as the workflows they call, and a bare import would be shadowed by
# the handler that uses it.
from visionagent.pipeline import context as context_workflow
from visionagent.utils.file_utils import get_safe_file_path, validate_context_filename

router = APIRouter()
logger = get_api_logger(__name__)

_CONTEXT_REQUEST_BODY = {
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


async def _copy_context_file(source: BinaryIO, file_path: str) -> None:
    """Finish a cancellation-safe atomic stream copy off the event loop."""
    task = asyncio.create_task(asyncio.to_thread(_write_file, source, file_path))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except BaseException:
            pass
        raise


async def _add_context_files(
    *,
    session_id: str,
    files: list[UploadFile],
    user_id: str,
) -> dict[str, Any]:
    """Validate parsed spools, persist atomically, and run context extraction."""
    if len(files) > settings.max_upload_files:
        raise HTTPException(
            status_code=413,
            detail=f"at most {settings.max_upload_files} files per upload",
        )
    aggregate = 0
    for file in files:
        name = file.filename or "upload"
        try:
            validate_context_filename(name)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        total = 0
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
        await file.seek(0)

    start_time = time.time()
    processed_files = []
    # Authorization above is a request-boundary check. The durable gate is
    # re-read under the same tenant lock account deletion uses before any
    # path or context JSON is written.
    try:
        async with context_workflow.context_write_scope(user_id, session_id):
            # Only now, with the id proven again inside the write fence,
            # does it become part of a path.
            # Persist canonical absolute paths in the deletion manifest even
            # when a local deployment configured STORAGE_DIR relatively.
            # The store subsequently validates every manifest entry against
            # this exact owned directory before it can unlink anything.
            session_dir = os.path.abspath(
                os.path.join(str(settings.storage_dir / "context_files"), session_id)
            )
            await asyncio.to_thread(os.makedirs, session_dir, exist_ok=True)

            for file in files:
                original_file_name = file.filename or "upload"
                file_name, file_path = await asyncio.to_thread(
                    get_safe_file_path,
                    session_dir,
                    original_file_name,
                )
                file_extension = os.path.splitext(file_name)[1].lower()
                committed = False
                try:
                    await _copy_context_file(file.file, file_path)

                    logger.info(
                        "context_file_processing_started",
                        file_extension=file_extension,
                    )

                    vlm_start = time.time()
                    extraction = asyncio.create_task(
                        context_workflow.add_file_to_context(
                            session_id,
                            file_path,
                            file_name,
                        )
                    )
                    try:
                        result = await asyncio.shield(extraction)
                    except asyncio.CancelledError:
                        extraction.cancel()
                        try:
                            await extraction
                        except context_workflow.ContextCommitCancelled as commit:
                            committed = bool(commit.result.get("success"))
                        except asyncio.CancelledError:
                            pass
                        except BaseException:
                            pass
                        raise
                    vlm_time = time.time() - vlm_start

                    if result["success"]:
                        committed = True
                        logger.info(
                            "context_file_processing_completed",
                            file_extension=file_extension,
                            duration_ms=round(vlm_time * 1000, 2),
                            content_length=result.get("content_length", 0),
                            pages_processed=result.get("pages_processed", 0),
                        )
                        processed_files.append({
                            "filename": file_name,
                            "processing_time": round(vlm_time, 2),
                            "content_length": result.get("content_length", 0),
                            "pages_processed": result.get("pages_processed", 0),
                            "status": "success"
                        })
                    else:
                        logger.error(
                            "context_file_processing_failed",
                            file_extension=file_extension,
                        )
                        processed_files.append({
                            "filename": file_name,
                            "processing_time": round(vlm_time, 2),
                            "status": "failed",
                            "error": "File processing failed"
                        })
                finally:
                    if not committed:
                        await asyncio.to_thread(_remove_file, file_path)
    except (context_workflow.ContextWriteUnavailable, TimeoutError) as exc:
        raise HTTPException(
            status_code=409,
            detail="Account deletion is in progress; context was not saved",
        ) from exc

    total_time = time.time() - start_time

    return {
        "status": "success",
        "message": "Files processed and added to session context successfully",
        "session_id": session_id,
        "timing": {
            "total_time": round(total_time, 2),
            "files_processed": len(files),
            "individual_files": processed_files
        }
    }


@router.post("/add_context/", openapi_extra=_CONTEXT_REQUEST_BODY)
async def add_context(
    request: Request,
    session_id: str = Query(...),
    credentials: Any = Depends(access_security),
) -> dict[str, Any]:
    """Authorize the session before parsing its bounded multipart body."""
    try:
        # Authentication freshness and ownership both consult PostgreSQL.
        # No File(...) parameter exists, so these checks run before spooling.
        user_id = await asyncio.to_thread(get_current_user_id, credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        await asyncio.to_thread(verify_session_owner, session_id, user_id)

        try:
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
                if not all(isinstance(value, UploadFile) for value in values):
                    raise HTTPException(
                        status_code=422,
                        detail="files must be multipart file parts",
                    )
                files = [value for value in values if isinstance(value, UploadFile)]
                return await _add_context_files(
                    session_id=session_id,
                    files=files,
                    user_id=user_id,
                )
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
    except HTTPException:
        raise
    except Exception as e:
        raise translated_http_error(
            e,
            operation="add_context",
            detail="Failed to add context files",
            session_id=session_id,
        ) from e


@router.get("/get_context/{session_id}")
async def get_context(
    session_id: str,
    credentials: Any = Depends(access_security),
) -> dict[str, Any]:
    """
    Get session context for a specific session
    """
    try:
        user_id = get_current_user_id(credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        verify_session_owner(session_id, user_id)
        
        return {"session_id": session_id, **context_workflow.get_context(session_id)}
    
    except HTTPException:
        raise
    except Exception as e:
        raise translated_http_error(
            e,
            operation="get_context",
            detail="Failed to get context",
            session_id=session_id,
        ) from e

@router.delete("/clear_context/{session_id}")
async def clear_context(
    session_id: str,
    credentials: Any = Depends(access_security),
) -> dict[str, Any]:
    """
    Clear all context for a session
    """
    try:
        user_id = await asyncio.to_thread(get_current_user_id, credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        await asyncio.to_thread(verify_session_owner, session_id, user_id)

        try:
            async with context_workflow.context_write_scope(user_id, session_id):
                await asyncio.to_thread(context_workflow.clear_context, session_id)
        except (context_workflow.ContextWriteUnavailable, TimeoutError) as exc:
            raise HTTPException(
                status_code=409,
                detail="Account deletion is in progress; context was not changed",
            ) from exc
        
        return {
            "status": "success",
            "message": f"Context cleared for session {session_id}",
            "session_id": session_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise translated_http_error(
            e,
            operation="clear_context",
            detail="Failed to clear context",
            session_id=session_id,
        ) from e

@router.delete("/remove_file/{session_id}")
async def remove_file_from_context(
    session_id: str,
    file_name: str = Query(...),
    credentials: Any = Depends(access_security),
) -> dict[str, Any]:
    """
    Remove a specific file from session context
    """
    try:
        user_id = await asyncio.to_thread(get_current_user_id, credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        await asyncio.to_thread(verify_session_owner, session_id, user_id)

        try:
            async with context_workflow.context_write_scope(user_id, session_id):
                success = await asyncio.to_thread(
                    context_workflow.remove_file_from_context,
                    session_id,
                    file_name,
                )
        except (context_workflow.ContextWriteUnavailable, TimeoutError) as exc:
            raise HTTPException(
                status_code=409,
                detail="Account deletion is in progress; context was not changed",
            ) from exc
        
        if success:
            return {
                "status": "success",
                "message": f"File {file_name} removed from session {session_id}",
                "session_id": session_id,
                "file_name": file_name
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_name} not found in session {session_id}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise translated_http_error(
            e,
            operation="remove_context_file",
            detail="Failed to remove context file",
            session_id=session_id,
            file_name=file_name,
        ) from e


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_file(file_path: str) -> None:
    """Remove an unpublished/failed raw context file durably."""
    path = Path(file_path)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _write_file(source: BinaryIO, file_path: str) -> None:
    """Stream one spool to an atomically published, fsynced session file."""
    target = Path(file_path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=".context-upload-",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    published = False
    try:
        source.seek(0)
        with os.fdopen(descriptor, "wb") as buffer:
            while chunk := source.read(1 << 20):
                buffer.write(chunk)
            buffer.flush()
            os.fsync(buffer.fileno())
        # Hard-link publication is atomic and refuses to overwrite a target
        # that appeared after get_safe_file_path selected this name.
        os.link(temporary, target)
        published = True
        temporary.unlink()
        _fsync_directory(target.parent)
    except BaseException:
        if published:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        try:
            _fsync_directory(target.parent)
        except OSError:
            pass
        raise
