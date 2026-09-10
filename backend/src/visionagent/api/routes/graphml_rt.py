
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from visionagent.api.security import access_security, get_current_user_id
from visionagent.config.settings import settings

router = APIRouter()

@router.get("/graphml/{filename}")
async def get_graphml_file(
    filename: str,
    credentials: Any = Depends(access_security)
) -> Any:
    """Serve the caller's tenant-owned GraphML file."""
    user_id = get_current_user_id(credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # A user may only read their own graph. Without this any authenticated
    # account could name another account's file and read its entities.
    if filename != f"graph_{user_id}.graphml":
        raise HTTPException(status_code=404, detail="GraphML file not found")

    graph_dir = settings.graph_dir
    file_path = graph_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="GraphML file not found")

    if not filename.endswith('.graphml'):
        raise HTTPException(status_code=400, detail="File must be a GraphML file")

    return FileResponse(
        path=str(file_path),
        media_type="application/xml",
        filename=filename
    )

@router.get("/graphml/")
async def list_graphml_files(
    credentials: Any = Depends(access_security)
) -> dict[str, Any]:
    """List only the caller's tenant-owned GraphML file."""
    user_id = get_current_user_id(credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    graph_dir = settings.graph_dir
    
    if not graph_dir.exists():
        return {"files": []}
    
    # Only the caller's own graph -- graphstore writes graph_<user_id>.graphml
    graphml_files = []
    for file_path in graph_dir.glob(f"graph_{user_id}.graphml"):
        graphml_files.append({
            "filename": file_path.name,
            "size": file_path.stat().st_size,
            "modified": file_path.stat().st_mtime
        })
    
    return {"files": graphml_files}
