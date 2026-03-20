"""
api/routes/download.py
======================
GET /api/v1/download/{filename}

Forces a browser download (Content-Disposition: attachment) for a rendered
collage file. The /output static mount in app.py serves files inline (for
<img> preview); this endpoint forces the Save-As dialog.

Security:
    • filename is validated to be alphanumeric + safe chars only.
    • Path traversal (../) is blocked by stripping the filename to its stem.
    • Only files inside cfg.output_dir are ever served.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from config import get_config

router = APIRouter()

_SAFE_FILENAME = re.compile(r"^[\w\-\.]+$")


@router.get(
    "/download/{filename}",
    summary="Download collage",
    description="Force-download a rendered collage PNG by filename.",
    responses={
        200: {"content": {"image/png": {}}, "description": "PNG file download"},
        404: {"description": "File not found"},
        422: {"description": "Invalid filename"},
    },
)
async def download_collage(filename: str) -> FileResponse:
    cfg = get_config()

    # Reject filenames with path traversal characters or unusual chars
    if not _SAFE_FILENAME.match(filename):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid filename. Only alphanumeric characters, hyphens, underscores, and dots are allowed.",
        )

    # Constrain to output directory — never serve arbitrary paths
    file_path = Path(cfg.output_dir) / Path(filename).name

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found.",
        )

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Requested path is not a file.",
        )

    media_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
