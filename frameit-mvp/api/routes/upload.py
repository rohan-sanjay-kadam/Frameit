"""
api/routes/upload.py
====================
POST /api/v1/upload

Accepts a multipart/form-data batch of image files, validates MIME types
and sizes, saves them to uploads/{session_id}/ on local disk, and returns
the list of saved filenames (photo_ids) together with a session_id.

The session_id is generated here and used by the generate endpoint to
locate the files. No database — just a UUID-named directory.

Limits enforced at this layer (before the pipeline's deeper validation):
  • max 10 files per request  (MAX_PHOTOS from config)
  • max 20 MB per file
  • JPEG and PNG only
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.schemas import UploadResponse
from config import get_config

router = APIRouter()

_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
_MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload photos",
    description="Upload 1–10 JPEG/PNG photos. Returns session_id and photo_ids for use with /generate.",
)
async def upload_photos(
    files: list[UploadFile] = File(..., description="Image files to upload"),
) -> UploadResponse:
    cfg = get_config()

    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one file is required.",
        )

    if len(files) > cfg.max_photos:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Maximum {cfg.max_photos} files per upload. Received {len(files)}.",
        )

    session_id = str(uuid.uuid4())
    session_dir = Path(cfg.upload_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    photo_ids: list[str] = []
    warnings: list[str] = []
    rejected = 0

    for idx, file in enumerate(files):
        # MIME type check
        if file.content_type not in _ALLOWED_TYPES:
            warnings.append(
                f"File '{file.filename}' skipped: unsupported type '{file.content_type}'. "
                f"Only JPEG and PNG are accepted."
            )
            rejected += 1
            continue

        # Read content (enforces memory limit implicitly via FastAPI)
        content = await file.read()

        if len(content) > _MAX_FILE_BYTES:
            warnings.append(
                f"File '{file.filename}' skipped: "
                f"{len(content) / 1024 / 1024:.1f} MB exceeds the 20 MB limit."
            )
            rejected += 1
            continue

        # Derive extension from content type for reliable filenames
        ext = "jpg" if file.content_type in ("image/jpeg", "image/jpg") else "png"
        filename = f"photo_{idx:02d}.{ext}"
        dest = session_dir / filename

        dest.write_bytes(content)
        photo_ids.append(filename)

    if not photo_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No valid image files were provided. " + " ".join(warnings),
        )

    return UploadResponse(
        session_id=session_id,
        photo_ids=photo_ids,
        accepted=len(photo_ids),
        rejected=rejected,
        warnings=warnings,
    )
