"""
api/routes/generate.py
======================
POST /api/v1/generate

Synchronous pipeline invocation. No Celery, no job queue.
The request blocks while the pipeline runs (~1–4 seconds locally),
then returns the full result in the response body.

Flow:
    1. Validate the request (session_id exists, photo files are on disk)
    2. Resolve the absolute paths of the requested photo_ids
    3. Run CollagePipeline.run() — the same orchestrator used in production,
       just called directly rather than via a Celery task
    4. Serialise PipelineResult → GenerateResponse and return

Error handling:
    • Session directory not found → 404
    • Photo file not found → 422
    • Pipeline FAILED status → 500 with error list
    • Unhandled exception → 500
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from api.schemas import GenerateRequest, GenerateResponse, TrackSchema
from config import get_config
from pipeline.pipeline import CollagePipeline, PipelineConfig
from pipeline.pipeline_types import PipelineStatus

log = logging.getLogger(__name__)

router = APIRouter()

# Module-level pipeline singleton — loaded once, reused across requests.
# The CLIP backend and template list are initialised here on first call.
_pipeline: CollagePipeline | None = None


def _get_pipeline() -> CollagePipeline:
    global _pipeline
    if _pipeline is None:
        cfg = get_config()
        pipeline_cfg = PipelineConfig(
            templates_dir=cfg.templates_dir,
            output_dir=cfg.output_dir,
            assets_dir=cfg.assets_dir,
            spotify_client_id=cfg.spotify_client_id or None,
            spotify_client_secret=cfg.spotify_client_secret or None,
            music_track_count=cfg.music_track_count,
            music_market=cfg.music_market,
            max_photos=cfg.max_photos,
            min_photos=cfg.min_photos,
            blur_threshold=cfg.blur_threshold,
            duplicate_threshold=cfg.duplicate_threshold,
            enable_clip=cfg.clip_backend != "fallback",
            skip_music=not cfg.spotify_configured,
            jpeg_quality=cfg.jpeg_quality,
        )
        _pipeline = CollagePipeline(pipeline_cfg)
        log.info("CollagePipeline initialised (backend=%s)", cfg.clip_backend)
    return _pipeline


@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate collage",
    description=(
        "Run the full pipeline: validate photos → analyse mood → select template "
        "→ render collage → recommend music. Returns the complete result synchronously."
    ),
)
async def generate_collage(req: GenerateRequest) -> GenerateResponse:
    cfg = get_config()

    # ── Resolve file paths ────────────────────────────────────────────────
    session_dir = Path(cfg.upload_dir) / req.session_id
    if not session_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{req.session_id}' not found. Upload photos first.",
        )

    photo_paths: list[Path] = []
    missing: list[str] = []
    for photo_id in req.photo_ids:
        p = session_dir / photo_id
        if not p.exists():
            missing.append(photo_id)
        else:
            photo_paths.append(p)

    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Photos not found in session: {missing}",
        )

    # ── Run pipeline ──────────────────────────────────────────────────────
    try:
        pipeline = _get_pipeline()
        result = pipeline.run(
            photo_paths=[str(p) for p in photo_paths],
            fmt=req.format,
            seed=req.seed,
            user_style_tags=[req.vibe] if req.vibe else [],
        )
    except Exception as exc:
        log.exception("Pipeline raised an unhandled exception")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Render failed: {exc}",
        ) from exc

    # ── Hard failure ──────────────────────────────────────────────────────
    if result.status == PipelineStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Pipeline failed.",
                "errors": result.errors,
                "warnings": result.warnings,
            },
        )

    # ── Derive collage URL from local path ────────────────────────────────
    collage_path = Path(result.collage_path)
    collage_filename = collage_path.name
    collage_url = f"/output/{collage_filename}"

    # ── Extract analysis summary fields (with safe defaults) ──────────────
    summary = result.analysis_summary or {}

    # ── Serialise music tracks ────────────────────────────────────────────
    tracks = [
        TrackSchema(
            id=t.get("id", ""),
            name=t.get("name", ""),
            artist=t.get("artist", ""),
            album=t.get("album", ""),
            preview_url=t.get("preview_url"),
            spotify_url=t.get("spotify_url", ""),
            popularity=t.get("popularity", 0),
        )
        for t in result.music_tracks
    ]

    return GenerateResponse(
        status=result.status.value,
        collage_filename=collage_filename,
        collage_url=collage_url,
        seed=result.seed,
        format=req.format,
        detected_mood=summary.get("primary_mood", "unknown"),
        detected_orientation=summary.get("dominant_orientation", "unknown"),
        energy=summary.get("energy", "medium"),
        scene_tags=summary.get("scene_tags", []),
        palette_hex=summary.get("palette_hex", []),
        selected_template=result.selected_template or "unknown",
        music_tracks=tracks,
        accepted_photos=result.accepted_photos,
        rejected_photos=result.rejected_photos,
        warnings=result.warnings,
        timing_ms=result.timing_ms,
    )
