"""
api/schemas.py
==============
All Pydantic models for request validation and response serialisation.
Imported by every route — the single source of shape truth for the API.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────────────────────────────────────
# Upload
# ──────────────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    session_id: str = Field(..., description="UUID identifying this upload session")
    photo_ids: list[str] = Field(..., description="Ordered list of saved filenames")
    accepted: int = Field(..., description="Number of files successfully saved")
    rejected: int = Field(..., description="Number of files that failed (too large, wrong type)")
    warnings: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Generate
# ──────────────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    session_id: str = Field(..., description="Session ID returned by /upload")
    photo_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Ordered photo filenames to include in the collage",
    )
    format: str = Field(
        default="post",
        description="Output canvas: 'post' (1080×1080) or 'story' (1080×1920)",
    )
    vibe: Optional[str] = Field(
        default=None,
        description=(
            "Optional mood override. One of: travel, party, aesthetic, romance, "
            "food, fitness, urban, family. If absent, mood is auto-detected."
        ),
    )
    seed: Optional[int] = Field(
        default=None,
        description="RNG seed for reproducible renders. Omit for a random result.",
    )

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v not in ("post", "story"):
            raise ValueError("format must be 'post' or 'story'")
        return v

    @field_validator("vibe")
    @classmethod
    def validate_vibe(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        valid = {"travel", "party", "aesthetic", "romance", "food", "fitness", "urban", "family"}
        if v not in valid:
            raise ValueError(f"vibe must be one of: {', '.join(sorted(valid))}")
        return v


class TrackSchema(BaseModel):
    id: str
    name: str
    artist: str
    album: str
    preview_url: Optional[str] = None
    spotify_url: str
    popularity: int


class PhotoValidationSchema(BaseModel):
    path: str
    accepted: bool
    reject_reason: Optional[str] = None
    warning: Optional[str] = None


class GenerateResponse(BaseModel):
    status: str = Field(..., description="'success' or 'partial' (completed with warnings)")

    # Collage output
    collage_filename: str = Field(..., description="Filename in output/ directory")
    collage_url: str = Field(..., description="URL path to preview the collage image")
    seed: int = Field(..., description="RNG seed used — save this to regenerate identically")
    format: str

    # Analysis summary
    detected_mood: str
    detected_orientation: str
    energy: str
    scene_tags: list[str] = Field(default_factory=list)
    palette_hex: list[str] = Field(default_factory=list)
    selected_template: str

    # Music
    music_tracks: list[TrackSchema] = Field(default_factory=list)

    # Diagnostics
    accepted_photos: list[str] = Field(default_factory=list)
    rejected_photos: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timing_ms: dict[str, int] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Error responses
# ──────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
