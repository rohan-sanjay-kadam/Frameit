"""
config.py — Frameit MVP
=======================
All runtime configuration in one place. Values come from environment variables
or .env file. Every field has a safe local default so the app starts with
zero configuration.

Usage anywhere in the codebase:
    from config import get_config
    cfg = get_config()
    print(cfg.output_dir)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    upload_dir: str = "uploads"
    output_dir: str = "output"
    templates_dir: str = "collage_templates"
    assets_dir: str = "assets"

    # ── Pipeline ───────────────────────────────────────────────────────────
    # "fallback" works with zero ML deps (colour-histogram embeddings).
    # Set to "open_clip" or "transformers" once you have a GPU / ML env.
    clip_backend: str = "fallback"

    max_photos: int = 10
    min_photos: int = 1
    blur_threshold: float = 80.0
    duplicate_threshold: float = 0.97
    jpeg_quality: int = 95

    # ── Spotify (optional) ─────────────────────────────────────────────────
    # Leave blank to skip music recommendations gracefully.
    spotify_client_id: str = ""
    spotify_client_secret: str = ""
    music_track_count: int = 10
    music_market: str = "US"

    # ── Server ─────────────────────────────────────────────────────────────
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
            output_dir=os.getenv("OUTPUT_DIR", "output"),
            templates_dir=os.getenv("TEMPLATES_DIR", "collage_templates"),
            assets_dir=os.getenv("ASSETS_DIR", "assets"),
            clip_backend=os.getenv("CLIP_BACKEND", "fallback"),
            max_photos=int(os.getenv("MAX_PHOTOS", "10")),
            min_photos=int(os.getenv("MIN_PHOTOS", "1")),
            blur_threshold=float(os.getenv("BLUR_THRESHOLD", "80.0")),
            duplicate_threshold=float(os.getenv("DUPLICATE_THRESHOLD", "0.97")),
            jpeg_quality=int(os.getenv("JPEG_QUALITY", "95")),
            spotify_client_id=os.getenv("SPOTIFY_CLIENT_ID", ""),
            spotify_client_secret=os.getenv("SPOTIFY_CLIENT_SECRET", ""),
            music_track_count=int(os.getenv("MUSIC_TRACK_COUNT", "10")),
            music_market=os.getenv("MUSIC_MARKET", "US"),
            cors_origins=os.getenv(
                "CORS_ORIGINS", "http://localhost:5173"
            ).split(","),
        )

    @property
    def spotify_configured(self) -> bool:
        return bool(self.spotify_client_id and self.spotify_client_secret)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the singleton Config instance (cached after first call)."""
    return Config.from_env()