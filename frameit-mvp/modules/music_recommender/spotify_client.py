"""
modules/music_recommender/spotify_client.py
=============================================
Thin HTTP wrapper around the Spotify Web API.

Responsibilities
----------------
    - Client Credentials OAuth (get and cache bearer tokens)
    - GET /recommendations
    - GET /audio-features (used for debugging / verification)
    - Parse raw API JSON into SpotifyTrack dataclasses

No business logic here — mood mapping, genre selection, and audio feature
targets all live in mood_profiles.py and music_recommender.py.

Token caching
-------------
Tokens are valid for 3600 seconds.  We cache the token module-level and
reuse it until 60 seconds before expiry.  This means a server process that
handles many requests never re-authenticates unnecessarily.

The cache is checked BEFORE credential validation so that callers that
pre-seed the cache (e.g. tests) don't need to provide credentials.

Uses only Python stdlib for HTTP (urllib) — no requests / httpx dependency.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_SPOTIFY_BASE = "https://api.spotify.com/v1"
_TOKEN_URL    = "https://accounts.spotify.com/api/token"
_TIMEOUT_SEC  = 15


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpotifyTrack:
    id:          str
    name:        str
    artist:      str
    album:       str
    preview_url: Optional[str]
    spotify_url: str
    duration_ms: int
    popularity:  int


# ──────────────────────────────────────────────────────────────────────────────
# Token cache
# ──────────────────────────────────────────────────────────────────────────────

class _TokenCache:
    def __init__(self) -> None:
        self.token:      str   = ""
        self.expires_at: float = 0.0

    def is_valid(self) -> bool:
        return bool(self.token) and time.time() < self.expires_at - 60


_TOKEN_CACHE = _TokenCache()


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_access_token(
    client_id:     str,
    client_secret: str,
) -> str:
    """
    Obtain a Spotify bearer token using the Client Credentials flow.

    The Client Credentials flow is for server-to-server requests that do NOT
    act on behalf of a specific user — perfect for recommendation lookups.

    Flow:
        POST https://accounts.spotify.com/api/token
        Authorization: Basic base64(client_id:client_secret)
        Body: grant_type=client_credentials

    Token is cached until 60 s before expiry to avoid hammering the auth endpoint.
    The cache is checked first so tests can inject a mock token without credentials.

    Args:
        client_id:     Spotify app client ID.
        client_secret: Spotify app client secret.

    Returns:
        Bearer token string.

    Raises:
        ValueError:  client_id or client_secret is empty.
        RuntimeError: Spotify returned a non-200 response.
    """
    # Cache check first — allows tests to pre-seed the cache
    if _TOKEN_CACHE.is_valid():
        return _TOKEN_CACHE.token

    if not client_id or not client_secret:
        raise ValueError(
            "Spotify credentials are missing. "
            "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env."
        )

    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    body        = urlencode({"grant_type": "client_credentials"}).encode()

    req = Request(
        _TOKEN_URL,
        data    = body,
        headers = {
            "Authorization": f"Basic {credentials}",
            "Content-Type":  "application/x-www-form-urlencoded",
        },
        method  = "POST",
    )

    try:
        with urlopen(req, timeout=_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read())
    except HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"Spotify auth failed (HTTP {exc.code}): {body_text}"
        ) from exc

    _TOKEN_CACHE.token      = data["access_token"]
    _TOKEN_CACHE.expires_at = time.time() + data.get("expires_in", 3600)
    return _TOKEN_CACHE.token


def fetch_recommendations(
    token:       str,
    params:      dict[str, str],
) -> list[SpotifyTrack]:
    """
    Call GET /recommendations and return parsed SpotifyTrack objects.

    Args:
        token:  Bearer token from get_access_token().
        params: Query parameters dict (seed_genres, limit, audio targets, etc.)

    Returns:
        List of SpotifyTrack.  Empty list on empty response.

    Raises:
        RuntimeError: Spotify returned a non-200 response.
    """
    data = _spotify_get("/recommendations", params, token)
    return [_parse_track(t) for t in data.get("tracks", [])]


def fetch_audio_features(
    token:     str,
    track_ids: list[str],
) -> list[dict]:
    """
    Call GET /audio-features for a batch of track IDs.
    Useful for verifying that returned tracks match the requested targets.

    Returns:
        List of feature dicts (may contain None entries for unavailable tracks).
    """
    if not track_ids:
        return []
    data = _spotify_get("/audio-features", {"ids": ",".join(track_ids[:100])}, token)
    return [f for f in data.get("audio_features", []) if f is not None]


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _spotify_get(endpoint: str, params: dict, token: str) -> dict:
    url = f"{_SPOTIFY_BASE}{endpoint}?{urlencode(params)}"
    req = Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urlopen(req, timeout=_TIMEOUT_SEC) as resp:
            return json.loads(resp.read())
    except HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"Spotify API error (HTTP {exc.code}) at {endpoint}: {body_text}"
        ) from exc


def _parse_track(raw: dict) -> SpotifyTrack:
    artists = ", ".join(a["name"] for a in raw.get("artists", []))
    return SpotifyTrack(
        id          = raw.get("id", ""),
        name        = raw.get("name", ""),
        artist      = artists,
        album       = raw.get("album", {}).get("name", ""),
        preview_url = raw.get("preview_url"),
        spotify_url = raw.get("external_urls", {}).get("spotify", ""),
        duration_ms = raw.get("duration_ms", 0),
        popularity  = raw.get("popularity", 0),
    )