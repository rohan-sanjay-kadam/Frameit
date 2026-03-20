"""
modules/music_recommender/music_recommender.py
================================================
Maps image analysis mood results to Spotify track recommendations.

Public API
----------
    get_recommendations(mood, limit, market, client_id, client_secret)
        -> RecommendationResult

    get_recommendations_from_analysis(analysis_result, ...)
        -> RecommendationResult

    format_result(result) -> str   (human-readable summary for logging)

How mood → music works
-----------------------
1. Look up the MoodFeatureProfile for the mood (mood_profiles.py).
2. Optionally nudge target_energy using the image analyzer's energy_label.
3. Serialise the SpotifyAudioTarget to query parameters.
4. Call GET /recommendations with seed_genres + audio targets.
5. Parse the response into RecommendationResult.

The audio features serve as a "search filter" inside the seed genre space.
Without targets, Spotify returns whatever is most popular in those genres.
With targets, it returns tracks that match the mood's energy, valence, and
tempo profile — dramatically improving relevance.

Soft targets (target_*) vs hard floors/ceilings (min_*/max_*)
--------------------------------------------------------------
Most moods use only soft targets.  Hard constraints are applied only where
a genre genuinely includes extreme outliers that would break the mood:
    - fitness: min_energy=0.70, min_tempo=120 (excludes warm-down tracks)
    - party:   min_danceability=0.65, min_tempo=110 (excludes slow pop ballads)
Hard constraints shrink the candidate pool — use them sparingly.

Energy nudge from image analysis
---------------------------------
The image analyzer returns energy as "low" | "medium" | "high".
We apply a small nudge (-0.10 / 0.0 / +0.10) to target_energy so that
a "travel" album of skydiving shots gets slightly more energetic music
than a "travel" album of quiet mountain hikes — same mood, different feel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from modules.music_recommender.mood_profiles  import (
    MoodFeatureProfile, SpotifyAudioTarget,
    get_mood_profile, MOOD_PROFILES,
)
from modules.music_recommender.spotify_client import (
    SpotifyTrack, get_access_token, fetch_recommendations,
)


# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RecommendationResult:
    mood:         str
    tracks:       list[SpotifyTrack]
    seed_genres:  list[str]
    audio_target: SpotifyAudioTarget
    total_found:  int
    query_params: dict     # kept for debugging and test assertions


# ──────────────────────────────────────────────────────────────────────────────
# Energy nudge map
# ──────────────────────────────────────────────────────────────────────────────

_ENERGY_NUDGE: dict[str, float] = {
    "low":    -0.10,
    "medium":  0.00,
    "high":   +0.10,
}


# ──────────────────────────────────────────────────────────────────────────────
# Public functions
# ──────────────────────────────────────────────────────────────────────────────

def get_recommendations(
    mood:          str,
    limit:         int           = 10,
    market:        str           = "US",
    client_id:     Optional[str] = None,
    client_secret: Optional[str] = None,
    energy_label:  Optional[str] = None,
    extra_params:  Optional[dict] = None,
) -> RecommendationResult:
    """
    Fetch Spotify track recommendations for a given mood.

    Steps:
        1. Look up the MoodFeatureProfile (falls back to DEFAULT_PROFILE).
        2. Optionally nudge target_energy using energy_label.
        3. Get a bearer token (cached between calls).
        4. Build query params: seed_genres + audio targets + market + limit.
        5. Call /recommendations and return parsed tracks.

    Args:
        mood:          Mood name (must match a key in MOOD_PROFILES, case-insensitive).
        limit:         Number of tracks to return (1–100; clamped automatically).
        market:        ISO 3166-1 alpha-2 country code for track availability.
        client_id:     Spotify client ID.  Falls back to SPOTIFY_CLIENT_ID env var.
        client_secret: Spotify client secret.  Falls back to SPOTIFY_CLIENT_SECRET env var.
        energy_label:  "low" | "medium" | "high" from image analysis.
                       If provided, target_energy is nudged by ±0.10.
        extra_params:  Additional raw query params to merge in (e.g. seed_artists).

    Returns:
        RecommendationResult with tracks and diagnostic metadata.

    Raises:
        ValueError:  Credentials missing.
        RuntimeError: Spotify API returned an error.
    """
    # Resolve credentials from env if not provided
    cid = client_id     or os.environ.get("SPOTIFY_CLIENT_ID",     "")
    sec = client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET", "")

    # Look up profile and optionally nudge energy
    profile = get_mood_profile(mood)
    target  = profile.audio_target
    if energy_label and energy_label in _ENERGY_NUDGE:
        target = target.nudge_energy(_ENERGY_NUDGE[energy_label])

    # Clamp limit to Spotify's allowed range
    limit = max(1, min(100, limit))

    # Build query parameters
    params: dict[str, str] = {
        "seed_genres": ",".join(profile.seed_genres[:5]),
        "limit":       str(limit),
        "market":      market,
        **target.to_query_params(),
    }
    if extra_params:
        params.update({k: str(v) for k, v in extra_params.items()})

    # Fetch token and call API
    token  = get_access_token(cid, sec)
    tracks = fetch_recommendations(token, params)

    return RecommendationResult(
        mood         = mood,
        tracks       = tracks,
        seed_genres  = profile.seed_genres,
        audio_target = target,
        total_found  = len(tracks),
        query_params = params,
    )


def get_recommendations_from_analysis(
    analysis_result,               # AnalysisResult from image_analyzer
    limit:         int           = 10,
    market:        str           = "US",
    client_id:     Optional[str] = None,
    client_secret: Optional[str] = None,
) -> RecommendationResult:
    """
    Full pipeline: AnalysisResult → Spotify recommendations.

    Extracts mood name and energy_label from the analysis result,
    then delegates to get_recommendations() with the energy nudge applied.

    Args:
        analysis_result: AnalysisResult from image_analyzer.analyze() or
                         a dict with 'primary_mood' and 'energy' keys
                         (as returned by aggregate_results()).
    """
    # Support both AnalysisResult objects and plain aggregate dicts
    if hasattr(analysis_result, "mood"):
        # AnalysisResult object from analyze()
        mood_name    = analysis_result.mood.primary.name
        energy_label = analysis_result.mood.primary.energy
    else:
        # Dict from aggregate_results()
        mood_name    = analysis_result.get("primary_mood", "aesthetic")
        energy_label = analysis_result.get("energy", "medium")

    return get_recommendations(
        mood          = mood_name,
        limit         = limit,
        market        = market,
        client_id     = client_id,
        client_secret = client_secret,
        energy_label  = energy_label,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helper (logging / CLI output)
# ──────────────────────────────────────────────────────────────────────────────

def format_result(result: RecommendationResult) -> str:
    """
    Return a human-readable summary of a RecommendationResult.
    Useful for logging and CLI debugging.
    """
    at   = result.audio_target
    lines = [
        f"Mood:    {result.mood}",
        f"Genres:  {', '.join(result.seed_genres)}",
        f"Tracks:  {result.total_found}",
        "",
        "Audio targets:",
    ]

    targets = [
        ("energy",       at.target_energy),
        ("valence",      at.target_valence),
        ("danceability", at.target_danceability),
        ("tempo (BPM)",  at.target_tempo),
        ("acousticness", at.target_acousticness),
    ]
    for name, val in targets:
        if val is not None:
            bar = ("█" * int(val * 20) + "░" * (20 - int(val * 20))) if val <= 1.0 else ""
            lines.append(f"  {name:<20} {val:.2f}  {bar}")

    if at.min_energy is not None:
        lines.append(f"  {'min_energy':<20} {at.min_energy:.2f}  (hard floor)")
    if at.min_tempo is not None:
        lines.append(f"  {'min_tempo':<20} {at.min_tempo:.0f} BPM  (hard floor)")

    lines.append("")
    lines.append("Tracks:")
    for i, t in enumerate(result.tracks, 1):
        dur     = f"{t.duration_ms // 60000}:{(t.duration_ms % 60000) // 1000:02d}"
        preview = "▶" if t.preview_url else " "
        lines.append(f"  {i:2}. {preview} {t.name} — {t.artist}")
        lines.append(f"       {t.album}  [{dur}]  pop={t.popularity}")

    return "\n".join(lines)