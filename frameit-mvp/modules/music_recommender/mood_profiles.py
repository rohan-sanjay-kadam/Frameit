"""
modules/music_recommender/mood_profiles.py
===========================================
Maps mood labels (from image_analyzer) to Spotify audio feature targets.

Spotify audio features — all 0.0–1.0 except tempo (BPM)
---------------------------------------------------------
energy          Physical intensity. High = loud, fast, dense. Low = quiet, sparse.
valence         Musical positivity. High = happy/euphoric. Low = sad/tense/dark.
danceability    Rhythmic suitability. Combines tempo stability + beat strength.
acousticness    Confidence the track is acoustic (no electronic production).
instrumentalness Likelihood of no vocals. >0.5 = instrumental.
tempo           BPM. Typical useful range 60–200. NOT normalised to 0–1.

API parameter slots per feature
--------------------------------
target_*   Soft guidance — Spotify tries to get close.
min_*/max_* Hard constraints — tracks outside these are excluded entirely.

Design rule: use hard min_/max_ only when a genre genuinely includes
slow/fast outliers that would break the mood feel (e.g. fitness, party).
Everywhere else, use only target_* for maximum variety.

Adding a new mood
-----------------
1. Add an entry to MOOD_PROFILES with all required fields.
2. Add matching prompts to MOOD_TAXONOMY in image_analyzer/mood_taxonomy.py.
No other files need changing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Audio target dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpotifyAudioTarget:
    """
    Desired audio feature values for a mood.
    Fields set to None are omitted from the Spotify API call entirely.
    """
    # Soft targets (Spotify tries to match these)
    target_energy:           Optional[float] = None
    target_valence:          Optional[float] = None
    target_danceability:     Optional[float] = None
    target_acousticness:     Optional[float] = None
    target_instrumentalness: Optional[float] = None
    target_tempo:            Optional[float] = None   # BPM

    # Hard floors/ceilings (exclude tracks outside these bounds)
    min_energy:              Optional[float] = None
    max_energy:              Optional[float] = None
    min_valence:             Optional[float] = None
    max_valence:             Optional[float] = None
    min_danceability:        Optional[float] = None
    max_danceability:        Optional[float] = None
    min_tempo:               Optional[float] = None
    max_tempo:               Optional[float] = None

    # Popularity guard — avoid one-hit wonders and pure obscurities
    min_popularity:          int = 20
    max_popularity:          int = 90

    def to_query_params(self) -> dict[str, str]:
        """
        Serialise to Spotify API query-string parameters.
        None fields are omitted so Spotify applies no constraint on that dimension.
        """
        out = {}
        for k, v in asdict(self).items():
            if v is not None:
                out[k] = str(v)
        return out

    def nudge_energy(self, delta: float) -> "SpotifyAudioTarget":
        """
        Return a copy with target_energy adjusted by delta (clamped to [0,1]).
        Used by the pipeline to fine-tune based on the image analyzer's energy signal.
        """
        import copy
        clone = copy.copy(self)
        if clone.target_energy is not None:
            clone.target_energy = max(0.0, min(1.0, clone.target_energy + delta))
        return clone


# ──────────────────────────────────────────────────────────────────────────────
# Mood profile dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MoodFeatureProfile:
    mood:         str
    display_name: str
    seed_genres:  list[str]          # up to 5 — Spotify's /recommendations limit
    audio_target: SpotifyAudioTarget
    energy_label: str = "medium"     # "low" | "medium" | "high"


# ──────────────────────────────────────────────────────────────────────────────
# Mood profiles — one per mood in MOOD_TAXONOMY
# ──────────────────────────────────────────────────────────────────────────────

MOOD_PROFILES: dict[str, MoodFeatureProfile] = {

    "travel": MoodFeatureProfile(
        mood="travel",
        display_name="Adventure & travel",
        seed_genres=["indie", "folk", "world-music", "ambient", "indie-pop"],
        energy_label="medium",
        audio_target=SpotifyAudioTarget(
            target_energy=0.60,
            target_valence=0.55,
            target_danceability=0.45,
            target_acousticness=0.55,
            target_instrumentalness=0.20,
            target_tempo=105.0,
            min_popularity=25, max_popularity=85,
        ),
    ),

    "party": MoodFeatureProfile(
        mood="party",
        display_name="Party & celebration",
        seed_genres=["pop", "dance", "edm", "hip-hop", "party"],
        energy_label="high",
        audio_target=SpotifyAudioTarget(
            target_energy=0.88,
            target_valence=0.80,
            target_danceability=0.85,
            target_acousticness=0.05,
            target_instrumentalness=0.02,
            target_tempo=128.0,
            min_energy=0.70,
            min_danceability=0.65,
            min_tempo=110.0, max_tempo=165.0,
            min_popularity=30, max_popularity=90,
        ),
    ),

    "aesthetic": MoodFeatureProfile(
        mood="aesthetic",
        display_name="Aesthetic & moody",
        seed_genres=["lo-fi", "chillwave", "indie-pop", "dream-pop", "alternative"],
        energy_label="low",
        audio_target=SpotifyAudioTarget(
            target_energy=0.30,
            target_valence=0.38,
            target_danceability=0.42,
            target_acousticness=0.45,
            target_instrumentalness=0.45,
            target_tempo=82.0,
            max_energy=0.58,
            max_tempo=105.0,
            min_popularity=15, max_popularity=75,
        ),
    ),

    "romance": MoodFeatureProfile(
        mood="romance",
        display_name="Romance & intimacy",
        seed_genres=["r-n-b", "soul", "acoustic", "jazz", "singer-songwriter"],
        energy_label="low",
        audio_target=SpotifyAudioTarget(
            target_energy=0.35,
            target_valence=0.62,
            target_danceability=0.40,
            target_acousticness=0.68,
            target_instrumentalness=0.10,
            target_tempo=76.0,
            max_energy=0.62,
            max_tempo=100.0,
            min_popularity=20, max_popularity=80,
        ),
    ),

    "food": MoodFeatureProfile(
        mood="food",
        display_name="Food & café",
        seed_genres=["jazz", "bossa-nova", "acoustic", "lounge", "easy-listening"],
        energy_label="low",
        audio_target=SpotifyAudioTarget(
            target_energy=0.38,
            target_valence=0.60,
            target_danceability=0.48,
            target_acousticness=0.60,
            target_instrumentalness=0.35,
            target_tempo=95.0,
            max_energy=0.65,
            min_popularity=15, max_popularity=75,
        ),
    ),

    "fitness": MoodFeatureProfile(
        mood="fitness",
        display_name="Fitness & sport",
        seed_genres=["hip-hop", "electronic", "rock", "metal", "work-out"],
        energy_label="high",
        audio_target=SpotifyAudioTarget(
            target_energy=0.90,
            target_valence=0.55,
            target_danceability=0.72,
            target_acousticness=0.04,
            target_instrumentalness=0.08,
            target_tempo=145.0,
            min_energy=0.70,
            min_tempo=120.0, max_tempo=185.0,
            min_popularity=20, max_popularity=88,
        ),
    ),

    "urban": MoodFeatureProfile(
        mood="urban",
        display_name="Urban & street",
        seed_genres=["hip-hop", "electronic", "indie-rock", "jazz", "trap"],
        energy_label="medium",
        audio_target=SpotifyAudioTarget(
            target_energy=0.68,
            target_valence=0.45,
            target_danceability=0.65,
            target_acousticness=0.12,
            target_instrumentalness=0.10,
            target_tempo=118.0,
            min_popularity=25, max_popularity=85,
        ),
    ),

    "family": MoodFeatureProfile(
        mood="family",
        display_name="Family & feel-good",
        seed_genres=["pop", "folk", "acoustic", "country", "indie-pop"],
        energy_label="medium",
        audio_target=SpotifyAudioTarget(
            target_energy=0.55,
            target_valence=0.75,
            target_danceability=0.55,
            target_acousticness=0.50,
            target_instrumentalness=0.05,
            target_tempo=105.0,
            min_valence=0.45,
            min_popularity=30, max_popularity=88,
        ),
    ),
}

# Fallback profile for any mood not in the map
DEFAULT_PROFILE = MoodFeatureProfile(
    mood="default",
    display_name="General",
    seed_genres=["pop", "indie-pop", "acoustic"],
    energy_label="medium",
    audio_target=SpotifyAudioTarget(
        target_energy=0.55,
        target_valence=0.60,
        target_danceability=0.55,
        min_popularity=25, max_popularity=85,
    ),
)


def get_mood_profile(mood_name: str) -> MoodFeatureProfile:
    """Return the profile for a mood name, case-insensitive. Falls back to DEFAULT_PROFILE."""
    return MOOD_PROFILES.get(mood_name.lower(), DEFAULT_PROFILE)