"""
modules/image_analyzer/mood_taxonomy.py
=======================================
Single source of truth for:
    MOOD_TAXONOMY  — maps mood labels -> CLIP prompts, style_tags, music_genres, energy
    SCENE_LABELS   — location vocabulary for zero-shot scene tagging

Adding a new mood: add one entry to MOOD_TAXONOMY with the four required keys.
No other file needs to change.
"""

from __future__ import annotations

MOOD_TAXONOMY: dict[str, dict] = {
    "travel": {
        "prompts": [
            "a scenic travel photo of a beautiful landscape",
            "a travel adventure photo with mountains or ocean",
            "a tourist exploring a foreign city or landmark",
            "a backpacker hiking through nature on vacation",
        ],
        "style_tags":    ["wanderlust", "outdoor", "editorial"],
        "music_genres":  ["indie", "folk", "world-music", "ambient"],
        "energy":        "medium",
    },
    "party": {
        "prompts": [
            "people celebrating at a party with drinks and music",
            "friends dancing and having fun at a nightclub",
            "a birthday party with balloons and confetti",
            "a group of people laughing together at a celebration",
        ],
        "style_tags":    ["vibrant", "colorful", "casual"],
        "music_genres":  ["pop", "dance", "hip-hop", "edm"],
        "energy":        "high",
    },
    "aesthetic": {
        "prompts": [
            "a moody aesthetic photograph with soft dreamy lighting",
            "a minimalist artistic photo with clean composition",
            "a dreamy pastel aesthetic lifestyle photograph",
            "a carefully composed flat lay with props and soft shadows",
        ],
        "style_tags":    ["clean", "editorial", "modern"],
        "music_genres":  ["indie-pop", "lo-fi", "chillwave", "dream-pop"],
        "energy":        "low",
    },
    "romance": {
        "prompts": [
            "a romantic couple photo at golden hour sunset",
            "a wedding or engagement photo with soft bokeh background",
            "two people sharing a tender intimate moment together",
            "a candlelit dinner for two in a cozy restaurant",
        ],
        "style_tags":    ["warm", "vintage", "soft"],
        "music_genres":  ["r-n-b", "soul", "jazz", "acoustic"],
        "energy":        "low",
    },
    "food": {
        "prompts": [
            "a beautifully plated gourmet dish in a restaurant",
            "a flat lay of food ingredients on a marble surface",
            "a close-up macro shot of dessert coffee or brunch",
            "an avocado toast smoothie bowl breakfast spread",
        ],
        "style_tags":    ["warm", "clean", "editorial"],
        "music_genres":  ["jazz", "bossa-nova", "acoustic", "lounge"],
        "energy":        "low",
    },
    "fitness": {
        "prompts": [
            "an athlete working out at the gym with weights",
            "a person running outdoors at sunrise on a trail",
            "yoga or meditation in nature at sunrise",
            "sports action photo with motion and energy",
        ],
        "style_tags":    ["high-contrast", "bold", "dynamic"],
        "music_genres":  ["hip-hop", "electronic", "rock", "workout"],
        "energy":        "high",
    },
    "urban": {
        "prompts": [
            "a street photography shot in a busy city at night",
            "an architectural photo of modern city buildings",
            "neon lights and urban nightlife in a metropolis",
            "graffiti wall or street art in an urban neighbourhood",
        ],
        "style_tags":    ["cinematic", "dark", "editorial"],
        "music_genres":  ["hip-hop", "electronic", "indie-rock", "jazz"],
        "energy":        "medium",
    },
    "family": {
        "prompts": [
            "a family portrait with smiling children and parents",
            "kids playing together in a backyard on a sunny day",
            "a cozy indoor family moment on the sofa",
            "grandparents with grandchildren laughing together",
        ],
        "style_tags":    ["warm", "casual", "bright"],
        "music_genres":  ["pop", "folk", "acoustic", "country"],
        "energy":        "medium",
    },
}

SCENE_LABELS: list[str] = [
    "beach", "forest", "mountain", "city street", "restaurant",
    "bedroom", "gym", "concert", "park", "airport", "cafe",
    "rooftop", "desert", "snow", "sunset",
]