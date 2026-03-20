"""
modules/collage_renderer/decorations.py
=========================================
Renders decoration layers (tape, stickers, stamps, text labels) and
returns them as positioned Layer objects ready for the main compositor.

Each decoration in the template's "decorations" array is processed by
render_decoration().  The function:
    1. Rolls the probability dice — skips if the roll fails.
    2. Loads the asset image from the assets registry.
    3. Computes position (absolute, or anchored to a slot).
    4. Reads rotation and opacity from the transform config.
    5. Returns a Layer dataclass.

Layer objects are collected by the renderer, sorted by z_index, and
pasted onto the canvas via paste_rotated() from compositor.py.

Adding a new decoration type
-----------------------------
1. Add an entry to _DEC_HANDLERS with your type name as the key.
2. Implement _render_<type>(dec_cfg, slots, assets_dir, ctx_rng) -> Layer | None.
3. No other files need to change.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


# ── Layer dataclass (shared with renderer.py) ─────────────────────────────────

@dataclass
class Layer:
    image:      Image.Image
    x:          int
    y:          int
    rotation:   float = 0.0
    opacity:    float = 1.0
    z_index:    int   = 0


# ── Asset loader ──────────────────────────────────────────────────────────────

def load_asset(key: str, assets_dir: str | Path) -> Optional[Image.Image]:
    """
    Load a decoration asset by registry key from the assets directory.
    Searches: assets_dir/{key}.png, assets_dir/decorations/{key}.png,
              assets_dir/textures/{key}.png.
    Returns None (graceful degradation) if the file is not found.
    """
    base = Path(assets_dir)
    candidates = [
        base / f"{key}.png",
        base / "decorations" / f"{key}.png",
        base / "textures"    / f"{key}.png",
        base / f"{key}.jpg",
    ]
    for p in candidates:
        if p.exists():
            try:
                return Image.open(p).convert("RGBA")
            except Exception:
                return None
    return None


def load_font(font_name: str, size: int, assets_dir: str | Path) -> ImageFont.ImageFont:
    """Load a font by name from assets/fonts/, falling back to PIL default."""
    font_path = Path(assets_dir) / "fonts" / f"{font_name}.ttf"
    if font_path.exists():
        try:
            return ImageFont.truetype(str(font_path), size)
        except Exception:
            pass
    # Try system fonts
    for system_path in [
        Path(f"/usr/share/fonts/truetype/{font_name}.ttf"),
        Path(f"/usr/share/fonts/{font_name}.ttf"),
    ]:
        if system_path.exists():
            try:
                return ImageFont.truetype(str(system_path), size)
            except Exception:
                pass
    return ImageFont.load_default()


# ── Individual decoration renderers ──────────────────────────────────────────

def _render_tape(
    dec_cfg:    dict,
    slots:      list[dict],
    assets_dir: Path,
    rng:        random.Random,
) -> Optional[Layer]:
    asset_key = dec_cfg.get("asset", "tape_clear")
    img       = load_asset(asset_key, assets_dir)
    if img is None:
        # Synthesise a simple semi-transparent rectangle as fallback
        img = _make_fallback_tape()

    transform = dec_cfg.get("transform", {})
    rotation  = float(transform.get("rotation", 0.0))
    opacity   = float(transform.get("opacity",  0.85))

    # Anchor strategy
    anchor_strategy = dec_cfg.get("anchor_strategy", "")
    if anchor_strategy == "random_slot_top_edge" and slots:
        slot = rng.choice(slots)
        x    = slot["x"] + slot["width"] // 2 - img.width // 2
        y    = slot["y"] - img.height // 2
    else:
        pos = dec_cfg.get("position", {})
        x   = int(pos.get("x", 0))
        y   = int(pos.get("y", 0))

    return Layer(image=img, x=x, y=y, rotation=rotation,
                 opacity=opacity, z_index=dec_cfg.get("z_index", 900))


def _render_sticker(
    dec_cfg:    dict,
    slots:      list[dict],
    assets_dir: Path,
    rng:        random.Random,
) -> Optional[Layer]:
    asset_key = dec_cfg.get("asset", "sticker_star")
    img       = load_asset(asset_key, assets_dir)
    if img is None:
        return None

    transform = dec_cfg.get("transform", {})
    rotation  = float(transform.get("rotation", 0.0))
    opacity   = float(transform.get("opacity",  1.0))
    pos       = dec_cfg.get("position", {})
    x         = int(pos.get("x", 0))
    y         = int(pos.get("y", 0))

    return Layer(image=img, x=x, y=y, rotation=rotation,
                 opacity=opacity, z_index=dec_cfg.get("z_index", 910))


def _render_stamp(
    dec_cfg:    dict,
    slots:      list[dict],
    assets_dir: Path,
    rng:        random.Random,
) -> Optional[Layer]:
    asset_key = dec_cfg.get("asset", "stamp_retro_01")
    img       = load_asset(asset_key, assets_dir)
    if img is None:
        return None

    transform = dec_cfg.get("transform", {})
    rotation  = float(transform.get("rotation", 0.0))
    opacity   = float(transform.get("opacity",  0.65))
    pos       = dec_cfg.get("position", {})
    x         = int(pos.get("x", 0))
    y         = int(pos.get("y", 0))

    return Layer(image=img, x=x, y=y, rotation=rotation,
                 opacity=opacity, z_index=dec_cfg.get("z_index", 920))


def _render_text_label(
    dec_cfg:    dict,
    slots:      list[dict],
    assets_dir: Path,
    rng:        random.Random,
) -> Optional[Layer]:
    """Render a text string onto a transparent RGBA surface."""
    content = dec_cfg.get("content", "")
    if not content and dec_cfg.get("content_strategy") == "sequential_numbers":
        content = "  ".join(str(i + 1) for i in range(len(slots)))
    if not content:
        return None

    font_name = dec_cfg.get("font", "Caveat-Regular")
    font_size = int(dec_cfg.get("size", 28))
    color     = dec_cfg.get("color", "#FFFFFF")
    pos       = dec_cfg.get("position", {})
    width     = int(pos.get("width", 600))

    font = load_font(font_name, font_size, assets_dir)
    # Approximate text dimensions
    text_height = font_size + 10

    img  = Image.new("RGBA", (width, text_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), content, font=font, fill=color)

    transform = dec_cfg.get("transform", {})
    rotation  = float(transform.get("rotation", 0.0))
    opacity   = float(transform.get("opacity",  1.0))

    return Layer(
        image=img,
        x=int(pos.get("x", 40)),
        y=int(pos.get("y", 0)),
        rotation=rotation,
        opacity=opacity,
        z_index=dec_cfg.get("z_index", 930),
    )


# ── Handler registry ──────────────────────────────────────────────────────────

_DEC_HANDLERS = {
    "tape":       _render_tape,
    "sticker":    _render_sticker,
    "stamp":      _render_stamp,
    "text_label": _render_text_label,
    "date_stamp": _render_stamp,    # reuse stamp logic for date stamps
}


# ── Public dispatcher ─────────────────────────────────────────────────────────

def render_decoration(
    dec_cfg:    dict,
    slots:      list[dict],
    assets_dir: str | Path,
    rng:        random.Random,
) -> Optional[Layer]:
    """
    Build a single decoration Layer from its config dict.

    Returns None if:
        - The probability roll fails.
        - The required asset file is not found.
        - The decoration type is unknown.

    Args:
        dec_cfg:    Resolved decoration config (all $params already evaluated).
        slots:      List of resolved slot dicts (used by anchor strategies).
        assets_dir: Path to the assets directory.
        rng:        Seeded Random instance (for probability + position choices).
    """
    probability = float(dec_cfg.get("probability", 1.0))
    if rng.random() > probability:
        return None

    dec_type = dec_cfg.get("type", "")
    handler  = _DEC_HANDLERS.get(dec_type)
    if handler is None:
        return None

    try:
        return handler(dec_cfg, slots, Path(assets_dir), rng)
    except Exception:
        # Decoration failure is non-fatal — skip it silently
        return None


# ── Fallback helpers ──────────────────────────────────────────────────────────

def _make_fallback_tape() -> Image.Image:
    """
    Synthesise a simple tape strip when the asset file is not found.
    Returns a 120×28 semi-transparent rectangle.
    """
    img  = Image.new("RGBA", (120, 28), (255, 255, 240, 140))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 119, 27], outline=(200, 200, 180, 80), width=1)
    return img