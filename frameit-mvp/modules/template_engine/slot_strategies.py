"""
modules/template_engine/slot_strategies.py
==========================================
Compute concrete slot geometry (x, y, width, height, rotation, z_index)
from a slot_strategy config block.

Three strategies
----------------
    scattered         — random positions within a safe_zone, optional overlap budget
    horizontal_strip  — photos side-by-side in a centred horizontal band
    grid              — even grid that fills the canvas with configurable gap/padding

Every function receives:
    strategy    dict   — the template's slot_strategy block (already $param-resolved)
    defaults    dict   — the template's slot_defaults block (already $param-resolved)
    photo_count int    — number of photos to place
    canvas_w    int    — canvas pixel width
    canvas_h    int    — canvas pixel height
    rng         random.Random — seeded RNG (for scatter / z-order randomness)

Every function returns:
    list[dict]  — one slot dict per photo, sorted by z_index ascending,
                  ready to be consumed by the renderer.

Each slot dict has these keys:
    photo_index   int   — 0-based index into the accepted_photos list
    x             int   — left edge in canvas pixels
    y             int   — top edge in canvas pixels
    width         int   — slot width in pixels
    height        int   — slot height in pixels
    rotation      float — degrees (positive = clockwise)
    z_index       int   — paint order; lower = painted first (further back)
    frame         dict  — resolved frame config
    crop          dict  — crop mode + focal_point
"""

from __future__ import annotations

import math
import random
from typing import Any


# ── Public dispatcher ─────────────────────────────────────────────────────────

def compute_slot_positions(
    strategy: dict,
    defaults: dict,
    photo_count: int,
    canvas_w: int,
    canvas_h: int,
    rng: random.Random,
) -> list[dict]:
    """
    Route to the correct layout algorithm based on strategy["type"].
    Returns slots sorted by z_index ascending (background → foreground).
    """
    s_type = strategy.get("type", "grid")

    if s_type == "scattered":
        slots = _scattered(strategy, defaults, photo_count, canvas_w, canvas_h, rng)
    elif s_type == "horizontal_strip":
        slots = _horizontal_strip(strategy, defaults, photo_count, canvas_w, canvas_h, rng)
    else:  # "grid" is the safe default
        slots = _grid(strategy, defaults, photo_count, canvas_w, canvas_h, rng)

    return sorted(slots, key=lambda s: s["z_index"])


# ── Strategy: scattered ───────────────────────────────────────────────────────

def _scattered(
    strategy: dict,
    defaults: dict,
    count: int,
    canvas_w: int,
    canvas_h: int,
    rng: random.Random,
) -> list[dict]:
    """
    Place photos at random positions within a safe_zone.

    Overlap is limited by overlap_budget (fraction of new slot area that
    may overlap an already-placed slot).  Up to 40 attempts are made per
    photo before giving up and accepting overlap.
    """
    zone = strategy.get("safe_zone", {
        "x": 60, "y": 60,
        "width":  canvas_w - 120,
        "height": canvas_h - 120,
    })
    size_cfg     = defaults.get("size", {})
    slot_w       = int(size_cfg.get("width",  320))
    slot_h       = int(size_cfg.get("height", 380))
    transform    = defaults.get("transform", {})
    overlap_bgt  = float(strategy.get("overlap_budget", 0.15))
    z_order      = strategy.get("z_order", "random")

    placed: list[tuple[int, int, int, int]] = []  # (x, y, w, h)
    slots: list[dict] = []

    for i in range(count):
        # Try to find a non-overlapping position
        for _ in range(40):
            x = rng.randint(zone["x"], max(zone["x"], zone["x"] + zone["width"]  - slot_w))
            y = rng.randint(zone["y"], max(zone["y"], zone["y"] + zone["height"] - slot_h))
            if _acceptable(x, y, slot_w, slot_h, placed, overlap_bgt):
                break

        placed.append((x, y, slot_w, slot_h))

        rotation = _get_rotation(transform, rng)
        z = rng.randint(1, count * 2) if z_order == "random" else i

        slots.append(_make_slot(i, x, y, slot_w, slot_h, rotation, z, defaults))

    return slots


# ── Strategy: horizontal_strip ────────────────────────────────────────────────

def _horizontal_strip(
    strategy: dict,
    defaults: dict,
    count: int,
    canvas_w: int,
    canvas_h: int,
    rng: random.Random,
) -> list[dict]:
    """
    Lay photos side-by-side in a horizontal strip centred on strip_y.
    Slot width is calculated to fill the available horizontal space evenly.
    """
    strip_y  = int(strategy.get("strip_y",      canvas_h // 2 - 190))
    strip_h  = int(strategy.get("strip_height", 380))
    gap      = int(strategy.get("gap",          10))
    pad_x    = int(strategy.get("padding_x",    40))
    transform = defaults.get("transform", {})

    available = canvas_w - pad_x * 2 - gap * max(count - 1, 0)
    slot_w    = max(1, available // count)

    slots = []
    for i in range(count):
        x        = pad_x + i * (slot_w + gap)
        rotation = _get_rotation(transform, rng)
        slots.append(_make_slot(i, x, strip_y, slot_w, strip_h, rotation, i, defaults))

    return slots


# ── Strategy: grid ────────────────────────────────────────────────────────────

def _grid(
    strategy: dict,
    defaults: dict,
    count: int,
    canvas_w: int,
    canvas_h: int,
    rng: random.Random,
) -> list[dict]:
    """
    Divide the canvas into an even grid.
    Columns = ceil(sqrt(count)); rows = ceil(count / cols).
    """
    gap     = int(strategy.get("gap",     6))
    padding = int(strategy.get("padding", 0))
    transform = defaults.get("transform", {})

    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)

    avail_w  = canvas_w - padding * 2 - gap * max(cols - 1, 0)
    avail_h  = canvas_h - padding * 2 - gap * max(rows - 1, 0)
    cell_w   = max(1, avail_w // cols)
    cell_h   = max(1, avail_h // rows)

    slots = []
    for i in range(count):
        col = i % cols
        row = i // cols
        x   = padding + col * (cell_w + gap)
        y   = padding + row * (cell_h + gap)
        rotation = _get_rotation(transform, rng)
        slots.append(_make_slot(i, x, y, cell_w, cell_h, rotation, i, defaults))

    return slots


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_slot(
    photo_index: int,
    x: int, y: int,
    w: int, h: int,
    rotation: float,
    z_index: int,
    defaults: dict,
) -> dict:
    return {
        "photo_index": photo_index,
        "x":           x,
        "y":           y,
        "width":       w,
        "height":      h,
        "rotation":    rotation,
        "z_index":     z_index,
        "frame":       defaults.get("frame", {"type": "none"}),
        "crop":        defaults.get("crop",  {"mode": "cover", "focal_point": "center"}),
        "filter":      defaults.get("filter", None),
    }


def _get_rotation(transform: dict, rng: random.Random) -> float:
    """Extract or sample a rotation value from the (already-resolved) transform dict."""
    r = transform.get("rotation", 0)
    # After $param resolution, rotation is already a concrete float.
    # Guard against any residual dict just in case.
    if isinstance(r, dict):
        return 0.0
    return float(r)


def _acceptable(
    x: int, y: int, w: int, h: int,
    placed: list[tuple[int, int, int, int]],
    budget: float,
) -> bool:
    """
    True if the proposed bbox overlaps each already-placed bbox by no more
    than budget * (new slot area).
    """
    new_area = w * h
    if new_area == 0:
        return True
    for px, py, pw, ph in placed:
        ix = max(0, min(x + w, px + pw) - max(x, px))
        iy = max(0, min(y + h, py + ph) - max(y, py))
        if (ix * iy) / new_area > budget:
            return False
    return True