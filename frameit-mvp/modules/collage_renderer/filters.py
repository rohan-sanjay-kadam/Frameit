"""
modules/collage_renderer/filters.py
=====================================
Colour grade presets applied to the fully composited canvas after all
slots and decorations have been laid down.

Available presets
-----------------
    none                  Pass-through.
    warm_vintage          Slightly warm, faded, lower contrast.
    faded_film            Milky lifted shadows, desaturated, soft.
    noir_bw               Full desaturation with boosted contrast.
    cinematic_teal_orange Hollywood split-tone: teal shadows, orange highlights.
    high_contrast         Punchy contrast, slightly saturated.

Each preset is implemented as a sequence of Pillow ImageEnhance operations
plus optional NumPy array manipulation for effects that Pillow can't do
(warmth shift, shadow lift, split-tone).

All functions operate on RGBA images:
    — RGB adjustments are applied to the RGB channels only.
    — The alpha channel is preserved unchanged.
    — The input image is not modified; a new image is returned.

Usage
-----
    from modules.collage_renderer.filters import apply_filter_preset

    canvas = apply_filter_preset(canvas, "warm_vintage")
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance


# ── Preset definitions ────────────────────────────────────────────────────────
#
# Each preset is a dict of float parameters consumed by _apply_params().
# Add a new preset by adding an entry here — no other code changes needed.

PRESETS: dict[str, dict] = {
    "none": {},

    "warm_vintage": {
        "brightness": 1.05,
        "contrast":   0.92,
        "saturation": 0.82,
        "warmth":     20,        # +R / -B shift (0-255 scale)
        "fade":       0.06,      # lift shadows toward white
    },

    "faded_film": {
        "brightness": 1.08,
        "contrast":   0.78,
        "saturation": 0.65,
        "fade":       0.14,
    },

    "noir_bw": {
        "saturation": 0.0,
        "contrast":   1.18,
        "brightness": 0.94,
    },

    "cinematic_teal_orange": {
        "contrast":       1.05,
        "teal_orange":    True,  # handled separately in _apply_params
    },

    "high_contrast": {
        "contrast":   1.32,
        "brightness": 0.96,
        "saturation": 1.08,
    },
}


# ── Public API ────────────────────────────────────────────────────────────────

def apply_filter_preset(img: Image.Image, preset_name: str) -> Image.Image:
    """
    Apply a named colour-grade preset to an RGBA image.

    Args:
        img:         RGBA source image.
        preset_name: Key from PRESETS (e.g. "warm_vintage").
                     Unknown names are treated as "none".

    Returns:
        New RGBA image with the grade applied.
    """
    params = PRESETS.get(preset_name, {})
    if not params:
        return img
    return _apply_params(img, params)


def list_presets() -> list[str]:
    """Return all available preset names."""
    return list(PRESETS.keys())


# ── Implementation ────────────────────────────────────────────────────────────

def _apply_params(img: Image.Image, params: dict) -> Image.Image:
    """Apply a parameter dict to an RGBA image. Returns a new RGBA image."""
    # Separate alpha before any RGB manipulation
    img  = img.convert("RGBA")
    alpha = img.split()[3]
    rgb   = img.convert("RGB")

    # ── Pillow ImageEnhance operations (in fixed order for predictability) ──
    if "brightness" in params:
        rgb = ImageEnhance.Brightness(rgb).enhance(params["brightness"])

    if "contrast" in params:
        rgb = ImageEnhance.Contrast(rgb).enhance(params["contrast"])

    if "saturation" in params:
        rgb = ImageEnhance.Color(rgb).enhance(params["saturation"])

    if "sharpness" in params:
        rgb = ImageEnhance.Sharpness(rgb).enhance(params["sharpness"])

    # ── NumPy operations for effects not available in Pillow ───────────────
    needs_numpy = any(k in params for k in ("warmth", "fade", "teal_orange"))
    if needs_numpy:
        arr = np.array(rgb, dtype=np.float32)  # (H, W, 3)

        if "warmth" in params:
            # Shift red channel up, blue channel down
            w = float(params["warmth"])
            arr[:, :, 0] = np.clip(arr[:, :, 0] + w * 0.55, 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] - w * 0.30, 0, 255)

        if "fade" in params:
            # Lift all channels toward white (milky shadow effect)
            f   = float(params["fade"])
            arr = arr * (1.0 - f) + 255.0 * f

        if params.get("teal_orange"):
            # Split-tone: shadows → teal, highlights → orange
            lum = arr.mean(axis=2, keepdims=True) / 255.0  # (H, W, 1)
            # Orange highlights: +R, -B proportional to brightness
            arr[:, :, 0] = np.clip(arr[:, :, 0] + 22.0 * lum[:, :, 0], 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] - 18.0 * lum[:, :, 0], 0, 255)
            # Teal shadows: -R, +B, +G proportional to darkness
            dark = 1.0 - lum[:, :, 0]
            arr[:, :, 0] = np.clip(arr[:, :, 0] - 12.0 * dark, 0, 255)
            arr[:, :, 1] = np.clip(arr[:, :, 1] + 8.0  * dark, 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] + 20.0 * dark, 0, 255)

        rgb = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    # Re-attach original alpha
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out