"""
modules/collage_renderer/grain.py
===================================
Procedural film-grain overlay using NumPy.

How it works
------------
Gaussian noise is sampled for each pixel, creating a spatially independent
per-pixel intensity fluctuation that mimics silver-halide grain.

The noise is rendered as a neutral-grey RGBA layer and blended onto the
canvas in "overlay" mode (via blend_layer in compositor.py).  Overlay mode
amplifies mid-tones while leaving deep shadows and bright highlights
relatively unaffected — which is how real film grain behaves.

intensity controls the standard deviation of the Gaussian (0.0–1.0 scale
where 1.0 = full 255 std dev).  Typical values:
    0.00   — no grain (clean digital look)
    0.02   — very subtle (minimal_grid template)
    0.03   — light grain (warm_vintage)
    0.05   — visible grain (faded_film, noir_bw)
    0.08+  — heavy grain (stylistic choice)

The RNG is seeded so the same seed always produces the same grain pattern,
keeping collage generation deterministic end-to-end.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image


def generate_grain_overlay(
    width:     int,
    height:    int,
    intensity: float        = 0.03,
    rng:       random.Random | None = None,
) -> Image.Image:
    """
    Generate a procedural film-grain overlay as an RGBA image.

    The returned image has the same dimensions as the canvas.  It should
    be blended onto the canvas using blend_layer(canvas, grain, "overlay")
    at a low opacity (0.4–0.7).

    Args:
        width, height: Canvas dimensions.
        intensity:     Noise std dev on 0–1 scale.  0 = no grain.
        rng:           Seeded random.Random for determinism.
                       If None, a random seed is used.

    Returns:
        RGBA Image.  RGB channels = neutral grey + noise.
        Alpha = fixed low value derived from intensity (so blending
        at opacity 1.0 already gives a subtle result).
    """
    if intensity <= 0.0:
        # Return a fully transparent layer — caller can skip blending
        return Image.new("RGBA", (width, height), (128, 128, 128, 0))

    # Seed the NumPy RNG from the Python rng for reproducibility
    np_seed = rng.randint(0, 2 ** 31) if rng else None
    np_rng  = np.random.default_rng(np_seed)

    # Gaussian noise centred on 128 (neutral grey)
    std   = intensity * 255.0
    noise = np_rng.normal(128.0, std, (height, width)).astype(np.float32)
    noise = np.clip(noise, 0, 255).astype(np.uint8)

    # Stack to 3-channel grey
    grey_rgb = np.stack([noise, noise, noise], axis=-1)  # (H, W, 3)

    # Alpha: low fixed value so blending at opacity 1 is already subtle
    # (scales with intensity so stronger grain stays subtle at low values)
    alpha_val = int(min(255, intensity * 255 * 2.5))
    alpha_arr = np.full((height, width), alpha_val, dtype=np.uint8)

    rgba = np.dstack([grey_rgb, alpha_arr])   # (H, W, 4)
    return Image.fromarray(rgba, "RGBA")