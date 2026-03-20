"""
modules/collage_renderer/compositor.py
=======================================
Low-level compositing primitives used by every other renderer file.

Functions
---------
    paste_rotated(canvas, layer, x, y, rotation, anchor, opacity)
        Paste an RGBA layer onto canvas with rotation + opacity.
        Handles the expand-then-recentre math so the anchor point stays
        where you specified even after the bounding box grows.

    blend_layer(base, top, mode, opacity)
        Composite `top` onto `base` using a Photoshop-style blend mode.
        Modes: "normal", "multiply", "screen", "overlay".
        All images must be RGBA.

    crop_to_fit(img, target_w, target_h, focal_point, face_box)
        Scale + crop to exactly fill (target_w × target_h) — CSS cover.

Why compositor.py is a separate file
-------------------------------------
Both the renderer and the decorations module need paste_rotated().
Both the renderer and the grain module need blend_layer().
Putting them here avoids circular imports between those files.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter


# ──────────────────────────────────────────────────────────────────────────────
# paste_rotated
# ──────────────────────────────────────────────────────────────────────────────

def paste_rotated(
    canvas:   Image.Image,
    layer:    Image.Image,       # RGBA source layer
    x:        int,
    y:        int,
    rotation: float = 0.0,       # degrees, positive = clockwise
    anchor:   str   = "top_left",  # "top_left" | "center"
    opacity:  float = 1.0,
) -> None:
    """
    Paste `layer` onto `canvas` at position (x, y) with rotation.

    The rotation is performed around the layer's own centre, then the
    expanded bounding box is repositioned so the specified anchor point
    lands on (x, y).

    Without the bounding-box correction a 45° rotated image drifts ~30%
    of its dimension away from the intended position.

    Args:
        canvas:   RGBA canvas to composite onto (modified in place).
        layer:    RGBA source image.
        x, y:     Anchor position in canvas pixels.
        rotation: Clockwise degrees. 0 = no rotation.
        anchor:   "top_left" positions the pre-rotation top-left corner
                  at (x, y). "center" positions the centre at (x, y).
        opacity:  0.0 = transparent, 1.0 = fully opaque.
    """
    if opacity < 1.0:
        r, g, b, a = layer.split()
        a = a.point(lambda p: int(p * opacity))
        layer = Image.merge("RGBA", (r, g, b, a))

    if rotation != 0.0:
        # expand=True gives us the full bounding box after rotation
        rotated = layer.rotate(-rotation, expand=True, resample=Image.BICUBIC)
    else:
        rotated = layer

    rw, rh = rotated.size
    lw, lh = layer.size

    if anchor == "center":
        paste_x = x - rw // 2
        paste_y = y - rh // 2
    else:
        # top_left: position refers to the pre-rotation corner
        # The bounding box expands equally on all sides after rotation,
        # so we shift back by half the expansion.
        paste_x = x - (rw - lw) // 2
        paste_y = y - (rh - lh) // 2

    # Clip to canvas bounds so PIL doesn't raise on out-of-range paste
    cx = max(0, paste_x)
    cy = max(0, paste_y)
    ox = cx - paste_x    # offset into rotated image if left/top was clipped
    oy = cy - paste_y
    cw = min(canvas.width  - cx, rw - ox)
    ch = min(canvas.height - cy, rh - oy)

    if cw <= 0 or ch <= 0:
        return   # entirely off-canvas

    region = rotated.crop((ox, oy, ox + cw, oy + ch))
    canvas.paste(region, (cx, cy), mask=region)


# ──────────────────────────────────────────────────────────────────────────────
# blend_layer
# ──────────────────────────────────────────────────────────────────────────────

def blend_layer(
    base:    Image.Image,
    top:     Image.Image,
    mode:    str   = "normal",
    opacity: float = 1.0,
) -> Image.Image:
    """
    Composite `top` onto `base` using a Photoshop-style blend mode.

    Supported modes
    ---------------
    normal    Standard alpha compositing (Porter-Duff over).
    multiply  base × top  — darkens, used for grain / texture overlays.
    screen    1 - (1-base)(1-top)  — lightens.
    overlay   Combines multiply (dark areas) and screen (light areas).

    Args:
        base:    RGBA background image.
        top:     RGBA layer to composite.
        mode:    Blend mode name (case-sensitive).
        opacity: Overall opacity of the `top` layer (applied before blending).

    Returns:
        New RGBA image — neither input is modified.
    """
    if top.size != base.size:
        top = top.resize(base.size, Image.LANCZOS)

    if opacity < 1.0:
        r, g, b, a = top.split()
        a = a.point(lambda p: int(p * opacity))
        top = Image.merge("RGBA", (r, g, b, a))

    if mode == "normal":
        return Image.alpha_composite(base, top)

    # NumPy path for mathematical blend modes
    base_arr = np.array(base, dtype=np.float32) / 255.0   # (H, W, 4)
    top_arr  = np.array(top,  dtype=np.float32) / 255.0

    top_alpha = top_arr[:, :, 3:4]   # (H, W, 1) broadcast-ready
    b_rgb = base_arr[:, :, :3]
    t_rgb = top_arr[:,  :, :3]

    if mode == "multiply":
        blended = b_rgb * t_rgb
    elif mode == "screen":
        blended = 1.0 - (1.0 - b_rgb) * (1.0 - t_rgb)
    elif mode == "overlay":
        mask    = b_rgb < 0.5
        blended = np.where(
            mask,
            2.0 * b_rgb * t_rgb,
            1.0 - 2.0 * (1.0 - b_rgb) * (1.0 - t_rgb),
        )
    else:
        # Unrecognised mode — fall back to normal
        return Image.alpha_composite(base, top)

    # Composite blended RGB over the base using the top layer's alpha
    result_rgb = b_rgb * (1.0 - top_alpha) + blended * top_alpha
    result_arr = np.clip(
        np.dstack([result_rgb, base_arr[:, :, 3:]]), 0, 1
    )
    return Image.fromarray((result_arr * 255).astype(np.uint8), "RGBA")


# ──────────────────────────────────────────────────────────────────────────────
# crop_to_fit
# ──────────────────────────────────────────────────────────────────────────────

def crop_to_fit(
    img:         Image.Image,
    target_w:    int,
    target_h:    int,
    focal_point: str                             = "center",
    face_box:    Optional[tuple[int,int,int,int]] = None,
) -> Image.Image:
    """
    Scale then crop `img` so it exactly fills (target_w × target_h).
    This is CSS object-fit: cover behaviour.

    focal_point controls where the crop window is centred:
        "center"  — centre of the scaled image  (default)
        "face"    — centre of `face_box` if provided, else falls back to center
        "top"     — top-centre of the scaled image (keeps faces in portrait shots)
        "bottom"  — bottom-centre

    Args:
        img:         Source PIL Image (any mode).
        target_w:    Target width in pixels.
        target_h:    Target height in pixels.
        focal_point: See above.
        face_box:    (x, y, w, h) bounding box in the *original* image's pixel
                     space, as returned by the image analyser.  Only used when
                     focal_point == "face".

    Returns:
        New RGBA image of exactly (target_w, target_h).
    """
    img = img.convert("RGBA")
    src_w, src_h = img.size

    # Scale so the image covers the target (larger dimension wins)
    scale = max(target_w / src_w, target_h / src_h)
    new_w = math.ceil(src_w * scale)
    new_h = math.ceil(src_h * scale)
    img   = img.resize((new_w, new_h), Image.LANCZOS)

    # Determine crop centre
    if focal_point == "face" and face_box is not None:
        fx, fy, fw, fh = face_box
        cx = int(fx * scale + fw * scale / 2)
        cy = int(fy * scale + fh * scale / 2)
    elif focal_point == "top":
        cx = new_w // 2
        cy = target_h // 2
    elif focal_point == "bottom":
        cx = new_w // 2
        cy = new_h - target_h // 2
    else:   # "center" or "auto"
        cx = new_w // 2
        cy = new_h // 2

    left = max(0, min(cx - target_w // 2, new_w - target_w))
    top  = max(0, min(cy - target_h // 2, new_h - target_h))

    return img.crop((left, top, left + target_w, top + target_h))