"""
modules/collage_renderer/frames.py
===================================
Frame builders.  Each function takes a cropped photo (RGBA) and wraps it
in a decorative frame, returning a new RGBA image that the renderer pastes
onto the canvas.

Frames available
----------------
    polaroid    White card with configurable borders; optional drop shadow.
    filmstrip   Dark borders top + bottom with procedurally drawn sprocket holes.
    border      Simple uniform border of any colour.
    none        Pass-through — returns the image unchanged.

All builders return RGBA images.  Shadow / border expansion means the
returned image is larger than the input photo.

Dispatch
--------
Use build_frame(photo, frame_cfg) as the single entry point.  It reads
frame_cfg["type"] and routes to the correct builder with the correct kwargs.
The renderer never calls individual builders directly.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    r, g, b = _hex_to_rgb(hex_color)
    return r, g, b, alpha


# ──────────────────────────────────────────────────────────────────────────────
# Polaroid
# ──────────────────────────────────────────────────────────────────────────────

def apply_polaroid_frame(
    photo:          Image.Image,
    border_top:     int   = 12,
    border_sides:   int   = 12,
    border_bottom:  int   = 56,
    frame_color:    str   = "#FFFFFF",
    shadow_blur:    int   = 12,
    shadow_opacity: float = 0.18,
) -> Image.Image:
    """
    Wrap a photo in a polaroid-style white card with an optional drop shadow.

    The returned image is larger than the input:
        width  = photo_w + border_sides * 2
        height = photo_h + border_top + border_bottom + shadow_expansion

    Shadow is achieved by blurring a semi-transparent copy of the card
    shape below the card layer, so it never clips the photo.

    Args:
        photo:          Source RGBA image.
        border_top:     Top border thickness (px).
        border_sides:   Left and right border thickness (px).
        border_bottom:  Bottom border thickness (px) — the "polaroid gap".
        frame_color:    Card colour as "#RRGGBB".
        shadow_blur:    Gaussian blur radius for the shadow.
        shadow_opacity: Shadow alpha factor (0 = none, 1 = full black).

    Returns:
        RGBA image including the card and shadow.
    """
    photo = photo.convert("RGBA")
    pw, ph = photo.size

    card_w = pw + border_sides * 2
    card_h = ph + border_top + border_bottom
    expand = shadow_blur * 2   # extra canvas for shadow bleed

    # Shadow layer: blurred dark rect behind the card
    out_w = card_w + expand
    out_h = card_h + expand
    result = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))

    shadow_alpha = int(255 * shadow_opacity)
    shadow_rect  = Image.new("RGBA", (card_w, card_h), (30, 20, 10, shadow_alpha))
    shadow_layer = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    # Offset shadow slightly down-right for depth
    shadow_offset = max(2, shadow_blur // 3)
    shadow_layer.paste(shadow_rect, (expand // 2 + shadow_offset,
                                     expand // 2 + shadow_offset))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_blur // 2))
    result = Image.alpha_composite(result, shadow_layer)

    # Card layer: white card at the standard position
    fr, fg, fb = _hex_to_rgb(frame_color)
    card = Image.new("RGBA", (card_w, card_h), (fr, fg, fb, 255))
    card.paste(photo, (border_sides, border_top))

    card_x = expand // 2
    card_y = expand // 2
    result.paste(card, (card_x, card_y), mask=card)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Filmstrip
# ──────────────────────────────────────────────────────────────────────────────

def apply_filmstrip_frame(
    photo:          Image.Image,
    sprocket_size:  int = 18,
    border_width:   int = 6,
    border_color:   str = "#0D0D0D",
    sprocket_color: str = "#111111",
) -> Image.Image:
    """
    Add filmstrip borders (top and bottom) with drawn sprocket holes.

    Strip height = sprocket_size * 2 + border_width * 2.
    Sprocket holes are ellipses spaced ~1.6× their width apart.

    Returns:
        RGBA image with film borders above and below the photo.
    """
    photo   = photo.convert("RGBA")
    pw, ph  = photo.size
    strip_h = sprocket_size * 2 + border_width * 2
    total_h = ph + strip_h * 2

    br, bg, bb = _hex_to_rgb(border_color)
    out = Image.new("RGBA", (pw, total_h), (br, bg, bb, 255))
    draw = ImageDraw.Draw(out)

    sr, sg, sb = _hex_to_rgb(sprocket_color)
    hole_w    = sprocket_size
    hole_h    = int(sprocket_size * 0.65)
    spacing   = int(sprocket_size * 1.6)
    hole_cy_t = strip_h // 2
    hole_cy_b = total_h - strip_h // 2

    x = spacing // 2
    while x + hole_w < pw:
        draw.ellipse(
            [x, hole_cy_t - hole_h // 2,
             x + hole_w, hole_cy_t + hole_h // 2],
            fill=(sr, sg, sb, 255),
        )
        draw.ellipse(
            [x, hole_cy_b - hole_h // 2,
             x + hole_w, hole_cy_b + hole_h // 2],
            fill=(sr, sg, sb, 255),
        )
        x += spacing

    out.paste(photo, (0, strip_h))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Border
# ──────────────────────────────────────────────────────────────────────────────

def apply_border_frame(
    photo:        Image.Image,
    border_width: int = 8,
    border_color: str = "#FFFFFF",
) -> Image.Image:
    """
    Add a simple uniform border around the photo.

    Returns:
        RGBA image (photo_w + 2*border, photo_h + 2*border).
    """
    photo = photo.convert("RGBA")
    pw, ph = photo.size
    br, bg, bb = _hex_to_rgb(border_color)
    out = Image.new("RGBA", (pw + border_width * 2, ph + border_width * 2),
                    (br, bg, bb, 255))
    out.paste(photo, (border_width, border_width))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def build_frame(photo: Image.Image, frame_cfg: dict) -> Image.Image:
    """
    Apply the frame described by `frame_cfg` to `photo`.

    frame_cfg is the already-resolved slot["frame"] dict from the template.
    The "type" key selects the builder; remaining keys are passed as kwargs.

    Unknown frame types fall through to "none" (no frame).
    """
    frame_type = frame_cfg.get("type", "none")

    if frame_type == "polaroid":
        return apply_polaroid_frame(
            photo,
            border_top    = int(frame_cfg.get("border_top",    12)),
            border_sides  = int(frame_cfg.get("border_sides",  12)),
            border_bottom = int(frame_cfg.get("border_bottom", 56)),
            frame_color   = _resolve_color(frame_cfg.get("color", "#FFFFFF")),
            shadow_blur   = int(frame_cfg.get("shadow", {}).get("blur",    12)),
            shadow_opacity= float(frame_cfg.get("shadow", {}).get("opacity", 0.18)),
        )

    if frame_type == "filmstrip":
        return apply_filmstrip_frame(
            photo,
            sprocket_size = int(frame_cfg.get("sprocket_size",   18)),
            border_width  = int(frame_cfg.get("border_width",     6)),
            border_color  = frame_cfg.get("border_color",  "#0D0D0D"),
            sprocket_color= frame_cfg.get("sprocket_color","#111111"),
        )

    if frame_type == "border":
        return apply_border_frame(
            photo,
            border_width = int(frame_cfg.get("border_width", 8)),
            border_color = frame_cfg.get("color", "#FFFFFF"),
        )

    # "none" or unknown — return photo unchanged (as RGBA)
    return photo.convert("RGBA")


def _resolve_color(value) -> str:
    """Accept a string or a list [r, g, b] and always return a hex string."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return "#{:02X}{:02X}{:02X}".format(*[int(c) for c in value])
    return "#FFFFFF"