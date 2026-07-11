"""
generate_assets.py
==================
Generates all required PNG assets programmatically using Pillow + NumPy.

Run once before starting the server:
    python generate_assets.py

This creates every asset file that the collage renderer references.
No internet access or external downloads required.

Assets generated
----------------
frames/
    polaroid_border.png    — white card overlay with drop shadow
    film_sprockets.png     — filmstrip edge tile (tiled horizontally)
    instant_white.png      — bright white instant-camera border
    instant_cream.png      — cream-tinted instant-camera border

textures/
    paper_grain_01.png     — fine paper grain (low frequency)
    paper_grain_02.png     — heavier craft paper grain
    noise_fine.png         — digital noise overlay

decorations/
    tape_clear.png         — transparent cellotape strip
    tape_washi_floral.png  — semi-opaque washi tape with printed flowers
    stamp_retro_01.png     — retro circular date stamp

fonts/
    README.txt             — instructions for downloading free fonts

Note on fonts
-------------
Fonts cannot be generated programmatically. The renderer falls back to
PIL's built-in bitmap font when TTF files are absent — captions will render
but look basic. To enable nicer captions, download:
    Caveat-Regular.ttf         https://fonts.google.com/specimen/Caveat
    CourierPrime-Regular.ttf   https://fonts.google.com/specimen/Courier+Prime
and place them in assets/fonts/.
"""

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)
(ASSETS / "frames").mkdir(exist_ok=True)
(ASSETS / "textures").mkdir(exist_ok=True)
(ASSETS / "decorations").mkdir(exist_ok=True)
(ASSETS / "fonts").mkdir(exist_ok=True)


def save(img: Image.Image, rel_path: str) -> None:
    out = ASSETS / rel_path
    img.save(str(out), "PNG")
    print(f"  [WRITE] assets/{rel_path}  {img.size}")


# ──────────────────────────────────────────────────────────────────────────────
# FRAMES
# ──────────────────────────────────────────────────────────────────────────────

def gen_polaroid_border():
    """
    A 380×460 white card with a soft drop shadow around it.
    The photo slot (320×380) sits centred with 12px side/top, 56px bottom.
    """
    W, H = 380, 460
    # Shadow layer
    shadow = Image.new("RGBA", (W + 40, H + 40), (0, 0, 0, 0))
    shadow_rect = Image.new("RGBA", (W, H), (30, 20, 10, 60))
    shadow.paste(shadow_rect, (22, 22))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=8))

    # White card on top
    card = Image.new("RGBA", (W + 40, H + 40), (0, 0, 0, 0))
    white = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    # Subtle edge highlight
    draw = ImageDraw.Draw(white)
    draw.rectangle([0, 0, W-1, H-1], outline=(220, 220, 215, 180), width=1)
    card.paste(white, (18, 18))

    result = Image.alpha_composite(shadow, card)
    save(result, "frames/polaroid_border.png")


def gen_film_sprockets():
    """
    A 1080×60 filmstrip edge tile.  Tiled horizontally to match any canvas width.
    Dark background with evenly spaced rounded sprocket holes.
    """
    W, H = 1080, 60
    img  = Image.new("RGBA", (W, H), (18, 14, 10, 255))
    draw = ImageDraw.Draw(img)

    hole_w, hole_h = 22, 16
    spacing = 36
    cy = H // 2
    x  = spacing // 2

    while x + hole_w < W:
        draw.rounded_rectangle(
            [x, cy - hole_h//2, x + hole_w, cy + hole_h//2],
            radius=4, fill=(8, 6, 4, 255),
        )
        x += spacing

    save(img, "frames/film_sprockets.png")


def gen_instant_border(color: tuple, filename: str):
    """
    400×480 instant-camera border.  Pure flat colour with a slight inner shadow
    to give depth around the photo area.
    """
    W, H = 400, 480
    r, g, b = color
    img  = Image.new("RGBA", (W, H), (r, g, b, 255))
    draw = ImageDraw.Draw(img)

    # Inner shadow — thin gradient-like darkening around the photo opening
    photo_x1, photo_y1 = 14, 14
    photo_x2, photo_y2 = W - 14, H - 66
    for i in range(4):
        alpha = 18 - i * 4
        draw.rectangle(
            [photo_x1 + i, photo_y1 + i, photo_x2 - i, photo_y2 - i],
            outline=(0, 0, 0, alpha), width=1,
        )

    # Outer edge highlight
    draw.rectangle([0, 0, W-1, H-1], outline=(255, 255, 255, 40), width=1)
    save(img, f"frames/{filename}")


# ──────────────────────────────────────────────────────────────────────────────
# TEXTURES
# ──────────────────────────────────────────────────────────────────────────────

def gen_paper_grain(filename: str, intensity: float, scale: float = 1.0):
    """
    Seamlessly tileable paper grain texture (512×512 RGBA).

    intensity: noise std dev (0–1).
    scale:     spatial frequency; >1 = coarser grain.
    """
    W, H = 512, 512
    rng  = np.random.default_rng(42)

    # Base: uniform warm off-white
    base = np.full((H, W, 3), [245, 240, 232], dtype=np.float32)

    # Layered noise for organic feel
    noise = np.zeros((H, W), dtype=np.float32)
    for freq, amp in [(1.0, 0.6), (2.0, 0.25), (4.0, 0.15)]:
        n = rng.normal(0, intensity * amp * 255, (H, W)).astype(np.float32)
        noise += n

    noise = np.clip(noise * scale, -60, 60)

    grain_rgb = np.clip(base + noise[:, :, np.newaxis], 0, 255).astype(np.uint8)

    # Alpha: low so blending at moderate opacity is subtle
    alpha = np.full((H, W), int(intensity * 200), dtype=np.uint8)
    rgba  = np.dstack([grain_rgb, alpha])

    save(Image.fromarray(rgba, "RGBA"), f"textures/{filename}")


def gen_noise_fine():
    """512×512 fine digital noise overlay — neutral grey with low alpha."""
    W, H = 512, 512
    rng  = np.random.default_rng(99)
    noise = rng.normal(128, 18, (H, W)).astype(np.float32)
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    grey  = np.stack([noise, noise, noise], axis=-1)
    alpha = np.full((H, W), 30, dtype=np.uint8)
    rgba  = np.dstack([grey, alpha])
    save(Image.fromarray(rgba, "RGBA"), "textures/noise_fine.png")


# ──────────────────────────────────────────────────────────────────────────────
# DECORATIONS
# ──────────────────────────────────────────────────────────────────────────────

def gen_tape_clear():
    """
    140×32 semi-transparent cellotape strip.
    Slight warm tint, visible edge lines, random micro-scratches for realism.
    """
    W, H = 140, 32
    rng  = np.random.default_rng(7)

    # Base: very slightly warm, semi-transparent
    arr = np.full((H, W, 4), [255, 252, 240, 110], dtype=np.uint8)

    # Edge lines (slightly darker top/bottom)
    arr[0:2,  :, :3] = [200, 195, 180]
    arr[0:2,  :,  3] = 140
    arr[-2:,  :, :3] = [200, 195, 180]
    arr[-2:,  :,  3] = 140

    # Micro-scratch lines (horizontal, random vertical position)
    for _ in range(4):
        y   = rng.integers(3, H - 3)
        x1  = rng.integers(0, W // 3)
        x2  = rng.integers(2 * W // 3, W)
        arr[y, x1:x2, :3] = [230, 225, 210]
        arr[y, x1:x2,  3] = 80

    save(Image.fromarray(arr, "RGBA"), "decorations/tape_clear.png")


def gen_tape_washi_floral():
    """
    160×34 washi tape with printed floral pattern.
    Warm beige base with small repeating flower motifs.
    """
    W, H = 160, 34
    img  = Image.new("RGBA", (W, H), (240, 210, 185, 200))
    draw = ImageDraw.Draw(img)

    # Repeating flower motifs
    petal_color  = (200, 120, 100, 180)
    center_color = (240, 180,  80, 200)
    spacing = 28
    offsets = [(spacing * i + 14, H // 2) for i in range(W // spacing + 1)]

    for cx, cy in offsets:
        r_petal = 6
        for angle in range(0, 360, 60):
            rad  = math.radians(angle)
            px   = int(cx + r_petal * math.cos(rad))
            py   = int(cy + r_petal * math.sin(rad))
            draw.ellipse([px-3, py-3, px+3, py+3], fill=petal_color)
        draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=center_color)

    # Small leaf marks between flowers
    leaf_color = (120, 160, 90, 160)
    for i in range(W // spacing):
        lx = spacing * i + spacing // 2 + 14
        draw.ellipse([lx-2, H//2 - 8, lx+2, H//2 - 4], fill=leaf_color)

    # Edge lines
    draw.line([(0, 1), (W, 1)], fill=(180, 150, 130, 160), width=1)
    draw.line([(0, H-2), (W, H-2)], fill=(180, 150, 130, 160), width=1)

    save(img, "decorations/tape_washi_floral.png")


def gen_stamp_retro():
    """
    120×120 retro circular rubber-stamp overlay.
    Distressed ink look with circular border + inner text area.
    """
    W, H = 120, 120
    img  = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    rng  = np.random.default_rng(13)
    cx, cy, r = W // 2, H // 2, 50

    # Outer circle
    stamp_color = (160, 50, 30, 200)
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=stamp_color, width=4)

    # Inner circle
    draw.ellipse([cx-r+12, cy-r+12, cx+r-12, cy+r-12], outline=stamp_color, width=2)

    # Horizontal lines in centre (simulates text block)
    for y_off in [-10, -3, 4, 11]:
        x1 = cx - 22
        x2 = cx + 22
        draw.line([(x1, cy+y_off), (x2, cy+y_off)], fill=stamp_color, width=2)

    # Star / asterisk in top arc
    for angle in range(0, 360, 45):
        rad = math.radians(angle)
        sx  = int(cx + (r - 7) * math.cos(rad))
        sy  = int(cy + (r - 7) * math.sin(rad))
        draw.ellipse([sx-1, sy-1, sx+1, sy+1], fill=stamp_color)

    # Distress effect: random pixel erasure
    arr = np.array(img)
    mask = rng.random((H, W)) < 0.12
    arr[mask, 3] = (arr[mask, 3] * 0.25).astype(np.uint8)
    img = Image.fromarray(arr, "RGBA")

    save(img, "decorations/stamp_retro_01.png")


# ──────────────────────────────────────────────────────────────────────────────
# FONTS README
# ──────────────────────────────────────────────────────────────────────────────

def gen_fonts_readme():
    readme = (
        "Font files for Frameit MVP\n"
        "==========================\n\n"
        "Place the following free Google Fonts TTF files in this directory:\n\n"
        "  Caveat-Regular.ttf\n"
        "    Download: https://fonts.google.com/specimen/Caveat\n"
        "    Used for: caption zones, text label decorations\n\n"
        "  CourierPrime-Regular.ttf\n"
        "    Download: https://fonts.google.com/specimen/Courier+Prime\n"
        "    Used for: film strip frame numbers\n\n"
        "The renderer falls back to PIL's built-in bitmap font if these files\n"
        "are absent. Captions will still render but look basic.\n\n"
        "Quick download with curl:\n"
        "  curl -L 'https://fonts.gstatic.com/s/caveat/v18/WnznHAc5bAfYB2QRah7pcpNvOx-pjfJ9eIWpZA.woff2' "
        "   -- (woff2 only; for TTF visit the Google Fonts page above)\n"
    )
    (ASSETS / "fonts" / "README.txt").write_text(readme)
    print(f"  [WRITE] assets/fonts/README.txt")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating Frameit MVP assets...\n")

    print("── frames ──")
    gen_polaroid_border()
    gen_film_sprockets()
    gen_instant_border((255, 255, 255), "instant_white.png")
    gen_instant_border((255, 248, 238), "instant_cream.png")

    print("\n── textures ──")
    gen_paper_grain("paper_grain_01.png", intensity=0.055, scale=0.9)
    gen_paper_grain("paper_grain_02.png", intensity=0.085, scale=1.3)
    gen_noise_fine()

    print("\n── decorations ──")
    gen_tape_clear()
    gen_tape_washi_floral()
    gen_stamp_retro()

    print("\n── fonts ──")
    gen_fonts_readme()

    print(f"\n{'─'*50}")
    print("Assets written to: assets/")
    print("Total files generated:")
    total = sum(1 for p in ASSETS.rglob("*") if p.is_file())
    print(f"  {total} files across frames/, textures/, decorations/, fonts/")
    print("\nNext: place TTF fonts in assets/fonts/ (see README.txt)")


if __name__ == "__main__":
    main()