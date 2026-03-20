"""
modules/collage_renderer/renderer.py
======================================
CollageRenderer — the main compositor.

Takes an uploaded photo list + a resolved template dict and produces a
final PNG image at the requested canvas size.

Layer order (bottom → top)
--------------------------
    0.  Background         solid colour or texture overlay
    1.  Photo slots        sorted by z_index; each slot = photo + frame
    2.  Decorations        tape, stickers, stamps, text labels
    3.  Global colour filter  applied to the flattened canvas
    4.  Grain overlay      procedural film grain blended in overlay mode

How randomness is applied
--------------------------
The caller supplies an optional seed.  If None, a random seed is generated
and stored.  The same seed always produces the same output:
    1. The entire template is $param-resolved with a ParamResolver(seed).
    2. Slot positions are computed using the same seeded Rng.
    3. Decoration probability rolls use the same Rng.
    4. Grain uses the same Rng.

Because all random draws are made from a single sequential Rng, the order
of draws must not change between calls with the same seed.  Adding new
$param fields to templates is safe; changing the order of existing fields
in a way that changes the draw sequence would produce different output.

Output formats
--------------
    "post"   1080 × 1080 PNG
    "story"  1080 × 1920 PNG

Usage
-----
    renderer = CollageRenderer(assets_dir="assets")
    meta = renderer.render_to_file(
        image_paths=["uploads/abc/photo_00.jpg"],
        template=template_dict,          # already loaded from JSON
        output_path="output/result.png",
        fmt="post",
        seed=42,
    )
    print(meta["seed"], meta["canvas_size"])
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image

from modules.collage_renderer.compositor  import paste_rotated, blend_layer, crop_to_fit
from modules.collage_renderer.frames      import build_frame
from modules.collage_renderer.filters     import apply_filter_preset
from modules.collage_renderer.grain       import generate_grain_overlay
from modules.collage_renderer.decorations import render_decoration, Layer
from modules.template_engine.param_resolver  import ParamResolver
from modules.template_engine.slot_strategies import compute_slot_positions


# ── Canvas size registry ──────────────────────────────────────────────────────

CANVAS_SIZES: dict[str, tuple[int, int]] = {
    "post":  (1080, 1080),
    "story": (1080, 1920),
}


# ── Image loader (with EXIF orientation) ─────────────────────────────────────

def _load_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGBA")


# ── Background builder ────────────────────────────────────────────────────────

def _build_background(bg_cfg: dict, canvas_w: int, canvas_h: int,
                       assets_dir: Path, rng: random.Random) -> Image.Image:
    """
    Render the background layer.

    Supports:
        solid    — flat colour with optional lightness_jitter.
        texture  — solid colour + asset image blended on top.
        gradient — falls back to solid (gradient requires Cairo; out of scope for MVP).
        image    — falls back to solid.
    """
    # Resolve base colour
    color_cfg = bg_cfg.get("color", {"value": "#FFFFFF"})

    if isinstance(color_cfg, str):
        base_hex = color_cfg
        jitter   = 0
    elif isinstance(color_cfg, dict):
        base_hex = color_cfg.get("value", "#FFFFFF")
        jitter   = int(color_cfg.get("lightness_jitter", 0))
    else:
        base_hex = "#FFFFFF"
        jitter   = 0

    # Handle case where "value" itself was a $pick and resolved to a string
    if isinstance(base_hex, dict):
        base_hex = "#FFFFFF"

    base_hex = base_hex.lstrip("#")
    if len(base_hex) == 3:
        base_hex = "".join(c * 2 for c in base_hex)

    r = max(0, min(255, int(base_hex[0:2], 16) + jitter))
    g = max(0, min(255, int(base_hex[2:4], 16) + jitter))
    b = max(0, min(255, int(base_hex[4:6], 16) + jitter))

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (r, g, b, 255))

    if bg_cfg.get("type") == "texture":
        tex_cfg   = bg_cfg.get("texture", {})
        asset_key = tex_cfg.get("asset", "")
        opacity   = float(tex_cfg.get("opacity", 0.12))
        blend     = tex_cfg.get("blend_mode", "multiply")

        tex_path  = _find_asset(asset_key, assets_dir)
        if tex_path:
            tex_img = Image.open(tex_path).convert("RGBA").resize(
                (canvas_w, canvas_h), Image.LANCZOS
            )
            canvas = blend_layer(canvas, tex_img, blend, opacity)

    return canvas


def _find_asset(key: str, assets_dir: Path) -> Optional[Path]:
    candidates = [
        assets_dir / f"{key}.png",
        assets_dir / "textures"    / f"{key}.png",
        assets_dir / "decorations" / f"{key}.png",
        assets_dir / f"{key}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ── Main renderer ─────────────────────────────────────────────────────────────

class CollageRenderer:
    """
    Stateless renderer.  Instantiate once; call render() or render_to_file()
    for each collage.

    Args:
        assets_dir: Path to the assets/ directory.  Default: "assets".
    """

    def __init__(self, assets_dir: Optional[str | Path] = None):
        self.assets_dir = Path(assets_dir) if assets_dir else Path("assets")

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        image_paths: Sequence[str | Path],
        template:    dict | str | Path,
        fmt:         str = "post",
        seed:        Optional[int] = None,
        face_boxes:  Optional[list[Optional[tuple[int,int,int,int]]]] = None,
    ) -> Image.Image:
        """
        Composite a collage and return a PIL RGBA Image.

        Args:
            image_paths: Ordered list of accepted photo file paths.
            template:    Resolved template dict, or path to a .json file.
            fmt:         "post" (1080×1080) or "story" (1080×1920).
            seed:        RNG seed.  None → random seed chosen here.
            face_boxes:  Per-image face bounding boxes from the image analyser,
                         aligned with image_paths.  Used for focal crop.
                         Pass None to use centre-crop for all photos.

        Returns:
            RGBA PIL Image at the requested canvas size.
        """
        # ── Setup ────────────────────────────────────────────────────────────
        if isinstance(template, (str, Path)):
            with open(template) as f:
                template = json.load(f)

        if fmt not in CANVAS_SIZES:
            fmt = "post"
        canvas_w, canvas_h = CANVAS_SIZES[fmt]

        if seed is None:
            seed = random.randint(0, 2 ** 32 - 1)

        rng      = random.Random(seed)
        resolver = ParamResolver(rng, canvas_w, canvas_h)

        # Resolve all $param fields using the seeded Rng
        template = resolver.resolve(template)

        # Clamp photo list to template constraints
        max_photos   = template.get("constraints", {}).get("max_photos", 10)
        image_paths  = list(image_paths)[:max_photos]
        photo_count  = len(image_paths)
        face_boxes   = list(face_boxes or [None] * photo_count)[:photo_count]

        # ── Stage 0: Background ───────────────────────────────────────────────
        canvas = _build_background(
            template.get("background", {"type": "solid", "color": {"value": "#FFFFFF"}}),
            canvas_w, canvas_h, self.assets_dir, rng,
        )

        # ── Stage 1: Compute slot positions ───────────────────────────────────
        slots = compute_slot_positions(
            strategy    = template.get("slot_strategy", {"type": "grid"}),
            defaults    = template.get("slot_defaults", {}),
            photo_count = photo_count,
            canvas_w    = canvas_w,
            canvas_h    = canvas_h,
            rng         = rng,
        )

        # ── Stage 2: Render and paste photo slots (sorted by z_index) ─────────
        for slot in slots:   # already sorted ascending by slot_strategies.py
            idx = slot.get("photo_index", 0)
            if idx >= len(image_paths):
                continue

            photo = _load_image(image_paths[idx])

            # Crop photo to slot dimensions
            crop_cfg    = slot.get("crop", {"mode": "cover", "focal_point": "center"})
            focal_point = crop_cfg.get("focal_point", "center")
            face_box    = face_boxes[idx] if idx < len(face_boxes) else None

            photo_cropped = crop_to_fit(
                photo,
                target_w    = slot["width"],
                target_h    = slot["height"],
                focal_point = focal_point,
                face_box    = face_box,
            )

            # Per-slot colour filter (optional)
            slot_filter = slot.get("filter")
            if slot_filter and slot_filter != "none":
                photo_cropped = apply_filter_preset(photo_cropped, slot_filter)

            # Build frame
            frame_cfg = slot.get("frame", {"type": "none"})
            framed    = build_frame(photo_cropped, frame_cfg)

            # Paste onto canvas
            paste_rotated(
                canvas,
                framed,
                x        = slot["x"],
                y        = slot["y"],
                rotation = slot.get("rotation", 0.0),
                anchor   = "top_left",
                opacity  = 1.0,
            )

        # ── Stage 3: Decorations ──────────────────────────────────────────────
        dec_layers: list[Layer] = []
        for dec_cfg in template.get("decorations", []):
            layer = render_decoration(dec_cfg, slots, self.assets_dir, rng)
            if layer is not None:
                dec_layers.append(layer)

        dec_layers.sort(key=lambda l: l.z_index)
        for layer in dec_layers:
            paste_rotated(
                canvas,
                layer.image,
                x        = layer.x,
                y        = layer.y,
                rotation = layer.rotation,
                anchor   = "top_left",
                opacity  = layer.opacity,
            )

        # ── Stage 4: Global colour filter ─────────────────────────────────────
        global_preset = (
            template.get("filters", {})
                    .get("global", {})
                    .get("preset", "none")
        )
        if global_preset and global_preset != "none":
            canvas = apply_filter_preset(canvas, str(global_preset))

        # ── Stage 5: Grain overlay ────────────────────────────────────────────
        grain_intensity = float(template.get("grain", {}).get("intensity", 0.0))
        if grain_intensity > 0.0:
            grain  = generate_grain_overlay(canvas_w, canvas_h, grain_intensity, rng)
            canvas = blend_layer(canvas, grain, mode="overlay", opacity=0.55)

        return canvas

    def render_to_file(
        self,
        image_paths:     Sequence[str | Path],
        template:        dict | str | Path,
        output_path:     str | Path,
        fmt:             str          = "post",
        seed:            Optional[int] = None,
        quality:         int          = 95,
        face_boxes:      Optional[list] = None,
    ) -> dict:
        """
        Render and save the collage to disk.

        Args:
            output_path: Destination file path (.png or .jpg).
            quality:     JPEG quality (1–95).  Ignored for PNG.

        Returns:
            Metadata dict: {output_path, seed, format, canvas_size, image_count}.
        """
        # If seed not provided, generate one now so it's recorded in metadata
        if seed is None:
            seed = random.randint(0, 2 ** 32 - 1)

        t_start = time.monotonic()
        result  = self.render(image_paths, template, fmt=fmt, seed=seed,
                              face_boxes=face_boxes)
        render_ms = int((time.monotonic() - t_start) * 1000)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() in (".jpg", ".jpeg"):
            result.convert("RGB").save(str(output_path), "JPEG", quality=quality)
        else:
            result.save(str(output_path), "PNG")

        return {
            "output_path": str(output_path),
            "seed":        seed,
            "format":      fmt,
            "canvas_size": CANVAS_SIZES.get(fmt, (1080, 1080)),
            "image_count": len(list(image_paths)),
            "render_ms":   render_ms,
        }