"""
pipeline/pipeline.py
====================
Orchestrates the full photo → collage + music pipeline.

Stages (run sequentially, each isolated):
    1. VALIDATE   PhotoValidator gates corrupt / blurry / duplicate / oversized photos
    2. ANALYSE    ImageAnalyzer extracts mood, palette, orientation per photo
    3. AGGREGATE  Combine per-photo signals into one collage-level summary
    4. SELECT     TemplateSelector scores and picks the best template JSON
    5. RENDER     CollageRenderer composites the final PNG
    6. MUSIC      MusicRecommender fetches Spotify tracks for the detected mood

Failures degrade gracefully:
    • ANALYSE failure  → orientation-only fallback, PARTIAL status
    • MUSIC failure    → empty track list, PARTIAL status, warning added
    • SELECT failure   → first available template used, warning added
    • RENDER failure   → hard FAILED (broken image is worse than no image)

Usage:
    config = PipelineConfig(templates_dir="collage_templates", output_dir="output")
    pipeline = CollagePipeline(config)     # init once, reuse across requests
    result = pipeline.run(
        photo_paths=["uploads/abc/photo_00.jpg"],
        fmt="post",
        seed=42,
    )
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from pipeline.pipeline_types import (
    OutputFormat,
    PhotoRejectReason,
    PipelineResult,
    PipelineStatus,
)
from pipeline.photo_validator import validate_photos

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Paths
    templates_dir:        str = "collage_templates"
    output_dir:           str = "output"
    assets_dir:           Optional[str] = None

    # Spotify
    spotify_client_id:     Optional[str] = None
    spotify_client_secret: Optional[str] = None
    music_track_count:     int = 10
    music_market:          str = "US"

    # Validation
    max_photos:           int   = 10
    min_photos:           int   = 1
    blur_threshold:       float = 80.0
    duplicate_threshold:  float = 0.97

    # Analysis
    enable_clip:          bool = True   # False → orientation-only, no ML
    clip_backend_name:    str  = "fallback"  # "fallback" | "open_clip" | "transformers"

    # Rendering
    jpeg_quality:         int  = 95

    # Behaviour
    skip_music:           bool = False
    skip_music_on_error:  bool = True

    def spotify_available(self) -> bool:
        return bool(self.spotify_client_id and self.spotify_client_secret)


# ──────────────────────────────────────────────────────────────────────────────
# Lazy module loaders
# ──────────────────────────────────────────────────────────────────────────────
# Each module lives in modules/<name>/<name>.py as a flat file.
# We load them with importlib so the pipeline doesn't have a hard import-time
# dependency on optional ML packages (torch, open_clip, etc.).

_HERE = Path(__file__).parent.parent  # project root


def _load_module(cache_key: str, rel_path: str):
    if cache_key not in sys.modules:
        full = _HERE / rel_path
        spec = importlib.util.spec_from_file_location(cache_key, full)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[cache_key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[cache_key]


def _import_analyzer():
    m = _load_module("_ia", "modules/image_analyzer/image_analyzer.py")
    return m.analyze_batch, m.aggregate_results, m.FallbackCLIPBackend


def _import_selector():
    m = _load_module("_sel", "modules/template_engine/template_selector.py")
    return m.SelectionContext, m.select_template, m.load_templates_from_dir


def _import_renderer():
    return _load_module("_rend", "modules/collage_renderer/renderer.py").CollageRenderer


def _import_recommender():
    m = _load_module("_rec", "modules/music_recommender/music_recommender.py")
    return m.get_recommendations


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class CollagePipeline:
    """
    One instance per server process.  Initialise once; reuse across requests.
    The CLIP backend and loaded templates are held as instance state to avoid
    re-loading on every request.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Load and validate templates at startup so bad templates fail fast
        SelectionContext, select_template, load_templates_from_dir = _import_selector()
        self._select_template   = select_template
        self._SelectionContext  = SelectionContext
        self._templates         = load_templates_from_dir(config.templates_dir)
        if not self._templates:
            raise RuntimeError(
                f"No templates found in '{config.templates_dir}'. "
                "Add at least one .json template file."
            )
        log.info(
            "Pipeline ready — %d template(s): %s",
            len(self._templates),
            [t.get("id", "?") for t in self._templates],
        )

        # Renderer (stateless, but instantiated once to warm asset loading)
        CollageRenderer = _import_renderer()
        self._renderer = CollageRenderer(assets_dir=config.assets_dir)

        # CLIP backend (lazy — built on first analysis call)
        self._clip_backend = None
        self._clip_ready   = False

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        photo_paths: Sequence[str | Path],
        fmt: str | OutputFormat = OutputFormat.POST,
        seed: Optional[int] = None,
        output_filename: Optional[str] = None,
        user_style_tags: Optional[list[str]] = None,
        recently_used_template_ids: Optional[list[str]] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline end-to-end and return a PipelineResult.

        Args:
            photo_paths:                 Ordered list of photo file paths.
            fmt:                         "post" (1080×1080) or "story" (1080×1920).
            seed:                        RNG seed for reproducible renders.
            output_filename:             Override the output PNG filename.
            user_style_tags:             Explicit style preferences (e.g. ["vintage"]).
            recently_used_template_ids:  IDs to de-prioritise in template selection.
        """
        fmt = OutputFormat(fmt) if isinstance(fmt, str) else fmt
        result = PipelineResult(status=PipelineStatus.FAILED, output_format=fmt)
        timings: dict[str, int] = {}

        try:
            # Stage 1 — Validate
            t0 = _ms()
            accepted_paths = self._stage_validate(photo_paths, result)
            timings["validate_ms"] = _ms() - t0

            if not accepted_paths:
                result.add_error("No valid photos remain after validation.")
                result.timing_ms = timings
                return result

            if len(accepted_paths) < self.config.min_photos:
                result.add_error(
                    f"Only {len(accepted_paths)} valid photo(s); "
                    f"minimum required is {self.config.min_photos}."
                )
                result.timing_ms = timings
                return result

            # Stage 2 — Analyse
            t0 = _ms()
            analysis_results, analysis_summary = self._stage_analyse(
                accepted_paths, result
            )
            timings["analyse_ms"] = _ms() - t0
            result.analysis_summary = analysis_summary

            # Stage 3 — Select template
            t0 = _ms()
            template = self._stage_select(
                analysis_summary,
                accepted_paths,
                fmt=fmt,
                user_style_tags=user_style_tags or [],
                recently_used_ids=recently_used_template_ids or [],
                result=result,
            )
            timings["select_ms"] = _ms() - t0
            result.selected_template = template.get("id", "unknown")

            # Stage 4 — Render
            t0 = _ms()
            collage_path, used_seed = self._stage_render(
                accepted_paths, template, fmt, seed, output_filename, result
            )
            timings["render_ms"] = _ms() - t0
            result.collage_path = collage_path
            result.seed = used_seed

            # Stage 5 — Music
            t0 = _ms()
            self._stage_music(analysis_summary, result)
            timings["music_ms"] = _ms() - t0

            # Final status
            result.status = (
                PipelineStatus.PARTIAL if result.warnings
                else PipelineStatus.SUCCESS
            )

        except Exception as exc:
            log.exception("Pipeline: unhandled exception")
            result.add_error(f"Unexpected error: {exc}")
            result.status = PipelineStatus.FAILED

        result.timing_ms = timings
        _log_summary(result)
        return result

    # ── Stage implementations ─────────────────────────────────────────────────

    def _stage_validate(
        self,
        photo_paths: Sequence[str | Path],
        result: PipelineResult,
    ) -> list[str]:
        clip_be = self._get_clip_backend() if self.config.enable_clip else None

        validations = validate_photos(
            paths=photo_paths,
            clip_backend=clip_be,
            max_photos=self.config.max_photos,
            blur_threshold=self.config.blur_threshold,
            duplicate_threshold=self.config.duplicate_threshold,
        )
        result.photo_validations = validations

        accepted = []
        for v in validations:
            if v.accepted:
                accepted.append(v.path)
                result.accepted_photos.append(v.path)
            else:
                result.rejected_photos.append(v.path)
                reason = v.reject_reason.value if v.reject_reason else "unknown"
                result.add_warning(
                    f"Photo '{Path(v.path).name}' rejected ({reason}): {v.warning}"
                )

        n_oversized = sum(
            1 for v in validations
            if v.reject_reason == PhotoRejectReason.OVERSIZED
        )
        if n_oversized:
            result.add_warning(
                f"{n_oversized} photo(s) dropped: batch exceeds "
                f"the {self.config.max_photos}-photo limit."
            )

        log.info(
            "Validate: %d/%d accepted", len(accepted), len(list(photo_paths))
        )
        return accepted

    def _stage_analyse(
        self,
        accepted_paths: list[str],
        result: PipelineResult,
    ) -> tuple[list, dict]:
        if not self.config.enable_clip:
            summary = _orientation_only_summary(accepted_paths)
            result.add_warning(
                "CLIP analysis disabled — using orientation-only fallback."
            )
            return [], summary

        try:
            analyze_batch, aggregate_results, _ = _import_analyzer()
            be = self._get_clip_backend()
            per_image = analyze_batch(accepted_paths, backend=be)
            summary   = aggregate_results(per_image)
            summary["analysis_backend"] = type(be).__name__
            log.info("Analyse: mood=%s orient=%s",
                     summary.get("primary_mood"), summary.get("dominant_orientation"))
            return per_image, summary

        except Exception as exc:
            log.warning("Analyse stage failed (%s) — using fallback", exc)
            result.add_warning(
                f"Image analysis failed: {exc}. Using orientation-only fallback."
            )
            return [], _orientation_only_summary(accepted_paths)

    def _stage_select(
        self,
        analysis_summary: dict,
        accepted_paths: list[str],
        fmt: OutputFormat,
        user_style_tags: list[str],
        recently_used_ids: list[str],
        result: PipelineResult,
    ) -> dict:
        try:
            SelectionContext = self._SelectionContext
            ctx = SelectionContext(
                photo_count=len(accepted_paths),
                dominant_orientation=analysis_summary.get("dominant_orientation", "square"),
                user_style_tags=user_style_tags + analysis_summary.get("style_tags", []),
                recently_used_ids=recently_used_ids,
                fmt=fmt.value,
            )
            template = self._select_template(self._templates, ctx)
            log.info("Select: template=%s", template.get("id"))
            return template

        except Exception as exc:
            log.warning("Template selection failed (%s) — using first available", exc)
            result.add_warning(f"Template selection failed: {exc}. Using default.")
            return self._templates[0]

    def _stage_render(
        self,
        accepted_paths: list[str],
        template: dict,
        fmt: OutputFormat,
        seed: Optional[int],
        output_filename: Optional[str],
        result: PipelineResult,
    ) -> tuple[str, int]:
        if not output_filename:
            ts      = int(time.time())
            tpl_id  = template.get("id", "collage")
            output_filename = f"{tpl_id}_{fmt.value}_{ts}.png"

        output_path = str(Path(self.config.output_dir) / output_filename)

        meta = self._renderer.render_to_file(
            image_paths=accepted_paths,
            template=template,
            output_path=output_path,
            fmt=fmt.value,
            seed=seed,
            quality=self.config.jpeg_quality,
        )
        log.info(
            "Render: %s  seed=%d  %dms",
            output_path, meta["seed"], meta.get("render_ms", 0),
        )
        return output_path, meta["seed"]

    def _stage_music(
        self,
        analysis_summary: dict,
        result: PipelineResult,
    ) -> None:
        if self.config.skip_music:
            return

        if not self.config.spotify_available():
            result.add_warning(
                "Spotify credentials not configured — music recommendations skipped. "
                "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env to enable."
            )
            return

        try:
            get_recommendations = _import_recommender()
            mood = analysis_summary.get("primary_mood", "aesthetic")
            rec  = get_recommendations(
                mood=mood,
                limit=self.config.music_track_count,
                market=self.config.music_market,
                client_id=self.config.spotify_client_id,
                client_secret=self.config.spotify_client_secret,
            )
            result.music_tracks = [
                {
                    "id":          t.id,
                    "name":        t.name,
                    "artist":      t.artist,
                    "album":       t.album,
                    "preview_url": t.preview_url,
                    "spotify_url": t.spotify_url,
                    "popularity":  t.popularity,
                }
                for t in rec.tracks
            ]
            log.info("Music: %d tracks for mood '%s'", len(rec.tracks), mood)

        except Exception as exc:
            msg = f"Music recommendation failed: {exc}"
            log.warning(msg)
            if self.config.skip_music_on_error:
                result.add_warning(msg)
            else:
                raise

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_clip_backend(self):
        """Lazy-init the CLIP backend singleton."""
        if not self._clip_ready:
            try:
                _, _, FallbackCLIPBackend = _import_analyzer()
                name = self.config.clip_backend_name

                if name == "open_clip":
                    from modules.image_analyzer.backends.open_clip_backend import (
                        OpenCLIPBackend,
                    )
                    self._clip_backend = OpenCLIPBackend()
                    log.info("CLIP backend: open_clip")

                elif name == "transformers":
                    from modules.image_analyzer.backends.transformers_backend import (
                        TransformersCLIPBackend,
                    )
                    self._clip_backend = TransformersCLIPBackend()
                    log.info("CLIP backend: transformers")

                else:
                    self._clip_backend = FallbackCLIPBackend()
                    log.info("CLIP backend: fallback (colour-histogram)")

            except Exception as exc:
                log.warning("Could not load CLIP backend (%s) — using fallback", exc)
                _, _, FallbackCLIPBackend = _import_analyzer()
                self._clip_backend = FallbackCLIPBackend()

            self._clip_ready = True
        return self._clip_backend


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper — used by tests and CLI tools
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    photo_paths: Sequence[str | Path],
    fmt: str = "post",
    seed: Optional[int] = None,
    templates_dir: str = "collage_templates",
    output_dir: str = "output",
    spotify_client_id: Optional[str] = None,
    spotify_client_secret: Optional[str] = None,
    user_style_tags: Optional[list[str]] = None,
    **config_kwargs,
) -> PipelineResult:
    """
    One-call convenience wrapper for scripts and tests.
    For the API server, use CollagePipeline directly (avoids re-init per request).
    """
    config = PipelineConfig(
        templates_dir=templates_dir,
        output_dir=output_dir,
        spotify_client_id=spotify_client_id,
        spotify_client_secret=spotify_client_secret,
        **config_kwargs,
    )
    pipeline = CollagePipeline(config)
    return pipeline.run(
        photo_paths=photo_paths,
        fmt=fmt,
        seed=seed,
        user_style_tags=user_style_tags,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ms() -> int:
    return int(time.monotonic() * 1000)


def _orientation_only_summary(paths: list[str]) -> dict:
    """
    Minimal analysis from pixel dimensions alone — used when CLIP is disabled
    or the analysis stage fails.  Returns enough fields to drive template
    selection and provide a basic response payload.
    """
    from PIL import Image as _PILImage

    orientations = []
    for p in paths:
        try:
            img = _PILImage.open(p)
            w, h = img.size
            ar = w / h
            if 0.95 <= ar <= 1.05:
                orientations.append("square")
            elif ar < 1.0:
                orientations.append("portrait")
            else:
                orientations.append("landscape")
        except Exception:
            orientations.append("square")

    counts: dict[str, int] = {}
    for o in orientations:
        counts[o] = counts.get(o, 0) + 1
    dominant = max(counts, key=counts.get) if counts else "square"

    return {
        "dominant_orientation": dominant,
        "primary_mood":         "aesthetic",
        "style_tags":           ["clean", "minimal"],
        "music_genres":         ["indie-pop", "lo-fi"],
        "energy":               "medium",
        "scene_tags":           [],
        "palette_hex":          [],
        "is_warm":              False,
        "is_dark":              False,
        "image_count":          len(paths),
        "analysis_backend":     "orientation_only",
    }


def _log_summary(result: PipelineResult) -> None:
    log.info(
        "Pipeline %s | template=%s | photos=%d/%d | "
        "music=%d tracks | render=%dms",
        result.status.value,
        result.selected_template or "—",
        len(result.accepted_photos),
        len(result.accepted_photos) + len(result.rejected_photos),
        len(result.music_tracks),
        result.timing_ms.get("render_ms", 0),
    )
    for w in result.warnings:
        log.warning("  ⚠  %s", w)
    for e in result.errors:
        log.error("  ✗  %s", e)