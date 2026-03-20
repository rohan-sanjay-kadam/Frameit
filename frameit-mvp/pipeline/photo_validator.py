"""
pipeline/photo_validator.py
===========================
Gate 1 of the pipeline.  Validates every uploaded photo before any ML or
rendering work runs.

Checks (in order — short-circuit on first failure per photo):
    1. Can PIL open and decode the file?         → CORRUPT
    2. Is the shorter side ≥ MIN_DIMENSION?      → TOO_SMALL
    3. Is the image sharp (Laplacian variance)?  → BLURRY
    4. Is it near-identical to an accepted photo?→ DUPLICATE  (uses CLIP embedding)
    5. Have we already accepted MAX_PHOTOS?      → OVERSIZED

Order matters:
    • CORRUPT check first — no other check can run on a broken file.
    • BLURRY before DUPLICATE — no point computing an embedding for a blurry photo.
    • DUPLICATE requires embeddings, so it runs last among per-photo checks.
    • OVERSIZED is a batch-level check, applied after all per-photo checks pass.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from pipeline.pipeline_types import PhotoRejectReason, PhotoValidation

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Thresholds (all overridable from PipelineConfig)
# ──────────────────────────────────────────────────────────────────────────────

MIN_DIMENSION       = 400     # px — shorter side must be at least this
BLUR_THRESHOLD      = 80.0    # Laplacian variance; below this → blurry
DUPLICATE_THRESHOLD = 0.97    # cosine similarity; at or above this → duplicate
MAX_PHOTOS          = 10      # hard cap on accepted photos per batch


# ──────────────────────────────────────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────────────────────────────────────

def check_openable(path: str | Path) -> tuple[bool, Image.Image | None, str]:
    """Try to fully open and decode the image. Returns (ok, image, error_msg)."""
    try:
        img = Image.open(path)
        img.verify()          # catches truncated / corrupt files
        img = Image.open(path)  # re-open — verify() consumes the file handle
        img.load()            # force decode of pixel data
        return True, img, ""
    except Exception as exc:
        return False, None, str(exc)


def check_resolution(img: Image.Image) -> tuple[bool, str]:
    """Reject images whose shorter side is below MIN_DIMENSION."""
    w, h = img.size
    short = min(w, h)
    if short < MIN_DIMENSION:
        return False, f"Shorter side {short}px < minimum {MIN_DIMENSION}px"
    return True, ""


def check_blur(img: Image.Image) -> tuple[bool, float]:
    """
    Estimate sharpness via the variance of the Laplacian operator.

    A sharp image has strong high-frequency edges → high Laplacian variance.
    A motion-blurred or out-of-focus image is low-pass → low variance.

    We use a pure-NumPy 4-neighbour Laplacian (no scipy needed):
        kernel = [ 0  1  0 ]
                 [ 1 -4  1 ]
                 [ 0  1  0 ]
    Applied by summing four shifted copies of the grey array.

    Returns (is_sharp, variance_value).
    """
    gray = np.array(img.convert("L"), dtype=np.float32)

    laplacian = (
          gray[1:-1, :-2]    # left neighbour
        + gray[1:-1, 2:]     # right neighbour
        + gray[:-2,  1:-1]   # top neighbour
        + gray[2:,   1:-1]   # bottom neighbour
        - 4.0 * gray[1:-1, 1:-1]
    )
    variance = float(laplacian.var())
    return variance >= BLUR_THRESHOLD, variance


def check_duplicate(
    embedding: np.ndarray,
    accepted_embeddings: list[np.ndarray],
) -> tuple[bool, float]:
    """
    Returns (is_duplicate, max_cosine_similarity).
    Compares the new embedding against all already-accepted embeddings.
    Embeddings must be L2-normalised unit vectors (cosine sim = dot product).
    """
    if not accepted_embeddings:
        return False, 0.0
    sims = [float(np.dot(embedding, ae)) for ae in accepted_embeddings]
    max_sim = max(sims)
    return max_sim >= DUPLICATE_THRESHOLD, max_sim


# ──────────────────────────────────────────────────────────────────────────────
# Lazy image-analyzer import (avoids circular imports and hard ML dep)
# ──────────────────────────────────────────────────────────────────────────────

def _get_embedding_fn(clip_backend: Any | None):
    """
    Return the get_image_embedding function from image_analyzer,
    or None if the module is unavailable.  Uses importlib to avoid a
    hard import-time dependency on the modules/ directory.
    """
    if clip_backend is None:
        return None
    try:
        _here = Path(__file__).parent.parent
        spec = importlib.util.spec_from_file_location(
            "_ia_pv",
            _here / "modules" / "image_analyzer" / "image_analyzer.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("_ia_pv", mod)
        spec.loader.exec_module(mod)
        return lambda img: mod.get_image_embedding(img, backend=clip_backend)
    except Exception as exc:
        log.warning("Could not load image_analyzer for duplicate detection: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Batch validator — public API
# ──────────────────────────────────────────────────────────────────────────────

def validate_photos(
    paths: Sequence[str | Path],
    clip_backend: Any | None = None,
    max_photos: int = MAX_PHOTOS,
    blur_threshold: float = BLUR_THRESHOLD,
    duplicate_threshold: float = DUPLICATE_THRESHOLD,
) -> list[PhotoValidation]:
    """
    Validate a batch of photo paths. Returns one PhotoValidation per input,
    in the same order.

    Args:
        paths:               Ordered list of photo file paths.
        clip_backend:        Optional CLIPBackend instance for duplicate detection.
                             Pass None to skip duplicate checking.
        max_photos:          Maximum number of photos to accept (extras → OVERSIZED).
        blur_threshold:      Laplacian variance threshold for sharpness.
        duplicate_threshold: Cosine similarity threshold for duplicate rejection.

    Returns:
        List[PhotoValidation] — one per input path.
    """
    results: list[PhotoValidation] = []
    accepted_count = 0
    accepted_embeddings: list[np.ndarray] = []

    embed_fn = _get_embedding_fn(clip_backend)

    for raw_path in paths:
        path = Path(raw_path)
        pv = PhotoValidation(path=str(path), accepted=False)

        # ── 1. Can PIL open it? ───────────────────────────────────────────
        ok, img, err = check_openable(path)
        if not ok:
            pv.reject_reason = PhotoRejectReason.CORRUPT
            pv.warning = f"Cannot open: {err}"
            log.warning("CORRUPT  %s — %s", path.name, err)
            results.append(pv)
            continue

        pv.width, pv.height = img.size

        # ── 2. Resolution ─────────────────────────────────────────────────
        ok, msg = check_resolution(img)
        if not ok:
            pv.reject_reason = PhotoRejectReason.TOO_SMALL
            pv.warning = msg
            log.warning("TOO_SMALL %s — %s", path.name, msg)
            results.append(pv)
            continue

        # ── 3. Blur ───────────────────────────────────────────────────────
        is_sharp, variance = check_blur(img)
        if not is_sharp:
            pv.reject_reason = PhotoRejectReason.BLURRY
            pv.warning = (
                f"Laplacian variance {variance:.1f} < threshold {blur_threshold}"
            )
            log.warning(
                "BLURRY   %s — variance=%.1f (threshold %.1f)",
                path.name, variance, blur_threshold,
            )
            results.append(pv)
            continue

        # ── 4. Duplicate (requires CLIP backend) ──────────────────────────
        if embed_fn is not None:
            try:
                embedding = embed_fn(img)
                is_dup, max_sim = check_duplicate(embedding, accepted_embeddings)
                if is_dup:
                    pv.reject_reason = PhotoRejectReason.DUPLICATE
                    pv.warning = (
                        f"Cosine similarity {max_sim:.3f} ≥ "
                        f"threshold {duplicate_threshold}"
                    )
                    log.warning(
                        "DUPLICATE %s — sim=%.3f", path.name, max_sim
                    )
                    results.append(pv)
                    continue
                accepted_embeddings.append(embedding)
            except Exception as exc:
                log.warning(
                    "Embedding failed for %s (%s) — skipping duplicate check",
                    path.name, exc,
                )

        # ── 5. Batch cap ──────────────────────────────────────────────────
        if accepted_count >= max_photos:
            pv.reject_reason = PhotoRejectReason.OVERSIZED
            pv.warning = (
                f"Batch already has {max_photos} photos (cap reached)"
            )
            log.warning("OVERSIZED %s — cap=%d", path.name, max_photos)
            results.append(pv)
            continue

        # ── Accepted ──────────────────────────────────────────────────────
        pv.accepted = True
        accepted_count += 1
        log.debug(
            "ACCEPTED  %s (%dx%d, blur_var=%.1f)",
            path.name, pv.width, pv.height, variance,
        )
        results.append(pv)

    n_accepted = sum(1 for r in results if r.accepted)
    n_rejected = len(results) - n_accepted
    log.info(
        "Validation complete — %d accepted, %d rejected (of %d)",
        n_accepted, n_rejected, len(results),
    )
    return results