"""
modules/image_analyzer/image_analyzer.py
=========================================
Extracts structured signals from uploaded photos to drive layout selection,
music recommendation, and caption generation.

Public API
----------
    get_image_embedding(image)   -> np.ndarray  (512-d unit vector)
    classify_mood(image)         -> MoodResult
    extract_colors(image)        -> ColorPalette
    detect_orientation(image)    -> OrientationResult
    classify_scenes(image)       -> list[str]
    analyze(image)               -> AnalysisResult       (full single-image pipeline)
    analyze_batch(images)        -> list[AnalysisResult]
    aggregate_results(results)   -> dict                 (collage-level summary)

CLIP backend selection (auto, in priority order)
-------------------------------------------------
    1. OpenCLIPBackend         -- pip install open-clip-torch
    2. TransformersCLIPBackend -- pip install transformers torch
    3. FallbackCLIPBackend     -- pure NumPy, zero extra deps (default for MVP)

How embeddings work
-------------------
CLIP maps images and text into the same 512-d space so cosine similarity is
meaningful cross-modally.  We use this for:
    - Zero-shot classification: compare image embedding to text prompt embeddings
    - Duplicate detection: cosine sim > 0.97 in photo_validator.py
    - Nearest-neighbour lookup (future): vector DB for similar collage retrieval
"""

from __future__ import annotations

import colorsys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from modules.image_analyzer.mood_taxonomy import MOOD_TAXONOMY, SCENE_LABELS


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MoodLabel:
    name:         str
    score:        float
    style_tags:   list[str] = field(default_factory=list)
    music_genres: list[str] = field(default_factory=list)
    energy:       str = "medium"   # "low" | "medium" | "high"


@dataclass
class MoodResult:
    primary:       MoodLabel
    secondary:     MoodLabel | None
    all_scores:    dict[str, float]
    raw_embedding: np.ndarray


@dataclass
class ColorSwatch:
    hex:    str
    rgb:    tuple[int, int, int]
    hsl:    tuple[float, float, float]
    weight: float
    role:   str = "accent"   # "dominant" | "accent" | "neutral" | "mid"


@dataclass
class ColorPalette:
    swatches:  list[ColorSwatch]
    is_warm:   bool
    is_dark:   bool
    dominant:  ColorSwatch


@dataclass
class OrientationResult:
    label:        str    # "portrait" | "landscape" | "square"
    width:        int
    height:       int
    aspect_ratio: float


@dataclass
class AnalysisResult:
    mood:        MoodResult
    palette:     ColorPalette
    orientation: OrientationResult
    embedding:   np.ndarray
    scene_tags:  list[str]
    image_path:  str = ""


# ---------------------------------------------------------------------------
# CLIP backend protocol
# ---------------------------------------------------------------------------

class CLIPBackend(Protocol):
    def encode_image(self, image: Image.Image) -> np.ndarray: ...
    def encode_text(self, texts: list[str])   -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Backend 1: open_clip (pip install open-clip-torch)
# ---------------------------------------------------------------------------

class OpenCLIPBackend:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        import open_clip  # type: ignore
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()

    def encode_image(self, image: Image.Image) -> np.ndarray:
        import torch
        tensor = self._preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
        return _normalise(features.squeeze(0).float().numpy())

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch
        tokens = self._tokenizer(texts)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
        return np.array([_normalise(row) for row in features.float().numpy()])


# ---------------------------------------------------------------------------
# Backend 2: HuggingFace transformers (pip install transformers torch)
# ---------------------------------------------------------------------------

class TransformersCLIPBackend:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor  # type: ignore
        self._model     = CLIPModel.from_pretrained(model_id)
        self._processor = CLIPProcessor.from_pretrained(model_id)
        self._model.eval()

    def encode_image(self, image: Image.Image) -> np.ndarray:
        import torch
        inputs = self._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
        return _normalise(features.squeeze(0).float().numpy())

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch
        inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        return np.array([_normalise(row) for row in features.float().numpy()])


# ---------------------------------------------------------------------------
# Backend 3: FallbackCLIPBackend (zero extra deps -- default for MVP)
# ---------------------------------------------------------------------------

class FallbackCLIPBackend:
    """
    Pure-NumPy stand-in when no CLIP library is installed.

    Image encoding: 512-d descriptor from colour histograms + spatial means.
    Text encoding:  deterministic hash-based vectors (same prompt = same vector).

    Mood/scene classification is NOT semantically accurate with this backend.
    Use OpenCLIPBackend or TransformersCLIPBackend in production.
    """

    EMBEDDING_DIM = 512

    def encode_image(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize((128, 128), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0  # (128, 128, 3)

        # Per-channel colour histograms (16 bins each = 48 values)
        hr, _ = np.histogram(arr[:, :, 0], bins=16, range=(0, 1))
        hg, _ = np.histogram(arr[:, :, 1], bins=16, range=(0, 1))
        hb, _ = np.histogram(arr[:, :, 2], bins=16, range=(0, 1))
        colour_feats = np.concatenate([hr, hg, hb]).astype(np.float32)

        # Luminance histogram (32 bins)
        lum = 0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2]
        lum_hist, _ = np.histogram(lum, bins=32, range=(0, 1))
        lum_feats = lum_hist.astype(np.float32)

        # Spatial quadrant means (4 quadrants x 3 channels = 12 values)
        h, w = arr.shape[:2]
        quads = [arr[:h//2, :w//2], arr[:h//2, w//2:],
                 arr[h//2:, :w//2], arr[h//2:, w//2:]]
        spatial_feats = np.array([q.mean(axis=(0,1)) for q in quads]).flatten()

        raw = np.concatenate([colour_feats, lum_feats, spatial_feats])
        padded = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        padded[:len(raw)] = raw
        return _normalise(padded)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vec = rng.standard_normal(self.EMBEDDING_DIM).astype(np.float32)
            vecs.append(_normalise(vec))
        return np.array(vecs)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _build_backend() -> CLIPBackend:
    try:
        return OpenCLIPBackend()
    except (ImportError, Exception):
        pass
    try:
        return TransformersCLIPBackend()
    except (ImportError, Exception):
        pass
    warnings.warn(
        "No CLIP library found. Using FallbackCLIPBackend -- "
        "mood/scene classification is not semantically accurate. "
        "Install open-clip-torch for production.",
        RuntimeWarning, stacklevel=2,
    )
    return FallbackCLIPBackend()


_BACKEND: CLIPBackend | None = None


def _get_backend() -> CLIPBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _build_backend()
    return _BACKEND


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def _rgb_to_hsl(r: int, g: int, b: int) -> tuple[float, float, float]:
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    return (h*360, s, l)


def _is_warm_hue(hue_deg: float) -> bool:
    return hue_deg <= 60 or hue_deg >= 300


def _load_image(source) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    img = Image.open(source)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_image_embedding(
    image,
    backend: CLIPBackend | None = None,
) -> np.ndarray:
    """
    Compute a 512-d L2-normalised embedding for an image.

    The embedding encodes visual semantics so cosine similarity is meaningful:
    similar images score close to 1.0, dissimilar images close to 0.0.

    Used for: zero-shot mood/scene classification, duplicate detection,
    and (future) nearest-neighbour collage retrieval.

    Returns: np.ndarray shape (512,) dtype float32, L2-normalised.
    """
    img = _load_image(image)
    be  = backend or _get_backend()
    return be.encode_image(img)


def classify_mood(
    image,
    embedding: np.ndarray | None = None,
    backend: CLIPBackend | None = None,
) -> MoodResult:
    """
    Zero-shot mood classification via CLIP.

    For each mood in MOOD_TAXONOMY we encode its 4 text prompts, compute
    cosine similarity to the image embedding, and average across prompts.
    The mood with the highest averaged score wins.

    Multi-prompt averaging is more robust than a single phrasing because
    no one phrase covers the full visual range of a concept.

    Args:
        image:     PIL Image or path (ignored if embedding is provided).
        embedding: Pre-computed embedding -- avoids a duplicate forward pass.
        backend:   Override the module-level CLIP backend.
    """
    be = backend or _get_backend()
    if embedding is None:
        embedding = get_image_embedding(image, backend=be)

    mood_scores: dict[str, float] = {}
    for mood_name, cfg in MOOD_TAXONOMY.items():
        text_embeddings = be.encode_text(cfg["prompts"])
        sims = [_cosine(embedding, te) for te in text_embeddings]
        mood_scores[mood_name] = float(np.mean(sims))

    ranked = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)

    def _make_label(name, score):
        c = MOOD_TAXONOMY[name]
        return MoodLabel(name=name, score=score,
                         style_tags=c["style_tags"],
                         music_genres=c["music_genres"],
                         energy=c["energy"])

    return MoodResult(
        primary=_make_label(*ranked[0]),
        secondary=_make_label(*ranked[1]) if len(ranked) > 1 else None,
        all_scores=dict(ranked),
        raw_embedding=embedding,
    )


def extract_colors(
    image,
    n_colors: int = 5,
    sample_size: int = 2000,
) -> ColorPalette:
    """
    Extract a dominant colour palette using k-means clustering.

    Near-white (lum > 0.92) and near-black (lum < 0.10) pixels are excluded
    from the k-means fit -- they would dominate a naive histogram and hide the
    actual palette.  They are returned as a single "neutral" swatch instead.

    Returns: ColorPalette with swatches sorted by weight descending.
    """
    img    = _load_image(image).resize((200, 200), Image.LANCZOS)
    arr    = np.array(img, dtype=np.float32)   # (200, 200, 3)
    pixels = arr.reshape(-1, 3)                 # (40000, 3)

    if len(pixels) > sample_size:
        idx    = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[idx]

    lum        = (0.2126*pixels[:,0] + 0.7152*pixels[:,1] + 0.0722*pixels[:,2]) / 255.0
    is_neutral = (lum < 0.10) | (lum > 0.92)
    colour_px  = pixels[~is_neutral]
    neutral_px = pixels[is_neutral]

    swatches: list[ColorSwatch] = []

    if len(colour_px) >= n_colors:
        km      = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
        km.fit(colour_px)
        centers = km.cluster_centers_.astype(int)
        counts  = np.bincount(km.labels_, minlength=n_colors)
        total   = len(pixels)

        for i in range(n_colors):
            r, g, b = int(centers[i,0]), int(centers[i,1]), int(centers[i,2])
            h, s, l = _rgb_to_hsl(r, g, b)
            swatches.append(ColorSwatch(
                hex=_rgb_to_hex(r, g, b), rgb=(r, g, b), hsl=(h, s, l),
                weight=float(counts[i]) / total,
                role="accent" if s > 0.4 else "mid",
            ))

    if len(neutral_px) > 0:
        nm     = neutral_px.mean(axis=0).astype(int)
        nr, ng, nb = int(nm[0]), int(nm[1]), int(nm[2])
        swatches.append(ColorSwatch(
            hex=_rgb_to_hex(nr, ng, nb), rgb=(nr, ng, nb),
            hsl=_rgb_to_hsl(nr, ng, nb),
            weight=float(len(neutral_px)) / len(pixels),
            role="neutral",
        ))

    swatches.sort(key=lambda s: s.weight, reverse=True)
    if swatches:
        swatches[0].role = "dominant"

    all_hsl = [s.hsl for s in swatches]
    avg_hue = float(np.mean([h for h,_,_ in all_hsl])) if all_hsl else 0.0
    avg_lum = float(np.mean([l for _,_,l in all_hsl])) if all_hsl else 0.5
    dominant = swatches[0] if swatches else ColorSwatch(
        "#808080", (128,128,128), (0.,0.,0.5), 1.0, "dominant"
    )

    return ColorPalette(swatches=swatches, is_warm=_is_warm_hue(avg_hue),
                        is_dark=avg_lum < 0.35, dominant=dominant)


def detect_orientation(image) -> OrientationResult:
    """
    Determine image orientation from pixel dimensions.  No ML needed.

    A +/-5% tolerance band around 1:1 is classified as "square" to avoid
    treating near-square crops as strongly portrait or landscape.
    """
    img  = _load_image(image)
    w, h = img.size
    ar   = w / h

    if 0.95 <= ar <= 1.05:
        label = "square"
    elif ar < 1.0:
        label = "portrait"
    else:
        label = "landscape"

    return OrientationResult(label=label, width=w, height=h, aspect_ratio=ar)


def classify_scenes(
    image,
    embedding: np.ndarray | None = None,
    backend: CLIPBackend | None = None,
    top_n: int = 3,
) -> list[str]:
    """
    Zero-shot scene tagging -- returns the top_n SCENE_LABELS most similar
    to the image embedding.

    Each label is turned into a prompt ("a photo taken at {scene}") and
    compared via cosine similarity.
    """
    be = backend or _get_backend()
    if embedding is None:
        embedding = get_image_embedding(image, backend=be)

    prompts = [f"a photo taken at {s}" for s in SCENE_LABELS]
    text_embeddings = be.encode_text(prompts)

    sims = [(_cosine(embedding, te), label)
            for te, label in zip(text_embeddings, SCENE_LABELS)]
    sims.sort(reverse=True)
    return [label for _, label in sims[:top_n]]


def analyze(
    image,
    image_path: str = "",
    backend: CLIPBackend | None = None,
) -> AnalysisResult:
    """
    Run the complete analysis pipeline on a single image.

    Computes the CLIP embedding once and reuses it for mood + scene
    classification -- no duplicate forward passes.

    Returns: AnalysisResult with all signals, ready for template selection,
    music recommendation, and the API response payload.
    """
    be  = backend or _get_backend()
    img = _load_image(image)

    embedding   = get_image_embedding(img, backend=be)
    mood        = classify_mood(img, embedding=embedding, backend=be)
    palette     = extract_colors(img)
    orientation = detect_orientation(img)
    scene_tags  = classify_scenes(img, embedding=embedding, backend=be)

    return AnalysisResult(
        mood=mood,
        palette=palette,
        orientation=orientation,
        embedding=embedding,
        scene_tags=scene_tags,
        image_path=str(image_path or getattr(image, "filename", "")),
    )


def analyze_batch(
    images: Sequence,
    backend: CLIPBackend | None = None,
) -> list[AnalysisResult]:
    """
    Analyse a list of images. Returns results in the same order as input.

    Sequential in the MVP.  With a real CLIP backend on a GPU, batch the
    encode_image() calls here for higher throughput.
    """
    be = backend or _get_backend()
    return [analyze(img, backend=be) for img in images]


def aggregate_results(results: list[AnalysisResult]) -> dict:
    """
    Combine per-image AnalysisResults into one collage-level summary dict.

    Aggregation rules:
        orientation  -- plurality vote
        primary_mood -- highest sum of scores across all images
        energy       -- maximum across images (one high-energy photo lifts the set)
        scene_tags   -- union of top-2 per image, deduplicated, max 5
        palette_hex  -- dominant hex per image, deduplicated, max 6
        is_warm/dark -- simple majority vote
    """
    if not results:
        return {}

    # Orientation: plurality vote
    counts: dict[str, int] = {}
    for r in results:
        o = r.orientation.label
        counts[o] = counts.get(o, 0) + 1
    dominant_orientation = max(counts, key=counts.get)  # type: ignore

    # Mood: sum scores across images
    mood_totals: dict[str, float] = {m: 0.0 for m in MOOD_TAXONOMY}
    for r in results:
        for mood_name, score in r.mood.all_scores.items():
            mood_totals[mood_name] = mood_totals.get(mood_name, 0.0) + score
    primary_mood_name = max(mood_totals, key=mood_totals.get)  # type: ignore
    primary_mood_cfg  = MOOD_TAXONOMY[primary_mood_name]

    # Scene tags: union of top-2 per image, deduplicated
    all_scenes: list[str] = []
    for r in results:
        all_scenes.extend(r.scene_tags[:2])
    unique_scenes = list(dict.fromkeys(all_scenes))[:5]

    # Palette hex: dominant swatch per image, deduplicated
    palette_hex = list(dict.fromkeys(r.palette.dominant.hex for r in results))[:6]

    # Energy: maximum
    _energy_rank = {"low": 0, "medium": 1, "high": 2}
    max_energy = max(
        results,
        key=lambda r: _energy_rank.get(r.mood.primary.energy, 0),
    ).mood.primary.energy

    return {
        "dominant_orientation": dominant_orientation,
        "primary_mood":         primary_mood_name,
        "style_tags":           primary_mood_cfg["style_tags"],
        "music_genres":         primary_mood_cfg["music_genres"],
        "energy":               max_energy,
        "scene_tags":           unique_scenes,
        "palette_hex":          palette_hex,
        "is_warm":              sum(r.palette.is_warm for r in results) > len(results)/2,
        "is_dark":              sum(r.palette.is_dark for r in results) > len(results)/2,
        "image_count":          len(results),
    }