"""
modules/template_engine/template_validator.py
==============================================
Two-pass template validation.

Pass 1 — Structural checks (no ML, no filesystem)
    Required fields present, types correct, enum values valid.

Pass 2 — Semantic checks
    $param ranges self-consistent (min <= max), slot IDs unique,
    decoration probability in [0,1].

Why two passes?
    Structural checks are cheap and catch author typos fast.
    Semantic checks need the full dict to be traversable, so they run after.
    Separating them gives clearer error messages.

Usage
-----
    from modules.template_engine.template_validator import TemplateValidator

    result = TemplateValidator().validate(raw_dict)
    if not result.valid:
        raise ValueError(str(result))

    # Or validate a whole directory (used in CI and on startup):
    results = validate_all_templates("collage_templates/")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_TOP_LEVEL = {
    "id", "name", "canvas", "constraints", "background",
    "slot_strategy", "slot_defaults",
}
VALID_FRAME_TYPES  = {"polaroid", "filmstrip", "border", "none"}
VALID_BG_TYPES     = {"solid", "texture", "gradient", "image"}
VALID_DEC_TYPES    = {"tape", "sticker", "stamp", "text_label", "date_stamp", "film_edge"}
VALID_SLOT_STRATS  = {"scattered", "horizontal_strip", "grid", "explicit"}
VALID_ORIENTATIONS = {"portrait", "landscape", "square", "any"}
VALID_ENERGIES     = {"low", "medium", "high"}


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  WARNING {w}")
        return "\n".join(lines) if lines else "  OK — no issues"


# ── Validator ─────────────────────────────────────────────────────────────────

class TemplateValidator:

    def validate(self, raw: dict) -> ValidationResult:
        r = ValidationResult()
        self._check_required_fields(raw, r)
        self._check_id_format(raw, r)
        self._check_canvas(raw, r)
        self._check_constraints(raw, r)
        self._check_background(raw, r)
        self._check_slot_strategy(raw, r)
        self._check_slot_defaults(raw, r)
        self._check_decorations(raw, r)
        self._check_param_ranges(raw, r)
        return r

    # ── Pass 1: structural ────────────────────────────────────────────────

    def _check_required_fields(self, raw: dict, r: ValidationResult) -> None:
        for key in REQUIRED_TOP_LEVEL:
            if key not in raw:
                r.errors.append(f"Missing required top-level field: '{key}'")
        if "version" not in raw:
            r.warnings.append("Missing 'version' field — recommended for template registry")

    def _check_id_format(self, raw: dict, r: ValidationResult) -> None:
        tid = raw.get("id", "")
        if tid and not tid.replace("_", "").replace("-", "").isalnum():
            r.errors.append(
                f"Template id '{tid}' must be alphanumeric with underscores/hyphens only"
            )

    def _check_canvas(self, raw: dict, r: ValidationResult) -> None:
        canvas = raw.get("canvas", {})
        for dim in ("width", "height"):
            v = canvas.get(dim)
            if v is None:
                r.errors.append(f"canvas.{dim} is required")
            elif isinstance(v, (int, float)) and v <= 0:
                r.errors.append(f"canvas.{dim} must be > 0, got {v}")

    def _check_constraints(self, raw: dict, r: ValidationResult) -> None:
        c  = raw.get("constraints", {})
        mn = c.get("min_photos", 1)
        mx = c.get("max_photos", 9)
        if isinstance(mn, (int, float)) and isinstance(mx, (int, float)):
            if mn > mx:
                r.errors.append(
                    f"constraints.min_photos ({mn}) > max_photos ({mx})"
                )
            if mn < 1:
                r.errors.append("constraints.min_photos must be >= 1")
            if mx > 20:
                r.warnings.append(
                    f"constraints.max_photos = {mx} is very high — may produce cluttered layouts"
                )
        for o in c.get("preferred_orientations", []):
            if o not in VALID_ORIENTATIONS:
                r.errors.append(
                    f"constraints.preferred_orientations: '{o}' is not valid. "
                    f"Must be one of {VALID_ORIENTATIONS}"
                )

    def _check_background(self, raw: dict, r: ValidationResult) -> None:
        bg = raw.get("background", {})
        bg_type = bg.get("type", "solid")
        if bg_type not in VALID_BG_TYPES:
            r.errors.append(
                f"background.type must be one of {VALID_BG_TYPES}, got '{bg_type}'"
            )

    def _check_slot_strategy(self, raw: dict, r: ValidationResult) -> None:
        strat  = raw.get("slot_strategy", {})
        s_type = strat.get("type")
        if s_type not in VALID_SLOT_STRATS:
            r.errors.append(
                f"slot_strategy.type must be one of {VALID_SLOT_STRATS}, got '{s_type}'"
            )

    def _check_slot_defaults(self, raw: dict, r: ValidationResult) -> None:
        sd    = raw.get("slot_defaults", {})
        frame = sd.get("frame", {})
        ft    = frame.get("type", "none")
        if ft not in VALID_FRAME_TYPES:
            r.errors.append(
                f"slot_defaults.frame.type must be one of {VALID_FRAME_TYPES}, got '{ft}'"
            )

    def _check_decorations(self, raw: dict, r: ValidationResult) -> None:
        seen_ids: set[str] = set()
        for i, dec in enumerate(raw.get("decorations", [])):
            did = dec.get("id", f"<unnamed #{i}>")
            if did in seen_ids:
                r.errors.append(f"Duplicate decoration id: '{did}'")
            seen_ids.add(did)

            dec_type = dec.get("type")
            if dec_type not in VALID_DEC_TYPES:
                r.errors.append(
                    f"decorations[{i}].type must be one of {VALID_DEC_TYPES}, "
                    f"got '{dec_type}'"
                )

            prob = dec.get("probability", 1.0)
            if isinstance(prob, (int, float)) and not 0.0 <= prob <= 1.0:
                r.errors.append(
                    f"decorations[{i}].probability must be in [0, 1], got {prob}"
                )

    # ── Pass 2: $param range checks ───────────────────────────────────────

    def _check_param_ranges(self, raw: dict, r: ValidationResult) -> None:
        self._walk(raw, "", r)

    def _walk(self, node: Any, path: str, r: ValidationResult) -> None:
        if isinstance(node, dict):
            if "$rand" in node:
                lo, hi = node["$rand"]
                if lo > hi:
                    r.errors.append(f"{path}.$rand: min ({lo}) > max ({hi})")
                return
            if "$rand_int" in node:
                lo, hi = node["$rand_int"]
                if lo > hi:
                    r.errors.append(f"{path}.$rand_int: min ({lo}) > max ({hi})")
                return
            if "$normal" in node:
                std = node["$normal"].get("std", 1)
                if isinstance(std, (int, float)) and std < 0:
                    r.errors.append(f"{path}.$normal.std must be >= 0, got {std}")
                return
            if "$pick" in node:
                choices = node["$pick"]
                weights = node.get("weights")
                if weights is not None and len(weights) != len(choices):
                    r.errors.append(
                        f"{path}.$pick: weights length ({len(weights)}) "
                        f"!= choices length ({len(choices)})"
                    )
                return
            for k, v in node.items():
                self._walk(v, f"{path}.{k}" if path else k, r)

        elif isinstance(node, list):
            for i, v in enumerate(node):
                self._walk(v, f"{path}[{i}]", r)


# ── Directory-level helper (used in CI and pipeline startup) ──────────────────

def validate_all_templates(
    templates_dir: str | Path,
) -> dict[str, ValidationResult]:
    """
    Validate every .json file in a directory.
    Returns {filename: ValidationResult}.  Used in CI and on server startup.
    """
    results: dict[str, ValidationResult] = {}
    validator = TemplateValidator()
    for p in sorted(Path(templates_dir).glob("*.json")):
        try:
            with open(p) as f:
                raw = json.load(f)
            results[p.name] = validator.validate(raw)
        except json.JSONDecodeError as e:
            vr = ValidationResult()
            vr.errors.append(f"JSON parse error: {e}")
            results[p.name] = vr
    return results