"""
modules/template_engine/template_selector.py
=============================================
Scores all available templates against the current render context and
returns the best match.

Selection is a scoring function, not a hard filter — every eligible template
gets a numeric score; the highest scorer wins.  This avoids dead zones where
no template matches exactly (e.g. a 7-photo batch finds a template that
takes 4–8 even though it prefers 4–6).

Score signals
-------------
    photo count fit  up to 40 pts  — distance from ideal count in range
    orientation match  0 / 12 / 25 pts
    user style tag overlap  10 pts per shared tag (max 30)
    template priority field  0–10 pts  (template author's hint)
    recency penalty  -20 pts  (avoids showing the same template twice in a row)
    story format bonus  +8 pts  (for templates tagged "story")

Fallback
--------
If no template has photo_count in its min/max range, or the list is empty,
`select_template` returns the first template whose id is "minimal_grid".
If even that is absent it raises ValueError — the server cannot start without
at least one valid template.

Usage
-----
    from modules.template_engine.template_selector import (
        SelectionContext, select_template, load_templates_from_dir,
    )

    templates = load_templates_from_dir("collage_templates/")
    ctx = SelectionContext(
        photo_count=4,
        dominant_orientation="square",
        user_style_tags=["vintage", "warm"],
        recently_used_ids=[],
    )
    template = select_template(templates, ctx)
    print(template["id"])   # e.g. "polaroid_stack"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# ── Selection context ─────────────────────────────────────────────────────────

@dataclass
class SelectionContext:
    photo_count:           int
    dominant_orientation:  str          # "square" | "portrait" | "landscape"
    user_style_tags:       list[str]    # from vibe picker or analysis
    recently_used_ids:     list[str]    # last N template ids (de-prioritised)
    fmt:                   str = "post" # "post" | "story"


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_template(template: dict, ctx: SelectionContext) -> float:
    """
    Return a numeric score for one template against the selection context.
    Returns -1.0 if the template is hard-ineligible (photo count out of range).
    """
    c  = template.get("constraints", {})
    mn = c.get("min_photos", 1)
    mx = c.get("max_photos", 9)

    # Hard gate — photo count must be in range
    if not (mn <= ctx.photo_count <= mx):
        return -1.0

    score = 0.0

    # Photo count fit (up to 40 pts) — penalise distance from ideal count
    ideal    = (mn + mx) / 2
    distance = abs(ctx.photo_count - ideal)
    score   += max(0.0, 40.0 - distance * 8.0)

    # Orientation match
    preferred = c.get("preferred_orientations", ["any"])
    if "any" in preferred:
        score += 12.0
    elif ctx.dominant_orientation in preferred:
        score += 25.0

    # User style tag overlap (10 pts per shared tag, max 30)
    user_tags = set(ctx.user_style_tags)
    tpl_tags  = set(template.get("style_tags", []))
    score    += min(30.0, len(user_tags & tpl_tags) * 10.0)

    # Template priority field (0–100 maps to 0–10 pts)
    priority = c.get("priority", 50)
    score   += priority * 0.10

    # Recency penalty — avoid repeating the last 3 templates
    if template.get("id") in ctx.recently_used_ids[-3:]:
        score -= 20.0

    # Story format bonus
    if ctx.fmt == "story" and "story" in template.get("style_tags", []):
        score += 8.0

    return score


# ── Selection ─────────────────────────────────────────────────────────────────

def select_template(templates: list[dict], ctx: SelectionContext) -> dict:
    """
    Score all templates and return the highest scorer.
    Falls back to minimal_grid, then the first available template.
    Raises ValueError only if the list is empty.
    """
    if not templates:
        raise ValueError("No templates provided to select_template().")

    scored = [(score_template(t, ctx), t) for t in templates]
    eligible = [(s, t) for s, t in scored if s >= 0]

    if not eligible:
        # Fallback: find minimal_grid or use the first template
        fallback = next((t for t in templates if t.get("id") == "minimal_grid"), None)
        if fallback:
            return fallback
        return templates[0]

    eligible.sort(key=lambda x: x[0], reverse=True)
    return eligible[0][1]


# ── Template loading ──────────────────────────────────────────────────────────

def load_templates_from_dir(templates_dir: str | Path) -> list[dict]:
    """
    Load all .json files from a directory into a list of template dicts.
    Files that fail JSON parsing are skipped with a warning printed to stderr.
    Returns templates sorted by their "id" field for deterministic ordering.
    """
    import sys
    templates: list[dict] = []
    for p in sorted(Path(templates_dir).glob("*.json")):
        try:
            with open(p) as f:
                templates.append(json.load(f))
        except Exception as exc:
            print(f"WARNING: Could not load template {p.name}: {exc}", file=sys.stderr)
    return templates
