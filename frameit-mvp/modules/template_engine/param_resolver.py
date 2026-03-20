"""
modules/template_engine/param_resolver.py
==========================================
Resolves $param objects in template JSON using a seeded RNG.

The resolver walks the template dict recursively and replaces every
$param object with a concrete value before any rendering happens.
This means by the time the renderer receives a template, it is a plain
dict of numbers and strings — no special cases needed in rendering code.

Supported param types
---------------------
    { "$rand":     [min, max] }                   -> float in [min, max]
    { "$rand_int": [min, max] }                   -> int   in [min, max]
    { "$normal":   {"mean": m, "std": s} }        -> float, Gaussian sample
    { "$pick":     ["A","B","C"], "weights":[...] }  -> one element (weighted)
    { "$canvas_pct": <inner_param_or_float> }     -> resolved * short canvas side

Static scalars (int, float, str, bool, None) pass through unchanged.
Lists and plain dicts are recursed into.

Seeded RNG
----------
The caller supplies a seed (or None for random).  The same seed always
produces the same resolved template, which is how "regenerate with same
seed" works — pass the stored seed back and you get an identical collage.

Usage
-----
    from modules.template_engine.param_resolver import ParamResolver
    import random

    rng      = random.Random(42)
    resolver = ParamResolver(rng, canvas_width=1080, canvas_height=1080)
    resolved = resolver.resolve(raw_template_dict)
"""

from __future__ import annotations

import random
from typing import Any


class ParamResolver:
    """
    Recursively resolve all $param objects in a template dict.

    Args:
        rng:           A seeded random.Random instance.
        canvas_width:  Canvas pixel width (used by $canvas_pct).
        canvas_height: Canvas pixel height (used by $canvas_pct).
    """

    def __init__(
        self,
        rng: random.Random,
        canvas_width: int = 1080,
        canvas_height: int = 1080,
    ):
        self.rng           = rng
        self.canvas_width  = canvas_width
        self.canvas_height = canvas_height

    # ── Public entry point ────────────────────────────────────────────────

    def resolve(self, value: Any) -> Any:
        """
        Resolve a value — scalar, list, dict, or $param object.
        Called recursively; the top-level call is always the full template dict.
        """
        if isinstance(value, dict):
            return self._resolve_dict(value)
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        return value  # scalar — pass through unchanged

    # ── $param handlers ───────────────────────────────────────────────────

    def _resolve_dict(self, d: dict) -> Any:
        # Detect $param objects by their leading "$" key
        if "$rand" in d:
            lo, hi = d["$rand"]
            return self.rng.uniform(lo, hi)

        if "$rand_int" in d:
            lo, hi = d["$rand_int"]
            return self.rng.randint(int(lo), int(hi))

        if "$normal" in d:
            mean = d["$normal"]["mean"]
            std  = d["$normal"]["std"]
            return self.rng.gauss(mean, std)

        if "$pick" in d:
            choices = d["$pick"]
            weights = d.get("weights")
            return self.rng.choices(choices, weights=weights, k=1)[0]

        if "$canvas_pct" in d:
            # Resolve the inner value (may itself be a $param), then multiply
            pct   = self.resolve(d["$canvas_pct"])
            short = min(self.canvas_width, self.canvas_height)
            return int(float(pct) * short)

        # Plain dict — recurse into all values
        return {k: self.resolve(v) for k, v in d.items()}