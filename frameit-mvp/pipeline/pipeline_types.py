"""
pipeline/pipeline_types.py
==========================
Shared dataclasses and enums used by the pipeline and every module that
feeds into it.  No business logic here — only types.

Imported by:
    pipeline.py, photo_validator.py, api/routes/generate.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class OutputFormat(str, Enum):
    POST  = "post"    # 1080 × 1080
    STORY = "story"   # 1080 × 1920


class PipelineStatus(str, Enum):
    SUCCESS = "success"          # all stages completed, no warnings
    PARTIAL = "partial"          # completed but with warnings / fallbacks used
    FAILED  = "failed"           # could not produce output


class PhotoRejectReason(str, Enum):
    CORRUPT    = "corrupt"       # PIL cannot open the file
    TOO_SMALL  = "too_small"     # shorter side < MIN_DIMENSION
    BLURRY     = "blurry"        # Laplacian variance below threshold
    DUPLICATE  = "duplicate"     # cosine similarity > DUPLICATE_THRESHOLD
    OVERSIZED  = "oversized"     # batch already hit MAX_PHOTOS cap


# ──────────────────────────────────────────────────────────────────────────────
# Per-photo validation result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PhotoValidation:
    path:          str
    accepted:      bool
    reject_reason: Optional[PhotoRejectReason] = None
    warning:       Optional[str]               = None
    width:         int                         = 0
    height:        int                         = 0


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline result — the final output of CollagePipeline.run()
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    status: PipelineStatus

    # Primary output
    collage_path:      Optional[str]  = None
    output_format:     OutputFormat   = OutputFormat.POST
    seed:              Optional[int]  = None

    # Module outputs
    analysis_summary:  Optional[dict] = None   # from aggregate_results()
    selected_template: Optional[str]  = None   # template id
    music_tracks:      list[dict]     = field(default_factory=list)

    # Validation detail
    photo_validations: list[PhotoValidation] = field(default_factory=list)
    accepted_photos:   list[str]             = field(default_factory=list)
    rejected_photos:   list[str]             = field(default_factory=list)

    # Diagnostics
    warnings:   list[str]        = field(default_factory=list)
    errors:     list[str]        = field(default_factory=list)
    timing_ms:  dict[str, int]   = field(default_factory=dict)
    metadata:   dict[str, Any]   = field(default_factory=dict)

    # ── Convenience mutators ──────────────────────────────────────────────

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def to_dict(self) -> dict:
        return {
            "status":            self.status.value,
            "collage_path":      self.collage_path,
            "output_format":     self.output_format.value,
            "seed":              self.seed,
            "selected_template": self.selected_template,
            "music_tracks":      self.music_tracks,
            "accepted_photos":   self.accepted_photos,
            "rejected_photos":   self.rejected_photos,
            "warnings":          self.warnings,
            "errors":            self.errors,
            "timing_ms":         self.timing_ms,
            "analysis_summary":  self.analysis_summary,
        }