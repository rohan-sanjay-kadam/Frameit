/**
 * src/api/types.ts
 * ================
 * TypeScript types that mirror the FastAPI Pydantic schemas in api/schemas.py.
 * Any field added to the backend schemas should be added here too.
 */

// ── Domain types ──────────────────────────────────────────────────────────────

export type Vibe =
  | "travel"
  | "party"
  | "aesthetic"
  | "romance"
  | "food"
  | "fitness"
  | "urban"
  | "family";

export type OutputFormat = "post" | "story";

// ── Spotify track ─────────────────────────────────────────────────────────────

export interface Track {
  id:          string;
  name:        string;
  artist:      string;
  album:       string;
  preview_url: string | null;   // 30-second preview MP3, null if unavailable
  spotify_url: string;
  popularity:  number;          // 0–100
}

// ── Upload ────────────────────────────────────────────────────────────────────

/** Response from POST /api/v1/upload */
export interface UploadResponse {
  session_id: string;       // UUID identifying the upload session
  photo_ids:  string[];     // ordered list of saved filenames on the server
  accepted:   number;
  rejected:   number;
  warnings:   string[];
}

// ── Generate ──────────────────────────────────────────────────────────────────

/** Body sent to POST /api/v1/generate */
export interface GenerateRequest {
  session_id: string;
  photo_ids:  string[];
  format:     OutputFormat;
  vibe?:      Vibe;
  seed?:      number;
}

/** Response from POST /api/v1/generate */
export interface GenerateResponse {
  status:               "success" | "partial";
  collage_filename:     string;   // filename in server's output/ directory
  collage_url:          string;   // /output/<filename> — served by static mount
  seed:                 number;   // RNG seed used (save to regenerate identically)
  format:               OutputFormat;
  detected_mood:        string;
  detected_orientation: string;
  energy:               string;
  scene_tags:           string[];
  palette_hex:          string[];
  selected_template:    string;
  music_tracks:         Track[];
  accepted_photos:      string[];
  rejected_photos:      string[];
  warnings:             string[];
  timing_ms:            Record<string, number>;
}

// ── Error ─────────────────────────────────────────────────────────────────────

/** Shape of FastAPI error responses */
export interface ApiErrorDetail {
  message:  string;
  errors:   string[];
  warnings: string[];
}

export interface ApiErrorResponse {
  error:   string;
  detail?: string | ApiErrorDetail;
}